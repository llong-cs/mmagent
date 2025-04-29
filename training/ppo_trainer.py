# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import re
import math
import json
import os
import textwrap
import time
import json
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Optional, Union, List
from qwen_omni_utils import process_mm_info
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback
from transformers.utils import is_peft_available
torch.set_printoptions(threshold=np.inf)
from ..core import masked_mean, masked_whiten
from ..models import create_reference_model
from ..models.utils import unwrap_model_for_generation
from .ppo_config import PPOConfig
from .utils import (
    OnlineTrainerState,
    disable_dropout_in_model,
    exact_div,
    generate_model_card,
    get_comet_experiment_url,
    log_table_to_comet_experiment,
    peft_module_casting_to_bf16,
    prepare_deepspeed,
    print_rich_table,
    selective_log_softmax,
    truncate_response,
    pad
)
import sys
from mmagent.retrieve import back_translate, translate
from mmagent.utils.chat_api import parallel_get_embedding, get_response_with_retry, generate_messages
from mmagent.utils.general import load_video_graph
from mmagent.retrieve import verify_qa
import mmagent.videograph
sys.modules["videograph"] = mmagent.videograph

def find_last_one_position(tensor):
    indices = (tensor == 1).nonzero(as_tuple=True)
    rows = indices[0]
    cols = indices[1]
    last_one_positions = [-1] * tensor.shape[0]
    for row, col in zip(rows, cols):
        row = row.item()
        last_one_positions[row] = col.item()
    return torch.tensor(last_one_positions)

def eval_answer(question, predict, ground_truth):
    if predict == "":
        return False
    response = verify_qa(question, ground_truth, predict).lower()
    return True if "yes" in response else False

def search(query, video_graph, history_clip=set(), threshold=0.05):
    model = "text-embedding-3-large"
    queries = back_translate(video_graph, [query])
    query_embedding = parallel_get_embedding(model, queries)[0]
    nodes = video_graph.search_text_nodes(query_embedding, threshold=threshold)
    nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
    resp = f'Search results of query "{query}":\n\n'
    resp_len = len(resp)
    _clip = set()
    raw_data = list()
    for node in nodes:
        if len(_clip) == 5:
            break
        node_id = node[0]
        node_score = node[1]
        clip_id = video_graph.nodes[node_id].metadata['timestamp']
        if clip_id in _clip:
            continue
        _clip.add(clip_id)
        if clip_id in history_clip:
            continue
        history_clip.add(clip_id)
        clip_node_id = video_graph.text_nodes_by_clip[clip_id]
        clip_node_id = sorted(clip_node_id)
        
        content = translate(video_graph, [video_graph.nodes[_node_id].metadata['contents'][0] for _node_id in clip_node_id])
        text = '\n'.join(content)

        raw_data.append({'clip_id': 'clip_' + str(clip_id), 'memory': content})
        
        resp = resp + 'ID=' + str(clip_id) + '\n' + text + '\n\n'
    if len(resp) < resp_len + 5:
        resp = resp + 'No results found.\n\n'
    return resp, history_clip, raw_data


if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

if is_wandb_available():
    import wandb

pattern = r"\[(.*)\](.*)"
INVALID_LOGPROB = 1.0

def get_reward(
    model: torch.nn.Module, query_responses: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lm_backbone = getattr(model, model.base_model_prefix)
    output = lm_backbone(
        inputs_embeds=query_responses,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
        use_cache=False
    )
    reward_logits = model.score(output.hidden_states[-1])
    return reward_logits


def forward(
    model: torch.nn.Module,
    query_responses: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.nn.Module:
    return model(
        inputs_embeds=query_responses,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )

def generate(
    lm_backbone: torch.nn.Module, queries: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor, generation_config: GenerationConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    output = lm_backbone.generate(
        inputs_embeds=queries,
        attention_mask=attention_mask,
        position_ids=position_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
    )
    logits = torch.stack(output.scores, 1)
    return output.sequences, logits

@torch.no_grad()
def batch_generation(
    model: torch.nn.Module,
    queries: torch.Tensor,
    attention_masks: torch.Tensor,
    position_ids: torch.Tensor,
    local_rollout_forward_batch_size: int,
    pad_token_id: int,
    generation_config: GenerationConfig,
):
    query_responses = []
    logitss = []
    batch_size = queries.shape[0]
    for i in range(0, batch_size, local_rollout_forward_batch_size):
        query = queries[i : i + local_rollout_forward_batch_size]
        attention_mask = attention_masks[i : i + local_rollout_forward_batch_size]
        position_id = position_ids[i : i + local_rollout_forward_batch_size]
        query_response, logits = generate(
            model,
            query,
            attention_mask,
            position_id,
            generation_config,
        )
        query_responses.append(query_response)
        logitss.append(logits)

    # padding tensors
    padded_query_responses = pad(query_responses, padding_value=pad_token_id, padding_side="right")
    padded_logitss = pad(logitss, padding_value=0, padding_side="right")

    # reshaping
    padded_query_responses = padded_query_responses.view(-1, padded_query_responses.shape[-1])[:batch_size]
    padded_logitss = padded_logitss.view(-1, *padded_logitss.shape[2:])[:batch_size]

    return padded_query_responses, padded_logitss

# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, value_model) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model
        self.critic_backbone = getattr(value_model, value_model.base_model_prefix)

    def forward(self, **kwargs):
        output = self.critic_backbone(**kwargs)
        logits = self.value_model.score(output.hidden_states[-1])
        return self.policy(**kwargs), logits


class PPOTrainer(Trainer):
    _tag_names = ["trl", "ppo"]

    def __init__(
        self,
        args: PPOConfig,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ],
        model: nn.Module,
        ref_model: Optional[nn.Module],
        train_dataset: Dataset,
        reward_model: Optional[nn.Module] = None,
        value_model: Optional[nn.Module] = None,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[list[TrainerCallback]] = None,
        peft_config: Optional["PeftConfig"] = None,
    ) -> None:
        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, you must make a copy of it, or `None` if you use peft."
            )

        self.args = args
        self.processing_class = processing_class
        self.policy_model = model
        self.value_model = value_model

        # Define the collator if not provided
        if data_collator is None:
            data_collator = DataCollatorWithPadding(self.processing_class)

        # Handle stop token settings: update policy model's generation_config to use provided stop token
        if args.stop_token and args.stop_token_id:
            raise ValueError("You cannot set both `stop_token` and `stop_token_id`.")
        elif args.stop_token:
            if args.stop_token == "eos":
                self.policy_model.generation_config.eos_token_id = self.stop_token_id = processing_class.eos_token_id
            else:
                raise ValueError(
                    f"Unknown `stop_token` {args.stop_token}. Allowed values are: `'eos'` and `None` (no stop token)."
                )
        else:
            self.policy_model.generation_config.eos_token_id = self.stop_token_id = args.stop_token_id  # None or int

        # peft support
        if not is_peft_available() and peft_config is not None:
            raise ImportError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_confg, we merge and unload it first
            if isinstance(self.policy_model, PeftModel):
                self.policy_model = self.policy_model.merge_and_unload()
            if isinstance(self.value_model, PeftModel):
                self.value_model = self.value_model.merge_and_unload()

            # get peft model with the given config
            self.policy_model = get_peft_model(self.policy_model, peft_config)
            self.value_model = get_peft_model(self.value_model, peft_config)
            if args.bf16 and getattr(self.policy_model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(self.policy_model)
            if args.bf16 and getattr(self.value_model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(self.value_model)

        self.is_peft_model = is_peft_available() and isinstance(self.policy_model, PeftModel)
        self.model_adapter_name = args.model_adapter_name
        self.ref_adapter_name = args.ref_adapter_name

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model:
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(self.policy_model)

        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47

        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        if args.whiten_rewards:
            assert args.local_mini_batch_size >= 8, (
                f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
            )
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        self.local_dataloader_batch_size = args.local_batch_size

        #########
        # setup model, optimizer, and others
        #########
        for module in [self.policy_model, self.ref_model, self.value_model]:
            if module is not None:
                disable_dropout_in_model(module)
        self.model = PolicyAndValueWrapper(self.policy_model, self.value_model)
        self.model.config = self.policy_model.config  # needed for pushing to hub
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # data = next(iter(self.dataloader))
        # print(data["inputs_embeds"].shape) # [bs, seq_len, hidden_size]
        # print(data["attention_mask"].shape) # [bs, seq_len]

        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:

            if self.ref_model is None:
                if not self.is_peft_model:
                    raise ValueError("No reference model and model is not a Peft model.")
            else:
                self.ref_model = prepare_deepspeed(
                    self.ref_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
        else:
            if self.ref_model is None:
                if not self.is_peft_model:
                    raise ValueError("No reference model and model is not a Peft model.")
            else:
                self.ref_model = self.ref_model.to(self.accelerator.device)

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with (
            self.accelerator.unwrap_model(self.model.policy).disable_adapter()
            if self.is_peft_model and not self.ref_adapter_name
            else nullcontext()
        ):
            if self.ref_adapter_name:
                self.model.policy.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.policy.set_adapter(self.model_adapter_name or "default")

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        backup_model = self.model
        self.model = self.model.policy  # save only the policy

        if self.is_deepspeed_enabled:
            backup_deepspeed = self.deepspeed
            self.deepspeed = self.model

        super().save_model(output_dir, _internal_call)

        self.model = backup_model

        if self.is_deepspeed_enabled:
            self.deepspeed = backup_deepspeed

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        # ref_policy = self.ref_model.cpu()
        ref_policy = self.ref_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=processing_class.pad_token_id,
            bos_token_id=processing_class.bos_token_id,
            eos_token_id=processing_class.eos_token_id,
            # use_cache=model.policy.config.text_config.use_cache
        )

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches * args.num_mini_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():
                conversations, answers, memories, actions, predict_answers, history_clips, complete, has_answer = [], [], [], [], [], [], [], []
                for i in data:
                    conversations.append(i["messages"])
                    answers.append(i["answer"])
                    memories.append(i["mem"])
                    actions.append(0)
                    predict_answers.append("")
                    complete.append("")
                    history_clips.append(set())
                    has_answer.append(False)
                for i in range(args.search_rounds):
                    add_generation_prompt = True if i == 0 else False
                    try:
                        text = processing_class.apply_chat_template(conversations, add_generation_prompt=add_generation_prompt, tokenize=False)
                        audios, images, videos = process_mm_info(conversations, use_audio_in_video=True)
                        inputs = processing_class(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
                    except:
                        print("tokenize error!!!!")
                        print(conversations)
                        print(text)
                        exit()
                    with unwrap_model_for_generation(
                        self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                    ) as unwrapped_model:
                        inputs = inputs.to(unwrapped_model.policy.device).to(unwrapped_model.policy.dtype)
                        text_ids = unwrapped_model.policy.generate(**inputs, generation_config=generation_config, use_audio_in_video=True)
                        output = processing_class.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        for idx in range(len(output)):
                            if len(conversations[idx]) == 1:
                                conversations[idx].append({
                                    "role": "assistant",
                                    "content": ""
                                })
                            cur_output = "<think>" + output[idx].split("<think>")[-1]
                            match_result = re.search(pattern, output[idx].split("</think>")[-1], re.DOTALL)
                            if match_result:
                                action = match_result.group(1)
                                action_content = match_result.group(2)
                            else:
                                action = "Search"
                                action_content = output[idx].split("</think>")[-1]
                            cur_output += "<|im_start|>"

                            if action == "Answer":
                                has_answer[idx] = True
                                if i < args.search_rounds - 1:
                                    if complete[idx] == "":
                                        complete[idx] = cur_output
                                        predict_answers[idx] = action_content
                                else:
                                    if complete[idx] == "":
                                        conversations[idx][1]["content"] += cur_output
                                    else:
                                        conversations[idx][1]["content"] += complete[idx]
                            else:
                                if complete[idx] != "":
                                    if i == args.search_rounds - 1:
                                        conversations[idx][1]["content"] += cur_output
                                    continue
                                _, history_clips[idx], raw_data = search(action_content, load_video_graph(memories[idx]), history_clips[idx])
                                conversations[idx][1]["content"] += cur_output + "Searched knowledge:\n" + json.dumps(raw_data, ensure_ascii=False).encode("utf-8", "ignore").decode("utf-8") + "\n"
                                actions[idx] += 1
                del cur_output, complete, history_clips
                
                text = processing_class.apply_chat_template(conversations, add_generation_prompt=False, tokenize=False)
                audios, images, videos = process_mm_info(conversations, use_audio_in_video=True)
                inputs = processing_class(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
                with unwrap_model_for_generation(
                    self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model:
                    inputs = inputs.to(unwrapped_model.policy.device).to(unwrapped_model.policy.dtype)
                    input_ids = inputs["input_ids"]
                    attention_masks = inputs["attention_mask"]
                    pixel_values_videos = inputs["pixel_values_videos"]
                    video_grid_thw = inputs["video_grid_thw"]
                    video_second_per_grid = inputs["video_second_per_grid"]
                    feature_attention_mask = inputs["feature_attention_mask"]
                    input_features = inputs["input_features"]

                    audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
                    input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
                    position_ids, _ = unwrapped_model.policy.get_rope_index(
                        input_ids,
                        None,
                        video_grid_thw,
                        attention_masks,
                        True,
                        audio_feature_lengths,
                        video_second_per_grid,
                    )

                    inputs_embeds = unwrapped_model.policy.get_input_embeddings()(input_ids)

                    # 2. Merge text , audios , image and video
                    if input_ids.shape[1] != 1:
                        if input_features is not None:
                            audio_feat_lengths, audio_output_lengths = unwrapped_model.policy.audio_tower._get_feat_extract_output_lengths(
                                audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
                            )
                            feature_lens = (
                                audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
                            )
                            audio_outputs = unwrapped_model.policy.audio_tower(
                                input_features,
                                feature_lens=feature_lens,
                                aftercnn_lens=audio_feat_lengths,
                            )
                            audio_features = audio_outputs.last_hidden_state
                            if audio_features.shape[0] != sum(audio_output_lengths.tolist()):
                                raise ValueError("length of audio_features should match audio_output_lengths")
                            audio_mask = (input_ids == unwrapped_model.policy.config.audio_token_index).unsqueeze(-1).expand_as(inputs_embeds)
                            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

                        if pixel_values_videos is not None:
                            pixel_values_videos = pixel_values_videos.type(unwrapped_model.policy.visual.get_dtype())
                            video_embeds = unwrapped_model.policy.visual(pixel_values_videos, grid_thw=video_grid_thw)
                            video_mask = (input_ids == unwrapped_model.policy.config.video_token_index).unsqueeze(-1).expand_as(inputs_embeds)
                            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

                        if attention_masks is not None:
                            attention_masks = attention_masks.to(inputs_embeds.device)

                del pixel_values_videos, video_grid_thw, video_second_per_grid

                query_responses = inputs_embeds.to(device)
                query_responses = query_responses[:, -args.max_tokens:]
                attention_masks = attention_masks.to(device)
                attention_masks = attention_masks[:, -args.max_tokens:]
                position_ids = position_ids.to(device)
                position_ids = position_ids[:, :, -args.max_tokens:]
                responses = input_ids.to(device)
                responses = responses[:, -args.max_tokens:]
                response_masks = torch.zeros_like(responses, dtype=torch.bool, device=device)
                # mask the model output
                for i in range(responses.shape[0]):
                    mask = False
                    for j in range(responses.shape[1]):
                        if responses[i][j] == 13708 and responses[i][j + 1] == 766:
                            mask = True
                            response_masks[i][j] = mask
                        elif responses[i][j] == 151644:
                            response_masks[i][j] = mask
                            mask = False
                        else:
                            response_masks[i][j] = mask
                
                logprobs, ref_logprobs, scores, values = [], [], [], []
                for i in range(0, query_responses.shape[0], args.local_rollout_forward_batch_size):
                    answer = answers[i : i + args.local_rollout_forward_batch_size] # [bs] str
                    predict_answer = predict_answers[i : i + args.local_rollout_forward_batch_size] # [bs] str
                    conversation = conversations[i : i + args.local_rollout_forward_batch_size]
                    action = actions[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size] # [bs, seq_len, hidden_states]
                    attention_mask = attention_masks[i : i + args.local_rollout_forward_batch_size] # [bs, seq_len]
                    position_id = position_ids[:, i : i + args.local_rollout_forward_batch_size]
                    response = responses[i : i + args.local_rollout_forward_batch_size] # [bs, seq_len] token_id
                    
                    unwrapped_policy_model = accelerator.unwrap_model(model).policy
                    policy_output = forward(unwrapped_policy_model, query_response, attention_mask, position_id)
                    logits = policy_output.logits
                    logits = torch.cat((logits[:, -1:], logits[:, :-1]), dim=1) # a trick operate to get the logits
                    logits /= args.temperature + 1e-7
                    logprob = selective_log_softmax(logits, response)
                    del policy_output, logits
                    torch.cuda.empty_cache()

                    if ref_policy is None:
                        with self.null_ref_context():
                            ref_output = forward(model.policy, query_response, attention_mask, position_id)
                    else:
                        ref_policy = ref_policy.to(self.accelerator.device)
                        ref_output = forward(ref_policy, query_response, attention_mask, position_id)
                        ref_policy = ref_policy.cpu()
                    ref_logits = ref_output.logits
                    ref_logits = torch.cat((ref_logits[:, -1:], ref_logits[:, :-1]), dim=1)
                    ref_logits /= args.temperature + 1e-7
                    ref_logprob = selective_log_softmax(ref_logits, response)
                    del ref_output, ref_logits
                    torch.cuda.empty_cache()

                    # # Response Processing 2. run reward model on the truncated responses
                    unwrapped_value_model = accelerator.unwrap_model(model).value_model
                    full_value = get_reward(unwrapped_value_model, query_response, attention_mask, position_id)
                    value = torch.cat((full_value[:, -1:], full_value[:, :-1]), dim=1).squeeze(-1)

                    score = []
                    for j in range(len(answer)):
                        if eval_answer(conversation[j][0]["content"][-1]["text"], predict_answer[j], answer[j]):
                            score.append(1)
                        else:
                            if has_answer[j]:
                                score.append(0)
                            else:
                                score.append(0 - args.action_cost * action[j])

                    score = torch.tensor(score, dtype=torch.float32, device=value.device)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    scores.append(score)
                    values.append(value)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                # sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                values = torch.cat(values, 0)
                del (logprob, ref_logprob, full_value, value, score, unwrapped_model)
                torch.cuda.empty_cache()
                gc.collect()

                # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
                # Completions not passing that filter will receive a lower score.
                if self.args.missing_eos_penalty is not None:
                    contain_eos_token = torch.any(responses == self.processing_class.eos_token_id, dim=-1)
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty
                    del contain_eos_token
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                # response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                # padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                # check padding_mask and attention_mask
                # assert attention_masks[:, context_length:] == ~padding_mask
                logprobs = torch.masked_fill(logprobs, ~response_masks, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, ~response_masks, INVALID_LOGPROB)
                # sequence_lengths_p1 = sequence_lengths + 1
                # response_masks_p1 = response_masks.clone()
                # response_masks_p1 = response_masks_p1[sequence_lengths + 1] = 1
                # assert torch.sum(response_masks_p1, dim=1) == torch.sum(response_masks, dim=1) + 1
                # padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                values = torch.masked_fill(values, ~response_masks, 0)

                # 4. compute rewards
                logr = ref_logprobs - logprobs
                kl = (logr.exp() - 1) - logr
                non_score_reward = -args.kl_coef * kl
                rewards = non_score_reward.clone()
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                # actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
                actual_end = find_last_one_position(response_masks)
                rewards[[actual_start, actual_end]] += scores
                # rewards[response_masks] += scores

                # 5. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(rewards, mask=response_masks, shift_mean=False)
                    rewards = torch.masked_fill(rewards, ~response_masks, 0)

                # 6. compute advantages and returns
                lastgaelam = 0
                advantages_reversed = []
                gen_length = responses.shape[1]
                for t in reversed(range(gen_length)):
                    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                advantages = torch.masked_fill(advantages, ~response_masks, 0)
                returns = advantages + values
                # if int(os.environ.get('LOCAL_RANK', 0)) == 0:
                #     print(advantages)
                #     print(values)
                #     print(returns)
                advantages = masked_whiten(advantages, response_masks)
                advantages = torch.masked_fill(advantages, ~response_masks, 0)
                torch.cuda.empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                            mb_advantage = advantages[micro_batch_inds]
                            mb_responses = responses[micro_batch_inds]
                            mb_attention_mask = attention_masks[micro_batch_inds]
                            mb_query_responses = query_responses[micro_batch_inds]
                            mb_position_ids = position_ids[:, micro_batch_inds]
                            mb_logprobs = logprobs[micro_batch_inds]
                            mb_return = returns[micro_batch_inds]
                            mb_values = values[micro_batch_inds]

                            output, vpred_temp = forward(model, mb_query_responses, mb_attention_mask, mb_position_ids)
                            logits = output.logits
                            logits = torch.cat((logits[:, -1:], logits[:, :-1]), dim=1)
                            logits /= args.temperature + 1e-7
                            new_logprobs = selective_log_softmax(logits, mb_responses)
                            new_logprobs = torch.masked_fill(
                                new_logprobs, ~response_masks[micro_batch_inds], INVALID_LOGPROB
                            )
                            vpred = torch.cat((vpred_temp[:, -1:], vpred_temp[:, :-1]), dim=1).squeeze(-1)
                            # vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                            vpred = torch.masked_fill(vpred, ~response_masks[micro_batch_inds], 0)
                            vpredclipped = torch.clamp(
                                vpred,
                                mb_values - args.cliprange_value,
                                mb_values + args.cliprange_value,
                            )
                            vf_losses1 = torch.square(vpred - mb_return)
                            vf_losses2 = torch.square(vpredclipped - mb_return)
                            vf_loss_max = torch.max(vf_losses1, vf_losses2)
                            vf_loss = 0.5 * masked_mean(vf_loss_max, response_masks[micro_batch_inds])
                            torch.distributed.barrier()
                            vf_clipfrac = masked_mean(
                                (vf_losses2 > vf_losses1).float(), response_masks[micro_batch_inds]
                            )
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = masked_mean(pg_loss_max, response_masks[micro_batch_inds])
                            loss = pg_loss + args.vf_coef * vf_loss
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                pg_clipfrac = masked_mean(
                                    (pg_losses2 > pg_losses).float(), response_masks[micro_batch_inds]
                                )
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac
                                )
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                                vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    vf_clipfrac
                                )
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, vpred_temp, logits, new_logprobs, vpred, vpredclipped,
                        vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max,
                        pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
                        mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                actions = torch.tensor(actions, dtype=torch.float32, device=self.accelerator.device).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.sum(1).mean()
                rlhf_reward = mean_non_score_reward + scores.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather_for_metrics(mean_kl).mean().item()
                metrics["objective/actions"] = self.accelerator.gather_for_metrics(actions).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather_for_metrics(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = (
                    self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
                )
                metrics["objective/rlhf_reward"] = self.accelerator.gather_for_metrics(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather_for_metrics(scores.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather_for_metrics(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather_for_metrics(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather_for_metrics(pg_loss_stats).mean().item()
                metrics["loss/value_avg"] = self.accelerator.gather_for_metrics(vf_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather_for_metrics(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather_for_metrics(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather_for_metrics(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather_for_metrics(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                self.log(metrics)

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores, metrics, non_score_reward
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)
                torch.cuda.empty_cache()
            del (
                query_responses,
                inputs_embeds,
                attention_masks,
                position_ids,
                responses,
                response_masks,
                logprobs,
                ref_logprobs,
                values,
                rewards,
                advantages,
                returns,
                actions,
                input_ids,
                # postprocessed_responses,
                # sequence_lengths,
                # contain_eos_token,
                # sequence_lengths_p1,
                # response_idxs,
                # padding_mask,
                # padding_mask_p1,
                # response_masks_p1,
                # actual_start,
                # actual_end,
            )
            torch.cuda.empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def generate_completions(self, sampling: bool = False):
        args = self.args
        processing_class = self.processing_class
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        table = defaultdict(list)
        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            for batch in self.eval_dataloader:
                query = batch["input_ids"]
                with torch.no_grad():
                    context_length = query.shape[1]
                    query_response, _ = batch_generation(
                        unwrapped_model.policy,
                        query,
                        query.shape[0],
                        processing_class.pad_token_id,
                        generation_config,
                    )
                    response = query_response[:, context_length:]
                    postprocessed_response = response
                    if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, response
                        )
                    table["query"].extend(
                        gather_object(processing_class.batch_decode(query, skip_special_tokens=True))
                    )
                    table["model response"].extend(
                        gather_object(processing_class.batch_decode(postprocessed_response))
                    )

                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    _, score, _ = get_reward(
                        self.reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                    )
                    table["score"].extend(self.accelerator.gather_for_metrics(score).float().cpu().numpy())

                if sampling:
                    break
        df = pd.DataFrame(table)

        if self.accelerator.is_main_process:
            print_rich_table(df.iloc[0 : 0 + 5])
            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            if "comet_ml" in args.report_to:
                log_table_to_comet_experiment(
                    name="completions.csv",
                    table=df,
                )

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent("""\
        @article{mziegler2019fine-tuning,
            title        = {{Fine-Tuning Language Models from Human Preferences}},
            author       = {Daniel M. Ziegler and Nisan Stiennon and Jeffrey Wu and Tom B. Brown and Alec Radford and Dario Amodei and Paul F. Christiano and Geoffrey Irving},
            year         = 2019,
            eprint       = {arXiv:1909.08593}
        }""")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="PPO",
            trainer_citation=citation,
            paper_title="Fine-Tuning Language Models from Human Preferences",
            paper_id="1909.08593",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
