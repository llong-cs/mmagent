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

import shutil
import json
import wandb
import torch
import torch.nn as nn
from transformers import (
    HfArgumentParser,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerModel,
    Qwen2_5OmniPreTrainedModelForConditionalGeneration,
    Qwen2_5OmniThinkerConfig,
    Trainer
)
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from trl import (
    ModelConfig,
    PPOConfig,
    PPOTrainer,
    ScriptArguments,
    get_peft_config
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder, Qwen2_5OmniVisionEncoder
from transformers.utils import ModelOutput
from dataclasses import dataclass, field
import warnings
import sys, os
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
# sys.stderr = open(os.devnull, 'w')
# import logging
# logging.basicConfig(level=logging.ERROR)
import numpy as np
torch.set_printoptions(threshold=np.inf)

def get_last_one_position(tensor):
    # 查找所有值为 1 的元素的索引
    indices = (tensor == 1).nonzero(as_tuple=False)

    # 初始化一个长度为张量行数的数组，初始值为 -1
    last_positions = -torch.ones(tensor.shape[0], dtype=torch.long)

    for index in indices:
        row, col = index[0], index[1]
        last_positions[row] = col

    return last_positions

class Qwen2_5OmniThinkerForSequenceClassification(Qwen2_5OmniPreTrainedModelForConditionalGeneration):
    config_class = Qwen2_5OmniThinkerConfig
    _no_split_modules = ["Qwen2_5OmniAudioEncoder", "Qwen2_5OmniVisionEncoder"]

    def __init__(self, config: Qwen2_5OmniThinkerConfig):
        super().__init__(config)
        self.audio_tower = Qwen2_5OmniAudioEncoder._from_config(
            config.audio_config, attn_implementation=config._attn_implementation
        )

        self.visual = Qwen2_5OmniVisionEncoder._from_config(
            config.vision_config, attn_implementation=config._attn_implementation
        )

        self.vocab_size = config.text_config.vocab_size
        self.model = Qwen2_5OmniThinkerModel._from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.score = nn.Linear(config.text_config.hidden_size, 1, bias=False)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_audio_in_video: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        video_second_per_grid: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None
        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0

            else:
                batch_size, seq_length = input_ids.shape
                delta = (
                    cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
                )
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)
            embeds_to_talker = inputs_embeds.clone()

            # 2. Merge text , audios , image and video
            if input_ids.shape[1] != 1:
                if input_features is not None:
                    audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
                        audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
                    )
                    feature_lens = (
                        audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
                    )
                    audio_outputs = self.audio_tower(
                        input_features,
                        feature_lens=feature_lens,
                        aftercnn_lens=audio_feat_lengths,
                    )
                    audio_features = audio_outputs.last_hidden_state
                    if audio_features.shape[0] != sum(audio_output_lengths.tolist()):
                        raise ValueError("length of audio_features should match audio_output_lengths")
                    audio_mask = (input_ids == self.config.audio_token_index).unsqueeze(-1).expand_as(inputs_embeds)
                    audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)
                    embeds_to_talker = embeds_to_talker.masked_scatter(audio_mask, torch.zeros_like(audio_features))

                if pixel_values is not None:
                    pixel_values = pixel_values.type(self.visual.get_dtype())
                    image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                    image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1).expand_as(inputs_embeds)
                    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                    embeds_to_talker = embeds_to_talker.masked_scatter(image_mask, torch.zeros_like(image_embeds))

                if pixel_values_videos is not None:
                    pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                    video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                    video_mask = (input_ids == self.config.video_token_index).unsqueeze(-1).expand_as(inputs_embeds)
                    video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
                    embeds_to_talker = embeds_to_talker.masked_scatter(video_mask, torch.zeros_like(video_embeds))

                if attention_mask is not None:
                    attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.score(hidden_states)

        batch_size = inputs_embeds.shape[0]
        last_non_pad_token = get_last_one_position(attention_mask)
        last_non_pad_token = last_non_pad_token.to(logits.device)
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        if not return_dict:
            output = (pooled_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_features=None,
        feature_attention_mask=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            use_audio_in_video=use_audio_in_video,
            video_second_per_grid=video_second_per_grid,
            **kwargs,
        )

        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        # update attention_mask
        if getattr(outputs, "attention_mask", None) is not None:
            model_kwargs["attention_mask"] = outputs.attention_mask

        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )

        if getattr(outputs, "rope_deltas", None) is not None:
            model_kwargs["rope_deltas"] = outputs.rope_deltas

        return model_kwargs
    
def collate_fn(examples) -> dict[str, torch.Tensor]:
    return examples

@dataclass
class OmniPPOConfig(PPOConfig):
    search_rounds: int = field(
        default=5,
        metadata={"help": "Number for search rounds during rolling out."},
    )
    action_cost: float = field(
        default=0.1,
        metadata={"help": "Cost of each action."},
    )
    top_p: float = field(
        default=0.9,
        metadata={"help": "Cost of each action."},
    )
    top_k: float = field(
        default=30,
        metadata={"help": "Cost of each action."},
    )
    max_tokens: int = field(
        default=29000,
        metadata={"help": "Number for search rounds during rolling out."},
    )

class FixZero3CheckpointPPOTrainer(PPOTrainer):

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        backup_model = self.model
        self.model = self.model.policy  # save only the policy

        Trainer.save_model(self, output_dir, _internal_call)

        self.model = backup_model

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if self.is_deepspeed_enabled:
            state_dict = {name.removeprefix('policy.'): param for name, param in state_dict.items()
                          if name.startswith('policy.')}

        super()._save(output_dir, state_dict)

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, OmniPPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    if training_args.report_to == "wandb":
        wandb.login(key="214125030792bd6cfd84015505ed93487f714a59")
        if int(os.environ.get('LOCAL_RANK', 0)) == 0:
            wandb.init(
                project="multimodal-agent",
            )

    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    
    processor = Qwen2_5OmniProcessor.from_pretrained(model_args.model_name_or_path)
    processor.pad_token_id = 151643
    processor.bos_token_id = 151644
    processor.eos_token_id = 151644 # add bos at the end of each action
    peft_config = get_peft_config(model_args)

    value_model = Qwen2_5OmniThinkerForSequenceClassification.from_pretrained(
        training_args.sft_model_path, 
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        num_labels=1
    )
    del value_model.audio_tower
    del value_model.visual
    if training_args.gradient_checkpointing:
        value_model.gradient_checkpointing_enable()
        value_model.config.use_reentrant = False
        value_model.config.text_config.use_cache = False
        value_model.enable_input_require_grads()

    policy = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        training_args.sft_model_path,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
    )
    if training_args.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
        policy.config.use_reentrant = False
        policy.config.text_config.use_cache = False
        policy.enable_input_require_grads()

    if peft_config is None:
        ref_policy = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            training_args.sft_model_path,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
        )
        del ref_policy.audio_tower
        del ref_policy.visual
    else:
        peft_config.target_modules = ["q_proj", "v_proj"]
        ref_policy = None

    train_dataset = []
    with open(script_args.dataset_name) as f:
        for line in f.readlines():
            train_dataset.append(json.loads(line))
    trainer = FixZero3CheckpointPPOTrainer(
        args=training_args,
        processing_class=processor,
        model=policy,
        ref_model=ref_policy,
        value_model=value_model,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    wandb.finish()