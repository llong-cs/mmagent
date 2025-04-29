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

# https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb
import shutil
import json
import torch
import numpy as np
torch.set_printoptions(threshold=np.inf)
from transformers import (
    HfArgumentParser,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from dataclasses import dataclass, field
from trl import (
    ModelConfig,
    GRPOConfig,
    GRPOTrainer,
    ScriptArguments,
    get_peft_config
)
import warnings
import os
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

@dataclass
class OmniGRPOConfig(GRPOConfig):
    search_rounds: int = field(
        default=5,
        metadata={"help": "Number for search rounds during rolling out."},
    )
    action_cost: float = field(
        default=0.1,
        metadata={"help": "Cost of each action."},
    )
    max_tokens: int = field(
        default=29000,
        metadata={"help": "Number for search rounds during rolling out."},
    )

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, OmniGRPOConfig, ModelConfig))
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
    processor.eos_token_id = 151644
    peft_config = get_peft_config(model_args)

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
    )
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_reentrant = False
        model.enable_input_require_grads()
    
    if peft_config is None:
        ref_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
        )
        del ref_model.audio_tower
        del ref_model.visual
    else:
        peft_config.target_modules = ["q_proj", "v_proj"]
        ref_model = None

    train_dataset = []
    with open(script_args.dataset_name) as f:
        for line in f.readlines():
            train_dataset.append(json.loads(line))
    
    trainer = GRPOTrainer(
        args=training_args,
        processing_class=processor,
        model=model,
        ref_model=ref_model,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    trainer.train()

"""
ScriptArguments(dataset_name='/mnt/hdfs/foundation/agent/heyc/qwen2_5_omni/data/ppo_data.jsonl', dataset_config=None, dataset_train_split='train', dataset_test_split='test', gradient_checkpointing_use_reentrant=False, ignore_bias_buffers=False)
GRPOConfig(
_n_gpu=1,
accelerator_config={'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None, 'use_configured_state': False},
adafactor=False,
adam_beta1=0.9,
adam_beta2=0.99,
adam_epsilon=1e-08,
auto_find_batch_size=False,
average_tokens_across_devices=False,
batch_eval_metrics=False,
beta=0.04,
bf16=True,
bf16_full_eval=False,
cache_implementation=None,
data_seed=None,
dataloader_drop_last=False,
dataloader_num_workers=0,
dataloader_persistent_workers=False,
dataloader_pin_memory=True,
dataloader_prefetch_factor=None,
ddp_backend=None,
ddp_broadcast_buffers=None,
ddp_bucket_cap_mb=None,
ddp_find_unused_parameters=None,
ddp_timeout=1800,
debug=[],
deepspeed=None,
disable_tqdm=False,
dispatch_batches=None,
do_eval=False,
do_predict=False,
do_train=False,
ds3_gather_for_generation=True,
epsilon=0.2,
epsilon_high=None,
eval_accumulation_steps=None,
eval_delay=0,
eval_do_concat_batches=True,
eval_on_start=False,
eval_steps=None,
eval_strategy=IntervalStrategy.NO,
eval_use_gather_object=False,
evaluation_strategy=None,
fp16=False,
fp16_backend=auto,
fp16_full_eval=False,
fp16_opt_level=O1,
fsdp=[],
fsdp_config={'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False},
fsdp_min_num_params=0,
fsdp_transformer_layer_cls_to_wrap=None,
full_determinism=False,
gradient_accumulation_steps=1,
gradient_checkpointing=True,
gradient_checkpointing_kwargs=None,
greater_is_better=None,
group_by_length=False,
half_precision_backend=auto,
hub_always_push=False,
hub_model_id=None,
hub_private_repo=None,
hub_strategy=HubStrategy.EVERY_SAVE,
hub_token=<HUB_TOKEN>,
ignore_data_skip=False,
include_for_metrics=[],
include_inputs_for_metrics=False,
include_num_input_tokens_seen=False,
include_tokens_per_second=False,
jit_mode_eval=False,
label_names=None,
label_smoothing_factor=0.0,
learning_rate=1e-06,
length_column_name=length,
load_best_model_at_end=False,
local_rank=0,
log_completions=False,
log_level=passive,
log_level_replica=warning,
log_on_each_node=False,
logging_dir=grpo_output/runs/Apr22_09-06-50_n117-196-097,
logging_first_step=False,
logging_nan_inf_filter=True,
logging_steps=1.0,
logging_strategy=IntervalStrategy.STEPS,
lr_scheduler_kwargs={},
lr_scheduler_type=SchedulerType.COSINE,
max_completion_length=786,
max_grad_norm=0.1,
max_prompt_length=512,
max_steps=-1,
metric_for_best_model=None,
min_p=None,
model_init_kwargs=None,
mp_parameters=,
neftune_noise_alpha=None,
no_cuda=False,
num_generations=8,
num_iterations=1,
num_train_epochs=5.0,
optim=OptimizerNames.ADAMW_TORCH,
optim_args=None,
optim_target_modules=None,
output_dir=grpo_output,
overwrite_output_dir=False,
past_index=-1,
per_device_eval_batch_size=8,
per_device_train_batch_size=4,
prediction_loss_only=False,
push_to_hub=False,
push_to_hub_model_id=None,
push_to_hub_organization=None,
push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
ray_scope=last,
ref_model_mixup_alpha=0.6,
ref_model_sync_steps=512,
remove_unused_columns=False,
repetition_penalty=1.0,
report_to=['wandb'],
restore_callback_states_from_checkpoint=False,
resume_from_checkpoint=None,
reward_weights=None,
run_name=grpo_output,
save_on_each_node=False,
save_only_model=False,
save_safetensors=True,
save_steps=100,
save_strategy=SaveStrategy.STEPS,
save_total_limit=None,
scale_rewards=True,
seed=42,
skip_memory_metrics=True,
split_batches=None,
sync_ref_model=False,
temperature=0.9,
tf32=None,
top_k=50,
top_p=1.0,
torch_compile=False,
torch_compile_backend=None,
torch_compile_mode=None,
torch_empty_cache_steps=None,
torchdynamo=None,
tp_size=0,
tpu_metrics_debug=False,
tpu_num_cores=None,
use_cpu=False,
use_ipex=False,
use_legacy_prediction_loop=False,
use_liger_kernel=False,
use_mps_device=False,
use_vllm=False,
vllm_device=None,
vllm_dtype=None,
vllm_enable_prefix_caching=None,
vllm_gpu_memory_utilization=None,
vllm_guided_decoding_regex=None,
vllm_max_model_len=None,
vllm_server_host=0.0.0.0,
vllm_server_port=8000,
vllm_server_timeout=120.0,
warmup_ratio=0.1,
warmup_steps=0,
weight_decay=0.1,
)
ModelConfig(model_name_or_path='/mnt/hdfs/foundation/agent/heyc/qwen2_5_omni/sft_output/checkpoint-230', model_revision='main', torch_dtype=None, trust_remote_code=False, attn_implementation=None, use_peft=False, lora_r=16, lora_alpha=32, lora_dropout=0.05, lora_target_modules=None, lora_modules_to_save=None, lora_task_type='CAUSAL_LM', use_rslora=False, use_dora=False, load_in_8bit=False, load_in_4bit=False, bnb_4bit_quant_type='nf4', use_bnb_nested_quant=False)
"""