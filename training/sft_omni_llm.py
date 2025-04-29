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

import os
import torch
import wandb
from torch.utils.data import Dataset
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer, TrlParser
import numpy as np
torch.set_printoptions(threshold=np.inf)

def collate_fn(examples) -> dict[str, torch.Tensor]:
    """Collate batch of examples for training."""
    max_len = 0
    for example in examples:
        # print(example.shape)
        if example["inputs_embeds"].shape[0] > max_len:
            max_len = example["inputs_embeds"].shape[0]
    inputs_embeds, attention_mask, labels, position_ids = None, None, None, None

    for example in examples:
        if example["inputs_embeds"].shape[0] < max_len:
            pad_len = max_len - example["inputs_embeds"].shape[0]
            mask = torch.cat((torch.zeros(pad_len, dtype=torch.long, device=example["inputs_embeds"].device), torch.ones(example["inputs_embeds"].shape[0], dtype=torch.long, device=example["inputs_embeds"].device)), dim=0).unsqueeze(0)
            inputs_embed = torch.cat((torch.zeros(pad_len, example["inputs_embeds"].shape[1], dtype=example["inputs_embeds"].dtype, device=example["inputs_embeds"].device), example["inputs_embeds"]), dim=0).unsqueeze(0)
            label = torch.cat((torch.full((pad_len,), -100, dtype=torch.long, device=example["labels"].device), example["labels"]), dim=0).unsqueeze(0)
            position_id = torch.cat((torch.full((3, pad_len), 1, dtype=torch.long, device=example["position_id"].device), example["position_id"]), dim=1).unsqueeze(1)
        else:
            mask = torch.ones(example["inputs_embeds"].shape[0], dtype=torch.long, device=example["inputs_embeds"].device).unsqueeze(0)
            inputs_embed = example["inputs_embeds"].unsqueeze(0)
            label = example["labels"].unsqueeze(0)
            position_id = example["position_id"].unsqueeze(1)
        if inputs_embeds is None:
            inputs_embeds = inputs_embed
            attention_mask = mask
            labels = label
            position_ids = position_id
        else:
            inputs_embeds = torch.cat((inputs_embeds, inputs_embed), dim=0)
            attention_mask = torch.cat((attention_mask, mask), dim=0)
            labels = torch.cat((labels, label), dim=0)
            position_ids = torch.cat((position_ids, position_id), dim=1)
    assert labels.shape == attention_mask.shape
    print(inputs_embeds.shape)
    inputs = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "labels": labels
    }
    return inputs

class SFTDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

    def __len__(self):
        return len(os.listdir(self.data_path))

    def __getitem__(self, idx):
        return torch.load(os.path.join(self.data_path, f"{idx}.pt"))

if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Load dataset
    # dataset = load_dataset("json", data_files={"train": script_args.dataset_name})

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
    )

    # Configure model modules for gradients
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_reentrant = False
        model.enable_input_require_grads()

    processor = Qwen2_5OmniProcessor.from_pretrained(model_args.model_name_or_path)

    # Prepare dataset
    prepared_dataset = SFTDataset(script_args.dataset_name)

    # Initialize wandb if specified
    if training_args.report_to == "wandb":
        wandb.login(key="214125030792bd6cfd84015505ed93487f714a59")
        if int(os.environ.get('LOCAL_RANK', 0)) == 0:
            wandb.init(
                project="multimodal agent",
            )

    model.audio_tower = model.audio_tower.cpu()
    model.visual = model.visual.cpu()
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset,
        data_collator=collate_fn,
        processing_class=processor,
    )

    # Train model
    trainer.train()

    # Save final model
    trainer.save_model(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    wandb.finish()
