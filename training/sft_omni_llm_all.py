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

import json
import os
# import random
from dataclasses import dataclass, field
from typing import Any

import requests
import torch
# import torch.nn.functional as F
import wandb
from datasets import load_dataset
from peft import LoraConfig
from qwen_omni_utils import process_mm_info
# from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, Qwen2VLProcessor
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration, AutoTokenizer
import torch
from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer, TrlParser

import numpy as np
torch.set_printoptions(threshold=np.inf)

def find_tensor_start(A, B):
    len_A = len(A)
    len_B = len(B)
    for i in range(len_B - len_A + 1):
        if torch.all(B[i:i + len_A] == A):
            return i + len_A
    return -1

def collate_fn(examples) -> dict[str, torch.Tensor]:
    """Collate batch of examples for training."""
    conversations = [example["messages"] for example in examples]
    text = processor.apply_chat_template(conversations, tokenize=False)
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=True)
    inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # ignore the input prompt
    ignore_end = torch.tensor([151644, 77091, 198]) # <|im_start|>assistant\n
    for i in range(len(labels)):
        start = find_tensor_start(ignore_end, labels[i])
        if start!= -1:
            labels[i][:start] = -100

    inputs["labels"] = labels
    
    return inputs

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
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    processor.tokenizer = tokenizer

    # Prepare dataset
    # prepared_dataset = torch.load(script_args.dataset_name)
    prepared_dataset = []
    with open(script_args.dataset_name) as f:
        for line in f.readlines():
            prepared_dataset.append(json.loads(line))

    # Initialize wandb if specified
    if training_args.report_to == "wandb":
        wandb.login(key="214125030792bd6cfd84015505ed93487f714a59")
        if int(os.environ.get('LOCAL_RANK', 0)) == 0:
            wandb.init(
                project="multimodal agent",
            )

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
