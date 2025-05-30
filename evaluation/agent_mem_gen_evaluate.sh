#!/bin/bash

# Define variables
model="gemini-1.5-pro-002"
val_path="data/sft/memgen/0511/conversations/val.jsonl"
output_dir="data/sft/memgen/0511/baseline/evaluation"
val_num=200

CUDA_VISIBLE_DEVICES=0 python3 evaluation/baseline_memory_evaluation.py \
    --model ${model} \
    --val_path ${val_path} \
    --output_dir ${output_dir} \
    --val_num ${val_num}