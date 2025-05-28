#!/bin/bash

sample_rounds=1
output_dir="data/annotations/results"
ours_version="0511"
dataset="data/annotations/small_test.jsonl"

retrieval_model="gpt-4o-2024-11-20"
# retrieval_model="gemini-1.5-pro-002"

model="qwen"
python3 evaluation/qa.py --dataset ${dataset} --sample_rounds ${sample_rounds} --version ${ours_version} --model ${model} --retrieval_model ${retrieval_model} --output_dir ${output_dir} > logs/qa.out

# model="gemini"
# python3 evaluation/qa.py --dataset ${dataset} --sample_rounds ${sample_rounds} --version ${ours_version} --model ${model} --retrieval_model ${retrieval_model} --output_dir ${output_dir} > logs/qa.out
