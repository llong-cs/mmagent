#!/bin/bash

sample_rounds=1
output_dir="data/annotations/results"
ours_version="0511"
dataset="data/annotations/small_test.jsonl"
model="qwen"

python3 evaluation/qa.py --dataset ${dataset} --sample_rounds ${sample_rounds} --version ${ours_version} --model ${model} --output_dir ${output_dir} > logs/qa.out
