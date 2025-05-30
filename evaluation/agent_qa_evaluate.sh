#!/bin/bash

# Default values
sample_rounds=${1:-1}
output_dir=${2:-"data/annotations/results"}
ours_version=${3:-"0511"}
dataset=${4:-"data/annotations/small_test.jsonl"}
retrieval_model=${5:-"gpt-4o-2024-11-20"}
model=${6:-"qwen"}

python3 evaluation/qa.py \
    --dataset ${dataset} \
    --sample_rounds ${sample_rounds} \
    --version ${ours_version} \
    --model ${model} \
    --retrieval_model ${retrieval_model} \
    --output_dir ${output_dir} > logs/qa.out

# Example usage:
# bash evaluation/agent_qa_evaluate.sh 1 "data/annotations/results" "0511" "data/annotations/small_test.jsonl" "gpt-4o-2024-11-20" "qwen"
