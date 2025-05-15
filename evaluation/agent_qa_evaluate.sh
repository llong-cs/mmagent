#!/bin/bash

sample_rounds=1
output_dir="data/annotations/results"

dataset="data/annotations/small_test.jsonl"
python3 evaluation/qa.py --dataset ${dataset} --sample_rounds ${sample_rounds} --output_dir ${output_dir} > logs/qa.out

# dataset="data/annotations/small_test_qwen.jsonl"
# python3 evaluation/qa.py --dataset ${dataset} --sample_rounds ${sample_rounds} --output_dir ${output_dir} > logs/qa_qwen.out
