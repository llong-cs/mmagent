#!/bin/bash

# Define variables
val_path="data/sft/memgen/0429/conversations/val.jsonl"
output_dir="data/sft/memgen/0429/evaluation"
node_num=8
val_num=200

# Define an array of checkpoint paths
ckpt_paths=(
    "/mnt/hdfs/foundation/longlin.kylin/mmagent/data/memgen_sft/0429/ckpts/checkpoint-3600"
)

# Loop through each checkpoint path
for ckpt_path in "${ckpt_paths[@]}"; do
    for i in $(seq 0 $((node_num - 1))); do
        CUDA_VISIBLE_DEVICES=$i python3 evaluation/sft_evaluation.py \
            --ckpt_path ${ckpt_path} \
            --val_path ${val_path} \
            --output_dir ${output_dir} \
            --generate \
            --cuda_id $i & \
    done
    wait

    python3 evaluation/sft_evaluation.py \
        --ckpt_path ${ckpt_path} \
        --val_path ${val_path} \
        --output_dir ${output_dir} \
        --val_num ${val_num}
done