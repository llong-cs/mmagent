#!/bin/bash

ckpt_path=${1:-"/mnt/hdfs/foundation/longlin.kylin/mmagent/data/memgen_sft/0429/ckpts/checkpoint-600"}
val_path=${2:-"data/sft/memgen/0429/conversations/val.jsonl"}
output_dir=${3:-"data/sft/memgen/0429/evaluation"}
node_num=${4:-8}
val_num=${5:-200}

# for i in $(seq 0 $((node_num - 1))); do
#     CUDA_VISIBLE_DEVICES=$i python3 evaluation/sft_evaluation.py \
#         --ckpt_path ${ckpt_path} \
#         --val_path ${val_path} \
#         --output_dir ${output_dir} \
#         --generate \
#         --cuda_id $i & \
# done
# wait

python3 evaluation/sft_evaluation.py \
    --ckpt_path ${ckpt_path} \
    --val_path ${val_path} \
    --output_dir ${output_dir} \
    --val_num ${val_num}