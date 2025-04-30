#!/bin/bash

mem_type=${1:-"episodic"}
input_dir=${2:-"data/sft/memgen/0429/conversations"}
output_dir=${2:-"/mnt/hdfs/foundation/longlin.kylin/mmagent/data/memgen_sft/0429"}

for i in {0..15}; do
    CUDA_VISIBLE_DEVICES=$i python preprocess_inputs.py \
        --conversations_path ${input_dir}/${mem_type}_conversations.jsonl \
        --output_dir ${output_dir} \
        --memory_type ${mem_type} \
        --cuda_id $i &
done
wait