#!/bin/bash

input_dir=${1:-"data/sft/memgen/0511/conversations"}
output_dir=${2:-"/mnt/hdfs/foundation/longlin.kylin/mmagent/data/memgen_sft/0511"}

for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i python3 data_preparation/prepare_memory_gen_sft_data.py \
        --conversations_dir ${input_dir} \
        --output_dir ${output_dir} \
        --cuda_id $i &
done
wait