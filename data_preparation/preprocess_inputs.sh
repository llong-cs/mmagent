#!/bin/bash

data_path="data/sft/memgen/0511/train_for_memory_6k.json"
output_dir="/mnt/hdfs/foundation/longlin.kylin/mmagent/data/memgen_sft/0511"

conversations_type="epi_then_sem_by_type"

CUDA_VISIBLE_DEVICES=0 python3 data_preparation/prepare_memory_gen_sft_data.py \
    --data_path ${data_path} \
    --conversations_type ${conversations_type} \
    --output_dir ${output_dir} \
    --prepare_conversations \
    --cuda_id 0

for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i python3 data_preparation/prepare_memory_gen_sft_data.py \
        --data_path ${data_path} \
        --conversations_type ${conversations_type} \
        --output_dir ${output_dir} \
        --cuda_id $i &
done
wait