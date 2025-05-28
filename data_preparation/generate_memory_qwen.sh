#!/bin/bash

# Define variables
data_list="/mnt/bn/videonasi18n/longlin.kylin/mmagent/data/annotations/small_test.jsonl"
machine_idx=$ARNOLD_ID
node_num=$((ARNOLD_WORKER_NUM * ARNOLD_WORKER_GPU))
node_per_machine=$ARNOLD_WORKER_GPU
version="0511"

# Calculate the starting cuda_id for this machine
start_cuda_id=$((machine_idx * node_per_machine))

for local_cuda_id in $(seq 0 $((node_per_machine - 1))); do
    global_cuda_id=$((start_cuda_id + local_cuda_id))
    CUDA_VISIBLE_DEVICES=$local_cuda_id python3 data_preparation/generate_memory_qwen.py \
        --data_list $data_list \
        --cuda_id $global_cuda_id \
        --node_num $node_num \
        --version $version \
        --preprocessing "" &
done
wait