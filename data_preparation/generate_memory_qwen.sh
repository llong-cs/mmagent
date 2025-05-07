#!/bin/bash

# Define variables
machine_idx=${1:-0}
node_num=24
node_per_machine=8

# Calculate the starting cuda_id for this machine
start_cuda_id=$((machine_idx * node_per_machine))

for local_cuda_id in $(seq 0 $((node_per_machine - 1))); do
    global_cuda_id=$((start_cuda_id + local_cuda_id))
    CUDA_VISIBLE_DEVICES=$local_cuda_id python3 data_preparation/generate_memory_qwen.py \
        --cuda_id $global_cuda_id \
        --node_num $node_num \
        --preprocessing "" &
done
wait
