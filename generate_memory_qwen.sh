#!/bin/bash

# Define variables
node_num=16
node_per_machine=8
machine_idx=0

for cuda_id in $(seq 0 $((node_per_machine - 1))); do
    CUDA_VISIBLE_DEVICES=$cuda_id python3 generate_memory_qwen.py \
        --cuda_id $((cuda_id + machine_idx * node_per_machine)) \
        --node_num $node_num & \
done
wait
