#!/bin/bash

# Default values
default_data_path="data/sft/memgen/0511/train_for_memory_6k.json"
default_output_dir="/mnt/hdfs/foundation/longlin.kylin/mmagent/data/memgen_sft/0511"
default_conversations_type="epi_then_sem"
default_mode="train"

# Parse command line arguments
data_path="${1:-$default_data_path}"
output_dir="${2:-$default_output_dir}"
conversations_type="${3:-$default_conversations_type}"
mode="${4:-$default_mode}"
num_gpus=$((ARNOLD_WORKER_NUM * ARNOLD_WORKER_GPU))
num_gpus_per_node=$ARNOLD_WORKER_GPU
nodes_id=$ARNOLD_ID

# Uncomment the following lines if you want to run the prepare_conversations step
# CUDA_VISIBLE_DEVICES=0 python3 data_preparation/prepare_memory_gen_sft_data.py \
#     --data_path ${data_path} \
#     --conversations_type ${conversations_type} \
#     --output_dir ${output_dir} \
#     --prepare_conversations \
#     --cuda_id 0

for local_cuda_id in $(seq 0 $((num_gpus_per_node - 1))); do
    global_cuda_id=$((nodes_id * num_gpus_per_machine + local_cuda_id))
    CUDA_VISIBLE_DEVICES=$local_cuda_id python3 data_preparation/prepare_memory_gen_sft_data.py \
        --data_path ${data_path} \
        --conversations_type ${conversations_type} \
        --output_dir ${output_dir} \
        --mode ${mode} \
        --num_gpus ${num_gpus} \
        --cuda_id $global_cuda_id &
done
wait

# an example to run the script
# bash data_preparation/preprocess_inputs.sh data/sft/memgen/0511/train_for_memory_6k.json /mnt/hdfs/foundation/longlin.kylin/mmagent/data/memgen_sft/0511 epi_then_sem train