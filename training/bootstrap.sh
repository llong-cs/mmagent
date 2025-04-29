#!/usr/bin/env bash

export PYTHONPATH=$PWD:$PYTHONPATH
set -e

sudo pip3 install transformers==4.51.0
pip3 uninstall -y torch
pip3 uninstall -y torchvision
pip3 uninstall -y torchaudio
sudo pip3 install torch==2.6.0
sudo pip3 install torchvision==0.21.0
sudo pip3 install torchaudio==2.6.0
cp -r /mnt/bn/videonasi18n/longlin.kylin/mmagent ./
cd mmagent && pip3 install -e . && cd ..

if [ -d "/usr/local/lib/python3.11/dist-packages/transformers/" ]; then
    sudo cp /opt/tiger/qwen2_5_omni/modeling_qwen2_5_omni.py /usr/local/lib/python3.11/dist-packages/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py
fi
if [ -d "/home/tiger/.local/lib/python3.11/site-packages/transformers/" ]; then
    sudo cp /opt/tiger/qwen2_5_omni/modeling_qwen2_5_omni.py /home/tiger/.local/lib/python3.11/site-packages/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py
fi
if [ -d "/usr/local/lib/python3.11/dist-packages/trl/" ]; then
    sudo cp /opt/tiger/qwen2_5_omni/ppo_trainer.py /usr/local/lib/python3.11/dist-packages/trl/trainer/ppo_trainer.py
    sudo cp /opt/tiger/qwen2_5_omni/grpo_trainer.py /usr/local/lib/python3.11/dist-packages/trl/trainer/grpo_trainer.py
fi
if [ -d "/home/tiger/.local/lib/python3.11/site-packages/trl/" ]; then
    sudo cp /opt/tiger/qwen2_5_omni/ppo_trainer.py /home/tiger/.local/lib/python3.11/site-packages/trl/trainer/ppo_trainer.py
    sudo cp /opt/tiger/qwen2_5_omni/grpo_trainer.py /home/tiger/.local/lib/python3.11/site-packages/trl/trainer/grpo_trainer.py
fi

args="${@:1}"

main_process_port=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)
num_processes=$((ARNOLD_WORKER_NUM * ARNOLD_WORKER_GPU))

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch \
    --config_file /opt/tiger/qwen2_5_omni/deepspeed_config/deepspeed_zero3.yaml \
    --main_process_port $main_process_port \
    --num_machines $ARNOLD_WORKER_NUM \
    --num_processes $num_processes \
    --machine_rank $ARNOLD_ID \
    --main_process_ip $ARNOLD_WORKER_0_HOST $args
