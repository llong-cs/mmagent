sudo pip3 uninstall -y transformers
sudo pip3 install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
pip3 install accelerate==0.34.2 # https://github.com/huggingface/trl/issues/2377
pip3 install qwen-omni-utils==0.0.4
pip3 install flash-attn==2.6.3 --no-build-isolation
pip3 uninstall -y torch
pip3 uninstall -y torchvision
pip3 uninstall -y torchaudio
sudo pip3 install torch==2.6.0
sudo pip3 install torchvision==0.21.0
sudo pip3 install torchaudio==2.6.0
pip3 install peft==0.15.2
pip3 install moviepy==2.1.2
sudo pip3 uninstall -y httpx
sudo pip3 install httpx==0.23.0
sudo pip3 install bytedlaplace==0.9.26
pip3 install pydub==0.25.1
cd /mnt/bn/videonasi18n/longlin.kylin/mmagent && pip3 install -e .

sudo pip3 install trl==0.16.0 # other versions may have problems
sudo apt-get -y install ffmpeg # load audio in video(mp4)

sudo chmod 777 /usr/local/lib/python3.11/dist-packages/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py
sudo chmod 777 /usr/local/lib/python3.11/dist-packages/trl/trainer/ppo_trainer.py
sudo chmod 777 /usr/local/lib/python3.11/dist-packages/trl/trainer/grpo_trainer.py

sudo cp /mnt/hdfs/foundation/agent/heyc/qwen2_5_omni/modeling_qwen2_5_omni.py /usr/local/lib/python3.11/dist-packages/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py
sudo cp /mnt/hdfs/foundation/agent/heyc/qwen2_5_omni/ppo_trainer.py /usr/local/lib/python3.11/dist-packages/trl/trainer/ppo_trainer.py
sudo cp /mnt/hdfs/foundation/agent/heyc/qwen2_5_omni/grpo_trainer.py /usr/local/lib/python3.11/dist-packages/trl/trainer/grpo_trainer.py

# code /usr/local/lib/python3.11/dist-packages/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py
# code /usr/local/lib/python3.11/dist-packages/trl/trainer/ppo_trainer.py
# code /usr/local/lib/python3.11/dist-packages/trl/trainer/grpo_trainer.py

# /usr/local/lib/python3.11/dist-packages/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py
# 1231 cos = freqs.cos().to(torch.float)  # .type_as(tensor_)
# 1232 sin = freqs.sin().to(torch.float)  # .type_as(tensor_)
# ppo need
# 2197 # causal_mask = self._update_causal_mask(
# 2198 #     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
# 2199 # )
# 2200 causal_mask = attention_mask
# all need
# 2684         else:
# 2685             embeds_to_talker = inputs_embeds.clone()

# /usr/local/lib/python3.11/dist-packages/trl/trainer/ppo_trainer.py 替换为 ppo_trainer.py