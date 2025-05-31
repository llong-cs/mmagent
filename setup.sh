pip3 install -r requirements.txt
pip3 install -e .

sudo pip3 install setuptools_scm torchdiffeq resampy x_transformers
pip3 install accelerate==0.34.2 # https://github.com/huggingface/trl/issues/2377
pip3 install qwen-omni-utils==0.0.4
sudo pip3 install ninja
sudo apt-get -y install ninja-build

# git clone -b qwen2_omni_public https://github.com/fyabc/vllm.git
# cd vllm
# git checkout de8f43fbe9428b14d31ac5ec45d065cd3e5c3ee0
# sudo pip3 install -r requirements/cuda.txt
# sudo pip3 install --upgrade setuptools wheel
# sudo pip3 install .

sudo pip3 uninstall -y transformers
sudo pip3 install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
pip3 install flash-attn==2.6.3 --no-build-isolation
pip3 uninstall -y torch
pip3 uninstall -y torchvision
pip3 uninstall -y torchaudio
sudo pip3 install torch==2.6.0
sudo pip3 install torchvision==0.21.0
sudo pip3 install torchaudio==2.6.0
pip3 install peft==0.15.2
pip3 install moviepy==2.1.2
sudo pip3 install jupyter
sudo pip3 install httpx==0.23.0
sudo pip3 install bytedlaplace==0.9.26
pip3 install pydub==0.25.1

sudo pip3 install trl==0.16.0 # other versions may have problems
sudo apt-get -y install ffmpeg # load audio in video(mp4)
# pip3 install ninja
# sudo pip3 uninstall -y transformers
# sudo pip3 install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
# pip3 install accelerate
# pip3 install qwen-omni-utils -U
# pip3 install flash-attn==2.1.0 --no-build-isolation