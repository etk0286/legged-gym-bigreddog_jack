

Himloco :https://github.com/InternRobotics/HIMLoco
Amp For Hardware : https://github.com/escontra/AMP_for_hardware?tab=readme-ov-file





Ubuntu 20.04

NVIDIA Driver:580.95.05 

CUDA 12.0

Python 3.8

PyTorch 1.10.0+cu113

Isaac Gym: Preview 4



Download

    git clone https://github.com/etk0286/legged-gym-bigreddog_jack.git 

Build Env

    -conda create -n amp_him python==3.8
    -conda activate amp_him


    pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

Install Isaac Gym Preview 4

Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym

    cd isaacgym/python && pip install -e .

Install Amp_Himloco

    cd rsl_rl && pip install -e .
    cd ../legged_gym && pip install -e .


EXAMPLE

train

    python legged_gym/scripts/train.py --task=big_reddog_him_jack`` --headless

play(export)

    python legged_gym/scripts/play_him.py --task=big_reddog_him_jack
