#!/bin/sh

# Update package list
sudo apt update

# Manually add the NVIDIA CUDA repository key
wget -O /tmp/7fa2af80.pub https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/7fa2af80.pub
sudo apt-key add /tmp/7fa2af80.pub

# Add NVIDIA CUDA repository keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Add NVIDIA CUDA repository
sudo sh -c 'echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" > /etc/apt/sources.list.d/cuda.list'

# Update package lists
sudo apt-get update

# Clean up
rm /tmp/7fa2af80.pub cuda-keyring_1.1-1_all.deb

# Continue with other installations
# Install NVIDIA CUDA packages
sudo apt-get install nvidia-cudnn nvidia-cuda-toolkit-gcc cuda-toolkit libcudnn

# Install TensorRT
sudo apt-get install tensorrt libnvinfer7

# Install Python 3 and pip3
sudo apt install -y python3 python3-pip libhdf5-dev

# Rename EXTERNALLY-MANAGED to EXTERNALLY-MANAGED.old to allow Python 3.11 to install packages
sudo mv /usr/lib/python3.11/EXTERNALLY-MANAGED /usr/lib/python3.11/EXTERNALLY-MANAGED.old

# Install Python packages
pip3 install yfinance numpy scikit-learn tensorflow alpaca-trade-api ta TensorRT
