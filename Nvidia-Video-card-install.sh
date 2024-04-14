#!/bin/sh

# the following is for Ubuntu 22.04 LTS

# Update package list
sudo apt update

# Add NVIDIA CUDA repository key
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/Release.gpg
sudo apt-key add Release.gpg

# Add NVIDIA CUDA repository keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Add NVIDIA CUDA repository
sudo sh -c 'echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" > /etc/apt/sources.list.d/cuda.list'

# Update package lists
sudo apt-get update

# Clean up
rm Release.gpg cuda-keyring_1.1-1_all.deb

# Update package lists
sudo apt-get update

# Clean up
rm cuda-keyring_1.0-1_all.deb

sudo apt update

sudo apt-get install cuda-toolkit libcudnn

sudo apt-get install tensorrt

sudo apt-get install libnvinfer7

# Install Python 3
sudo apt install -y python3

# Install pip3
sudo apt install -y python3-pip libhdf5-dev

# making sure python3.11 can install packages by renaming EXTERNALLY-MANAGED to EXTERNALLY-MANAGED.old
sudo mv /usr/lib/python3.11/EXTERNALLY-MANAGED /usr/lib/python3.11/EXTERNALLY-MANAGED.old



# Install Python packages
pip3 install yfinance numpy scikit-learn tensorflow alpaca-trade-api ta TensorRT

