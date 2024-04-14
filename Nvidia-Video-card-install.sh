#!/bin/sh

# Update package list
sudo apt update

sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu$(lsb_release -cs) main"

sudo apt update

sudo apt-get install libnvinfer7

sudo apt-get install tensorrt


# Install Python 3
sudo apt install -y python3

# Install pip3
sudo apt install -y python3-pip libhdf5-dev

# making sure python3.11 can install packages by renaming EXTERNALLY-MANAGED to EXTERNALLY-MANAGED.old
sudo mv /usr/lib/python3.11/EXTERNALLY-MANAGED /usr/lib/python3.11/EXTERNALLY-MANAGED.old



# Install Python packages
pip3 install yfinance numpy scikit-learn tensorflow alpaca-trade-api ta TensorRT

