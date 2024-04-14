#!/bin/sh

# Update package list
sudo apt update

# Install Python 3 and pip3
sudo apt install -y python3 python3-pip

# Install required packages
sudo apt install -y libhdf5-dev

# Making sure python3.11 can install packages by renaming EXTERNALLY-MANAGED to EXTERNALLY-MANAGED.old
sudo mv /usr/lib/python3.11/EXTERNALLY-MANAGED /usr/lib/python3.11/EXTERNALLY-MANAGED.old

# Install Python packages
pip3 install yfinance numpy scikit-learn alpaca-trade-api ta

# Install PyTorch
pip3 install torch torchvision

# Ignore errors about not finding TensorRT to run this code without NVidia video cards on
# every Linux device like a Raspberry Pi.

# Add '/home/x800/.local/bin' to PATH if not already present
if ! grep -q '/home/x800/.local/bin' ~/.bashrc; then
    echo 'export PATH="$PATH:/home/x800/.local/bin"' >> ~/.bashrc
    source ~/.bashrc
fi
