#!/bin/sh

# Update package list
sudo apt update

# Install Python 3
sudo apt install -y python3

# Install pip3
sudo apt install -y python3-pip python3-h5py

# Install Python packages
pip3 install yfinance numpy scikit-learn tensorflow alpaca-trade-api ta
