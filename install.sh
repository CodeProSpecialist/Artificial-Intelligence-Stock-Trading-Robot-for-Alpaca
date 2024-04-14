#!/bin/sh

# Install Python3 and pip3 if not already installed
sudo apt update
sudo apt install python3 python3-pip -y

# Install required Python packages using pip3
pip3 install yfinance numpy tensorflow alpaca-trade-api
