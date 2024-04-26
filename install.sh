#!/bin/sh

# Update package list
sudo apt update

# Prompt the user
echo "We need to remove pip and pip3 before installing Anaconda. "
echo "Uninstall python-pip and python3-pip? (y/n)"
read response

# Check the response
if [ "$response" = "y" ]; then
  # Uninstall the packages as root
  sudo apt purge python-pip python3-pip
else
  echo "Uninstallation cancelled"
  exit 1
fi

sudo apt update

# Install required packages
sudo apt install -y libhdf5-dev

# Install TA-Lib dependencies
echo "Installing TA-Lib dependencies ..."
sudo apt-get install libatlas-base-dev gfortran -y

# Download and install TA-Lib
echo "Downloading TA-Lib..."
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzvf ta-lib-0.4.0-src.tar.gz

cd ta-lib/
echo "Configuring TA-Lib..."
./configure --prefix=/usr/local --build=x86_64-unknown-linux-gnu
echo "Building TA-Lib..."
sudo make -s ARCH=x86_64
echo "Installing TA-Lib..."
sudo make -s ARCH=x86_64 install

# For Raspberry Pi 4 (aarch64):
# ./configure --prefix=/usr/local --build=aarch64-unknown-linux-gnu
# sudo make -s ARCH=aarch64
# sudo make -s ARCH=aarch64 install

cd ..
sudo rm -r -f -I ta-lib
rm ta-lib-0.4.0-src.tar.gz

# Initialize conda
conda init bash

# Activate Anaconda environment
conda activate

# Update Python packages using Anaconda's pip
pip3 install yfinance numpy scikit-learn alpaca-trade-api pytz ta-lib torch torchvision

# Inform the user about Anaconda installation
echo "Your Python commands will be the Python commands that run with Anaconda's Python programs."
echo "You can install anything else with pip3 ."

# Inform the user about the virtual environment
echo "Your Python commands in the directory for Anaconda will be the Python commands that run this installed virtual environment's Python programs."
