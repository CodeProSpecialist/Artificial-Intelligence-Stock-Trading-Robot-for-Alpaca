#!/bin/sh

# Update package list
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
./configure --prefix=/usr
echo "Installing TA-Lib..."
make
sudo make install
cd ..
rm -r ta-lib
rm ta-lib-0.4.0-src.tar.gz

# Update Python packages using Anaconda's pip (assuming Anaconda is already installed)
conda activate
pip3 install yfinance numpy scikit-learn alpaca-trade-api pytz ta-lib torch torchvision

# Inform the user about Anaconda installation
echo "Your Python commands will be the Python commands that run with Anaconda's Python programs."
echo "You can activate Anaconda by running 'conda activate' and then install anything else with pip."
