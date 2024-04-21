#!/bin/sh

# Update package list
sudo apt update

# Install Python 3 and pip3
sudo apt install -y python3 python3-pip

# Install required packages
sudo apt install -y libhdf5-dev


# Install Python packages
pip3 install yfinance numpy scikit-learn alpaca-trade-api pytz ta-lib

# Install PyTorch
pip3 install torch torchvision

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


# Add '/home/x800/.local/bin' to PATH if not already present
if ! grep -q '/home/x800/.local/bin' ~/.bashrc; then
    echo 'export PATH="$PATH:/home/x800/.local/bin"' >> ~/.bashrc
    source ~/.bashrc
fi
