#!/bin/sh

# Update package list
sudo apt update

# Install Python 3 and pip3
sudo apt install -y python3 python3-pip

# Install required packages
sudo apt install -y libhdf5-dev

# Create and activate virtual environment in the user's home directory
HOME_VENV_PATH="$HOME/My-Python-Virtual-Environment-Packages"
VENV_PATH="$HOME_VENV_PATH/venv"

# Check if the virtual environment directory exists, if not, create it
if [ ! -d "$VENV_PATH" ]; then
    mkdir -p "$VENV_PATH"
fi

# Create the virtual environment
python3 -m venv "$VENV_PATH"

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Install Python packages within the virtual environment
pip install yfinance numpy scikit-learn alpaca-trade-api pytz ta-lib torch torchvision

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

# Add the virtual environment's bin directory to PATH if not already present
if ! grep -q "$HOME_VENV_PATH/venv/bin" ~/.bashrc; then
    echo 'export PATH="$PATH:'"$HOME_VENV_PATH"'/venv/bin"' >> ~/.bashrc
    source ~/.bashrc
fi

# Deactivate the virtual environment
deactivate
