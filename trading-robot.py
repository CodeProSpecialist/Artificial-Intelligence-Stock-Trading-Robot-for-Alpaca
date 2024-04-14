import logging
import os
import pickle
import time
from datetime import datetime, timedelta
import numpy as np
import pytz
import yfinance as yf
import talib
import alpaca_trade_api as tradeapi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from torch.nn import functional as F

# Configure logging to write to a file
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Load environment variables for Alpaca API
API_KEY_ID = os.getenv('APCA_API_KEY_ID')
API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
API_BASE_URL = os.getenv('APCA_API_BASE_URL')

# Initialize the Alpaca API
api = tradeapi.REST(API_KEY_ID, API_SECRET_KEY, API_BASE_URL)

# Function to calculate MACD, RSI, and Volume
def calculate_technical_indicators(symbol, lookback_days=90):
    stock_data = yf.Ticker(symbol)
    historical_data = stock_data.history(period=f'{lookback_days}d')

    # Calculate MACD
    short_window = 12
    long_window = 26
    signal_window = 9
    macd, signal, _ = talib.MACD(historical_data['Close'],
                                  fastperiod=short_window,
                                  slowperiod=long_window,
                                  signalperiod=signal_window)
    historical_data['macd'] = macd
    historical_data['signal'] = signal

    # Calculate RSI
    rsi_period = 14
    rsi = talib.RSI(historical_data['Close'], timeperiod=rsi_period)
    historical_data['rsi'] = rsi

    # Volume is already present in historical_data

    return historical_data


# Function to preprocess data
def preprocess_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Function to create sequences for LSTM
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, 0])  # Predict next close price
    return np.array(X), np.array(y)

# Function to build and train the LSTM model
def build_and_train_lstm_model(X_train, y_train, window_size):
    class LSTMModel(nn.Module):
        def __init__(self):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size=X_train.shape[2], hidden_size=64, num_layers=2, batch_first=True)
            self.dropout = nn.Dropout(0.2)
            self.fc1 = nn.Linear(64, 32)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, 1)

        def forward(self, x):
            lstm_output, _ = self.lstm(x)
            lstm_output_last = lstm_output[:, -1, :]  # Select output of last time step
            dropout_output = self.dropout(lstm_output_last)
            fc1_output = self.fc1(dropout_output)
            relu_output = self.relu(fc1_output)
            output = self.fc2(relu_output)
            return output

    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 50
    batch_size = 32

    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            batch_X = torch.tensor(X_train[i:i + batch_size], dtype=torch.float32)
            batch_y = torch.tensor(y_train[i:i + batch_size], dtype=torch.float32).unsqueeze(-1)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    return model


# Function to submit buy order
def submit_buy_order(symbol, quantity, cash_available):
    current_price = api.get_last_trade(symbol).price
    api.submit_order(
        symbol=symbol,
        qty=quantity,
        side='buy',
        type='market',
        time_in_force='gtc'
    )
    # Update cash available after buying stocks
    cash_available -= quantity * current_price
    return cash_available

# Function to submit sell order
def submit_sell_order(symbol, quantity, cash_available):
    current_price = api.get_last_trade(symbol).price
    api.submit_order(
        symbol=symbol,
        qty=quantity,
        side='sell',
        type='market',
        time_in_force='gtc'
    )
    # Update cash available after selling stocks
    cash_available += quantity * current_price
    return cash_available

# Function to get account information
def get_account_info():
    account_info = api.get_account()
    cash_available = float(account_info.cash)
    day_trade_count = account_info.daytrade_count
    positions = {position.symbol: float(position.qty) for position in api.list_positions()}
    return cash_available, day_trade_count, positions

# Main loop
while True:
    try:
        symbols_to_buy = ['AGQ', 'UGL']  # Example symbols to buy
        window_size = 10  # Example window size for LSTM

        # Fetch and calculate technical indicators
        data = []
        for symbol in symbols_to_buy:
            technical_data = calculate_technical_indicators(symbol)
            data.append(technical_data.values)
            time.sleep(1)
        if data:
            combined_data = np.concatenate(data)
            scaled_data = preprocess_data(combined_data)
            X, y = create_sequences(scaled_data, window_size)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            lstm_model = build_and_train_lstm_model(X_train, y_train, window_size)

            # Make predictions
            lstm_predictions = lstm_model(torch.tensor(X_test, dtype=torch.float32))

            # Execute buy/sell orders based on predictions and account information
            cash_available, day_trade_count, positions = get_account_info()
            for symbol in symbols_to_buy:
                index = -1, symbols_to_buy.index(symbol) * 5 + 3
                current_price = data[index[0]][index[1]]

                # Convert current_price * 0.998 to a PyTorch tensor
                current_price_tensor = torch.tensor(current_price * 0.998)

                # Perform the comparison operation with tensors
                if lstm_predictions[-1][0] < current_price_tensor.item() and cash_available >= current_price:
                    quantity = int(cash_available // current_price)  # Buy as many shares as possible
                    cash_available = submit_buy_order(symbol, quantity, cash_available)
                if day_trade_count < 3:
                    purchase_price = positions[symbol]
                    if current_price > purchase_price * 1.005:  # Sell if price is 0.5% or greater than purchase price
                        quantity = int(positions[symbol])  # Sell all shares
                        cash_available = submit_sell_order(symbol, quantity, cash_available)
        else:
            print("No data fetched.")
            time.sleep(60)

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        logging.info("Restarting in 60 seconds...")
        time.sleep(60)
