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
    return scaled_data, scaler

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

    # Start training
    print("Training LSTM model...")
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            batch_X = torch.tensor(X_train[i:i + batch_size], dtype=torch.float32)
            batch_y = torch.tensor(y_train[i:i + batch_size], dtype=torch.float32).unsqueeze(-1)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

    print("LSTM model training complete.")
    return model


# Function to save the model
def save_model(model, model_type):
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = f"{model_dir}/{model_type}_model.pkl"
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"{model_type} model saved successfully.")

# Function to load the model
def load_model(model_type):
    model_dir = "models"
    model_path = f"{model_dir}/{model_type}_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print(f"{model_type} model loaded successfully.")
        return model
    else:
        print(f"No saved {model_type} model found. Creating new model...")
        return None

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

        # Print current date and time in Eastern Time
        eastern = pytz.timezone('US/Eastern')
        current_time = datetime.now(eastern)
        print(f"Starting main loop at {current_time.strftime('%Y-%m-%d %H:%M:%S')} (Eastern Time)")

        # Initialize lists to store data for all symbols
        X_all, y_all = [], []

        # Process each symbol
        for symbol in symbols_to_buy:
            print(f"Processing {symbol}...")

            # Fetch and calculate technical indicators
            technical_data = calculate_technical_indicators(symbol)
            if technical_data is None:
                print(f"No data fetched for {symbol}.")
                continue

            # Preprocess data and create sequences
            scaled_data, _ = preprocess_data(technical_data.values)
            X, y = create_sequences(scaled_data, window_size)

            # Append data to lists
            X_all.extend(X)
            y_all.extend(y)

        # Convert lists to numpy arrays
        X_all, y_all = np.array(X_all), np.array(y_all)

        # Train LSTM model
        lstm_model = build_and_train_lstm_model(X_all, y_all, window_size)
        save_model(lstm_model, "lstm")

        # Make predictions and execute orders
        for symbol in symbols_to_buy:
            # Fetch and preprocess data for the symbol
            technical_data = calculate_technical_indicators(symbol)
            if technical_data is None:
                print(f"No data fetched for {symbol}.")
                continue

            scaled_data, scaler = preprocess_data(technical_data.values)
            X = scaled_data[-window_size:].reshape(1, window_size, -1)

            # Make prediction using the trained model
            with torch.no_grad():
                lstm_predictions = lstm_model(torch.tensor(X, dtype=torch.float32))

            # Execute buy/sell orders based on the prediction and account information
            cash_available, day_trade_count, positions = get_account_info()
            current_price = scaled_data[-1, -1]  # Last close price
            purchase_price = positions.get(symbol, 0)

            predicted_price = lstm_predictions.item() * scaler.scale_[0] + scaler.min_[0]

            print(f"Predicted price for {symbol}: {predicted_price:.2f}")

            if predicted_price < current_price * 0.998 and cash_available >= current_price:
                quantity = int(cash_available // current_price)
                cash_available = submit_buy_order(symbol, quantity, cash_available)
                print(f"Bought {quantity} shares of {symbol}.")

            if day_trade_count < 3 and current_price > purchase_price * 1.005:
                quantity = int(positions[symbol])
                cash_available = submit_sell_order(symbol, quantity, cash_available)
                print(f"Sold {quantity} shares of {symbol}.")

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        print(f"Error occurred: {str(e)}")

    print("Waiting for next iteration...")
    time.sleep(60)
