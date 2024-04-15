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

# Define the LSTMModel class outside the function
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, num_layers=2, batch_first=True)
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

# Configure logging to write to a file
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Load environment variables for Alpaca API
API_KEY_ID = os.getenv('APCA_API_KEY_ID')
API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
API_BASE_URL = os.getenv('APCA_API_BASE_URL')

# Initialize the Alpaca API
api = tradeapi.REST(API_KEY_ID, API_SECRET_KEY, API_BASE_URL)

# Function to preprocess data
def preprocess_data(data):
    # Fill NaN values with zeros
    data = np.nan_to_num(data)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Function to calculate MACD, RSI, and Volume for the last 14 days
def calculate_technical_indicators(symbol):
    try:
        stock_data = yf.Ticker(symbol)
        historical_data = stock_data.history(period='14d')  # Fetch data for the last 14 days

        if historical_data.empty:
            print(f"No historical data found for {symbol}.")
            return None

        # Print the latest closing price
        latest_closing_price = historical_data['Close'].iloc[-1]
        print(f"Latest closing price for {symbol}: {latest_closing_price:.2f}")

        # Check for NaN values and remove corresponding rows
        historical_data = historical_data.dropna()

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

    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        return None

# Function to create sequences for LSTM
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, 0])  # Predict next close price
    return np.array(X), np.array(y)

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

        # Load or create LSTM model
        lstm_model = load_model("lstm")
        if lstm_model is None:
            lstm_model = build_and_train_lstm_model(X_all, y_all, window_size)
            save_model(lstm_model, "lstm")
        else:
            lstm_model = build_and_train_lstm_model(X_all, y_all, window_size, lstm_model)
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
