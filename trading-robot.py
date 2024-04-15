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

# Function to load the LSTM model from a file
def load_model(filename):
    try:
        with open(f"models/{filename}.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

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
                continue

            # Preprocess data
            data, _ = preprocess_data(technical_data.values)

            # Create sequences for LSTM
            X, y = create_sequences(data, window_size)

            # Append data for this symbol to the overall lists
            X_all.append(X)
            y_all.append(y)

        # Concatenate data for all symbols
        X_combined = np.concatenate(X_all)
        y_combined = np.concatenate(y_all)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, shuffle=False)

        # Train the LSTM model
        model = LSTMModel(input_size=X_train.shape[2])

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

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")

        # Save the trained model
        model_filename = 'lstm_model'
        with open(f"models/{model_filename}.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"{model_filename} saved successfully.")

        # Make predictions
        predictions = model(torch.tensor(X_test, dtype=torch.float32))

        # Evaluate the model
        test_loss = criterion(predictions, torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1))
        print(f"Test Loss: {test_loss.item():.4f}")

        # Execute buy/sell orders based on predictions and account information
        # (Code for this part is omitted for brevity)

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        print(f"Error occurred: {str(e)}")

    print("Waiting for next iteration...")
    time.sleep(60)
