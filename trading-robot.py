import logging
import os
import pickle
import time
from datetime import datetime, timedelta

import alpaca_trade_api as tradeapi
import numpy as np
import pytz
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features
import torch
import torch.nn as nn
import torch.optim as optim

# Configure logging to write to a file
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Load environment variables for Alpaca API
API_KEY_ID = os.getenv('APCA_API_KEY_ID')
API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
API_BASE_URL = os.getenv('APCA_API_BASE_URL')

# Initialize the Alpaca API
api = tradeapi.REST(API_KEY_ID, API_SECRET_KEY, API_BASE_URL)

global symbols_to_buy, symbol

# Define global variable
symbols_to_buy = []

# Function to read the list of stock symbols from a text file and export as symbols_to_buy
def read_stock_symbols_list():
    global symbols_to_buy
    symbols = []
    # Read list of stocks to buy from text file
    with open("list-of-stocks-to-buy.txt", "r") as file:
        for line in file:
            symbols.append(line.strip())
    symbols_to_buy = symbols
    return symbols_to_buy

# Function to fetch historical data
def fetch_data():
    try:
        time.sleep(1)
        print("Currently downloading stock price data.....")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 1)  # One year of data
        data = yf.download(symbols_to_buy, start=start_date, end=end_date.strftime('%Y-%m-%d'))
        for symbol in symbols_to_buy:
            time.sleep(1)  # Sleep for 1 second between fetching data for each symbol
        data = add_all_ta_features(data, open='Open', high='High', low='Low', close='Close', volume='Volume',
                                   colprefix='ta_')
        return data
    except Exception as e:
        logging.error(f"Error fetching data: {str(e)}")
        print(f"Error getting data for {str(e)}")
        time.sleep(60)
        return None


# Function to preprocess data
def preprocess_data(data):
    try:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        return scaled_data
    except Exception as e:
        logging.error(f"Error preprocessing data: {str(e)}")
        time.sleep(60)
        return None


# Function to create sequences for LSTM
def create_sequences(data, window_size):
    try:
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size, 0])  # Predict next close price
        return np.array(X), np.array(y)
    except Exception as e:
        logging.error(f"Error creating sequences: {str(e)}")
        time.sleep(60)
        return None, None


# Function to build and train the LSTM model
def build_and_train_lstm_model(X_train, y_train, window_size):
    try:
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])  # Get output from last time step
                return out

        # Define parameters
        input_size = X_train.shape[2]
        hidden_size = 64
        num_layers = 2
        output_size = 1

        # Initialize model
        model = LSTMModel(input_size, hidden_size, num_layers, output_size)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train).float()
        y_train_tensor = torch.tensor(y_train).float()

        # Train the model
        num_epochs = 50
        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        return model
    except Exception as e:
        logging.error(f"Error building and training LSTM model: {str(e)}")
        time.sleep(60)
        return None


# Function to calculate moving averages for stock prices, RSI, and MACD
def calculate_moving_averages(data, window):
    try:
        close_prices = data[:, 3]  # Close prices are in the fourth column
        rsi = data[:, -2]  # RSI is the second last column
        macd = data[:, -1]  # MACD is the last column
        price_avg = np.convolve(close_prices, np.ones(window) / window, mode='valid')
        rsi_avg = np.convolve(rsi, np.ones(window) / window, mode='valid')
        macd_avg = np.convolve(macd, np.ones(window) / window, mode='valid')
        return price_avg, rsi_avg, macd_avg
    except Exception as e:
        logging.error(f"Error calculating moving averages: {str(e)}")
        time.sleep(60)
        return None, None, None


# Function to submit buy order
def submit_buy_order(symbol, quantity, cash_available):
    try:
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
    except Exception as e:
        logging.error(f"Error submitting buy order: {str(e)}")
        time.sleep(60)
        return cash_available


# Function to submit sell order
def submit_sell_order(symbol, quantity, cash_available):
    try:
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
    except Exception as e:
        logging.error(f"Error submitting sell order: {str(e)}")
        time.sleep(60)
        return cash_available


# Function to check if current time is within trading hours
def is_trading_hours():
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    trading_start = now.replace(hour=9, minute=30, second=0)
    trading_end = now.replace(hour=16, minute=0, second=0)
    return now.weekday() < 5 and trading_start <= now <= trading_end


# Function to check if model files exist
def does_model_exist(model_name):
    return os.path.isfile(model_name)


# Function to load LSTM model
def load_lstm_model(model_path):
    try:
        return torch.load(model_path)
    except Exception as e:
        logging.error(f"Error loading LSTM model: {str(e)}")
        return None


# Function to save model
def save_model(model, model_path):
    try:
        torch.save(model, model_path)
    except Exception as e:
        logging.error(f"Error saving model: {str(e)}")
        time.sleep(60)


# Function to get account information
def get_account_info():
    try:
        account_info = api.get_account()
        cash_available = float(account_info.cash)
        day_trade_count = account_info.daytrade_count
        positions = {position.symbol: float(position.qty) for position in api.list_positions()}
        return cash_available, day_trade_count, positions
    except Exception as e:
        logging.error(f"Error getting account information: {str(e)}")
        time.sleep(60)
        return None, None, None


# Main loop
while True:
    try:
        print("Starting main loop...")

        # Fetch data
        print("Fetching data...")
        data = fetch_data()
        if data is None:
            continue
        data = data.values  # Convert to numpy array

        # Get account information
        print("Fetching account information...")
        cash_available, day_trade_count, positions = get_account_info()
        if cash_available is None or day_trade_count is None or positions is None:
            continue

        # Calculate moving averages for stock prices, RSI, and MACD
        print("Calculating moving averages...")
        price_avg, rsi_avg, macd_avg = calculate_moving_averages(data, window=50)
        if price_avg is None or rsi_avg is None or macd_avg is None:
            continue

        # Print current date and time in Eastern Time Zone
        eastern = pytz.timezone('US/Eastern')
        current_time = datetime.now(eastern).strftime("%Y-%m-%d %H:%M")
        print(f"Current Eastern Time: {current_time}")

        # Print list of stocks being monitored to buy along with their current price
        print("List of stocks being monitored to buy:")
        for symbol in symbols_to_buy:
            current_price = data[-1, symbols_to_buy.index(symbol) * 6 + 3]  # Close price is at every 6th index
            print(f"{symbol}: ${current_price:.2f}")

        # Print current account cash balance
        print(f"Current account cash balance: ${cash_available:.2f}")

        # Print current day trade number out of 3 in 5 days
        print(f"Current day trade count: {day_trade_count}/3 in 5 days")

        if is_trading_hours():
            print("Trading hours. Looking for stocks to trade.....")

            # Preprocess data
            print("Preprocessing data...")
            scaled_data = preprocess_data(data)

            # Create sequences for LSTM
            print("Creating sequences...")
            X, y = create_sequences(scaled_data, window_size)
            if X is None or y is None:
                continue

            # Split data into training and testing sets
            print("Splitting data...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # Create or load LSTM model
            lstm_model_path = 'lstm_model.pth'
            print("Creating or loading LSTM model...")
            if does_model_exist(lstm_model_path):
                lstm_model = load_lstm_model(lstm_model_path)
            else:
                lstm_model = build_and_train_lstm_model(X_train, y_train, window_size)
                if lstm_model is not None:
                    save_model(lstm_model, lstm_model_path)

            # Predict using LSTM model
            print("Predicting using LSTM model...")
            with torch.no_grad():
                outputs = lstm_model(torch.tensor(X_test).float())

            # Execute buy orders based on predictions and available cash
            print("Executing buy orders...")
            for symbol in symbols_to_buy:
                current_price = data[-1, symbols_to_buy.index(symbol) * 6 + 3]  # Close price is at every 6th index
                rsi = data[-1, symbols_to_buy.index(symbol) * 6 + 10]  # RSI is at every 6th index + 7
                macd = data[-1, symbols_to_buy.index(symbol) * 6 + 11]  # MACD is at every 6th index + 8
                # Perform your buy logic here using the LSTM model predictions

            # Check day trade count and sell positions if less than 3 and at a higher price
            print("Checking day trade count and selling positions...")
            # Perform your sell logic here based on day trade count and other conditions

        else:
            print("Not trading hours - waiting...")

        print("This program runs Monday - Friday, 9:30am - 4:00pm.")

        # Print a message about sleeping for 60 seconds
        print("Sleeping for 60 seconds...")
        time.sleep(60)  # Sleep for 1 minute before checking again

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        logging.info("Restarting in 60 seconds...")
        time.sleep(60)  # Restart after 60 seconds
