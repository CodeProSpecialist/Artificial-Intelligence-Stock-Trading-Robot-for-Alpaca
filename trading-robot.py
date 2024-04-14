import logging
import os
import pickle
import time
from datetime import datetime, timedelta

import alpaca_trade_api as tradeapi
import numpy as np
import pytz
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features
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

global symbols_to_buy

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
        symbols = read_stock_symbols_list()  # Fetch stock symbols
        if not symbols:
            print("No stock symbols found.")
            return None
        data = yf.download(symbols, start=start_date, end=end_date.strftime('%Y-%m-%d'))
        if data.empty:
            print("No data available for the specified symbols.")
            return None
        # Ensure data is not empty
        if len(data) < 1:
            print("Insufficient data for technical analysis features.")
            return None
        # Drop any NaN values
        data = data.dropna()
        data.reset_index(inplace=True)
        # Check if data has enough rows for technical analysis features
        if len(data) < 200:
            print("Insufficient data for technical analysis features.")
            return None
        # Calculate technical analysis features
        data = add_all_ta_features(data, open='Open', high='High', low='Low', close='Close', volume='Volume',
                                   colprefix='ta_')
        # Flatten the data array
        data = data.values.flatten()
        return data
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Exiting...")
        sys.exit(0)
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
        model = nn.Sequential(
            nn.LSTM(input_size=X_train.shape[2], hidden_size=64, num_layers=2, batch_first=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        epochs = 50
        batch_size = 32

        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                batch_X = torch.tensor(X_train[i:i + batch_size], dtype=torch.float32)
                batch_y = torch.tensor(y_train[i:i + batch_size], dtype=torch.float32).unsqueeze(-1)
                optimizer.zero_grad()
                output, _ = model(batch_X)
                loss = criterion(output.squeeze(-1), batch_y)
                loss.backward()
                optimizer.step()

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
        years_ago = 1

        # Fetch data
        data = fetch_data()
        if data is None:
            continue
        data = data.values  # Convert to numpy array

        # Get account information
        cash_available, day_trade_count, positions = get_account_info()
        if cash_available is None or day_trade_count is None or positions is None:
            continue

        # Calculate moving averages for stock prices, RSI, and MACD
        price_avg, rsi_avg, macd_avg = calculate_moving_averages(data, window=50)
        if price_avg is None or rsi_avg is None or macd_avg is None:
            continue

        # Preprocess data
        scaled_data = preprocess_data(data)

        # Create sequences for LSTM
        X, y = create_sequences(scaled_data, window_size)
        if X is None or y is None:
            continue

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Create and train LSTM model
        lstm_model = build_and_train_lstm_model(X_train, y_train, window_size)
        if lstm_model is None:
            continue

        # Create RandomForestRegressor model
        rf_model = RandomForestRegressor()
        rf_model.fit(X_train_rf, y_train_rf)

        # Make predictions using LSTM model
        lstm_predictions = lstm_model(X_test)

        # Make predictions using RandomForestRegressor model
        rf_predictions = rf_model.predict(X_test_rf)

        # Combine predictions
        combined_predictions = (lstm_predictions + rf_predictions) / 2

        # Execute buy/sell orders based on combined predictions and available cash
        for symbol in symbols_to_buy:
            current_price = data[-1, symbols_to_buy.index(symbol) * 6 + 3]  # Close price is at every 6th index
            rsi = data[-1, symbols_to_buy.index(symbol) * 6 + 10]  # RSI is at every 6th index + 7
            macd = data[-1, symbols_to_buy.index(symbol) * 6 + 11]  # MACD is at every 6th index + 8
            if (combined_predictions[-1] < current_price * 0.998 and  # Buy if price drops more than 0.2%
                    cash_available >= current_price and
                    rsi < rsi_avg[-1] and
                    macd < 0):
                quantity = int(cash_available // current_price)  # Buy as many shares as possible
                cash_available = submit_buy_order(symbol, quantity, cash_available)

            # Check day trade count and sell positions if less than 3 and at a higher price
            if day_trade_count < 3:
                purchase_price = positions[symbol]  # Get the purchase price from positions dictionary
                if (symbol in symbols_to_buy and
                        current_price > purchase_price * 1.005 and  # Sell if price is 0.5% or greater than purchase price
                        rsi > rsi_avg[-1] and
                        macd > 0):
                    quantity = int(positions[symbol])  # Sell all shares
                    cash_available = submit_sell_order(symbol, quantity, cash_available)

        print("Not trading hours - waiting...")

        print("This program runs Monday - Friday, 9:30am - 4:00pm.")

        # Print a message about sleeping for 60 seconds
        print("Sleeping for 60 seconds...")
        time.sleep(60)  # Sleep for 1 minute before checking again

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        logging.info("Restarting in 60 seconds...")
        time.sleep(60)  # Restart after 60 seconds
