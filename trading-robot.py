import os
import pickle
import time
import numpy as np
import yfinance as yf
import talib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
import alpaca_trade_api as tradeapi
import logging
from datetime import datetime, timedelta, date
from datetime import time as time2
import pytz

# Configure Alpaca API
API_KEY_ID = os.getenv('APCA_API_KEY_ID')
API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
API_BASE_URL = os.getenv('APCA_API_BASE_URL')

# Initialize Alpaca API
api = tradeapi.REST(API_KEY_ID, API_SECRET_KEY, API_BASE_URL)


def get_current_price(symbol):
    stock_data = yf.Ticker(symbol)
    return round(stock_data.history(period='1d')['Close'].iloc[0], 4)

# Define the LSTMModel class
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=3, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        lstm_output_last = lstm_output[:, -1, :]  # Select output of last time step
        dropout_output = self.dropout(lstm_output_last)
        fc1_output = self.fc1(dropout_output)
        relu_output = self.relu(fc1_output)
        output = self.fc2(relu_output)
        return output

def stop_if_stock_market_is_closed():
    # Check if the current time is within the stock market hours
    # Set the stock market open and close times
    market_open_time = time2(9, 27)
    market_close_time = time2(16, 0)

    while True:
        # Get the current time in Eastern Time
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        current_time = now.time()

        # Check if the current time is within market hours
        if now.weekday() <= 4 and market_open_time <= current_time <= market_close_time:
            break

        print("\n")
        print('''

            2024 Edition of the Artificial Intelligence Stock Trading Robot 
           _____   __                   __             ____            __            __ 
          / ___/  / /_  ____   _____   / /__          / __ \  ____    / /_   ____   / /_
          \__ \  / __/ / __ \ / ___/  / //_/         / /_/ / / __ \  / __ \ / __ \ / __/
         ___/ / / /_  / /_/ // /__   / ,<           / _, _/ / /_/ / / /_/ // /_/ // /_  
        /____/  \__/  \____/ \___/  /_/|_|         /_/ |_|  \____/ /_.___/ \____/ \__/  

                                                  https://github.com/CodeProSpecialist

                       Featuring Neural Network Learning and Decision Making   

         ''')
        print(f'Current date & time (Eastern Time): {now.strftime("%A, %B %d, %Y, %I:%M:%S %p")}')
        print("Stockbot only works Monday through Friday: 9:30 am - 4:00 pm Eastern Time.")
        print("Stockbot begins watching stock prices early at 9:27 am Eastern Time.")
        print("Waiting until Stock Market Hours to begin the Stockbot Trading Program.")
        print("\n")
        print("\n")
        time.sleep(60)  # Sleep for 1 minute and check again. Keep this under the p in print.


logging.basicConfig(filename='trading-bot-program-logging-messages.txt', level=logging.INFO)

# Function to log error messages
def log_error(message):
    logging.error(message)

# Function to preprocess data
def preprocess_data(data):
    try:
        # Fill NaN values with zeros
        data = np.nan_to_num(data)

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        return scaled_data, scaler
    except Exception as e:
        log_error(f"Error in preprocess_data: {str(e)}")
        raise

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

        # Include volume
        volume = historical_data['Volume']
        historical_data['volume'] = volume

        return historical_data

    except Exception as e:
        log_error(f"Error in calculate_technical_indicators for symbol {symbol}: {str(e)}")
        raise

# Function to create sequences for LSTM
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, 0])  # Predict next close price
    return np.array(X), np.array(y)


# Function to load or create the LSTM model
def load_or_create_model(filename, input_size):
    model_dir = 'brain_models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, f"{filename}.pkl")
    if os.path.exists(model_path):
        # Load the existing model
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"Loaded existing model from {model_path}")
    else:
        # Create a new model
        model = LSTMModel(input_size)
        print(f"Created new model")

    return model


# Function to submit buy order
def submit_buy_order(symbol, quantity, target_buy_price):
    account_info = api.get_account()
    cash_available = float(account_info.cash)
    current_price = get_current_price(symbol)

    if current_price <= target_buy_price and cash_available >= current_price:
        api.submit_order(
            symbol=symbol,
            qty=quantity,
            side='buy',
            type='market',
            time_in_force='gtc'
        )
        print(f"Bought {quantity} shares of {symbol} at ${current_price:.2f}")

def submit_sell_order(symbol, quantity, target_sell_price):
    account_info = api.get_account()
    day_trade_count = account_info.daytrade_count

    current_price = get_current_price(symbol)
    
    try:
        position = api.get_position(symbol)
    except Exception as e:
        print(f"Error: {e}")
        return

    if position.qty != '0':
        bought_price = float(position.avg_entry_price)

        if current_price >= target_sell_price and day_trade_count < 3 and current_price >= bought_price * 1.005:
            api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            print(f"Sold {quantity} shares of {symbol} at ${current_price:.2f}")
    else:
        print(f"You don't own any shares of {symbol}, so no sell order was submitted.")


def get_stocks_to_trade():
    try:
        with open('list-of-stocks-to-buy.txt', 'r') as file:
            symbols = [line.strip() for line in file.readlines()]

        if not symbols:  # keep this under the w in with
            print("\n")
            print(
                "********************************************************************************************************")
            print(
                "*   Error: The file list-of-stocks-to-buy.txt doesn't contain any stock symbols.                       *")
            print(
                "*   This Robot does not work until you place stock symbols in the file named:                          *")
            print(
                "*                     list-of-stocks-to-buy.txt                                                        *")
            print(
                "********************************************************************************************************")
            print("\n")

        return symbols  # keep this under the i in if

    except FileNotFoundError:  # keep this under the t in try
        print("\n")
        print("****************************************************************************")
        print("*   Error: File not found: list-of-stocks-to-buy.txt                       *")
        print("****************************************************************************")
        print("\n")
        return []  # keep this under the p in print


# Main loop
while True:
    try:
        stop_if_stock_market_is_closed()  # comment this line to debug the Python code
        now = datetime.now(pytz.timezone('US/Eastern'))
        current_time_str = now.strftime("Eastern Time | %I:%M:%S %p | %m-%d-%Y |")

        cash_balance = round(float(api.get_account().cash), 2)
        print("------------------------------------------------------------------------------------")
        print(" 2024 Edition of the Artificial Intelligence Stock Trading Robot ")
        print("by https://github.com/CodeProSpecialist")
        print("------------------------------------------------------------------------------------")
        print(f"  {current_time_str} Cash Balance: ${cash_balance}")
        day_trade_count = api.get_account().daytrade_count
        print("\n")
        print(f"Current day trade number: {day_trade_count} out of 3 in 5 business days")
        print("\n")

        print("------------------------------------------------------------------------------------")
        print("\n")

        window_size = 10  # Example window size for LSTM

        # Get the list of stock symbols to trade
        symbols = get_stocks_to_trade()

        for symbol in symbols:
            print("\n")
            print(f"Processing {symbol}...")
            print("\n")
            time.sleep(1)
            # Fetch historical data for the stock symbol
            historical_data = calculate_technical_indicators(symbol)

            if historical_data is not None:
                # Preprocess data
                data = historical_data[['Open', 'High', 'Low', 'Close', 'Volume', 'macd', 'signal', 'rsi']].values
                scaled_data, scaler = preprocess_data(data)

                # Create sequences for LSTM
                X, y = create_sequences(scaled_data, window_size)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                # Load or create LSTM model
                print("\n")
                print(f"Training LSTM brain model for {symbol}.....")
                print("\n")
                model = load_or_create_model(symbol, input_size=X_train.shape[2])

                # Train LSTM model
                # Adjust training parameters
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Increase or Decrease lr or learning rate
                epochs = 100
                batch_size = 64  # Increase batch size

                # Create sequences for LSTM
                X, y = create_sequences(scaled_data, window_size)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                # Train LSTM model
                for epoch in range(epochs):
                    for i in range(0, len(X_train), batch_size):
                        batch_X = torch.tensor(X_train[i:i + batch_size], dtype=torch.float32)
                        batch_y = torch.tensor(y_train[i:i + batch_size], dtype=torch.float32).unsqueeze(-1)
                        optimizer.zero_grad()
                        output = model(batch_X)
                        loss = criterion(output, batch_y)
                        loss.backward()
                        optimizer.step()

                print(f"Training of LSTM brain model for {symbol} completed.")
                print("\n")

                # Save LSTM model
                model_path = os.path.join('brain_models', f"{symbol}.pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                print(f"LSTM model for {symbol} saved successfully")

                # Make predictions
                with torch.no_grad():
                    predictions = model(torch.tensor(X_test, dtype=torch.float32))

                # Get target buy price and target sell price
                target_buy_price = np.min(historical_data['Low'])
                target_sell_price = np.max(historical_data['High'])

                print("\n")
                # Print target buy and sell prices
                print(f"Predicted target buy price for {symbol}: {target_buy_price:.2f}")
                print(f"Predicted target sell price for {symbol}: {target_sell_price:.2f}")
                print("\n")

                # Submit buy and sell orders
                submit_buy_order(symbol, 1, target_buy_price)
                submit_sell_order(symbol, 1, target_sell_price)

        # Wait for next iteration
        print("\n")
        print("Waiting 30 seconds.....")
        print("\n")
        time.sleep(30)

    except Exception as e:
        log_error(f"Error occurred: {str(e)}")
        print(f"Error occurred: {str(e)}")
        time.sleep(60)
