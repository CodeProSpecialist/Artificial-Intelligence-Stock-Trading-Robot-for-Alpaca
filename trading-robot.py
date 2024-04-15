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

# Function to submit sell order
def submit_sell_order(symbol, quantity, target_sell_price):
    account_info = api.get_account()
    day_trade_count = account_info.daytrade_count
    current_price = get_current_price(symbol)

    if current_price >= target_sell_price and day_trade_count < 3:
        api.submit_order(
            symbol=symbol,
            qty=quantity,
            side='sell',
            type='market',
            time_in_force='gtc'
        )
        print(f"Sold {quantity} shares of {symbol} at ${current_price:.2f}")

# Main loop
while True:
    try:
        symbols_to_buy = ['AGQ', 'UGL']  # Example symbols to buy
        window_size = 10  # Example window size for LSTM

        for symbol in symbols_to_buy:
            print(f"Processing {symbol}...")
            historical_data = calculate_technical_indicators(symbol)

            if historical_data is not None:
                # Preprocess data
                data = historical_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
                scaled_data, scaler = preprocess_data(data)

                # Create sequences for LSTM
                X, y = create_sequences(scaled_data, window_size)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                # Load or create LSTM model
                model = load_or_create_model(symbol, input_size=X_train.shape[2])

                # Train LSTM model
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

                # Print target buy and sell prices
                print(f"Predicted target buy price for {symbol}: {target_buy_price:.2f}")
                print(f"Predicted target sell price for {symbol}: {target_sell_price:.2f}")

                # Submit buy and sell orders
                submit_buy_order(symbol, 1, target_buy_price)
                submit_sell_order(symbol, 1, target_sell_price)

        # Wait for next iteration
        print("Waiting for next iteration...")
        time.sleep(60)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        time.sleep(60)
