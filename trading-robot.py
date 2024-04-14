import os
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import alpaca_trade_api as tradeapi
import time
from datetime import datetime, time as dt_time, timedelta
import pytz

# Load environment variables for Alpaca API
API_KEY_ID = os.getenv('APCA_API_KEY_ID')
API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
API_BASE_URL = os.getenv('APCA_API_BASE_URL')

# Initialize the Alpaca API
api = tradeapi.REST(API_KEY_ID, API_SECRET_KEY, API_BASE_URL)

# Define ETF symbols
symbols = ['AGQ', 'UGL']

# Function to fetch historical data
def fetch_data():
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(years=1)
    data = yf.download(symbols, start=start_date, end=end_date.strftime('%Y-%m-%d'))
    return data

# Function to preprocess data
def preprocess_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Function to create sequences for LSTM
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size, 0])  # Predict next close price
    return np.array(X), np.array(y)

# Function to build and train the model
def build_and_train_model(X_train, y_train, window_size):
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(window_size, X_train.shape[2])),
        Dropout(0.2),
        LSTM(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

    return model

# Function to submit buy order
def submit_buy_order(symbol, quantity):
    api.submit_order(
        symbol=symbol,
        qty=quantity,
        side='buy',
        type='market',
        time_in_force='gtc'
    )

# Function to submit sell order
def submit_sell_order(symbol, quantity):
    api.submit_order(
        symbol=symbol,
        qty=quantity,
        side='sell',
        type='market',
        time_in_force='gtc'
    )

# Function to check if current time is within trading hours
def is_trading_hours():
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    trading_start = now.replace(hour=9, minute=30, second=0)
    trading_end = now.replace(hour=16, minute=0, second=0)
    return now.weekday() < 5 and trading_start <= now <= trading_end

# Function to check if brain model file exists
def does_model_exist():
    return os.path.isfile('brain_model.h5')

# Main loop
while True:
    if does_model_exist():
        # Load existing brain model
        model = load_model('brain_model.h5')
    else:
        # Fetch data
        data = fetch_data()

        # Preprocess data
        scaled_data = preprocess_data(data)

        # Create sequences for LSTM
        X, y = create_sequences(scaled_data, window_size=10)

        # Split data into train and test sets
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Build and train the model
        model = build_and_train_model(X_train, y_train, window_size=10)
        model.save('brain_model.h5')

    if is_trading_hours():
        # Predict using the trained model
        predictions = model.predict(X_test)

        # Execute buy orders based on predictions and available cash
        for symbol in symbols:
            current_price = data[symbol].iloc[-1]['Close']
            if predictions[-1] < current_price and cash_available >= current_price:
                quantity = int(cash_available // current_price)  # Buy as many shares as possible
                submit_buy_order(symbol, quantity)
                cash_available -= quantity * current_price

        # Check day trade count and sell positions if less than 3 and at a higher price
        account_info = api.get_account()
        day_trade_count = account_info.daytrade_count
        if day_trade_count < 3:
            positions = api.list_positions()
            for position in positions:
                symbol = position.symbol
                purchase_price = float(position.avg_entry_price)
                current_price = float(position.current_price)
                if symbol in symbols and current_price > purchase_price:
                    quantity = int(position.qty)
                    submit_sell_order(symbol, quantity)

    # Retrain the model every 24 hours
    if datetime.now().hour == 0 and datetime.now().minute == 0:
        # Fetch new data
        new_data = fetch_data()

        # Preprocess new data
        new_scaled_data = preprocess_data(new_data)

        # Create new sequences for LSTM
        new_X, new_y = create_sequences(new_scaled_data, window_size=10)

        # Update the existing model with new data
        model.fit(new_X, new_y, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)
        model.save('brain_model.h5')

    # Sleep for some time before checking again
    time.sleep(60)  # Sleep for 1 minute before checking again
