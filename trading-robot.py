import os
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import alpaca_trade_api as tradeapi
import time
from datetime import datetime, time as dt_time, timedelta
import pytz
from ta import add_all_ta_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

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
    start_date = end_date - pd.DateOffset(years=2)
    data = yf.download(symbols, start=start_date, end=end_date.strftime('%Y-%m-%d'))
    data = add_all_ta_features(data, colprefix='ta_')
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

# Function to build and train the LSTM model
def build_and_train_lstm_model(X_train, y_train, window_size):
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

# Function to build and train the RandomForest model
def build_and_train_rf_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    return model

# Function to add noise to data
def add_noise(data):
    noise = np.random.normal(0, 0.01, data.shape)
    noisy_data = data + noise
    return noisy_data

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
def does_model_exist(model_name):
    return os.path.isfile(model_name)

# Main loop
while True:
    if does_model_exist('lstm_model.h5') and does_model_exist('rf_model.pkl'):
        # Load existing LSTM model
        lstm_model = load_model('lstm_model.h5')

        # Load existing RandomForest model
        with open('rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
    else:
        # Fetch data
        data = fetch_data()

        # Preprocess data
        scaled_data = preprocess_data(data)

        # Create sequences for LSTM
        X_lstm, y_lstm = create_sequences(scaled_data, window_size=10)

        # Train LSTM model
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)
        lstm_model = build_and_train_lstm_model(X_train_lstm, y_train_lstm, window_size=10)
        lstm_model.save('lstm_model.h5')

        # Train RandomForest model
        X_rf, y_rf = scaled_data[:, :-1], scaled_data[:, -1]
        X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
        rf_model = build_and_train_rf_model(X_train_rf, y_train_rf)
        with open('rf_model.pkl', 'wb') as f:
            pickle.dump(rf_model, f)

    if is_trading_hours():
        # Predict using LSTM model
        predictions_lstm = lstm_model.predict(X_test_lstm)

        # Predict using RandomForest model
        predictions_rf = rf_model.predict(X_test_rf)

        # Ensemble predictions
        ensemble_predictions = (predictions_lstm + predictions_rf) / 2

        # Execute buy orders based on ensemble predictions and available cash
        for symbol in symbols:
            current_price = data[symbol].iloc[-1]['Close']
            rsi = data[symbol].iloc[-1]['ta_rsi']
            macd = data[symbol].iloc[-1]['ta_macd']
            volume = data[symbol].iloc[-1]['Volume']
            if ensemble_predictions[-1] < current_price and cash_available >= current_price and rsi < 30 and macd < 0 and volume < 1000000:
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
                rsi = data[symbol].iloc[-1]['ta_rsi']
                macd = data[symbol].iloc[-1]['ta_macd']
                volume = data[symbol].iloc[-1]['Volume']
                if symbol in symbols and current_price > purchase_price and rsi > 70 and macd > 0 and volume > 1000000:
                    quantity = int(position.qty)
                    # Check if ensemble prediction is higher than purchase price
                    if ensemble_predictions[-1] > purchase_price:
                        submit_sell_order(symbol, quantity)

    # Sleep for some time before checking again
    time.sleep(60)  # Sleep for 1 minute before checking again
