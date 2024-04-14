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
from ta import add_all_ta_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
import pickle
import logging

# Configure logging to write to a file
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

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
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # Two years of data
        data = yf.download(symbols, start=start_date, end=end_date.strftime('%Y-%m-%d'))
        data = add_all_ta_features(data, colprefix='ta_')
        return data
    except Exception as e:
        logging.error(f"Error fetching data: {str(e)}")
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
            X.append(data[i:i+window_size])
            y.append(data[i+window_size, 0])  # Predict next close price
        return np.array(X), np.array(y)
    except Exception as e:
        logging.error(f"Error creating sequences: {str(e)}")
        time.sleep(60)
        return None, None

# Function to build and train the LSTM model
def build_and_train_lstm_model(X_train, y_train, window_size):
    try:
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
    except Exception as e:
        logging.error(f"Error building and training LSTM model: {str(e)}")
        time.sleep(60)
        return None

# Function to build and train the RandomForest model
def build_and_train_rf_model(X_train, y_train):
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        return model
    except Exception as e:
        logging.error(f"Error building and training RandomForest model: {str(e)}")
        time.sleep(60)
        return None

# Function to calculate moving averages for stock prices, RSI, and MACD
def calculate_moving_averages(data, window):
    try:
        close_prices = data[:, 3]  # Close prices are in the fourth column
        rsi = data[:, -2]  # RSI is the second last column
        macd = data[:, -1]  # MACD is the last column
        price_avg = np.convolve(close_prices, np.ones(window)/window, mode='valid')
        rsi_avg = np.convolve(rsi, np.ones(window)/window, mode='valid')
        macd_avg = np.convolve(macd, np.ones(window)/window, mode='valid')
        return price_avg, rsi_avg, macd_avg
    except Exception as e:
        logging.error(f"Error calculating moving averages: {str(e)}")
        time.sleep(60)
        return None, None, None

# Function to submit buy order
def submit_buy_order(symbol, quantity):
    try:
        api.submit_order(
            symbol=symbol,
            qty=quantity,
            side='buy',
            type='market',
            time_in_force='gtc'
        )
    except Exception as e:
        logging.error(f"Error submitting buy order: {str(e)}")
        time.sleep(60)

# Function to submit sell order
def submit_sell_order(symbol, quantity):
    try:
        api.submit_order(
            symbol=symbol,
            qty=quantity,
            side='sell',
            type='market',
            time_in_force='gtc'
        )
    except Exception as e:
        logging.error(f"Error submitting sell order: {str(e)}")
        time.sleep(60)

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

# Function to train the brain models to recognize price patterns
def train_price_patterns(data):
    try:
        for symbol in symbols:
            symbol_data = data[data['Symbol'] == symbol]
            # Extract relevant features
            prices = symbol_data['Close'].values
            dates = symbol_data.index.date
            times = symbol_data.index.time
            rsi = symbol_data['ta_RSI']
            macd = symbol_data['ta_MACD']
            volume = symbol_data['Volume']

            # Train models to recognize times of the day when prices are lower for buying
            # and times of the day when prices are higher for selling
            X = np.column_stack((prices, rsi, macd, volume))
            y_buy = np.where(prices[:-1] < prices[1:], 1, 0)  # 1 if price is increasing, else 0
            y_sell = np.where(prices[:-1] > prices[1:], 1, 0)  # 1 if price is decreasing, else 0

            # Train model for buying
            buying_model = RandomForestClassifier()
            buying_model.fit(X[:-1], y_buy)

            # Train model for selling
            selling_model = RandomForestClassifier()
            selling_model.fit(X[:-1], y_sell)

            # Save models for future use
            with open(f'{symbol}_buying_model.pkl', 'wb') as f_buy:
                pickle.dump(buying_model, f_buy)
            with open(f'{symbol}_selling_model.pkl', 'wb') as f_sell:
                pickle.dump(selling_model, f_sell)

            print(f"Trained models for {symbol} successfully.")
    except Exception as e:
        logging.error(f"Error training price patterns: {str(e)}")
        time.sleep(60)

# Main loop
while True:
    try:
        if does_model_exist('lstm_model.h5') and does_model_exist('rf_model.pkl'):
            # Load existing LSTM model
            lstm_model = load_model('lstm_model.h5')

            # Load existing RandomForest model
            with open('rf_model.pkl', 'rb') as f:
                rf_model = pickle.load(f)
        else:
            # Fetch data
            data = fetch_data()
            if data is None:
                continue
            data = data.values  # Convert to numpy array

            # Preprocess data
            scaled_data = preprocess_data(data)
            if scaled_data is None:
                continue

            # Create sequences for LSTM
            X_lstm, y_lstm = create_sequences(scaled_data, window_size=10)
            if X_lstm is None or y_lstm is None:
                continue

            # Train LSTM model
            X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)
            lstm_model = build_and_train_lstm_model(X_train_lstm, y_train_lstm, window_size=10)
            if lstm_model is None:
                continue
            lstm_model.save('lstm_model.h5')

            # Train RandomForest model
            X_rf, y_rf = scaled_data[:, :-1], scaled_data[:, -1]
            X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
            rf_model = build_and_train_rf_model(X_train_rf, y_train_rf)
            if rf_model is None:
                continue
            with open('rf_model.pkl', 'wb') as f:
                pickle.dump(rf_model, f)

            # Train buying and selling models
            train_price_patterns(data)

        if is_trading_hours():
            # Fetch data
            data = fetch_data()
            if data is None:
                continue
            data = data.values  # Convert to numpy array

            # Calculate moving averages for stock prices, RSI, and MACD
            price_avg, rsi_avg, macd_avg = calculate_moving_averages(data, window=50)
            if price_avg is None or rsi_avg is None or macd_avg is None:
                continue

            # Predict using LSTM model
            predictions_lstm = lstm_model.predict(X_test_lstm)

            # Predict using RandomForest model
            predictions_rf = rf_model.predict(X_test_rf)

            # Ensemble predictions
            ensemble_predictions = (predictions_lstm + predictions_rf) / 2

            # Execute buy orders based on ensemble predictions and available cash
            for symbol in symbols:
                current_price = data[-1, symbols.index(symbol) * 6 + 3]  # Close price is at every 6th index
                rsi = data[-1, symbols.index(symbol) * 6 + 10]  # RSI is at every 6th index + 7
                macd = data[-1, symbols.index(symbol) * 6 + 11]  # MACD is at every 6th index + 8
                if (ensemble_predictions[-1] < current_price * 0.998 and  # Buy if price drops more than 0.2%
                    cash_available >= current_price and 
                    rsi < rsi_avg[-1] and 
                    macd < 0):
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
                    current_price = data[-1, symbols.index(symbol) * 6 + 3]  # Close price is at every 6th index
                    rsi = data[-1, symbols.index(symbol) * 6 + 10]  # RSI is at every 6th index + 7
                    macd = data[-1, symbols.index(symbol) * 6 + 11]  # MACD is at every 6th index + 8
                    if (symbol in symbols and 
                        current_price > purchase_price * 1.005 and  # Sell if price is 0.5% or greater than purchase price
                        rsi > rsi_avg[-1] and 
                        macd > 0):
                        quantity = int(position.qty)
                        submit_sell_order(symbol, quantity)

        # Sleep for some time before checking again
        time.sleep(60)  # Sleep for 1 minute before checking again

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        logging.info("Restarting in 60 seconds...")
        time.sleep(60)  # Restart after 60 seconds
