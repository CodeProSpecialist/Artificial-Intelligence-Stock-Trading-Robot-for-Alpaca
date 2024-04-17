# Artificial-Intelligence-Stock-Trading-Robot-for-Alpaca
An Artificial Intelligence Robot for stock trading. 

Upgrade to the newest version of this Python Robot today because some Python code updates were finished and some errors were recently fixed on 4-16-2024. 

This robot features neural network functionality to train, learn, and make trading decisions.  

This is one of the favorite trading robots of CodeProSpecialist. 

You will need at least a recommended 25 to 50 Gigabytes of free space 
on your hard drive when installing the dependencies for this robot. 

How does this robot work?

Begin by fine tuning the buying and selling prices from the past 14 days of historical price data. 


        Make sure to fine tune the file trading-robot.py to change the value of the following 
        in the python code lines 263 and 265: 
         percent_to_add_to_low_buy_price = 1.07    # 1.07 is the default setting ( + 7% )
        
         percent_to_subtract_from_high_sell_price = 0.985  # Factor to reduce the sell price by 1.5%

This stock trading bot buys stocks based on a target buy price, which is calculated as ( lowest_price * 1.07 ) 7% above the lowest price observed over the last 14 days. Here's how it works:

Calculate Technical Indicators: The bot fetches historical data for the last 14 days for each stock in the list. It then calculates technical indicators such as MACD, RSI, and volume.
Preprocess Data: The historical data is preprocessed, which includes handling NaN values and scaling the data using Min-Max scaling.
Create Sequences for LSTM: The preprocessed data is divided into input sequences (X) and corresponding output values (y) for training the LSTM model.
Train LSTM Model: The LSTM model is either loaded from a saved file or created if it doesn't exist. The model is trained using the input sequences (X_train) and output values (y_train).
Make Predictions: The trained LSTM model is used to predict the target buy price and target sell price.
Submit Buy and Sell Orders: If the current price is below the target buy price and there's enough cash available, a buy order is submitted. If the current price is above the target sell price and certain conditions are met (such as not exceeding the day trade limit), a sell order is submitted.
Wait and Repeat: After processing all stocks in the list, the bot waits for 30 seconds before starting the next iteration of the main loop.
Overall, the bot uses LSTM-based machine learning to predict buy and sell prices, with target buy prices being determined based on historical lows plus a percentage increase.
Is the LSTM model actually being used in the robot?

Yes, the LSTM model is being used in the robot. Here's how it's utilized:

Training the LSTM Model: The LSTM model is trained using historical data to learn patterns and relationships between various technical indicators and stock prices.
Making Predictions: Once the model is trained, it's used to make predictions on unseen data (test data) to forecast the target buy price and target sell price for each stock.
Submitting Orders: Based on the predictions made by the LSTM model, the robot decides whether to submit buy or sell orders for each stock.
Saving and Loading the Model: The trained LSTM model is saved to disk after training and loaded back when needed, ensuring that the model's learning is persistent across different runs of the program.
So, yes, the LSTM model plays a crucial role in the decision-making process of the stock trading robot.

Disclaimer:

This software is not affiliated with or endorsed Alpaca Securities, LLC. It aims to be a valuable tool for stock market trading, but all trading involves risks. Use it responsibly and consider seeking advice from financial professionals.

Ready to elevate your trading game? Download the Artificial-Intelligence-Stock-Trading-Robot, and get started today!

Important: Don't forget to regularly update your list of stocks to buy and keep an eye on the market conditions. Happy trading!

Remember that all trading involves risks. The ability to successfully implement these strategies depends on both market conditions and individual skills and knowledge. As such, trading should only be done with funds that you can afford to lose. Always do thorough research before making investment decisions, and consider consulting with a financial advisor. This is use at your own risk software. This software does not include any warranty or guarantees other than the useful tasks that may or may not work as intended for the software application end user. The software developer shall not be held liable for any financial losses or damages that occur as a result of using this software for any reason to the fullest extent of the law. Using this software is your agreement to these terms. This software is designed to be helpful and useful to the end user.

Place your alpaca code keys in the location: /home/name-of-your-home-folder/.bashrc Be careful to not delete the entire .bashrc file. Just add the 4 lines to the bottom of the .bashrc text file in your home folder, then save the file. .bashrc is a hidden folder because it has the dot ( . ) in front of the name. Remember that the " # " pound character will make that line unavailable. To be helpful, I will comment out the real money account for someone to begin with an account that does not risk using real money. The URL with the word "paper" does not use real money. The other URL uses real money. Making changes here requires you to reboot your computer or logout and login to apply the changes.

The 4 lines to add to the bottom of .bashrc are:

export APCA_API_KEY_ID='zxzxzxzxzxzxzxzxzxzxz'

export APCA_API_SECRET_KEY='zxzxzxzxzxzxzxzxzxzxzxzxzxzxzxzxzxzxzxzx'

#export APCA_API_BASE_URL='https://api.alpaca.markets'

export APCA_API_BASE_URL='https://paper-api.alpaca.markets'
