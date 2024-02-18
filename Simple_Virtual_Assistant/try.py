import yfinance as yf

stock_tickers = ['AAPL', 'META', 'TSLA', 'NVDA']

# for ticker in stock_tickers:
#     try:
#         stock_data = yf.download(ticker, start='2020-01-01', end='2024-01-01')
#         last_price = stock_data['Close'].iloc[-1]
#         print(f"The last price of {ticker} was {last_price}")
#         print(stock_data)
#     except Exception as e:
#         print(f"Error retrieving stock data for {ticker}: {e}")

import numpy as np

# Given points
x_values = [1, 2, 3, 5]
y_values = [1, 2, 1, 3]

# Fit a linear regression model
coefficients = np.polyfit(x_values, y_values, 1)
slope, intercept = coefficients

# Print the equation of the line
print(f"The best-fitting line equation is: y = {slope}x + {intercept}")
