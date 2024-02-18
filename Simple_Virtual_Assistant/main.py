from neuralintents.assistants import BasicAssistant
import yfinance as yf
import pandas_datareader as web
stock_tickers = ['AAPL', 'META', 'TSLA', 'NVDA']


def stock_function():
    for ticker in stock_tickers:
        try:
            stock_data = yf.download(ticker, start='2020-01-01', end='2024-01-01')
            last_price = stock_data['Close'].iloc[-1]
            print(f"The last price of {ticker} was {last_price}")
        except Exception as e:
            print(f"Error retrieving stock data for {ticker}: {e}")


assistant = BasicAssistant('intents.json', method_mappings={
    "stocks": stock_function,
    "goodbye": lambda: exit(0)
})

assistant.fit_model(epochs=50)
assistant.save_model()

done = False

while not done:
    message = input("Enter a message: ")
    if message == "STOP":
        done = True
    else:
        print(assistant.process_input(message))