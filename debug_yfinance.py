import yfinance as yf
import pandas as pd

def test_ticker(ticker):
    print(f"Testing {ticker}...")
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="1y")
        print(f"Data Shape: {df.shape}")
        if df.empty:
            print("DATAFRAME IS EMPTY")
        else:
            print(f"Columns: {df.columns}")
            print(f"Last Close: {df['Close'].iloc[-1]}")
    except Exception as e:
        print(f"ERROR: {e}")

test_ticker("^TWII")
test_ticker("0050.TW")
