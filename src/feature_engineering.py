import pandas as pd
import numpy as np

data = pd.read_csv('data/processed_data.csv')
data.set_index('Date', inplace=True)

# Technical Indicators

# RSI (Relative Strength Index)
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi