import pandas as pd
import numpy as np

df = pd.read_csv('data/nvidia_stock.csv', parse_dates=['Date'])
df = df.sort_values('Date')

# Technical Indicators
# EMA (Exponential Moving Average)
df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()

# SMA (Simple Moving Average)
df['SMA_10'] = df['Close'].rolling(window=10).mean()

# MACD (Moving Average Convergence Divergence)
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema_12 - ema_26
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# RSI (Relative Strength Index)
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = calculate_rsi(df)

# Sentiment data
df_sentiment = pd.read_csv('data/daily_sentiment.csv', parse_dates=['date'])

df_merged = pd.merge(df, df_sentiment, left_on='Date', right_on='date', how='inner')

df_merged.to_csv('data/final_dataset.csv', index=False)

print("Feature engineering completed")
