import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score
from joblib import parallel_backend


def run_stock_technical_prediction():
    try:
        df = pd.read_csv('../data/final_dataset.csv')
    except FileNotFoundError:
        print("Error: 'final_dataset.csv' not found.")
        return

    # Standardize column names (lowercase)
    df.columns = df.columns.str.lower()

    if 'date' not in df.columns:
        print("Error: 'Date' column is required.")
        return

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Detect stock columns (case-insensitive)
    stock_keywords = ['open', 'high', 'low', 'close', 'volume', 'price']
    stock_cols = [col for col in df.columns
                  if any(keyword in col for keyword in stock_keywords)]

    # Detect technical indicators (only if they exist in the dataset)
    tech_keywords = ['sma', 'ema', 'macd', 'rsi', 'bollinger', 'bb', 'cci',
                     'stoch', 'williams', 'momentum', 'roc', 'atr', 'adx',
                     'obv', 'vwap', 'ppo', 'trix', 'ultimate', 'commodity']
    tech_cols = [col for col in df.columns
                 if any(keyword in col for keyword in tech_keywords)]

    # Include avg_sentiment (required)
    if 'avg_sentiment' not in df.columns:
        print("Error: 'avg_sentiment' column is required.")
        return
    sentiment_cols = ['avg_sentiment']

    # Combine features + ensure 'close' is target
    selected_cols = stock_cols + tech_cols + sentiment_cols
    if 'close' not in selected_cols:
        print("Error: No 'close' price column found.")
        return

    # Remove duplicates
    selected_cols = list(dict.fromkeys(selected_cols))
    df = df[['date'] + selected_cols].dropna()

    if len(df) < 61:
        print("Error: Not enough data (need ≥61 days).")
        return

    # Scale features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[selected_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=selected_cols)

    # Create sequences (60-day lookback, 1-day ahead)
    X, y = [], []
    close_index = selected_cols.index('close')

    for i in range(60, len(scaled_df)):
        X.append(scaled_df.iloc[i - 60:i].values.flatten())  # Flatten window
        y.append(scaled_df.iloc[i, close_index])  # Next day's close

    X, y = np.array(X), np.array(y)

    # Train-test split (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    model = RandomForestRegressor(random_state=42)

    with parallel_backend('threading'):
        grid_search = GridSearchCV(
            model, param_grid, cv=tscv, scoring='r2', n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)

    # Evaluate
    y_pred = grid_search.best_estimator_.predict(X_test)
    print(f"Final R² Accuracy (Test Set): {r2_score(y_test, y_pred):.2f}")


if __name__ == '__main__':
    run_stock_technical_prediction()
