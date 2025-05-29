import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score
from joblib import parallel_backend


def run_stock_sentiment_prediction():
    # Load data
    try:
        df = pd.read_csv('../Data/final_dataset.csv', parse_dates=['Date'])
    except FileNotFoundError:
        print("Error: 'final_dataset.csv' not found.")
        return
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    # Validate required columns
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'avg_sentiment']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns - {missing_cols}")
        return

    # Prepare data
    df = df[required_cols].dropna().sort_values('Date')
    if len(df) < 61:
        print("Error: Insufficient data (need at least 61 days).")
        return

    # Scale features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.drop(columns=['Date']))
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns.drop('Date'))

    # Create time-series dataset (60-day lookback, 1-day ahead)
    X, y = [], []
    close_index = scaled_df.columns.get_loc('Close')

    for i in range(60, len(scaled_df)):
        X.append(scaled_df.iloc[i - 60:i].values.flatten())  # Flatten 60-day window
        y.append(scaled_df.iloc[i, close_index])  # Next day's close price

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

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    model = RandomForestRegressor(random_state=42)

    # Train with threading
    with parallel_backend('threading'):
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=tscv,
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)

    # Evaluate
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    print(f"Final R² Accuracy (Test Set): {r2:.2f}")


if __name__ == '__main__':
    run_stock_sentiment_prediction()
