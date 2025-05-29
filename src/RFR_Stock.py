import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score
from joblib import parallel_backend


def run_stock_only_prediction():
    try:
        df = pd.read_csv('../Data/final_dataset.csv')
    except FileNotFoundError:
        print("Error: 'final_dataset.csv' not found.")
        return

    if 'Date' not in df.columns:
        print("Error: 'Date' column is required.")
        return

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Automatically detect stock price columns by keywords
    stock_keywords = ['open', 'high', 'low', 'close', 'volume']

    # Find columns matching stock keywords (case-insensitive)
    feature_cols = []

    for col in df.columns:
        col_lower = col.lower()
        for keyword in stock_keywords:
            if keyword in col_lower:
                feature_cols.append(col)
                break

    # Ensure Close is included as target
    if 'Close' not in feature_cols:
        close_candidates = [col for col in df.columns if 'close' in col.lower()]
        if close_candidates:
            feature_cols.append(close_candidates[0])

    if not feature_cols:
        print("Error: No stock price columns found.")
        return

    # Filter dataset to only include Date and feature columns
    df = df[['Date'] + feature_cols].dropna()

    if df.empty or len(df) < 61:
        print("Error: Not enough data after filtering and dropping NaNs.")
        return

    # Scale features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=feature_cols)

    # Set parameters
    look_back = 60
    pred_horizon = 1
    target_col = 'Close'

    # Find target column (Close or close variant)
    if target_col not in feature_cols:
        close_candidates = [col for col in feature_cols if 'close' in col.lower()]
        if close_candidates:
            target_col = close_candidates[0]
        else:
            print("Error: No Close price column found.")
            return

    close_index = feature_cols.index(target_col)

    # Create sequences
    X, y = [], []
    for i in range(look_back, len(scaled_df) - pred_horizon + 1):
        X.append(scaled_df.iloc[i - look_back:i].values)
        y.append(scaled_df.iloc[i + pred_horizon - 1, close_index])

    X = np.array(X).reshape(len(X), -1)
    y = np.array(y)

    # 80/20 train-test split
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: Insufficient train/test data split.")
        return

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    rf = RandomForestRegressor(random_state=42)

    # Use threading backend to avoid joblib errors
    with parallel_backend('threading'):
        grid_search = GridSearchCV(rf, param_grid, cv=tscv, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)

    # Get best model and make predictions
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Calculate and print final accuracy
    test_accuracy = r2_score(y_test, y_pred)
    print(f"Final R² Accuracy (Test Set): {test_accuracy:.2f}")


if __name__ == '__main__':
    run_stock_only_prediction()
