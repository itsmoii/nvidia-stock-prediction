import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score
from joblib import parallel_backend

def run_stock_sentiment_prediction():
    try:
        df = pd.read_csv('../data/final_dataset.csv')
    except FileNotFoundError:
        print("Error: 'final_dataset.csv' not found.")
        return

    # --- User Configuration ---
    user_sentiment_cols = ['compound', 'neg', 'neu', 'pos']  # Change based on your dataset
    stock_feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    # ---------------------------

    if 'Date' not in df.columns:
        print("Error: 'Date' column is required.")
        return

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    stock_cols = [col for col in stock_feature_cols if col in df.columns]
    sentiment_cols = [col for col in user_sentiment_cols if col in df.columns]

    if 'Close' not in stock_cols and 'Close' in df.columns:
        stock_cols.append('Close')

    selected_cols = stock_cols + sentiment_cols
    if not selected_cols:
        print("Error: No valid stock or sentiment columns found.")
        return

    df = df[['Date'] + selected_cols].dropna()
    if df.empty or len(df) < 61:
        print("Error: Not enough data after filtering and dropping NaNs.")
        return

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[selected_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=selected_cols)

    look_back = 60
    pred_horizon = 1
    target_col = 'Close'
    close_index = selected_cols.index(target_col)

    X, y = [], []
    for i in range(look_back, len(scaled_df) - pred_horizon + 1):
        X.append(scaled_df.iloc[i - look_back:i].values)
        y.append(scaled_df.iloc[i + pred_horizon - 1, close_index])

    X = np.array(X).reshape(len(X), -1)
    y = np.array(y)

    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: Insufficient train/test data split.")
        return

    # --- Grid Search with TimeSeriesSplit and thread backend to avoid error ---
    param_grid = {
        'n_estimators': [100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    rf = RandomForestRegressor(random_state=42)

    with parallel_backend('threading'):
        grid_search = GridSearchCV(rf, param_grid, cv=tscv, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    test_accuracy = r2_score(y_test, y_pred)
    print(f"Final R² Accuracy (Test Set): {test_accuracy:.2f}")

if __name__ == '__main__':
    run_stock_sentiment_prediction()
