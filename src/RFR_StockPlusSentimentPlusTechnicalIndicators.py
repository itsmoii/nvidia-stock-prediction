import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from joblib import parallel_backend

# --- Load Data ---
df = pd.read_csv('../data/final_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# --- Feature Selection ---
stock_price_features = ['Close', 'High', 'Low', 'Open', 'Volume']
stock_price_features = [col for col in stock_price_features if col in df.columns]
sentiment_features = ['avg_sentiment'] if 'avg_sentiment' in df.columns else []
technical_indicator_features = [col for col in ['EMA_10', 'SMA_10', 'MACD', 'MACD_Signal', 'RSI'] if col in df.columns]

all_features = sorted(list(set(stock_price_features + sentiment_features + technical_indicator_features)))
if 'Close' not in all_features:
    all_features.append('Close')

df_selected = df[['Date'] + all_features].copy()

# --- Preprocessing ---
numeric_cols = [col for col in all_features if pd.api.types.is_numeric_dtype(df_selected[col])]
df_selected[numeric_cols] = df_selected[numeric_cols].ffill().bfill()

# --- Final Features ---
final_features = sorted(list(set(all_features)))
df_processed = df_selected.dropna(subset=final_features)

# --- Scaling ---
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(df_processed[final_features])
close_index = final_features.index('Close')

# --- Create Rolling Window Data ---
look_back = 60
pred_horizon = 1
X, y = [], []
for i in range(look_back, len(scaled_values) - pred_horizon + 1):
    X.append(scaled_values[i - look_back:i, :])
    y.append(scaled_values[i + pred_horizon - 1, close_index])

X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], -1)

# --- Time Series Split & Grid Search ---
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 5]
}

# Use threading backend to avoid joblib errors
with parallel_backend('threading'):
    grid_model = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), param_grid, cv=tscv, scoring='r2')
    grid_model.fit(X, y)
    best_model = grid_model.best_estimator_

# --- Train-Test Split for Final Evaluation ---
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- Train Final Model ---
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
test_accuracy = r2_score(y_test, y_pred)

# --- Final Accuracy ---
print(f"Final R² Accuracy (Test Set): {test_accuracy:.2f}")
