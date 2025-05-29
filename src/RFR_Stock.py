import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# Load the stock price data (replace 'final_dataset.csv' with your actual file name)
df = pd.read_csv('../data/final_dataset.csv')

# Keep only stock price-related columns
df_stock = df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']].dropna()
df_stock['Date'] = pd.to_datetime(df_stock['Date'])
df_stock = df_stock.sort_values('Date').reset_index(drop=True)

# Normalize the features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df_stock[['Close', 'High', 'Low', 'Open', 'Volume']])

# Set look-back period and prediction horizon
look_back = 60
pred_horizon = 1

# Create rolling windows for features and targets
X, y = [], []
for i in range(look_back, len(scaled_features) - pred_horizon + 1):
    X.append(scaled_features[i - look_back:i])
    y.append(scaled_features[i:i + pred_horizon, 0])  # Predicting 'Close' price

X, y = np.array(X), np.array(y).flatten()

# Flatten X for Random Forest (convert 3D -> 2D)
X = X.reshape(X.shape[0], -1)

# Split data into train/test (80/20 split)
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Train Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on test set
y_pred = rf.predict(X_test)

# Calculate R² score (accuracy)
test_accuracy = r2_score(y_test, y_pred)
print(f"Final R² Accuracy (Test Set): {test_accuracy:.2f}")
