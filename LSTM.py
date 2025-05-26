import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

# Load dataset
df = pd.read_csv('data/final_dataset.csv', parse_dates=['Date'])
df = df.dropna().reset_index(drop=True)

# Feature and target columns
features = ['Close', 'Volume', 'SMA_10', 'EMA_10', 'MACD', 'MACD_Signal', 'RSI', 'avg_sentiment']
target = 'Close'

# Parameters
sequence_length = 30
forecast_horizon = 1  # 1-day ahead prediction

def rolling_lstm_forecast(df, sequence_length, forecast_horizon):
    predictions, actuals, test_dates = [], [], []

    for i in range(sequence_length, len(df) - forecast_horizon):
        # Split train/test windows
        train_df = df.iloc[i - sequence_length:i]
        test_df = df.iloc[i + forecast_horizon - 1]  # Single row for prediction day

        # Save the test date and actual close
        test_dates.append(test_df['Date'])
        actuals.append(test_df['Close'])

        # Scale train + test together to keep them aligned
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_df[features + [target]])
        test_scaled = scaler.transform([test_df[features + [target]].values])

        # Create single training sequence
        X_train = np.expand_dims(train_scaled[:, :-1], axis=0)
        y_train = np.array([train_scaled[-1, -1]])  # last 'Close' value in training set

        # Build and train LSTM model
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=5, batch_size=1, verbose=0)

        # Create test input sequence
        test_sequence = np.expand_dims(train_scaled[:, :-1], axis=0)  # Use same X for predicting next day
        pred_scaled = model.predict(test_sequence, verbose=0)

        # Inverse scale prediction
        dummy = np.zeros((1, len(features)))
        pred_combined = np.hstack((dummy, pred_scaled))
        prediction = scaler.inverse_transform(pred_combined)[0, -1]

        predictions.append(prediction)

    

    return predictions, actuals, test_dates


print("Starting Rolling LSTM Forecast...")

predictions, actuals, test_dates = rolling_lstm_forecast(df, sequence_length, forecast_horizon)

# Evaluation
mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
r2 = r2_score(actuals, predictions)

print(f"\nFinal Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(test_dates, actuals, label='Actual Close')
plt.plot(test_dates, predictions, label='Predicted Close')
plt.title('Rolling LSTM Prediction vs Actual Close Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()
