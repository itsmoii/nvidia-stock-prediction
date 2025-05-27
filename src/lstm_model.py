import numpy as np
import pandas as pd
import plotly.express as px
import joblib
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def create_sequences(data, window_size):
    x, y = [], []
    for i in range(len(data) - window_size):
        x.append(data[i:i + window_size])
        y.append(data[i + window_size][0])
    return np.array(x), np.array(y)

def evaluate_lstm(model_path, scaler_path, data, window_size, feature_count):
    scaler = joblib.load(scaler_path)
    scaled_data = scaler.transform(data)
    x, y = create_sequences(scaled_data, window_size)
    x = x.reshape((x.shape[0], x.shape[1], feature_count))
    
    model = tf.keras.models.load_model(model_path)
    predicted = model.predict(x)
    
    predicted_extended = np.zeros((predicted.shape[0], feature_count))
    predicted_extended[:, 0] = predicted[:, 0]
    actual_extended = np.zeros((y.shape[0], feature_count))
    actual_extended[:, 0] = y

    predicted_inversed = scaler.inverse_transform(predicted_extended)[:, 0]
    actual_inversed = scaler.inverse_transform(actual_extended)[:, 0]

    rmse = np.sqrt(mean_squared_error(actual_inversed, predicted_inversed))
    mae = mean_absolute_error(actual_inversed, predicted_inversed)
    r2 = r2_score(actual_inversed, predicted_inversed)

    return predicted_inversed, actual_inversed, {"RMSE": rmse, "MAE": mae, "R2": r2}

def plot_predictions(dates, actual, predicted, title):
    df_plot = pd.DataFrame({
        'Date': dates,
        'Actual Price': actual,
        'Predicted Price': predicted
    })

    fig = px.line(
        df_plot,
        x='Date',
        y=['Actual Price', 'Predicted Price'],
        labels={'value': 'Closing Price (USD)', 'variable': 'Legend'},
        title=title
    )

    return fig
