import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

st.set_page_config(layout="wide")

# load stock data 
df  = pd.read_csv('data/nvidia_stock.csv', parse_dates=['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

st.title("NVIDIA Stock Prediction with Sentiment Analysis")
st.sidebar.header("Navigation")
selection = st.sidebar.selectbox("Go to", ["Project Overview", "Explore Data", 
                                            "Train and predict", "Compare Models"])
if selection == "Project Overview":
    st.subheader("Project Overview")
    st.write("""
    This project aims to predict NVIDIA stock prices using sentiment analysis of YouTube comments.
    The process involves:
    1. Collecting YouTube comments related to NVIDIA.
    2. Performing sentiment analysis on the comments.
    3. Training a machine learning model to predict stock prices based on sentiment scores.
    4. Visualizing the results and comparing different models.
    """)

elif selection == "Explore Data":
    st.header("Explore Data")
    st.subheader("NVIDIA Stock Movement March 2024 - March 2025")
    # Line Graph price movement
    fig = px.line(df, x=df.index, y='Close',
                labels={'Close': 'Closing Price (USD)', 'index': 'Date'})

    st.plotly_chart(fig)

elif selection == "Train and predict":
    
    model_choice = st.sidebar.selectbox("Select Model", ["LSTM", "XGBoost", "Random Forest"])

    if model_choice == "LSTM":
        dataset_choice = st.sidebar.selectbox("Select Dataset", ["Stock", "Stock and Sentiment", 
                                                            "Stock + Technical Indicators",
                                                            "Stock + Sentiment + Technical Indicators"])
        
        if dataset_choice == "Stock":
            st.header("LSTM Model - Stock Data Only")
            data = df[['Close']].values

            scaler = joblib.load('models/scaler_stock.save')
            scaled_data = scaler.transform(data)

            def create_sequences(data, window_size):
                x, y  = [], []
                for i in range(len(data) - window_size):
                    x.append(data[i:i + window_size])
                    y.append(data[i + window_size])
                return np.array(x), np.array(y)
            
            window_size = 60 

            x, y = create_sequences(scaled_data, window_size)
            x = x.reshape((x.shape[0], x.shape[1], 1))
        
            
            model = tf.keras.models.load_model('models/lstm_stock.h5')
            predicted = model.predict(x)

            predicted = scaler.inverse_transform(predicted)
            actual = scaler.inverse_transform(y)

            # Evaluate model performance
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            mae = mean_absolute_error(actual, predicted)
            r2 = r2_score(actual, predicted)

            metrics_data = {
                'Metric': ['RMSE', 'MAE', 'R² Score'],
                'Value': [rmse, mae, r2]
            
            }

            metric_df = pd.DataFrame(metrics_data)

            plt.figure(figsize=(12,6))
            plt.plot(actual, label='Actual Price')
            plt.plot(predicted, label='Predicted Price')
            plt.legend()
            plt.title("LSTM Predictions vs Actual NVIDIA Stock Prices")
            plt.show()

            df_plot = pd.DataFrame({
                'Date':  df.index[window_size:],         # Use same index as actual/predicted
                'Actual Price': actual.flatten(),
                'Predicted Price': predicted.flatten()
            })

            fig = px.line(
                df_plot,
                x='Date',
                y=['Actual Price', 'Predicted Price'],
                labels={'value': 'Closing Price (USD)', 'variable': 'Legend'},
                title='LSTM Predictions using only Closing Price'
            )

            st.plotly_chart(fig)

            st.table(metric_df.reset_index(drop=True))
        
        elif dataset_choice == "Stock and Sentiment":
            st.header("LSTM Model - Stock and Sentiment Data")

            df  = pd.read_csv('data/final_dataset.csv', parse_dates=['Date'])
            df = df.sort_values('Date')
            df.set_index('Date', inplace=True)

            data = df[['Close', 'avg_sentiment']].values

            scaler = joblib.load('models/scaler_stock&sentiment.save')
            scaled_data = scaler.transform(data)

            def create_sequences(data, window_size):
                x, y  = [], []
                for i in range(len(data) - window_size):
                    x.append(data[i:i + window_size])
                    y.append(data[i + window_size][0]) # Predict only the stock price
                return np.array(x), np.array(y)
            
            window_size = 60 

            x, y = create_sequences(scaled_data, window_size)
            x = x.reshape((x.shape[0], x.shape[1], 2))  # 2 features: Close price and sentiment
        
            
            model = tf.keras.models.load_model('models/lstm_stock&sentiment.h5')
            predicted = model.predict(x)

            predicted_extended = np.zeros((predicted.shape[0], 2))
            predicted_extended[:, 0] = predicted[:, 0]

            actual_extended = np.zeros((y.shape[0], 2))
            actual_extended[:, 0] = y

            predicted_inversed = scaler.inverse_transform(predicted_extended)[:, 0]
            actual_inversed = scaler.inverse_transform(actual_extended)[:, 0]

            # Evaluate model performance
            rmse = np.sqrt(mean_squared_error(actual_inversed, predicted_inversed))
            mae = mean_absolute_error(actual_inversed, predicted_inversed)
            r2 = r2_score(actual_inversed, predicted_inversed)

            metrics_data = {
                'Metric': ['RMSE', 'MAE', 'R² Score'],
                'Value': [rmse, mae, r2]
            
            }

            metric_df = pd.DataFrame(metrics_data)

            # Plotting the actual vs predicted prices
            df_plot = pd.DataFrame({
                'Date':  df.index[window_size:],         # Use same index as actual/predicted
                'Actual Price': actual_inversed,
                'Predicted Price': predicted_inversed
            })

            fig = px.line(
                df_plot,
                x='Date',
                y=['Actual Price', 'Predicted Price'],
                labels={'value': 'Closing Price (USD)', 'variable': 'Legend'},
                title='LSTM Predictions using only Closing Price'
            )

            st.plotly_chart(fig)

            st.table(metric_df.reset_index(drop=True))










elif selection == "Compare Models":
    st.header("Compare Models")
    comparison_df = pd.DataFrame({
    "Model": ["LSTM", "XGBoost Regressor", "Random Forest"],
    "MSE": ["", "", ""],
    "RMSE": ["", "", ""],
    "MAE": ["", "", ""],
    "R² Score": ["", "", ""]
    })
    st.table(comparison_df)








