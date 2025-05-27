import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.lstm_model import evaluate_lstm, plot_predictions
import tensorflow as tf

st.set_page_config(layout="wide")

# load stock data 
df  = pd.read_csv('data/final_dataset.csv', parse_dates=['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)
df.dropna(inplace=True)
window_size=60

lstm_metrics_summary = []

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
        # Stock only 
        data1 = df[['Close']].values
        scaler_path1 = 'models/scaler_stock.save'
        model_path1 = 'models/lstm_stock.h5'
        feature_count1 = 1

        predicted1, actual1, metrics1 = evaluate_lstm(model_path1, scaler_path1, data1, window_size, feature_count1)
        lstm_metrics_summary.append({
            "Model": "Stock Only",
            "RMSE": metrics1["RMSE"],
            "MAE": metrics1["MAE"],
            "R2": metrics1["R2"]
        })

        fig1 = plot_predictions(df.index[window_size:], actual1, predicted1, "Stock Only")

        # Stock and Sentiment
        data2 = df[['Close', 'avg_sentiment']].values

        scaler_path2 = 'models/scaler_stock&sentiment.save'
        model_path2 = 'models/lstm_stock&sentiment.h5'
        feature_count2 =2

        predicted2, actual2, metrics2 = evaluate_lstm(model_path2, scaler_path2, data2, window_size, feature_count2)
        
        lstm_metrics_summary.append({
            "Model": "Stock and Sentiment",
            "RMSE": metrics2["RMSE"],
            "MAE": metrics2["MAE"],
            "R2": metrics2["R2"]
        })

        fig2 = plot_predictions(df.index[window_size:], actual2, predicted2, "Stock and Sentiment")

        # Stock and Technical Indicators
        data3 = df[['Close', 'EMA_10', 'SMA_10', 'MACD', 'MACD_Signal', 'RSI']].values
            
        scaler_path3 = 'models/scaler_stock&indicators.save'
        model_path3 = 'models/lstm_stock&indicators.h5'
        feature_count3=6

        predicted3, actual3, metrics3 = evaluate_lstm(model_path3, scaler_path3, data3, window_size, feature_count3)

        lstm_metrics_summary.append({
            "Model": "Stock and Indicators",
            "RMSE": metrics3["RMSE"],
            "MAE": metrics3["MAE"],
            "R2": metrics3["R2"]
        })

        fig3 = plot_predictions(df.index[window_size:], actual3, predicted3, "Stock and Indicators")

        # Combined Stock, Sentiment, and Technical Indicators
        data4 = df[['Close', 'EMA_10', 'SMA_10', 'MACD', 'MACD_Signal', 'RSI', 'avg_sentiment', 'Volume']].values

        scaler_path4 = 'models/scaler_combined.save'
        model_path4 = 'models/lstm_combined.h5'
        feature_count4=8

        predicted4, actual4, metrics4 = evaluate_lstm(model_path4, scaler_path4, data4, window_size, feature_count4)
        lstm_metrics_summary.append({
            "Model": "Stock + Indicators + Sentiment",
            "RMSE": metrics4["RMSE"],
            "MAE": metrics4["MAE"],
            "R2": metrics4["R2"]
        })

        fig4 = plot_predictions(df.index[window_size:], actual4, predicted4, "Stock + Indicators + Sentiment")


        dataset_choice = st.sidebar.selectbox("Select Dataset", ["Stock", "Stock and Sentiment", 
                                                            "Stock + Technical Indicators",
                                                            "Stock + Sentiment + Technical Indicators",
                                                            "comparison"])
        
        if dataset_choice == "Stock":
            st.header("LSTM Model - Stock Data Only")
            st.plotly_chart(fig1)
            st.subheader("Evaluation Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{metrics1['RMSE']:.3f}")
            col2.metric("MAE", f"{metrics1['MAE']:.3f}")
            col3.metric("R² Score", f"{metrics1['R2']:.3f}")
       
        
        elif dataset_choice == "Stock and Sentiment":
            st.header("LSTM Model - Stock and Sentiment Data")
            st.plotly_chart(fig2)
            st.subheader("Evaluation Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{metrics2['RMSE']:.3f}")
            col2.metric("MAE", f"{metrics2['MAE']:.3f}")
            col3.metric("R² Score", f"{metrics2['R2']:.3f}")

       

        elif dataset_choice == "Stock + Technical Indicators":
            st.header("LSTM Model - Stock and Technical Indicators")
            st.plotly_chart(fig3)
            st.subheader("Evaluation Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{metrics3['RMSE']:.3f}")
            col2.metric("MAE", f"{metrics3['MAE']:.3f}")
            col3.metric("R² Score", f"{metrics3['R2']:.3f}")

        elif dataset_choice == "Stock + Sentiment + Technical Indicators":
            st.header("LSTM Model - Stock and Technical Indicators")
            st.plotly_chart(fig4)
            st.subheader("Evaluation Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{metrics4['RMSE']:.3f}")
            col2.metric("MAE", f"{metrics4['MAE']:.3f}")
            col3.metric("R² Score", f"{metrics4['R2']:.3f}")

        elif dataset_choice == "comparison":
            st.subheader("📊 LSTM Model Comparison")
            summary_df = pd.DataFrame(lstm_metrics_summary)
            summary_df = summary_df.round(3)
            st.dataframe(summary_df)

            fig_grouped = px.bar(
                summary_df.melt(id_vars="Model", value_vars=["RMSE", "MAE", "R2"]),
                x="Model",
                y="value",
                color="variable",
                barmode="group",
                text="value",
                title="LSTM Models - Metrics Comparison",
                template="plotly_white"
            )
            fig_grouped.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig_grouped, use_container_width=True)


            












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








