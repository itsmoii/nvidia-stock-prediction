import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

st.set_page_config(page_title="NVIDIA Stock Price Prediction", layout="wide")

st.title("🖥️ NVIDIA Stock Price Prediction with Random Forest (No Date Fallback)")
st.markdown("---")

st.sidebar.header("Model Parameters")
look_back_days = st.sidebar.slider("Look Back Days", 3, 10, 5)
n_estimators = st.sidebar.slider("Number of Trees", 50, 200, 100)


@st.cache_data
def load_and_process_data():
    try:
        data = pd.read_csv('Data/nvidia_stock.csv')
        st.write("Columns in CSV:", list(data.columns))

        # Check if date column exists
        if 'date' not in [col.lower() for col in data.columns]:
            st.warning("⚠️ No 'date' column found in CSV. Using row index as date proxy for demo only.")
            # Rename Close column to 'close' (case insensitive)
            close_col = None
            for col in data.columns:
                if col.lower() == 'close':
                    close_col = col
                    break
            if close_col is None:
                raise ValueError("CSV must contain a 'Close' column for price data.")

            data = data.rename(columns={close_col: 'close'})
            # Use index as pseudo-date
            data['date'] = pd.to_datetime(data.index, unit='D', origin=pd.Timestamp('2000-01-01'))
            return data[['date', 'close']]
        else:
            # If date exists, process normally
            data.columns = [col.lower() for col in data.columns]
            data['date'] = pd.to_datetime(data['date'])
            if 'close' not in data.columns:
                if 'adj close' in data.columns:
                    data = data.rename(columns={'adj close': 'close'})
                else:
                    raise ValueError("CSV must contain a 'close' or 'adj close' column")
            data = data.sort_values('date').reset_index(drop=True)
            return data[['date', 'close']]
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def run_prediction(stock_data, look_back, n_est):
    dataset = stock_data['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    rf = RandomForestRegressor(n_estimators=n_est, random_state=42)
    rf.fit(X_train, y_train)
    train_predict = rf.predict(X_train)
    test_predict = rf.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
    test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    train_accuracy = r2_score(y_train, train_predict)
    test_accuracy = r2_score(y_test, test_predict)
    train_dates = stock_data['date'][look_back:split + look_back].values
    test_dates = stock_data['date'][split + look_back:].values
    return {
        'train_predict': train_predict.flatten(),
        'test_predict': test_predict.flatten(),
        'train_dates': train_dates,
        'test_dates': test_dates,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'stock_data': stock_data
    }


nvidia_data = load_and_process_data()

if nvidia_data is not None and len(nvidia_data) > 0:
    if st.button("🚀 Run Prediction", type="primary"):
        with st.spinner('Training model and making predictions...'):
            results = run_prediction(nvidia_data, look_back_days, n_estimators)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Train R² Score", value=f"{results['train_accuracy']:.4f}")
            with col2:
                st.metric(label="Test R² Score", value=f"{results['test_accuracy']:.4f}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results['stock_data']['date'],
                y=results['stock_data']['close'],
                mode='lines',
                name='Actual Price',
                line=dict(color='blue', width=2),
                hovertemplate='<b>Actual Price</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=results['train_dates'],
                y=results['train_predict'],
                mode='lines',
                name='Train Prediction',
                line=dict(color='green', width=2),
                opacity=0.7,
                hovertemplate='<b>Train Prediction</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=results['test_dates'],
                y=results['test_predict'],
                mode='lines',
                name='Test Prediction',
                line=dict(color='red', width=2),
                hovertemplate='<b>Test Prediction</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ))
            fig.update_layout(
                title='NVIDIA Stock Price Prediction with Random Forest',
                xaxis_title='Date',
                yaxis_title='Close Price ($)',
                hovermode='x unified',
                width=1000,
                height=600,
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")
            st.subheader("📊 Model Information")
            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.info(f"**Look Back Days:** {look_back_days}")
                st.info(f"**Number of Trees:** {n_estimators}")
            with info_col2:
                st.info(f"**Train Split:** 80%")
                st.info(f"**Test Split:** 20%")
            with info_col3:
                st.info(f"**Data Points:** {len(nvidia_data)}")
                st.info(f"**Stock:** NVIDIA (NVDA)")
else:
    st.error("Unable to load data. Please check if the data file exists and is formatted correctly.")
