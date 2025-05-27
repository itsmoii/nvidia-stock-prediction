import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

st.set_page_config(page_title="NVIDIA Stock Price Prediction", layout="wide")

st.title("🖥️ NVIDIA Stock Price Prediction with Random Forest")
st.markdown("---")

st.sidebar.header("Model Parameters")
look_back_days = st.sidebar.slider("Look Back Days", 3, 10, 5)
n_estimators = st.sidebar.slider("Number of Trees", 50, 200, 100)


@st.cache_data
def load_and_process_data():
    try:
        # Read the CSV file
        data = pd.read_csv('Data/nvidia_stock.csv')
        st.write("Original CSV shape:", data.shape)
        st.write("First few rows:")
        st.write(data.head())

        # Skip the first 3 rows (headers, ticker info, empty date row) and read actual data
        data = pd.read_csv('Data/nvidia_stock.csv', skiprows=3)

        # The first column should be the date, rename it properly
        data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

        # Convert date column to datetime
        data['Date'] = pd.to_datetime(data['Date'])

        # Convert Close to numeric (remove any non-numeric characters if present)
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

        # Remove any rows with NaN values
        data = data.dropna()

        # Sort by date
        data = data.sort_values('Date').reset_index(drop=True)

        # Rename columns to lowercase for consistency
        data.columns = [col.lower() for col in data.columns]

        st.success(f"✅ Successfully loaded {len(data)} data points")
        st.write("Data date range:", data['date'].min(), "to", data['date'].max())

        return data[['date', 'close']]

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.write("Attempting alternative loading method...")

        try:
            # Alternative: Read all data and try to identify the actual data rows
            raw_data = pd.read_csv('Data/nvidia_stock.csv')

            # Find the row where actual date data starts
            date_start_row = None
            for i, row in raw_data.iterrows():
                try:
                    # Try to parse the first column as a date
                    pd.to_datetime(str(row.iloc[0]))
                    date_start_row = i
                    break
                except:
                    continue

            if date_start_row is not None:
                # Read from the identified starting row
                data = pd.read_csv('Data/nvidia_stock.csv', skiprows=date_start_row)
                data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
                data['Date'] = pd.to_datetime(data['Date'])
                data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
                data = data.dropna()
                data = data.sort_values('Date').reset_index(drop=True)
                data.columns = [col.lower() for col in data.columns]

                st.success(f"✅ Successfully loaded {len(data)} data points using alternative method")
                return data[['date', 'close']]
            else:
                raise Exception("Could not find valid date data in CSV")

        except Exception as e2:
            st.error(f"Alternative loading method also failed: {e2}")
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

    # Split into train and test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train Random Forest model
    rf = RandomForestRegressor(n_estimators=n_est, random_state=42)
    rf.fit(X_train, y_train)

    # Make predictions
    train_predict = rf.predict(X_train)
    test_predict = rf.predict(X_test)

    # Inverse transform predictions
    train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
    test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate accuracy
    train_accuracy = r2_score(y_train, train_predict)
    test_accuracy = r2_score(y_test, test_predict)

    # Get corresponding dates
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


# Load data
nvidia_data = load_and_process_data()

if nvidia_data is not None and len(nvidia_data) > 0:
    st.write("✅ Data loaded successfully!")
    st.write(f"📊 Dataset contains {len(nvidia_data)} records")
    st.write(
        f"📅 Date range: {nvidia_data['date'].min().strftime('%Y-%m-%d')} to {nvidia_data['date'].max().strftime('%Y-%m-%d')}")

    # Show sample data
    st.subheader("Sample Data")
    st.write(nvidia_data.head(10))

    if st.button("🚀 Run Prediction", type="primary"):
        if len(nvidia_data) > look_back_days + 10:  # Ensure enough data
            with st.spinner('Training model and making predictions...'):
                results = run_prediction(nvidia_data, look_back_days, n_estimators)

                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Train R² Score", value=f"{results['train_accuracy']:.4f}")
                with col2:
                    st.metric(label="Test R² Score", value=f"{results['test_accuracy']:.4f}")

                # Create plot
                fig = go.Figure()

                # Actual prices
                fig.add_trace(go.Scatter(
                    x=results['stock_data']['date'],
                    y=results['stock_data']['close'],
                    mode='lines',
                    name='Actual Price',
                    line=dict(color='blue', width=2),
                    hovertemplate='<b>Actual Price</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ))

                # Training predictions
                fig.add_trace(go.Scatter(
                    x=results['train_dates'],
                    y=results['train_predict'],
                    mode='lines',
                    name='Train Prediction',
                    line=dict(color='green', width=2),
                    opacity=0.7,
                    hovertemplate='<b>Train Prediction</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ))

                # Test predictions
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

                # Model information
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
            st.error(
                f"Not enough data for prediction. Need at least {look_back_days + 10} data points, but only have {len(nvidia_data)}")
else:
    st.error("❌ Unable to load data. Please check if the data file exists and is formatted correctly.")
    st.write("**Expected CSV format:**")
    st.write("- First column should contain dates")
    st.write("- Second column should contain closing prices")
    st.write("- Data should start from row 4 (after headers)")
