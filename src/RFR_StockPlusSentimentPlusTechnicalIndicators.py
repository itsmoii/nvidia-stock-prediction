import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns  # Import seaborn correctly

# Streamlit configuration
st.set_page_config(layout="wide", page_title="Stock Price Prediction")
st.title("NVIDIA Stock Price Prediction with Random Forest Regressor")

# Sidebar parameters
st.sidebar.header("Model Parameters")
n_estimators = st.sidebar.selectbox("Number of estimators", [100, 200, 300], index=1)
max_depth = st.sidebar.selectbox("Max depth", [None, 10, 20, 30], index=2)
test_size = st.sidebar.slider("Test set size (%)", 10, 40, 20)

# Progress bar
progress_bar = st.progress(0)
status_text = st.empty()

def update_progress(step, total_steps=6):
    progress_bar.progress(int((step / total_steps) * 100))

# --- Data Loading and Preprocessing ---
update_progress(1)
status_text.text("Loading and preprocessing data...")

@st.cache_data
def load_and_preprocess_data():
    # Fixed file path - adjust according to your folder structure
    df = pd.read_csv('Data/nvidia_stock.csv')  # ✅ Corrected path
    df = df.iloc[2:].copy()
    df = df.rename(columns={'Price': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    # Convert numeric columns
    numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.ffill().copy()

    return df

df = load_and_preprocess_data()

# --- Feature Engineering ---
update_progress(2)
status_text.text("Calculating technical indicators...")

def calculate_technical_indicators(df):
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['RSI'] = ta.momentum.rsi(df['Close'])
    df['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
    df['Stoch_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
    bollinger = ta.volatility.BollingerBands(close=df['Close'])
    df['BB_Upper'] = bollinger.bollinger_hband()
    df['BB_Lower'] = bollinger.bollinger_lband()
    df['BB_Mid'] = bollinger.bollinger_mavg()
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

    for col in ['Close', 'Volume', 'RSI', 'MACD']:
        for lag in [1, 2, 3]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    df['Future_Close'] = df['Close'].shift(-3)
    df = df.dropna().copy()
    return df

df = calculate_technical_indicators(df)

# Feature selection
features = ['Close', 'High', 'Low', 'Open', 'Volume',
            'MACD', 'MACD_Signal', 'RSI', 'Stoch_K', 'Stoch_D',
            'BB_Upper', 'BB_Lower', 'BB_Mid', 'ATR',
            'Close_lag_1', 'Close_lag_2', 'Close_lag_3',
            'Volume_lag_1', 'Volume_lag_2', 'Volume_lag_3',
            'RSI_lag_1', 'RSI_lag_2', 'RSI_lag_3',
            'MACD_lag_1', 'MACD_lag_2', 'MACD_lag_3']

X = df[features]
y = df['Future_Close']

# --- Train-Test Split ---
update_progress(3)
status_text.text("Splitting data into training and test sets...")

split_idx = int(len(X) * (1 - test_size / 100))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

col1, col2 = st.columns(2)
with col1:
    st.subheader("Training Data")
    st.write(f"Shape: {X_train.shape}")
    st.write(f"Date range: {df.index[0].date()} to {df.index[split_idx - 1].date()}")

with col2:
    st.subheader("Test Data")
    st.write(f"Shape: {X_test.shape}")
    st.write(f"Date range: {df.index[split_idx].date()} to {df.index[-1].date()}")

# --- Model Training ---
update_progress(4)
status_text.text("Training Random Forest Regressor...")

@st.cache_resource
def train_model(X_train, y_train, n_estimators, max_depth):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(
            random_state=42,
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1
        ))
    ])

    param_grid = {
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4],
        'regressor__max_features': ['sqrt', 'log2', 0.5],
        'regressor__bootstrap': [True, False]
    }

    tscv = TimeSeriesSplit(n_splits=5)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    with st.spinner("Training in progress..."):
        grid_search.fit(X_train, y_train)

    return grid_search

grid_search = train_model(X_train, y_train, n_estimators, max_depth)
best_model = grid_search.best_estimator_

st.subheader("Best Model Parameters")
st.write(grid_search.best_params_)

# --- Model Evaluation ---
update_progress(5)
status_text.text("Evaluating model performance...")

y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("RMSE", f"${rmse:.2f}")
col2.metric("MAE", f"${mae:.2f}")
col3.metric("R² Score", f"{r2:.4f}")
col4.metric("Training Size", f"{len(X_train)} days")

# --- Visualization ---
update_progress(6)
status_text.text("Creating visualizations...")

st.subheader("Actual vs Predicted Prices")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index[split_idx:], y_test, color='navy', label='Actual Price', linewidth=2)
ax.plot(df.index[split_idx:], y_pred, color='red', linestyle='--', label='Predicted Price', linewidth=2)

errors = y_test - y_pred
ax.fill_between(df.index[split_idx:], y_pred, y_test,
                where=(errors > 0), facecolor='green', alpha=0.2, interpolate=True, label='Overprediction')
ax.fill_between(df.index[split_idx:], y_pred, y_test,
                where=(errors < 0), facecolor='red', alpha=0.2, interpolate=True, label='Underprediction')

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Price ($)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_title('Actual vs Predicted NVIDIA Stock Prices', fontsize=16, pad=20)
ax.legend(loc='upper left')

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
fig.autofmt_xdate()

st.pyplot(fig)

# Feature importance
feature_importance = best_model.named_steps['regressor'].feature_importances_
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values('Importance', ascending=False).head(15)

st.subheader("Top 15 Important Features")
fig_importance, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
ax.set_title('Feature Importance')
st.pyplot(fig_importance)

status_text.text("Analysis complete!")
progress_bar.empty()
