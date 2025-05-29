import os
import json
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit

# Load and sort dataset
df = pd.read_csv("../data/final_dataset.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").dropna().reset_index(drop=True)

# Define feature sets
variants = {
    "Stock Only": ["Open", "High", "Low", "Volume"],
    "Stock + Technical Indicators": ["Open", "High", "Low", "Volume", "EMA_10", "SMA_10", "MACD", "MACD_Signal", "RSI"],
    "Stock + Sentiment": ["Open", "High", "Low", "Volume", "avg_sentiment"],
    "All Combined": ["Open", "High", "Low", "Volume", "EMA_10", "SMA_10", "MACD", "MACD_Signal", "RSI", "avg_sentiment"]
}

summary_metrics = []
window_size = 60
split_index = int(len(df) * 0.8)

for variant_name, features in variants.items():
    folder_name = variant_name.lower().replace(" ", "_").replace("+", "plus").replace("__", "_")
    print(f"\n>>> Running {variant_name}...")

    out_dir = f"../outputs/xgboost_regressor/{folder_name}"
    os.makedirs(out_dir, exist_ok=True)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    results = {
        "test_dates": [],
        "test_actual": [],
        "test_predicted": []
    }

    X_full = train_df[features]
    y_full = train_df["Close"]

    tscv = TimeSeriesSplit(n_splits=3)
    best_model = None
    best_score = float("inf")

    print("Training with TimeSeriesSplit:")
    for fold, (train_idx, val_idx) in enumerate(tqdm(tscv.split(X_full), total=tscv.get_n_splits(), desc="Folds")):
        X_train, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
        y_train, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]

        model_temp = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            eval_metric=["rmse", "mae"],
            early_stopping_rounds=10
        )

        model_temp.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )

        val_pred = model_temp.predict(X_val)
        mae_score = mean_absolute_error(y_val, val_pred)
        print(f"  Fold {fold+1} MAE: {mae_score:.4f}")

        if mae_score < best_score:
            best_score = mae_score
            model = model_temp
            evals_result = model_temp.evals_result()

    combined_df = pd.concat([train_df, test_df]).reset_index(drop=True)

    for i in tqdm(range(split_index, len(combined_df)), desc=f"Forecasting: {variant_name}"):
        start_idx = i - window_size
        if start_idx < 0:
            continue
        X_window = combined_df[features].iloc[start_idx:i]
        X_test = combined_df[features].iloc[[i]]
        y_test = combined_df["Close"].iloc[i]
        y_pred = model.predict(X_test)[0]

        results["test_dates"].append(combined_df["Date"].iloc[i])
        results["test_actual"].append(y_test)
        results["test_predicted"].append(y_pred)

    mae = mean_absolute_error(results["test_actual"], results["test_predicted"])
    rmse = np.sqrt(mean_squared_error(results["test_actual"], results["test_predicted"]))
    r2 = r2_score(results["test_actual"], results["test_predicted"])

    print(f"Metrics for {variant_name}:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2:   {r2:.4f}")

    summary_metrics.append({
        "Variant": variant_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    })

    pd.DataFrame({
        "Date": results["test_dates"],
        "Actual_Close": results["test_actual"],
        "Predicted_Close": results["test_predicted"]
    }).to_csv(f"{out_dir}/predictions.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(results["test_dates"], results["test_actual"], label="Actual")
    plt.plot(results["test_dates"], results["test_predicted"], label="Predicted")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title(f"Forecast Plot: {variant_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/forecast_plot.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(evals_result["validation_0"]["rmse"], label="Train RMSE")
    plt.plot(evals_result["validation_1"]["rmse"], label="Val RMSE")
    plt.plot(evals_result["validation_0"]["mae"], label="Train MAE")
    plt.plot(evals_result["validation_1"]["mae"], label="Val MAE")
    plt.xlabel("Boosting Iteration")
    plt.ylabel("Loss")
    plt.title(f"Training Loss: {variant_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/training_loss_plot.png")
    plt.close()

    with open(f"{out_dir}/evals_result.json", "w") as f:
        json.dump(evals_result, f, indent=4)

pd.DataFrame(summary_metrics).to_csv("../outputs/xgboost_regressor/model_comparison_metrics_xgboost_regressor.csv", index=False)