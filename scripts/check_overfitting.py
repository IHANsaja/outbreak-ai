import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
import joblib
import os
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = r'd:\Projects\outbreakAI\outbreak-ai\datasets\water_levels_90_rivers_ready.csv'
LSTM_MODEL_PATH = r'd:\Projects\outbreakAI\outbreak-ai\models\flood_lstm_retrained_accurate.pth'
XGB_MODEL_PATH = r'd:\Projects\outbreakAI\outbreak-ai\models\flood_xgboost_retrained_accurate.pkl'
OUTPUT_DIR = r'd:\Projects\outbreakAI\outbreak-ai\reports\diagnostics'

FEATURES = ['station_id', 'river_id', 'hour', 'month', 'alert_level', 
            'minor_flood', 'major_flood', 'water_level_lag1', 
            'water_level_lag2', 'rainfall_roll3']
TARGET = 'water_level_now'

class FloodLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(FloodLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(DEVICE)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, target, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(target[i + window_size])
    return np.array(X), np.array(y)

def calculate_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred))
    }

def run_diagnostic():
    print(f"Starting Overfitting Diagnostic on {DEVICE}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load & Split Data (Same as evaluation)
    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=FEATURES + [TARGET])
    
    train_dfs, test_dfs = [], []
    for station in df['station_id'].unique():
        s_df = df[df['station_id'] == station].sort_values('datetime')
        if len(s_df) < 50: continue
        split_idx = int(0.8 * len(s_df))
        train_dfs.append(s_df.iloc[:split_idx])
        test_dfs.append(s_df.iloc[split_idx:])
    
    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)
    print(f"Data: {len(train_df)} train, {len(test_df)} test rows.")

    diagnostic_results = {}

    # 2. XGBoost Feature Importance & Overfitting
    print("\nAnalyzing XGBoost...")
    if os.path.exists(XGB_MODEL_PATH):
        model_xgb = joblib.load(XGB_MODEL_PATH)
        
        # Train Metrics
        train_preds = model_xgb.predict(train_df[FEATURES])
        test_preds = model_xgb.predict(test_df[FEATURES])
        
        diagnostic_results["XGBoost"] = {
            "train": calculate_metrics(train_df[TARGET], train_preds),
            "test": calculate_metrics(test_df[TARGET], test_preds)
        }
        
        # Feature Importance
        importance = model_xgb.feature_importances_
        feat_imp = pd.Series(importance, index=FEATURES).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        feat_imp.plot(kind='bar')
        plt.title("XGBoost Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "xgb_feature_importance.png"))
        print(f"XGBoost Feature Importance plot saved.")
        
        # Overfitting Gap
        gap = diagnostic_results["XGBoost"]["test"]["MAE"] - diagnostic_results["XGBoost"]["train"]["MAE"]
        print(f"XGBoost MAE Gap (Test-Train): {gap:.5f}")

    # 3. LSTM Overfitting
    print("\nAnalyzing LSTM...")
    if os.path.exists(LSTM_MODEL_PATH):
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        scaler_x.fit(train_df[FEATURES])
        scaler_y.fit(train_df[[TARGET]])
        
        lstm_model = FloodLSTM(len(FEATURES), 128, 2, 1).to(DEVICE)
        lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=DEVICE))
        lstm_model.eval()
        
        def get_lstm_metrics(dataset_df, limit=50000): # Limit rows for speed in diagnostic
            if len(dataset_df) > limit:
                dataset_df = dataset_df.iloc[:limit]
            
            X_scaled = scaler_x.transform(dataset_df[FEATURES])
            y_scaled = scaler_y.transform(dataset_df[[TARGET]])
            X_seq, y_seq = create_sequences(X_scaled, y_scaled, 12)
            
            X_seq_t = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                preds_scaled = lstm_model(X_seq_t).cpu().numpy()
            
            preds = scaler_y.inverse_transform(preds_scaled).flatten()
            actuals = scaler_y.inverse_transform(y_seq).flatten()
            return calculate_metrics(actuals, preds)

        diagnostic_results["LSTM"] = {
            "train": get_lstm_metrics(train_df),
            "test": get_lstm_metrics(test_df)
        }
        
        gap_lstm = diagnostic_results["LSTM"]["test"]["MAE"] - diagnostic_results["LSTM"]["train"]["MAE"]
        print(f"LSTM MAE Gap (Test-Train): {gap_lstm:.5f}")

    # 4. Persistence Baseline Comparison (Naive Forecast: y_t = y_{t-1})
    print("\nComparing with Persistence Baseline...")
    p_actuals = test_df[TARGET].iloc[1:]
    p_preds = test_df[TARGET].iloc[:-1] # Predict current as previous
    persistence_mae = mean_absolute_error(p_actuals, p_preds)
    diagnostic_results["Persistence_Baseline"] = {"MAE": float(persistence_mae)}
    print(f"Persistence Baseline MAE: {persistence_mae:.5f}")

    # Save Diagnostic Results
    with open(os.path.join(OUTPUT_DIR, "diagnostic_results.json"), "w") as f:
        json.dump(diagnostic_results, f, indent=4)
    print(f"\nDiagnostic report saved to {OUTPUT_DIR}/diagnostic_results.json")

if __name__ == "__main__":
    run_diagnostic()
