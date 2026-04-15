import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import joblib
import json

# --- Google Colab Setup ---
# Upload 'water_levels_90_rivers_ready.csv' to the /content/ folder before running.
DATA_PATH = '/content/water_levels_90_rivers_ready.csv'
LSTM_MODEL_NAME = 'flood_lstm_retrained_accurate.pth'
XGB_MODEL_NAME = 'flood_xgboost_retrained_accurate.pkl'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {DEVICE}")

# 1. Architecture Definitions
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

def run_retrain_pipeline():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}. Please upload it to Colab.")
        return

    # --- 2. Load & Preprocess ---
    print("Loading 1.5M row dataset...")
    df = pd.read_csv(DATA_PATH)
    
    features = ['station_id', 'river_id', 'hour', 'month', 'alert_level', 
                'minor_flood', 'major_flood', 'water_level_lag1', 
                'water_level_lag2', 'rainfall_roll3']
    target = 'water_level_now'

    df = df.dropna(subset=features + [target])
    
    # Chronological Split (80/20 per station)
    print("Splitting data chronologically...")
    train_dfs, test_dfs = [], []
    for station_id in df['station_id'].unique():
        s_df = df[df['station_id'] == station_id].sort_values('datetime')
        if len(s_df) < 50: continue # Skip stations with insufficient data
        split_idx = int(0.8 * len(s_df))
        train_dfs.append(s_df.iloc[:split_idx])
        test_dfs.append(s_df.iloc[split_idx:])
    
    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)
    print(f"Split: {len(train_df)} train rows, {len(test_df)} test rows")

    # --- 3. XGBoost Training (High Accuracy via GridSearch) ---
    print("\nStarting XGBoost Retraining...")
    X_train_xgb, y_train_xgb = train_df[features], train_df[target]
    X_test_xgb, y_test_xgb = test_df[features], test_df[target]

    param_grid = {
        'n_estimators': [500, 1000],
        'max_depth': [6, 10],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.8, 1.0]
    }
    xgb_base = xgb.XGBRegressor(objective='reg:squarederror', tree_method='gpu_hist' if torch.cuda.is_available() else 'hist', random_state=42)
    grid_search = GridSearchCV(xgb_base, param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=1)
    grid_search.fit(X_train_xgb, y_train_xgb)
    
    best_xgb = grid_search.best_estimator_
    joblib.dump(best_xgb, XGB_MODEL_NAME)
    print(f"XGBoost saved to {XGB_MODEL_NAME}")

    # --- 4. LSTM Training (High Accuracy - 100 Epochs) ---
    print("\nStarting LSTM Retraining...")
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_x.fit(train_df[features])
    scaler_y.fit(train_df[[target]])

    # Generate Sequences per station
    WINDOW_SIZE = 12
    X_train_seq_list, y_train_seq_list = [], []
    X_test_seq_list, y_test_seq_list = [], []

    print("Generating sequences...")
    for station_id in train_df['station_id'].unique():
        s_train = train_df[train_df['station_id'] == station_id]
        X_s, y_s = create_sequences(scaler_x.transform(s_train[features]), scaler_y.transform(s_train[[target]]), WINDOW_SIZE)
        X_train_seq_list.append(X_s); y_train_seq_list.append(y_s)

    for station_id in test_df['station_id'].unique():
        s_test = test_df[test_df['station_id'] == station_id]
        X_s, y_s = create_sequences(scaler_x.transform(s_test[features]), scaler_y.transform(s_test[[target]]), WINDOW_SIZE)
        X_test_seq_list.append(X_s); y_test_seq_list.append(y_s)

    X_train_l = np.concatenate(X_train_seq_list); y_train_l = np.concatenate(y_train_seq_list)
    X_test_l = np.concatenate(X_test_seq_list); y_test_l = np.concatenate(y_test_seq_list)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_l, dtype=torch.float32).to(DEVICE), 
                                            torch.tensor(y_train_l, dtype=torch.float32).to(DEVICE)), 
                              batch_size=256, shuffle=True)

    lstm_model = FloodLSTM(len(features), 128, 2, 1).to(DEVICE)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    EPOCHS = 100
    for epoch in range(EPOCHS):
        lstm_model.train()
        total_loss = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = lstm_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"LSTM Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.7f}")

    torch.save(lstm_model.state_dict(), LSTM_MODEL_NAME)
    print(f"LSTM saved to {LSTM_MODEL_NAME}")

    # --- 5. Final Comparison Report ---
    print("\n--- Final Unified Evaluation (Large Dataset) ---")
    
    # XGB Metrics
    xgb_preds = best_xgb.predict(X_test_xgb)
    xgb_metrics = {
        "MAE": mean_absolute_error(y_test_xgb, xgb_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test_xgb, xgb_preds)),
        "R2": r2_score(y_test_xgb, xgb_preds)
    }

    # LSTM Metrics
    lstm_model.eval()
    with torch.no_grad():
        lt_preds_scaled = lstm_model(torch.tensor(X_test_l, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    lt_preds = scaler_y.inverse_transform(lt_preds_scaled)
    lt_true = scaler_y.inverse_transform(y_test_l)
    
    lstm_metrics = {
        "MAE": mean_absolute_error(lt_true, lt_preds),
        "RMSE": np.sqrt(mean_squared_error(lt_true, lt_preds)),
        "R2": r2_score(lt_true, lt_preds)
    }

    # Result Table
    report = pd.DataFrame({
        "XGBoost": xgb_metrics,
        "LSTM": lstm_metrics
    }).T
    print(report)

    # Save summary
    report.to_json('retrain_results.json')
    print("\nAll tasks complete. Download models and retrain_results.json from Colab files.")

if __name__ == "__main__":
    run_retrain_pipeline()
