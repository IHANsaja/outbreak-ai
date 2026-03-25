import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

# COLLAB TIP: If using Google Colab, upload your CSV to the 'content' folder.
# Defaulting to Colab-style path if local path is not found.
LOCAL_PATH = r'd:\Projects\outbreakAI\dataset-collect\datasets\water_levels_global_ml.csv'
COLAB_PATH = '/content/water_levels_global_ml.csv'

INPUT_CSV = COLAB_PATH if os.path.exists(COLAB_PATH) else LOCAL_PATH
MODEL_SAVE_NAME = 'flood_lstm_model_v2.pth'

# Device configuration (GPU vs CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Sequence Generation
def create_sequences(data, target, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(target[i + window_size])
    return np.array(X), np.array(y)

# 2. LSTM Model Architecture
class FloodLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(FloodLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) 
        return out

def run_training_pipeline():
    if not os.path.exists(INPUT_CSV):
        print(f"Dataset not found at {INPUT_CSV}. Please check file path.")
        return

    # Load and Clean
    print(f"Loading dataset from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    features = ['station_id', 'river_id', 'hour', 'month', 'alert_level', 
                'minor_flood', 'major_flood', 'water_level_lag1', 
                'water_level_lag2', 'rainfall_roll3']
    target = 'water_level_now'

    df = df.dropna(subset=features + [target])
    df = df[~df[features + [target]].isin([np.inf, -np.inf]).any(axis=1)]

    # 2. Per-Station Sequence Generation and Chronological Split
    print("Normalizing and generating sequences...")
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # Fit scalers on the whole dataset first to ensure consistent ranges
    scaler_x.fit(df[features])
    scaler_y.fit(df[[target]])

    WINDOW_SIZE = 12
    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []

    # Iterate through each station to split its local history correctly
    for station_id in df['station_id'].unique():
        station_df = df[df['station_id'] == station_id]
        
        # Scale locally for this station (optional, but we use global scaler for simplicity)
        X_station = scaler_x.transform(station_df[features])
        y_station = scaler_y.transform(station_df[[target]])
        
        # Create sequences for this station's timeline
        X_s, y_s = create_sequences(X_station, y_station, WINDOW_SIZE)
        
        if len(X_s) < 1:
            continue # Skip stations with too little data
            
        # Split this station's data chronologically: 80% old (train), 20% new (test)
        split_idx = int(0.8 * len(X_s))
        
        X_train_list.append(X_s[:split_idx])
        X_test_list.append(X_s[split_idx:])
        y_train_list.append(y_s[:split_idx])
        y_test_list.append(y_s[split_idx:])

    # Combine all results
    X_train = np.concatenate(X_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    print(f"Total sequences created: {len(X_train) + len(X_test)}")
    print(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    # Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Initialize Model & Optimizer
    model = FloodLSTM(len(features), 128, 2, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Training
    EPOCHS = 100
    batch_size = 128
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True) # Shuffle training batches but within per-station timelines
    
    train_losses = []
    
    print(f"Starting Training for {EPOCHS} Epochs on {device}...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for b_x, b_y in train_loader:
            optimizer.zero_grad()
            out = model(b_x)
            loss = criterion(out, b_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_train_loss:.7f}")

    # Evaluation
    print("\nFinalizing Evaluation...")
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test_t).cpu().numpy()
        y_true_scaled = y_test_t.cpu().numpy()
    
    # Inverse scaling to get real-world metrics (meters)
    preds = scaler_y.inverse_transform(preds_scaled)
    y_true = scaler_y.inverse_transform(y_true_scaled)

    mse = mean_squared_error(y_true, preds)
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)
    rmse = np.sqrt(mse)

    print("\n--- Model Accuracy Metrics (Real Meters) ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score (Confidence): {r2:.4f}")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss over 100 Epochs')
    plt.yscale('log')
    plt.legend()
    plt.show() # In Colab, this shows the plot instantly

    torch.save(model.state_dict(), MODEL_SAVE_NAME)
    print(f"\nModel architecture saved as {MODEL_SAVE_NAME}")

if __name__ == "__main__":
    run_training_pipeline()
