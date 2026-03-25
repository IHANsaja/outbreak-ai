import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# Device configuration (GPU vs CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

INPUT_CSV = r'd:\Projects\outbreakAI\dataset-collect\datasets\water_levels_global_ml.csv'
MODEL_PATH = r'd:\Projects\outbreakAI\dataset-collect\scripts\flood_lstm_model.pth'

# 1. Load and Prepare Sequence Data
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
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # Take last output of sequence
        return out

def train_model():
    if not os.path.exists(INPUT_CSV):
        print(f"Dataset not found: {INPUT_CSV}")
        return

    # Load Dataset
    print("Loading dataset...")
    df = pd.read_csv(INPUT_CSV)
    
    # Feature Selection
    features = ['station_id', 'river_id', 'hour', 'month', 'alert_level', 
                'minor_flood', 'major_flood', 'water_level_lag1', 
                'water_level_lag2', 'rainfall_roll3']
    target = 'water_level_now'

    # STRICT Cleanup: ensure no NaNs in features OR target
    df = df.dropna(subset=features + [target])
    
    # Check for infinity
    df = df[~df[features + [target]].isin([np.inf, -np.inf]).any(axis=1)]

    # Normalization (Crucial for Neural Networks)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_data = scaler_x.fit_transform(df[features])
    y_data = scaler_y.fit_transform(df[[target]])

    # Sequence Generation (Using 10 time-steps window)
    WINDOW_SIZE = 10
    print(f"Creating sequences (Window Size: {WINDOW_SIZE})...")
    X_seq, y_seq = create_sequences(X_data, y_data, WINDOW_SIZE)
    
    # Chronological Split
    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # Convert to Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Data Loaders
    batch_size = 64
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=False)

    # Initialize Model
    model = FloodLSTM(input_dim=len(features), hidden_dim=64, num_layers=2, output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) # Lower LR

    # Training Loop
    print("Starting Training...")
    epochs = 3 
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            if torch.isnan(loss):
                print("NaN detected in loss! Skipping batch.")
                continue
                
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient Clipping
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    # Save Model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Evaluation
    model.eval()
    with torch.no_grad(): # Use no_grad for eval
        test_preds = model(X_test_t)
        test_loss = criterion(test_preds, y_test_t).item()
    
    print(f"\nTraining Complete. Test Loss (MSE): {test_loss:.6f}")
    
    # Optional Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.title('LSTM Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.savefig(r'd:\Projects\outbreakAI\dataset-collect\docs\training_history.png')
    print("Training plot saved to docs/training_history.png")

if __name__ == "__main__":
    train_model()
