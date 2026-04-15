import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import os

# --- Configuration ---
DATA_DIR = r'd:\Projects\outbreakAI\outbreak-ai\reports\comparisons'
LEADERBOARD_PATH = os.path.join(DATA_DIR, 'comparison_leaderboard.json')
SAMPLE_PATH = os.path.join(DATA_DIR, 'sample_predictions.json')
OUTPUT_DIR = DATA_DIR

# Set modern style
plt.style.use('dark_background')
COLORS = {
    'Actual': '#ffffff',
    'LSTM': '#00d4ff',
    'XGBoost': '#ff4b2b',
    'TFT': '#6200ee'
}

def plot_leaderboard(leaderboard):
    models = list(leaderboard.keys())
    metrics = ['MAE', 'RMSE', 'R2']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model Performance Comparison', fontsize=20, fontweight='bold', color='white', y=1.05)

    for i, metric in enumerate(metrics):
        values = [leaderboard[m][metric] for m in models]
        bars = axes[i].bar(models, values, color=[COLORS.get(m, '#ffffff') for m in models], alpha=0.8)
        axes[i].set_title(metric, fontsize=16, color='white')
        axes[i].grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), 
                         va='bottom', ha='center', color='white', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'leaderboard_metrics.png'), dpi=300, bbox_inches='tight')
    print("Leaderboard metrics plot saved.")

def plot_battle(sample_data):
    actual = sample_data['actual']
    lstm = sample_data['LSTM']
    xgboost = sample_data['XGBoost']
    dates = sample_data['dates']
    station_id = sample_data['station_id']

    # Sample only every N points if too dense
    if len(dates) > 200:
        step = len(dates) // 100
        actual = actual[::step]
        lstm = lstm[::step]
        xgboost = xgboost[::step]
        dates = dates[::step]

    plt.figure(figsize=(15, 7))
    plt.plot(dates, actual, label='Actual Water Level', color=COLORS['Actual'], linewidth=2.5, alpha=0.9)
    plt.plot(dates, lstm, label='LSTM Prediction', color=COLORS['LSTM'], linewidth=2, linestyle='--', alpha=0.8)
    plt.plot(dates, xgboost, label='XGBoost Prediction', color=COLORS['XGBoost'], linewidth=2, linestyle=':', alpha=0.8)

    plt.title(f'Battle Plot: Model Forecasts vs Reality (Station {station_id})', fontsize=18, fontweight='bold')
    plt.xlabel('Time (Sampled Steps)', fontsize=14)
    plt.ylabel('Water Level', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(fontsize=12, facecolor='#222222', edgecolor='white')
    plt.grid(alpha=0.1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'battle_plot_sample.png'), dpi=300, bbox_inches='tight')
    print("Battle plot saved.")

def main():
    # 1. Load Leaderboard
    if os.path.exists(LEADERBOARD_PATH):
        with open(LEADERBOARD_PATH, 'r') as f:
            leaderboard = json.load(f)
        plot_leaderboard(leaderboard)
    else:
        print("Leaderboard file not found.")

    # 2. Load Sample Predictions
    if os.path.exists(SAMPLE_PATH):
        with open(SAMPLE_PATH, 'r') as f:
            sample_data = json.load(f)
        plot_battle(sample_data)
    else:
        print("Sample predictions file not found.")

if __name__ == "__main__":
    main()
