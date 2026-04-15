import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import joblib

"""Purpose: Load our high-quality features and clean any errors"""

# Load the engineered dataset
# Path updated to Google Colab standard
df = pd.read_csv('/content/water_levels_global_ml.csv')

# Define features and target (Same as LSTM for fair comparison)
features = ['station_id', 'river_id', 'hour', 'month', 'alert_level',
            'minor_flood', 'major_flood', 'water_level_lag1',
            'water_level_lag2', 'rainfall_roll3']
target = 'water_level_now'

# Cleanup
df = df.dropna(subset=features + [target])
df = df[~df[features + [target]].isin([np.inf, -np.inf]).any(axis=1)]

print(f"Ready: {len(df)} rows loaded for XGBoost.")

"""Purpose: Split the data so we use the most recent 20% of every station for testing"""

# Per-station chronological split (80/20)
train_dfs = []
test_dfs = []

for station_id in df['station_id'].unique():
    station_df = df[df['station_id'] == station_id].copy()
    if len(station_df) < 5: continue # Minimum samples for a split
    
    split_idx = int(0.8 * len(station_df))
    train_dfs.append(station_df.iloc[:split_idx])
    test_dfs.append(station_df.iloc[split_idx:])

train_df = pd.concat(train_dfs)
test_df = pd.concat(test_dfs)

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

print(f"Split Complete: {len(X_train)} training samples, {len(X_test)} testing samples.")

"""Purpose: Create the model structure and define tuning grid"""

param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    tree_method='hist', # Efficient for large datasets
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

"""Purpose: Perform hyperparameter tuning and training"""

print("\n--- Starting XGBoost Hyperparameter Tuning ---")
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

"""Purpose: See the real-world accuracy of the model in meters"""

# Evaluation
preds = best_model.predict(X_test)

print("\n--- Model Performance (Meters) ---")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, preds):.4f} m")
print(f"Confidence (R2 Score): {r2_score(y_test, preds):.4f}")

# Plot 1: Actual vs Predicted (Validation Set)
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:300], label='Actual Water Level', color='blue', alpha=0.7)
plt.plot(preds[:300], label='XGBoost Predicted', color='red', linestyle='--', alpha=0.8)
plt.title('River Flood Prediction: Actual vs XGBoost Predicted (Sample)')
plt.xlabel('Time Steps (Hourly)')
plt.ylabel('Water Level (Meters)')
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Feature Importance
plt.figure(figsize=(10, 8))
xgb.plot_importance(best_model, importance_type='weight', max_num_features=10)
plt.title('XGBoost: Features Driving Flood Predictions')
plt.show()

# Save the final model
joblib.dump(best_model, 'flood_xgboost_model_final.pkl')
print("Model saved to: flood_xgboost_model_final.pkl")
