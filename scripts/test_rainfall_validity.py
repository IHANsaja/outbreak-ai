import pandas as pd
import numpy as np
import os

file_path = r'd:\Projects\outbreakAI\dataset-collect\datasets\water_levels_global_ml.csv'

def validate_rainfall_impact(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    df = pd.read_csv(path)
    
    # Calculate water level change (Delta)
    df['wl_delta'] = df['water_level_now'] - df['water_level_lag1']
    
    print("--- 1. Statistical Correlation ---")
    # Correlation between current rainfall and the change in water level
    correlation = df['rainfall'].corr(df['wl_delta'])
    print(f"Correlation between Rainfall and Water Level Increase: {correlation:.4f}")

    print("\n--- 2. Probabilistic Proof ---")
    # Define "Heavy Rain" as > 50mm (adjustable)
    heavy_rain_threshold = 50
    df['is_heavy_rain'] = df['rainfall'] > heavy_rain_threshold
    
    # Group by rain intensity and check how often the water is rising
    stats = df.groupby('is_heavy_rain').agg(
        avg_increase=('wl_delta', 'mean'),
        rising_probability=('flow_trend', lambda x: (x == 'rising').mean() * 100),
        count=('wl_delta', 'count')
    )
    
    stats.index = ['Low/No Rain (<50mm)', 'Heavy Rain (>50mm)']
    print(stats)

    print("\n--- 3. Top 5 Dramatic Rises during Heavy Rain ---")
    # Show real examples
    rises = df[df['is_heavy_rain']].sort_values('wl_delta', ascending=False).head(5)
    cols_to_show = ['date', 'time', 'river', 'station', 'rainfall', 'water_level_lag1', 'water_level_now', 'wl_delta']
    print(rises[cols_to_show])

if __name__ == "__main__":
    validate_rainfall_impact(file_path)
