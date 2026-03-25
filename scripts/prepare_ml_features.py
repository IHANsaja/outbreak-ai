import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

INPUT_CSV = r'd:\Projects\outbreakAI\dataset-collect\water_levels_ml_ready.csv'
OUTPUT_CSV = r'd:\Projects\outbreakAI\dataset-collect\water_levels_global_ml.csv'

def prepare_features(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Date/Time Features
    print("Converting timestamps...")
    # Attempt to handle common formats
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    df = df.sort_values(['station', 'datetime'])
    
    # Extract time-based features
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek

    # 2. Categorical Encodings (Station ID, River ID)
    print("Encoding category IDs...")
    le_river = LabelEncoder()
    df['river_id'] = le_river.fit_transform(df['river'].astype(str))
    
    le_station = LabelEncoder()
    df['station_id'] = le_station.fit_transform(df['station'].astype(str))

    # 3. Lag Features (Per Station)
    # Important: Group by station so we don't lag data from one station into another
    print("Adding lag features (previous water levels and rainfall)...")
    df['water_level_lag1'] = df.groupby('station')['water_level_now'].shift(1)
    df['water_level_lag2'] = df.groupby('station')['water_level_now'].shift(2)
    
    df['rainfall_lag1'] = df.groupby('station')['rainfall'].shift(1)
    df['rainfall_lag2'] = df.groupby('station')['rainfall'].shift(2)
    
    # 4. Rolling Features
    print("Adding rolling features (3-step average rainfall)...")
    df['rainfall_roll3'] = df.groupby('station')['rainfall'].transform(lambda x: x.rolling(window=3).mean())

    # 5. Threshold Deltas (Relative Features)
    # How close are we to flood levels?
    print("Adding threshold relative features...")
    df['dist_to_minor'] = df['minor_flood'] - df['water_level_now']
    df['dist_to_major'] = df['major_flood'] - df['water_level_now']

    # 6. Cleanup
    # Dropping the first few lags as they will be NaN
    initial_count = len(df)
    df = df.dropna(subset=['water_level_lag1', 'rainfall_roll3'])
    print(f"Dropped {initial_count - len(df)} rows due to lag/rolling initialization.")

    # 7. Save Final ML File
    print(f"Saving ML-ready dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    
    print("\n--- Features Created ---")
    print(f"Final training rows: {len(df)}")
    print(f"Features: station_id, river_id, hour, month, alert_level, minor_flood, major_flood, "
          f"water_level_lag1, water_level_lag2, rainfall_lag1, rainfall_lag2, rainfall_roll3")
    
    # Save the mapping for later use in prediction
    mapping = df[['river', 'river_id', 'station', 'station_id']].drop_duplicates()
    mapping.to_csv(output_path.replace('.csv', '_mapping.csv'), index=False)
    print(f"Mapping saved to {output_path.replace('.csv', '_mapping.csv')}")

if __name__ == "__main__":
    prepare_features(INPUT_CSV, OUTPUT_CSV)
