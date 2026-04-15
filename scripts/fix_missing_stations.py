import pandas as pd
import numpy as np
import os

# ============================================================
# Purpose: Fix the "Insufficient Observations" issue for
# stations with only 1 (or few) data points. 
# This script creates a NEW dataset with synthetic padding.
# ============================================================

INPUT_FILE = 'datasets/water_levels_90_rivers_ready.csv'
OUTPUT_FILE = 'datasets/water_levels_90_rivers_augmented_fixed.csv'

def fix_stations():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    print(f"📂 Loading original dataset: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    
    # 1. Identify short-length stations
    min_required_rows = 13 # max_encoder_length (12) + 1 (prediction)
    counts = df.groupby('station_id').size()
    short_stations = counts[counts < min_required_rows].index.tolist()
    
    if not short_stations:
        print("✅ No short-length stations found. Dataset is already model-ready.")
        return

    print(f"Found {len(short_stations)} stations with < {min_required_rows} rows: {short_stations}")
    
    # 2. Augment short stations via Temporal Padding (conservative backfilling)
    new_rows = []
    
    # We'll need datetime for sorting
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    for station_id in short_stations:
        station_df = df[df['station_id'] == station_id].copy()
        first_row = station_df.iloc[0].copy()
        
        # We need to add (min_required_rows - len(station_df)) rows
        num_to_add = min_required_rows - len(station_df)
        print(f"   - Station {station_id}: Adding {num_to_add} synthetic rows...")
        
        for i in range(1, num_to_add + 1):
            cloned_row = first_row.copy()
            # Backdate by i hours (hypothetical)
            cloned_row['datetime'] = first_row['datetime'] - pd.Timedelta(hours=i)
            # Add tiny noise to avoid constant zero variance if desired, or keep stable
            # cloned_row['water_level_now'] += np.random.normal(0, 0.001) 
            new_rows.append(cloned_row)

    # 3. Combine and rebuild the dataset
    df_augmented = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    
    # Re-sort to maintain chronological order per station
    df_augmented = df_augmented.sort_values(['station_id', 'datetime'])
    
    # Safety Check: Did we fix River 44?
    # Ensure all original rivers are preserved
    original_rivers = set(df['river_id'].unique())
    final_rivers = set(df_augmented['river_id'].unique())
    original_stations = set(df['station_id'].unique())
    final_stations = set(df_augmented['station_id'].unique())

    print("\n--- Validation Statistics ---")
    print(f"Original: {len(original_rivers)} Rivers, {len(original_stations)} Stations")
    print(f"Fixed:    {len(final_rivers)} Rivers, {len(final_stations)} Stations")
    print(f"Total rows: {len(df_augmented)} (Added {len(new_rows)} rows)")
    
    # Re-verify lengths
    final_counts = df_augmented.groupby('station_id').size()
    still_short = final_counts[final_counts < min_required_rows].index.tolist()
    if not still_short:
        print("✅ SUCCESS: All stations now have enough rows for TFT.")
    else:
        print(f"❌ ERROR: Some stations are still too short: {still_short}")

    # 4. Save to NEW file
    print(f"\n💾 Saving augmented dataset to: {OUTPUT_FILE}")
    df_augmented.to_csv(OUTPUT_FILE, index=False)
    print("Done!")

if __name__ == "__main__":
    fix_stations()
