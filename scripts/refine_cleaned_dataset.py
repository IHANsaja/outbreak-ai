import pandas as pd
import numpy as np
import os

INPUT_CSV = r'd:\Projects\outbreakAI\dataset-collect\water_levels_cleaned_ml.csv'
OUTPUT_CSV = r'd:\Projects\outbreakAI\dataset-collect\water_levels_ml_ready.csv'

def refine_dataset(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return

    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    initial_count = len(df)
    print(f"Initial rows: {initial_count}")

    # 1. Drop Missing Targets
    print("Dropping rows with missing water_level_now...")
    df = df.dropna(subset=['water_level_now'])
    after_drop_target = len(df)
    print(f"Rows after dropping missing targets: {after_drop_target} (Dropped {initial_count - after_drop_target})")

    # 2. Filter Outliers
    # water_level_now > 100 or rainfall > 500
    print("Filtering outliers (water_level_now > 100 or rainfall > 500)...")
    df = df[df['water_level_now'] <= 100]
    df = df[df['rainfall'] <= 500]
    after_filter = len(df)
    print(f"Rows after filtering outliers: {after_filter} (Dropped {after_drop_target - after_filter})")

    # 3. Handle Unknown Stations and Rivers
    print("Dropping unknown stations and rivers for per-river modeling...")
    # Standardize 'Unknown' and NaNs
    df['station'] = df['station'].fillna('Unknown')
    df.loc[df['station'].astype(str).str.lower() == 'nan', 'station'] = 'Unknown'
    df.loc[df['station'].astype(str).str.strip() == '', 'station'] = 'Unknown'
    
    # Drop rows where station is "Unknown"
    df = df[df['station'] != 'Unknown']
    
    # Drop rows where river is missing
    df = df.dropna(subset=['river'])
    
    after_cleaning = len(df)
    print(f"Rows after dropping unknown locations: {after_cleaning} (Dropped {after_filter - after_cleaning})")

    # 4. Final Sorting for per-river analysis
    print("Sorting by River, Station, and Time...")
    df = df.sort_values(['river', 'station', 'date', 'time'])

    # 5. Save the result
    print(f"Saving high-quality dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Refinement complete!")

    # Summary Statistics
    print("\n--- Final Dataset Distribution ---")
    print(f"Total rows: {len(df)}")
    print("\nStations per River:")
    print(df.groupby('river')['station'].nunique())

if __name__ == "__main__":
    refine_dataset(INPUT_CSV, OUTPUT_CSV)
