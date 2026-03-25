import pandas as pd
import numpy as np
import re
import csv
import os

INPUT_CSV = "water_levels_total.csv"
OUTPUT_CSV = "water_levels_ml_cleaned.csv"

def is_numeric(val):
    if pd.isna(val) or val == "":
        return False
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False

def clean_time(t):
    if pd.isna(t):
        return t
    # Remove newlines, dots, and extra spaces
    t_str = str(t).replace('\n', ' ').replace('\r', ' ').replace('.', '').strip()
    t_str = re.sub(r'\s+', ' ', t_str)
    return t_str

def parse_date_from_source(source):
    if pd.isna(source):
        return np.nan
    # Extract timestamp like __1539046006.pdf
    m = re.search(r'__(\d{10})\.', str(source))
    if m:
        ts = int(m.group(1))
        return pd.to_datetime(ts, unit='s').strftime('%d-%b-%Y')
    return np.nan

def realign_row(row, coord_map):
    """
    Detects and fixes column shifts and recovers station names.
    PDF Columns: Basin, River, Station, Coord, Catchment, Unit, Alert, Minor, Major, Prev, Now, Remarks, Flow, Rainfall
    CSV Columns: date, time, river_basin, river, station, unit, alert_level, minor_flood, major_flood, water_level_prev, water_level_now, remarks, flow_trend, rainfall, source
    """
    
    # CASE 1: 'station' contains a coordinate (°)
    # Realign and recover station Name from coord_map
    if is_numeric(row['station']) is False and '°' in str(row['station']):
        coord = str(row['station']).strip()
        if coord in coord_map:
            # Observed shift when station name was skipped:
            # station = coordinate, unit = catchment, alert = empty, minor = Alert, major = Minor ...
            # We'll fix thresholds and levels
            # Actually, standard realign logic handles the numeric shift part.
            # We just need to replace the coordinate with the station name.
            row['station'] = coord_map[coord]
            # Since coordinate was in station, everything to the right is shifted.
            # However, usually Case 2 handles the numeric shift.
    
    # CASE 2: 'remarks' is numeric OR alert is missing but thresholds exist in wrong slots
    # This covers the general "shifted right" case.
    is_shifted = False
    if is_numeric(row['remarks']):
        is_shifted = True
    elif (pd.isna(row['alert_level']) or row['alert_level'] == "") and \
         is_numeric(row['minor_flood']) and is_numeric(row['major_flood']) and is_numeric(row['water_level_prev']):
        is_shifted = True
        
    if is_shifted:
        # Shift values right-to-left into correct slots
        row['alert_level'] = row['minor_flood']
        row['minor_flood'] = row['major_flood']
        row['major_flood'] = row['water_level_prev']
        row['water_level_prev'] = row['water_level_now']
        row['water_level_now'] = row['remarks']
        row['remarks'] = row['flow_trend']
        row['flow_trend'] = np.nan # Recalculate later
        
    return row

def get_remarks(row):
    try:
        now = float(row['water_level_now'])
        major = float(row['major_flood'])
        minor = float(row['minor_flood'])
        
        if now >= major: return "major flood"
        elif now >= minor: return "minor flood"
        else: return "normal"
    except:
        return "normal"

def get_flow_trend(curr, prev):
    if pd.isna(curr) or pd.isna(prev):
        return "normal"
    try:
        c = float(curr)
        p = float(prev)
        if c > p: return "rising"
        elif c < p: return "falling"
        else: return "normal"
    except:
        return "normal"

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"File not found: {INPUT_CSV}")
        return

    print(f"Loading {INPUT_CSV} robustly...")
    rows = []
    with open(INPUT_CSV, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return
        
        for row in reader:
            if not row: continue
            # Source is always the last field
            src = row[-1]
            new_row = [np.nan] * 15
            for i in range(min(len(row)-1, 14)):
                new_row[i] = row[i]
            new_row[14] = src
            rows.append(new_row)

    df = pd.DataFrame(rows, columns=['date', 'time', 'river_basin', 'river', 'station', 'unit', 'alert_level', 'minor_flood', 'major_flood', 'water_level_prev', 'water_level_now', 'remarks', 'flow_trend', 'rainfall', 'source'])
    
    # 1. Coordinate to Station Mapping
    print("Building coordinate mapping...")
    coord_map = {}
    for _, row in df.iterrows():
        s_val = str(row['station'])
        r_val = str(row['river'])
        u_val = str(row['unit'])
        if '°' in u_val and '°' not in s_val and not is_numeric(s_val):
            coord_map[u_val.strip()] = s_val.strip()
        elif '°' in s_val and '°' not in r_val and not is_numeric(r_val):
            coord_map[s_val.strip()] = r_val.strip()

    # 2. Basic Cleaning
    print("Cleaning time and basic fields...")
    df['time'] = df['time'].apply(clean_time)
    df['date'] = df['date'].ffill()
    
    # 3. Handle 'River' column shifting into 'Station'
    def fix_names(row):
        # Case where station name is in river column and station column has catchment
        if is_numeric(row['station']) and not is_numeric(row['river']) and pd.notna(row['river']):
            row['station'] = row['river']
        return row
    df = df.apply(fix_names, axis=1)
    
    # 4. Re-alignment and station recovery
    print("Fixing column shifts and recovering stations...")
    df = df.apply(lambda r: realign_row(r, coord_map), axis=1)
    
    if 'river' in df.columns:
        df = df.drop(columns=['river'])
    
    # 5. Filling Dates
    print("Filling missing dates...")
    df['date_parsed'] = df['source'].apply(parse_date_from_source)
    df['date'] = df['date'].fillna(df['date_parsed'])
    df['date'] = df.groupby('source')['date'].ffill().bfill()
    df = df.drop(columns=['date_parsed'])
    
    # 6. Handling Numeric Conversions
    print("Converting numeric fields...")
    num_cols = ['alert_level', 'minor_flood', 'major_flood', 'water_level_prev', 'water_level_now', 'rainfall']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.-]', '', regex=True), errors='coerce')
    
    # 7. Fill Missing Basin and Thresholds
    print("Filling missing metadata...")
    df['river_basin'] = df.groupby('source')['river_basin'].ffill()
    df['station'] = df['station'].astype(str).str.strip()
    
    for col in ['alert_level', 'minor_flood', 'major_flood']:
        df[col] = df.groupby('station')[col].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))
    df['rainfall'] = df['rainfall'].fillna(0.0)
    
    # 8. Final categorical labels
    print("Calculating final labels...")
    df['remarks'] = df.apply(get_remarks, axis=1)
    
    # Chronological sort and flow trend
    df = df.sort_values(['station', 'date', 'time'])
    df['prev_water_level'] = df.groupby('station')['water_level_now'].shift(1)
    df['flow_trend'] = df.apply(lambda r: get_flow_trend(r['water_level_now'], r['prev_water_level']), axis=1)
    df = df.drop(columns=['prev_water_level'])
    
    # 9. Clean up and Drop Source
    if 'source' in df.columns:
        df = df.drop(columns=['source'])
    
    print(f"Saving cleaned dataset to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
