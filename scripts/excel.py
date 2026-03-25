import os
import re
import csv
import pdfplumber
import pandas as pd
import easyocr
import numpy as np

# ✅ YOUR FOLDER
INPUT_FOLDER = "dmc_downloads"
OUTPUT_CSV = "water_levels_total.csv"

# Columns for our final CSV
CSV_COLUMNS = [
    "date", "time", "river_basin", "river", "station", "unit",
    "alert_level", "minor_flood", "major_flood", "water_level_prev",
    "water_level_now", "remarks", "flow_trend", "rainfall", "source"
]

# Global reader
READER = None

def get_reader():
    global READER
    if READER is None:
        print("🤖 Initializing OCR Reader...")
        READER = easyocr.Reader(['en'], gpu=False)
    return READER

def safe_float(val):
    if val in ["NA", "-", None, "", "N.A", "N.A."]:
        return None
    try:
        if isinstance(val, str):
            val = re.sub(r"[^\d.\-]", "", val)
        return float(val)
    except (ValueError, IndexError):
        return None

def extract_date_time(text):
    date_match = re.search(r"DATE\s*[:.-]\s*(\d{1,2}[-/\s]\w+[-/\s]\d{4})", text, re.I)
    time_match = re.search(r"TIME\s*[:.-]\s*([\d:APM\s.-]+)", text, re.I)
    date = date_match.group(1) if date_match else None
    time = time_match.group(1).strip() if time_match else None
    return date, time

def parse_pdf(file_path):
    data = []
    try:
        with pdfplumber.open(file_path) as pdf:
            first_page_text = pdf.pages[0].extract_text() or ""
            date, time = extract_date_time(first_page_text)
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        row = [str(cell).replace("\n", " ").strip() if cell is not None else "" for cell in row]
                        if len(row) < 10: continue
                        if "ft" not in row and "m" not in row: continue
                        if row[4].lower() == "alert": continue
                        data.append({
                            "date": date, "time": time,
                            "river_basin": row[0], "river": row[1], "station": row[2], "unit": row[3],
                            "alert_level": safe_float(row[4]), "minor_flood": safe_float(row[5]),
                            "major_flood": safe_float(row[6]), "water_level_prev": safe_float(row[7]),
                            "water_level_now": safe_float(row[8]), "remarks": row[9],
                            "flow_trend": row[10] if len(row) > 10 else "",
                            "rainfall": safe_float(row[11]) if len(row) > 11 else None,
                            "source": os.path.basename(file_path)
                        })
    except Exception as e:
        print(f"  ❌ PDF Error: {e}")
    return data

def parse_xlsx(file_path):
    data = []
    try:
        df = pd.read_excel(file_path)
        for _, row in df.iterrows():
            data.append({
                "river": row.get("River", ""), "station": row.get("Station", ""),
                "water_level_now": safe_float(row.get("Level", None)),
                "source": os.path.basename(file_path)
            })
    except Exception as e:
        print(f"  ⚠️ Excel Error: {e}")
    return data

def parse_image(file_path):
    data = []
    try:
        reader = get_reader()
        results = reader.readtext(file_path)
        if not results: return []
        results.sort(key=lambda x: x[0][0][1])
        lines = []
        curr_line = [results[0]]
        last_y = results[0][0][0][1]
        for i in range(1, len(results)):
            y = results[i][0][0][1]
            if abs(y - last_y) < 15:
                curr_line.append(results[i])
            else:
                curr_line.sort(key=lambda x: x[0][0][0])
                lines.append(" ".join([res[1] for res in curr_line]))
                curr_line = [results[i]]
                last_y = y
        lines.append(" ".join([res[1] for res in curr_line]))
        full_text = "\n".join(lines)
        date, time = extract_date_time(full_text)
        row_pattern = re.compile(
            r"^(.*?)\s+(ft|m)\s+([.\d]+|NA|N\.A)\s+([.\d]+|NA|N\.A)\s+([.\d]+|NA|N\.A)\s+([.\d-]+|NA|N\.A)\s+([.\d-]+|NA|N\.A)\s+(Normal|Alert|Minor|Major|Rising|Falling|Steady|[\w\s]+?)\s+([.\d-]+|NA|N\.A)?",
            re.I
        )
        for line in lines:
            m = row_pattern.search(line)
            if m:
                data.append({
                    "date": date, "time": time, "river_basin": "", "river": "",
                    "station": re.sub(r"^[^\w]+|[^\w]+$", "", m.group(1)).strip(),
                    "unit": m.group(2), "alert_level": safe_float(m.group(3)),
                    "minor_flood": safe_float(m.group(4)), "major_flood": safe_float(m.group(5)),
                    "water_level_prev": safe_float(m.group(6)), "water_level_now": safe_float(m.group(7)),
                    "remarks": m.group(8), "flow_trend": "", 
                    "rainfall": safe_float(m.group(9)) if m.group(9) else None,
                    "source": os.path.basename(file_path)
                })
    except Exception as e:
        print(f"  ❌ OCR Error: {e}")
    return data

def get_processed_files(csv_path):
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["source"])
        return set(df["source"].unique())
    except (pd.errors.EmptyDataError, ValueError, KeyError):
        return set()

def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Folder not found: {INPUT_FOLDER}")
        return

    processed = get_processed_files(OUTPUT_CSV)
    files = sorted(os.listdir(INPUT_FOLDER))
    
    # Open CSV in append mode
    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()

        print(f"📂 Found {len(files)} files in {INPUT_FOLDER} ({len(processed)} already processed)")

        for i, file in enumerate(files):
            if file in processed:
                continue
            
            path = os.path.join(INPUT_FOLDER, file)
            ext = file.lower().split('.')[-1]
            rows = []
            
            if ext == "pdf":
                rows = parse_pdf(path)
            elif ext in ["jpg", "jpeg", "png"]:
                print(f"[{i+1}/{len(files)}] 👁️ OCR processing: {file}")
                rows = parse_image(path)
            elif ext in ["xlsx", "xls"]:
                print(f"[{i+1}/{len(files)}] 📊 Excel processing: {file}")
                rows = parse_xlsx(path)
            else:
                continue

            if rows:
                for row in rows:
                    # Fill missing columns with empty string
                    clean_row = {col: row.get(col, "") for col in CSV_COLUMNS}
                    writer.writerow(clean_row)
                f.flush() # Force write to disk
                if ext == "pdf":
                    print(f"[{i+1}/{len(files)}] ✅ Extracted: {file}")
            else:
                if ext == "pdf":
                    print(f"[{i+1}/{len(files)}] ⏭️ Skipped (no data): {file}")

    print(f"\n✅ Processing cycle complete. Check {OUTPUT_CSV} for results.")

if __name__ == "__main__":
    main()