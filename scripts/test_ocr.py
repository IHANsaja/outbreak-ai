import excel
import os

path = os.path.join("dmc_downloads", "Water_level-Rainfall_2024__1732611743.jpg")
print(f"Testing OCR on: {path}")

try:
    data = excel.parse_image(path)
    print(f"Extracted {len(data)} rows.")
    for i, row in enumerate(data[:10]):
        print(f"Row {i+1}: {row['station']} - {row['water_level_now']} {row['unit']}")
except Exception as e:
    import traceback
    traceback.print_exc()
