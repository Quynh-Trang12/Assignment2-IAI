import json
import csv
import os

base_dir = r"E:\Assignment2-IAI"

# 1. Get the 40 SCATS sites
with open(os.path.join(base_dir, "baseline_avg_flow.json"), "r") as f:
    flows = json.load(f)
scats_ids = set(int(k) for k in flows.keys())

# 2. Get Site Descriptions
site_names = {}
with open(os.path.join(base_dir, "scats_mapping.csv"), "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 2 and row[0].isdigit():
            sid = int(row[0])
            if sid in scats_ids:
                site_names[sid] = row[1].strip()

# Print the 40 names
print("SCATS site descriptions:")
for sid, name in site_names.items():
    print(f"  {sid}: {name}")

# Try to find matching coordinates in Traffic_Count...csv
coords = {}
with open(os.path.join(base_dir, "Traffic_Count_Locations_with_LONG_LAT.csv"), "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if 'SITE_DESC' not in row: continue
        desc = (row.get('SITE_DESC', '') + " " + row.get('TFM_DESC', '')).upper()
        # Very naive match: check if ANY of our 40 site names are in this desc
        for sid, name in site_names.items():
            if sid in coords: continue
            
            # Simple heuristic: split by '/' or '&' and check if both parts are in desc
            parts = name.replace(" BD", "").replace(" N OF", "").replace(" S OF", "").replace(" E OF", "").replace(" W OF", "").split("/")
            parts = [p.strip().upper() for p in parts]
            if len(parts) >= 2:
                street1, street2 = parts[0], parts[1]
                if street1 in desc and street2 in desc:
                    coords[sid] = (float(row['X']), float(row['Y']))
            else:
                if name.upper() in desc:
                    coords[sid] = (float(row['X']), float(row['Y']))

missing = []
for sid, name in site_names.items():
    if sid not in coords:
        missing.append((sid, name))
print(f"Missing coordinates for: {missing}")

