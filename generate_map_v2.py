import json
import csv
import os
import math
from collections import defaultdict

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

# 3. Extract Coordinates
coords = {}
with open(os.path.join(base_dir, "Traffic_Count_Locations_with_LONG_LAT.csv"), "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if 'SITE_DESC' not in row: continue
        desc = (row.get('SITE_DESC', '') + " " + row.get('TFM_DESC', '')).upper()
        for sid, name in site_names.items():
            if sid in coords: continue
            parts = name.replace(" BD", "").replace(" N OF", "").replace(" S OF", "").replace(" E OF", "").replace(" W OF", "").split("/")
            parts = [p.strip().upper() for p in parts]
            if len(parts) >= 2:
                if parts[0] in desc and parts[1] in desc:
                    coords[sid] = (float(row['X']), float(row['Y']))
            else:
                if name.upper() in desc:
                    coords[sid] = (float(row['X']), float(row['Y']))

# Missing coords hardcoded based on map search
coords[4821] = (145.08614, -37.77979) # Eastern Freeway On Ramp
coords[2200] = (145.10600, -37.83000) # Maroondah/Union

def calc_dist_km(n1, n2):
    x1, y1 = coords[n1]
    x2, y2 = coords[n2]
    # Simple equirectangular approximation
    dx = (x2 - x1) * 111.0 * math.cos(math.radians((y1+y2)/2))
    dy = (y2 - y1) * 111.0
    return math.hypot(dx, dy)

# Step 1 & 2: Group by road
streets = defaultdict(list)
for sid, name in site_names.items():
    # Clean up and split streets
    clean_name = name.replace(" BD", "").replace(" N OF", "").replace(" S OF", "").replace(" E OF", "").replace(" W OF", "")
    parts = [p.strip().upper() for p in clean_name.split("/")]
    
    # We ignore generic single names like 'EASTERN FREEWAY ON RAMP' from grouping if they don't have interconnects
    for p in parts:
        if len(p) > 2: # Ignore blank or tiny abbreviations
            streets[p].append(sid)

edges = {}

# Step 3 & 4: Order them and connect sequential neighbors
for street_name, sites_on_street in streets.items():
    if len(sites_on_street) < 2:
        continue
        
    # Find the two sites furthest apart to establish a "line of sight"
    max_d = -1
    extreme_a, extreme_b = None, None
    for s1 in sites_on_street:
        for s2 in sites_on_street:
            if s1 != s2:
                d = calc_dist_km(s1, s2)
                if d > max_d:
                    max_d = d
                    extreme_a = s1
                    extreme_b = s2
                    
    # Sort all sites on this street by their distance to extreme_a
    # This guarantees a sequential ordering along the street path
    sorted_sites = sorted(sites_on_street, key=lambda s: calc_dist_km(extreme_a, s))
    
    # Connect sequential pairs
    for i in range(len(sorted_sites) - 1):
        u = sorted_sites[i]
        v = sorted_sites[i+1]
        dist = calc_dist_km(u, v)
        
        # Don't connect if it's insanely far (bad string match) -> Cap at 10km
        if dist < 10.0:
            edges[(u, v)] = dist
            edges[(v, u)] = dist

# --- MST COMPONENTS FALLBACK ---
# 1. Find Connected Components
adj = defaultdict(list)
for (u, v) in edges:
    adj[u].append(v)

visited = set()
components = []
for n in coords.keys():
    if n not in visited:
        comp = set()
        stack = [n]
        while stack:
            curr = stack.pop()
            if curr not in comp:
                comp.add(curr)
                visited.add(curr)
                stack.extend(adj[curr])
        components.append(comp)

if len(components) > 1:
    print(f"Found {len(components)} disconnected components. Building MST bridges...")
    comp_edges = []
    for i in range(len(components)):
        for j in range(i+1, len(components)):
            min_d = float('inf')
            best_pair = None
            for u in components[i]:
                for v in components[j]:
                    d = calc_dist_km(u, v)
                    if d < min_d:
                        min_d = d
                        best_pair = (u, v)
            if best_pair:
                comp_edges.append((min_d, i, j, best_pair[0], best_pair[1]))
                
    # Kruskal's to find MST of components
    comp_edges.sort()
    parent = list(range(len(components)))
    
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]
        
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            return True
        return False
        
    bridges_added = 0
    for d, i, j, u, v in comp_edges:
        if union(i, j):
            penalized_dist = d * 1.5
            edges[(u, v)] = penalized_dist
            edges[(v, u)] = penalized_dist
            bridges_added += 1
            print(f"Added bridge {u}-{v} (real dist: {d:.2f}km, penalised: {penalized_dist:.2f}km)")
    print(f"Successfully bridged graph with {bridges_added} penalized fallback edges.")
# -----------------------------

# Write to file
output_path = os.path.join(base_dir, "Boroondara-map.txt")
sorted_nodes = sorted(list(coords.keys()))

with open(output_path, "w") as f:
    f.write("Nodes:\n")
    for n in sorted_nodes:
        x, y = coords[n]
        # Store as X, Y or pseudo-integers? The graph parser handles floats or ints?
        # graph.py does int(coords[0]), int(coords[1]). We need to scale them or fix graph.py
        # Actually in graph.py: `(int(coords[0]), int(coords[1]))`. So passing floats scaled by 10000 works.
        ix, iy = int((x - 144.0) * 10000), int((y + 38.0) * 10000)
        f.write(f"{n}: ({ix},{iy})\n")
        
    f.write("Edges:\n")
    # Sort edges for clean output
    for (u, v), d in sorted(edges.items()):
        f.write(f"({u},{v}): {max(0.1, d):.3f}\n") # format to 3 decimal places km
        
    f.write("\nOrigin:\n")
    f.write(f"{sorted_nodes[0]}\n")
    f.write("Destinations:\n")
    f.write(f"{sorted_nodes[-1]}; {sorted_nodes[-2]}\n")

print(f"Generated data-driven map with {len(coords)} nodes and {len(edges)} directional edges!")
