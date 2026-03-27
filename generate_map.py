import json
import random
import math
import os

base_dir = os.path.dirname(__file__)
json_path = os.path.join(base_dir, "baseline_avg_flow.json")

with open(json_path, "r") as f:
    flows = json.load(f)

sites = sorted(list(flows.keys()))
nodes = []
for s in sites:
    nodes.append(int(s))

# Generate random coordinates
random.seed(42) # For reproducibility
coords = {}
for n in nodes:
    coords[n] = (random.randint(0, 100), random.randint(0, 100))

def dist(n1, n2):
    x1, y1 = coords[n1]
    x2, y2 = coords[n2]
    return int(math.hypot(x2 - x1, y2 - y1))

edges = []
# Build a Minimum Spanning Tree to ensure full connectivity
unvisited = set(nodes)
visited = set()

start = unvisited.pop()
visited.add(start)

while unvisited:
    best_dist = float('inf')
    best_edge = None
    
    for v in visited:
        for u in unvisited:
            d = dist(v, u)
            if d < best_dist:
                best_dist = d
                best_edge = (v, u, d)
                
    v, u, d = best_edge
    edges.append((v, u, max(1, d)))
    edges.append((u, v, max(1, d))) # Bidirectional
    visited.add(u)
    unvisited.remove(u)

# Add some nearest neighbors to create alternative paths (Grid-like)
for n in nodes:
    # find 3 closest nodes that aren't already connected
    connected = set([v for u, v, d in edges if u == n])
    distances = []
    for other in nodes:
        if other != n and other not in connected:
            distances.append((dist(n, other), other))
    distances.sort()
    for i in range(min(2, len(distances))):
        d, other = distances[i]
        edges.append((n, other, max(1, d)))
        edges.append((other, n, max(1, d)))

# Deduplicate edges
unique_edges = {}
for u, v, d in edges:
    if (u, v) not in unique_edges or unique_edges[(u, v)] > d:
        unique_edges[(u, v)] = d

output_path = os.path.join(base_dir, "Boroondara-map.txt")
with open(output_path, "w") as f:
    f.write("Nodes:\n")
    for n in nodes:
        x, y = coords[n]
        f.write(f"{n}: ({x},{y})\n")
    f.write("Edges:\n")
    for (u, v), d in unique_edges.items():
        f.write(f"({u},{v}): {d}\n")
    f.write("\nOrigin:\n")
    f.write(f"{nodes[0]}\n")
    f.write("Destinations:\n")
    f.write(f"{nodes[-1]}; {nodes[-2]}\n")

print(f"Generated {output_path} with {len(nodes)} nodes and {len(unique_edges)} edges.")
