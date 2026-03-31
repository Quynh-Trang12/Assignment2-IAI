import pandas as pd
import math
import sys
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List

# Dynamic Path Resolution
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logger_setup import get_colored_logger

# Instantiate the centralized, color-coded logger
logger = get_colored_logger(__name__)


class BoroondaraMapCompiler:
    """
    Compiles VicRoads SCATS CSV data into a connected, traversable mathematical graph.
    Uses semantic road-name grouping to create realistic street links and drops isolated nodes.
    """

    def __init__(self, csv_input: Path, map_output: Path) -> None:
        self.csv_input = csv_input
        self.map_output = map_output
        self.nodes: Dict[int, Tuple[float, float]] = {}
        self.descriptions: Dict[int, str] = {}
        self.edges: List[Tuple[int, int, float]] = []

    def _haversine(self, lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """
        Calculates the great-circle distance between two spatial coordinates in kilometers.
        """
        earth_radius_km = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        )

        # Clamp 'a' to 1.0 to prevent fatal domain errors caused by float inaccuracies
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - min(1.0, a)))
        return round(earth_radius_km * c, 3)

    def _is_in_boroondara(self, longitude: float, latitude: float) -> bool:
        """
        Ray-casting algorithm to strictly filter nodes inside the Boroondara polygon.
        """
        polygon_vertices = [
            (144.985, -37.815),  # West edge (Hawthorn / Yarra River)
            (145.010, -37.765),  # NW edge (Kew / Yarra Bend)
            (145.050, -37.755),  # Mid-North edge (Kew East / River Bend)
            (145.090, -37.775),  # NE edge (Balwyn North / Eastern Freeway)
            (145.120, -37.830),  # East edge (Surrey Hills / Warrigal Rd)
            (145.100, -37.880),  # SE edge (Ashburton)
            (145.040, -37.875),  # South edge (Glen Iris / Gardiners Creek)
            (145.020, -37.845),  # SW edge (Kooyong)
            (144.990, -37.835),  # SW edge (Hawthorn)
        ]

        vertex_count = len(polygon_vertices)
        is_inside = False

        p1x, p1y = polygon_vertices[0]
        for i in range(1, vertex_count + 1):
            p2x, p2y = polygon_vertices[i % vertex_count]
            if latitude > min(p1y, p2y) and latitude <= max(p1y, p2y):
                if longitude <= max(p1x, p2x):
                    if p1y != p2y:
                        x_intersection = (latitude - p1y) * (p2x - p1x) / (
                            p2y - p1y
                        ) + p1x
                        if p1x == p2x or longitude <= x_intersection:
                            is_inside = not is_inside
            p1x, p1y = p2x, p2y

        return is_inside

    def load_nodes(self) -> None:
        """
        Ingests the CSV, maps SCATS IDs to coordinates and descriptions,
        and applies the spatial boundary filter.
        """
        if not self.csv_input.exists():
            logger.error(f"File not found: {self.csv_input}")
            sys.exit(1)

        dataframe = pd.read_csv(self.csv_input)

        try:
            id_col = next(
                col
                for col in dataframe.columns
                if any(x in col.lower() for x in ["scats", "site", "id"])
            )
            lon_col = next(
                col
                for col in dataframe.columns
                if any(x in col.lower() for x in ["long", "x"])
            )
            lat_col = next(
                col
                for col in dataframe.columns
                if any(x in col.lower() for x in ["lat", "y"])
            )
        except StopIteration:
            logger.error(
                "Ensure your CSV has columns for SCATS ID, Longitude, and Latitude."
            )
            sys.exit(1)

        for _, row in dataframe.iterrows():
            raw_id = str(row[id_col])
            match = re.search(r"\d+", raw_id)
            if not match:
                continue

            scats_id = int(match.group())
            lon, lat = float(row[lon_col]), float(row[lat_col])

            if self._is_in_boroondara(lon, lat):
                self.nodes[scats_id] = (lon, lat)

                # Capture and merge descriptions for semantic road mapping
                desc = (
                    str(row.get("SITE_DESC", "")) + " " + str(row.get("TFM_DESC", ""))
                )
                self.descriptions[scats_id] = desc.upper()

        logger.info(f"Loaded {len(self.nodes)} nodes within the Boroondara boundary.")

    def build_topology(self) -> None:
        """
        Groups nodes by shared street names, sequences them geographically,
        and extracts only the largest connected component to drop isolated sites.
        """
        logger.info("Building topology based on real street names...")

        # 1. Group nodes by extracted road names
        roads = defaultdict(list)
        for scats_id, desc in self.descriptions.items():
            # Strip out generic directional words to isolate core road names
            clean_name = re.sub(r"\b(BD|N OF|S OF|E OF|W OF|HWY|RD|ST|AVE)\b", "", desc)

            # Split by delimiters like '/', '&', 'AND', or 'AT'
            parts = re.split(r"/|&|\bAND\b|\bAT\b", clean_name)
            for part in parts:
                p = part.strip()
                if len(p) > 2:  # Ignore tiny artifacts
                    roads[p].append(scats_id)

        # 2. Sequence nodes along each street
        raw_edges = set()
        for street_name, sites in roads.items():
            if len(sites) < 2:
                continue

            # Find the two extreme ends of the street
            max_d = -1
            extreme_a = sites[0]
            for s1 in sites:
                for s2 in sites:
                    d = self._haversine(*self.nodes[s1], *self.nodes[s2])
                    if d > max_d:
                        max_d = d
                        extreme_a = s1

            # Sort all sites on the street by distance from extreme_a
            sorted_sites = sorted(
                sites,
                key=lambda s: self._haversine(*self.nodes[extreme_a], *self.nodes[s]),
            )

            # Connect consecutive points along the sequence
            for i in range(len(sorted_sites) - 1):
                u, v = sorted_sites[i], sorted_sites[i + 1]
                dist = self._haversine(*self.nodes[u], *self.nodes[v])

                # Sanity check: cap crazy connections resulting from bad string matches
                if dist < 10.0:
                    # Store canonical edge tuple
                    edge_pair = tuple(sorted((u, v)))
                    raw_edges.add((edge_pair[0], edge_pair[1], dist))

        # 3. Find connected components to isolate unlinked nodes
        adj_list = defaultdict(list)
        for u, v, _ in raw_edges:
            adj_list[u].append(v)
            adj_list[v].append(u)

        visited = set()
        components = []
        for n in self.nodes.keys():
            if n not in visited:
                comp = set()
                stack = [n]
                while stack:
                    curr = stack.pop()
                    if curr not in comp:
                        comp.add(curr)
                        visited.add(curr)
                        stack.extend(adj_list[curr])
                components.append(comp)

        # 4. Filter graph to guarantee traversability (drop isolated intersections)
        if not components:
            logger.error("No edges could be formed. Check street name extraction.")
            return

        largest_comp = max(components, key=len)
        dropped_count = len(self.nodes) - len(largest_comp)

        # Apply the filter
        self.nodes = {k: v for k, v in self.nodes.items() if k in largest_comp}

        for u, v, d in raw_edges:
            if u in largest_comp and v in largest_comp:
                self.edges.append((u, v, d))
                self.edges.append((v, u, d))  # Bidirectional

        logger.info(f"Dropped {dropped_count} isolated intersections.")
        logger.info(
            f"Final Graph: {len(self.nodes)} nodes, {len(self.edges)//2} bidirectional edges."
        )

    def export(self) -> None:
        """
        Serializes the fully connected graph topology to disk.
        """
        if not self.nodes:
            logger.warning("Graph is empty. Skipping export.")
            return

        with open(self.map_output, "w", encoding="utf-8") as file_stream:
            file_stream.write("Nodes:\n")
            for node_id, coordinates in sorted(self.nodes.items()):
                file_stream.write(f"{node_id}: {coordinates}\n")

            file_stream.write("\nEdges:\n")
            # Sort edges for clean output reading
            for u, v, distance in sorted(self.edges):
                file_stream.write(f"({u}, {v}): {distance:.3f}\n")

            sorted_keys = sorted(self.nodes.keys())
            file_stream.write(f"\nOrigin:\n{sorted_keys[0]}\n")
            file_stream.write(f"\nDestinations:\n{sorted_keys[-1]}\n")

        logger.info(f"Map successfully exported to {self.map_output}")


if __name__ == "__main__":
    compiler = BoroondaraMapCompiler(
        _PROJECT_ROOT / "data" / "raw" / "Traffic_Count_Locations_with_LONG_LAT.csv",
        _PROJECT_ROOT / "data" / "maps" / "map.txt",
    )
    compiler.load_nodes()
    compiler.build_topology()
    compiler.export()
