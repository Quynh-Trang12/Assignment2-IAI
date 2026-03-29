import pandas as pd
import math
import sys
import re
from pathlib import Path
from typing import Dict, Tuple, List, Set

# ---------------------------------------------------------------------------
# Dynamic Path Resolution
# ---------------------------------------------------------------------------
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logger_setup import get_colored_logger

# Instantiate the centralized, color-coded logger
logger = get_colored_logger(__name__)


class BoroondaraMapCompiler:
    """
    Compiles VicRoads SCATS CSV data into a fully connected, traversable
    mathematical graph constrained strictly to the Boroondara Local Government Area.
    """

    def __init__(self, csv_input: Path, map_output: Path) -> None:
        self.csv_input = csv_input
        self.map_output = map_output
        self.nodes: Dict[int, Tuple[float, float]] = {}
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
        # Polygon vertices tracing the real Boroondara borders.
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

            # Evaluate if the infinite horizontal ray intersects the polygon's segment
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
        Ingests the CSV, maps SCATS IDs to coordinates, and applies the spatial boundary filter.
        """
        if not self.csv_input.exists():
            logger.error(f"File not found: {self.csv_input}")
            sys.exit(1)

        dataframe = pd.read_csv(self.csv_input)

        try:
            id_column = next(
                col
                for col in dataframe.columns
                if any(x in col.lower() for x in ["scats", "site", "id"])
            )
            lon_column = next(
                col
                for col in dataframe.columns
                if any(x in col.lower() for x in ["long", "x"])
            )
            lat_column = next(
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
            raw_identifier = str(row[id_column])
            numeric_match = re.search(r"\d+", raw_identifier)
            if not numeric_match:
                continue

            scats_id = int(numeric_match.group())
            lon, lat = float(row[lon_column]), float(row[lat_column])

            if self._is_in_boroondara(lon, lat):
                self.nodes[scats_id] = (lon, lat)

        logger.info(f"Successfully loaded {len(self.nodes)} strict Boroondara nodes.")

    def build_topology(self) -> None:
        """
        Connects local nodes into a mesh, and leverages Kruskal's Algorithm
        (Union-Find) to bridge isolated clusters, guaranteeing global connectivity.
        """
        logger.info("Building local grid topology...")
        node_ids = list(self.nodes.keys())

        # Establish base connectivity by linking each node to its 3 nearest geographic neighbors
        for i, source_id in enumerate(node_ids):
            distances = []
            for j, target_id in enumerate(node_ids):
                if i == j:
                    continue
                spatial_distance = self._haversine(
                    *self.nodes[source_id], *self.nodes[target_id]
                )

                # Prevent mathematical loops on identically placed sensors or excessively distant nodes
                if 0.001 < spatial_distance < 2.0:
                    distances.append((target_id, spatial_distance))

            distances.sort(key=lambda x: x[1])
            for neighbor_id, dist in distances[:3]:
                self.edges.append((source_id, neighbor_id, dist))
                self.edges.append((neighbor_id, source_id, dist))

        # Instantiate the Union-Find Disjoint Set to track network fragmentation
        parent_pointers = {n: n for n in node_ids}

        def find_root(node: int) -> int:
            """Recursively resolves the root of a node utilizing path compression."""
            if parent_pointers[node] == node:
                return node
            parent_pointers[node] = find_root(parent_pointers[node])
            return parent_pointers[node]

        def perform_union(node_a: int, node_b: int) -> bool:
            """Merges two discrete clusters. Returns True if a successful linkage occurred."""
            root_a = find_root(node_a)
            root_b = find_root(node_b)
            if root_a != root_b:
                parent_pointers[root_a] = root_b
                return True
            return False

        # Register existing baseline connectivity to the Disjoint Set
        for u, v, _ in self.edges:
            perform_union(u, v)

        # Analyze the topology to identify completely isolated islands
        unique_roots = set(find_root(n) for n in node_ids)
        if len(unique_roots) > 1:
            logger.info(
                f"Detected {len(unique_roots)} isolated clusters. Applying Kruskal's Stitching..."
            )

            # Compute distance bridges exclusively between nodes in separate clusters
            cross_island_edges = []
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    u = node_ids[i]
                    v = node_ids[j]
                    if find_root(u) != find_root(v):
                        dist = self._haversine(*self.nodes[u], *self.nodes[v])
                        cross_island_edges.append((dist, u, v))

            cross_island_edges.sort(key=lambda x: x[0])

            # Sequentially zip the clusters together until the global graph is whole
            for dist, u, v in cross_island_edges:
                if perform_union(u, v):
                    self.edges.append((u, v, dist))
                    self.edges.append((v, u, dist))

            logger.info("Graph topology is now 100% fully connected across Boroondara.")
        else:
            logger.info("Graph topology is natively fully connected.")

    def export(self) -> None:
        """
        Serializes the fully connected graph topology to disk matching Assignment 2A specifications.
        """
        with open(self.map_output, "w", encoding="utf-8") as file_stream:
            file_stream.write("Nodes:\n")
            for node_id, coordinates in self.nodes.items():
                file_stream.write(f"{node_id}: {coordinates}\n")

            file_stream.write("\nEdges:\n")
            for u, v, distance in self.edges:
                file_stream.write(f"({u}, {v}): {distance}\n")

            file_stream.write(f"\nOrigin:\n{list(self.nodes.keys())[0]}\n")
            file_stream.write(f"\nDestinations:\n{list(self.nodes.keys())[-1]}\n")

        logger.info(f"Map successfully exported to {self.map_output}")


if __name__ == "__main__":
    compiler = BoroondaraMapCompiler(
        _PROJECT_ROOT / "data" / "raw" / "Traffic_Count_Locations_with_LONG_LAT.csv",
        _PROJECT_ROOT / "data" / "maps" / "map.txt",
    )
    compiler.load_nodes()
    compiler.build_topology()
    compiler.export()
