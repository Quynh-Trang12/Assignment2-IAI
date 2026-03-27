from typing import Dict, Tuple, Optional
from travel_time import flow_to_travel_time_minutes


class Graph:
    """
    Graph structure for route finding.

    Stores:
    - adjacency list with distances
    - FlowPredictor for dynamic edge cost (flow → travel time)
    """

    def __init__(self, predictor=None):
        # adjacency: {u: {v: distance_km}}
        self.adjacency_list: Dict[int, Dict[int, float]] = {}
        self.nodes: Dict[int, Tuple[int, int]] = {}
        self.origin = None
        self.destinations = []

        # predictor (mock / baseline / lstm / gru)
        self.predictor = predictor

        # debug flag for printing
        self.debug = False

    def add_edge(self, u: int, v: int, distance: float):
        """
        Add directed edge u -> v with distance (km).
        """
        if u not in self.adjacency_list:
            self.adjacency_list[u] = {}

        self.adjacency_list[u][v] = distance

    def get_neighbors(self, node: int):
        """
        Return neighbors of a node.
        """
        return self.adjacency_list.get(node, {}).keys()

    def heuristic(self, node: int) -> float:
        """
        Compute Euclidean distance heuristic to the closest destination.
        """
        if node not in self.nodes or not self.destinations:
            return 0.0
            
        x1, y1 = self.nodes[node]
        import math
        min_dist = float('inf')
        for d in self.destinations:
            if d in self.nodes:
                x2, y2 = self.nodes[d]
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < min_dist:
                    min_dist = dist
        return min_dist if min_dist != float('inf') else 0.0

    def get_edge_cost(
        self,
        u: int,
        v: int,
        time_context: Optional[dict] = None
    ) -> float:
        """
        Compute travel time cost for edge (u, v).

        Flow is predicted using the selected model,
        then converted to travel time.
        """
        distance_km = self.adjacency_list[u][v]

        # Predict flow (ML or baseline)
        if self.predictor is not None:
            flow = self.predictor.predict_flow(u, v, time_context)
        else:
            flow = 100.0  # fallback

        # Convert flow → travel time
        cost = flow_to_travel_time_minutes(
            flow=flow,
            distance_km=distance_km,
            intersection_delay_sec=30.0,
            use_under_capacity_branch=True
        )

        if self.debug:
            print(
                f"[{self.predictor.model_type if self.predictor else 'none'}] "
                f"Edge {u}->{v}: distance={distance_km}, flow={flow}, cost={cost:.4f}"
            )

        return cost

    def load_from_file(self, filename: str):
        """
        Load graph from text file (Assignment 2A format).
        """
        with open(filename, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        mode = None

        for line in lines:
            if line.startswith("Nodes:"):
                mode = "nodes"
                continue
            elif line.startswith("Edges:"):
                mode = "edges"
                continue
            elif line.startswith("Origin:"):
                mode = "origin"
                continue
            elif line.startswith("Destinations:"):
                mode = "destinations"
                continue

            if mode == "edges":
                # Format: (2,1): 4
                parts = line.split(":")
                nodes_part = parts[0].strip()[1:-1]  # remove ()
                cost = float(parts[1].strip())

                u_str, v_str = nodes_part.split(",")
                u = int(u_str)
                v = int(v_str)

                self.add_edge(u, v, cost)
            elif mode == "nodes":
                parts = line.split(":")
                node_id = int(parts[0].strip())
                coords = parts[1].strip()[1:-1].split(",")
                self.nodes[node_id] = (int(coords[0]), int(coords[1]))
            elif mode == "origin":
                self.origin = int(line.strip())
            elif mode == "destinations":
                self.destinations = [int(x.strip()) for x in line.split(";")]