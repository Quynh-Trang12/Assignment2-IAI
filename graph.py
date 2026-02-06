import math


class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.origin = None
        self.destinations = []

    def load_from_file(self, filepath: str) -> None:
        try:
            with open(filepath, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            raise

        section = None
        for line in lines:
            if line.endswith(":"):
                section = line[:-1]
                continue

            if section == "Nodes":
                # Format: 1: (4,1)
                parts = line.split(":")
                nid = int(parts[0])
                coords = parts[1].strip(" ()").split(",")
                self.nodes[nid] = (float(coords[0]), float(coords[1]))
            elif section == "Edges":
                # Format: (2,1): 4
                parts = line.split(":")
                u, v = map(int, parts[0].strip(" ()").split(","))
                cost = float(parts[1])
                if u not in self.edges:
                    self.edges[u] = {}
                self.edges[u][v] = cost
            elif section == "Origin":
                self.origin = int(line)
            elif section == "Destinations":
                # Format: 5; 4
                self.destinations = [int(x) for x in line.split(";")]

    def get_neighbors(self, u: int) -> list:
        if u not in self.edges:
            return []
        return sorted(self.edges[u].items())

    def heuristic(self, u: int) -> float:
        if not self.destinations:
            return 0.0
        x1, y1 = self.nodes[u]
        min_h = float("inf")
        for d in self.destinations:
            x2, y2 = self.nodes[d]
            dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if dist < min_h:
                min_h = dist
        return min_h
