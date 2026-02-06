import math
from dataclasses import dataclass, field
from typing import Optional

@dataclass(order=False)
class SearchState:
    """
    Represents a node in the search tree.
    Enforces strict tie-breaking rules:
    1. Priority (f, g, or h)
    2. Node ID (Ascending)
    3. Timestamp (Chronological)
    """
    node_id: int
    parent: Optional['SearchState']
    g: float
    h: float
    f_method: str
    timestamp: int
    priority: float = field(init=False)

    def __post_init__(self):
        # Calculate priority based on method strategy
        if self.f_method in ('bfs', 'dfs'):
            self.priority = 0.0
        elif self.f_method == 'gbfs':
            self.priority = self.h
        elif self.f_method in ('as', 'cus2'):
            self.priority = self.g + self.h
        elif self.f_method == 'cus1':
            self.priority = self.g
        else:
            self.priority = 0.0

    def __lt__(self, other):
        # 1. Primary Priority Comparison
        # Use simple float comparison, but handle close floats if necessary. 
        # Standard strict inequality is usually fine for priority queues unless rigid deterministic equality is needed.
        if not math.isclose(self.priority, other.priority, rel_tol=1e-9):
            return self.priority < other.priority

        # 2. Node ID Tie-Breaker (Ascending)
        if self.node_id != other.node_id:
            return self.node_id < other.node_id

        # 3. Chronological Tie-Breaker
        return self.timestamp < other.timestamp
