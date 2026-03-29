# ---------------------------------------------------------------------------
# Imports & Dependencies
# ---------------------------------------------------------------------------
import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data Structures & State Management
# ---------------------------------------------------------------------------
@dataclass(order=False)
class SearchState:
    """
    An immutable-style Data Transfer Object (DTO) representing a single node
    (or state) within the search tree during traversal.

    Architectural Note:
    This class strictly encapsulates the tie-breaking rules required for the assignment.
    By implementing the `__lt__` (less than) magic method, it allows Python's native `heapq`
    to automatically sort nodes based on:
    1. Primary: Priority value (f-cost, computed dynamically based on the algorithm).
    2. Secondary: Node Identifier (Ascending numerical order).
    3. Tertiary: Chronological generation order (nodes generated earlier are expanded first).

    Attributes:
        node_id (int): The unique integer identifier of the graph node represented by this state.
        parent (Optional[SearchState]): The predecessor state that generated this state.
                                        Crucial for backtracking to reconstruct the final path.
        g (float): The cumulative path cost from the origin to this specific node.
                   (Retained as 'g' to universally adhere to standard mathematical pathfinding notation).
        h (float): The estimated heuristic cost from this node to the nearest goal.
                   (Retained as 'h' to universally adhere to standard mathematical pathfinding notation).
        search_method (str): The identifier of the search algorithm evaluating this state (e.g., 'as', 'gbfs').
        timestamp (int): A strictly increasing integer marking exactly when this state was instantiated.
        priority_score (float): The computed evaluation metric used to rank this node in a priority queue.
    """

    node_id: int
    parent: Optional["SearchState"]
    g: float
    h: float
    search_method: str
    timestamp: int

    # field(init=False) tells the dataclass NOT to expect this in the constructor,
    # as we will dynamically compute it immediately after initialization.
    priority_score: float = field(init=False)

    # ---------------------------------------------------------------------------
    # Lifecycle Hooks
    # ---------------------------------------------------------------------------
    def __post_init__(self) -> None:
        """
        A built-in dataclass lifecycle hook that executes immediately after instantiation.

        It computes the specific evaluation priority score (f-value) for the node
        based strictly on the designated search algorithm's mathematical strategy.

        Internal Variables:
            method (str): The normalized, lowercase identifier of the active search algorithm.
        """
        method = self.search_method.lower()

        if method in ("bfs", "dfs"):
            self.priority_score = 0.0
        elif method == "gbfs":
            self.priority_score = self.h
        elif method in ("as", "cus2"):
            self.priority_score = self.g + self.h
        elif method == "cus1":
            self.priority_score = self.g
        else:
            self.priority_score = 0.0

        # Round once at creation. 6 decimal places is perfect for handling tiny travel time fractions without losing mathematical integrity.
        self.priority_score = round(self.priority_score, 6)

    # ---------------------------------------------------------------------------
    # Operator Overloading (Tie-Breaking Engine)
    # ---------------------------------------------------------------------------
    def __lt__(self, other: "SearchState") -> bool:
        """
        Overrides the standard "less than" (<) operator. This allows priority queues (like heapq)
        to natively compare two SearchState objects and correctly order them.

        Args:
            other (SearchState): The adjacent state being compared against within the priority heap.

        Returns:
            bool: True if THIS state is mathematically "better" and should be expanded before the 'other' state.
        """
        # 1. Primary Priority (Lightning fast, no math functions slowing down the queue)
        if self.priority_score != other.priority_score:
            return self.priority_score < other.priority_score

        # 2. Secondary Tie-Breaker: Node Identifier (Ascending)
        if self.node_id != other.node_id:
            return self.node_id < other.node_id

        # 3. Tertiary Tie-Breaker: Chronological Order
        return self.timestamp < other.timestamp
