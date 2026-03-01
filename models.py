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
    parent: Optional['SearchState']
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
        
        # Uninformed searches (BFS, DFS) do not utilize priority scoring for exploration.
        # Setting a baseline of 0.0 maintains structural compatibility across the engine.
        if method in ('bfs', 'dfs'):
            self.priority_score = 0.0
            
        # Greedy Best-First Search (GBFS) evaluates purely on estimated heuristic cost: f(n) = h(n)
        elif method == 'gbfs':
            self.priority_score = self.h
            
        # A* Search (AS) and Iterative Deepening A* (CUS2) evaluate on total estimated cost: f(n) = g(n) + h(n)
        elif method in ('as', 'cus2'):
            self.priority_score = self.g + self.h
            
        # Uniform Cost Search (CUS1) evaluates purely on the cumulative path cost: f(n) = g(n)
        elif method == 'cus1':
            self.priority_score = self.g
            
        # Fail-safe fallback to prevent math operation exceptions on undefined methods
        else:
            self.priority_score = 0.0

    # ---------------------------------------------------------------------------
    # Operator Overloading (Tie-Breaking Engine)
    # ---------------------------------------------------------------------------
    def __lt__(self, other: 'SearchState') -> bool:
        """
        Overrides the standard "less than" (<) operator. This allows priority queues (like heapq) 
        to natively compare two SearchState objects and correctly order them.
        
        Args:
            other (SearchState): The adjacent state being compared against within the priority heap.
            
        Returns:
            bool: True if THIS state is mathematically "better" and should be expanded before the 'other' state.
        """
        # 1. Primary Priority Comparison
        # Floating point arithmetic can introduce micro-inaccuracies in Python (e.g., 0.1 + 0.2 != 0.30000000000000004). 
        # We use math.isclose to ensure that practically equal costs are treated as true ties.
        # If we used strict inequality (<) on close floats, it would arbitrarily break ties, 
        # which violates the strict secondary tie-breaking assignment requirements.
        if not math.isclose(self.priority_score, other.priority_score, rel_tol=1e-9):
            return self.priority_score < other.priority_score

        # 2. Secondary Tie-Breaker: Node Identifier (Ascending)
        # If priority scores are perfectly equal, the node with the smaller numerical ID is expanded first (e.g.,Node 4 before Node 7).
        if self.node_id != other.node_id:
            return self.node_id < other.node_id

        # 3. Tertiary Tie-Breaker: Chronological Order
        # If priority scores and node IDs are both equal (e.g., cyclic redundant paths), 
        # the node added to the frontier earliest (lowest timestamp) is expanded first.
        return self.timestamp < other.timestamp