import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass(order=False)
class SearchState:
    """
    Represents an individual node (or state) within the search tree during traversal.
    
    This class strictly enforces the tie-breaking rules required for the assignment:
    1. Primary: Priority value (calculated dynamically based on the search algorithm).
    2. Secondary: Node Identifier (Ascending order).
    3. Tertiary: Chronological generation order (nodes generated earlier are expanded first).
    
    Attributes:
        node_id (int): The unique identifier of the graph node represented by this state.
        parent (Optional[SearchState]): The predecessor state that generated this state. 
                                        Used to backtrack and reconstruct the final path.
        g (float): The cumulative path cost from the origin to this node. 
                   (Kept as 'g' to adhere to standard mathematical pathfinding notation).
        h (float): The estimated heuristic cost from this node to the goal. 
                   (Kept as 'h' to adhere to standard mathematical pathfinding notation).
        search_method (str): The identifier of the search algorithm evaluating this state (e.g., 'as', 'gbfs').
        timestamp (int): A strictly increasing counter marking when this state was generated.
        priority_score (float): The calculated evaluation metric used by priority queues. 
                                Computed automatically upon instantiation.
    """
    node_id: int
    parent: Optional['SearchState']
    g: float
    h: float
    search_method: str
    timestamp: int
    priority_score: float = field(init=False)

    def __post_init__(self) -> None:
        """
        Calculates the evaluation priority score for the node based on the selected search algorithm.
        This score dictates the node's position within a priority queue (e.g., in A* or GBFS).
        """
        method = self.search_method.lower()
        
        # Uninformed searches (BFS, DFS) do not rely on priority queues for ordering 
        # in the same way, but setting a baseline priority ensures structural compatibility.
        if method in ('bfs', 'dfs'):
            self.priority_score = 0.0
            
        # Greedy Best-First Search (GBFS) evaluates purely on the estimated heuristic cost: f(n) = h(n)
        elif method == 'gbfs':
            self.priority_score = self.h
            
        # A* Search (AS) and Iterative Deepening A* (CUS2) evaluate on total estimated cost: f(n) = g(n) + h(n)
        elif method in ('as', 'cus2'):
            self.priority_score = self.g + self.h
            
        # Uniform Cost Search (CUS1) evaluates purely on the cumulative path cost: f(n) = g(n)
        elif method == 'cus1':
            self.priority_score = self.g
            
        # Fallback for undefined methods to prevent mathematical errors
        else:
            self.priority_score = 0.0

    def __lt__(self, other: 'SearchState') -> bool:
        """
        Defines the "less than" comparison to strictly enforce the priority queue ordering 
        and the assignment's specific tie-breaking rules.
        
        Args:
            other (SearchState): The adjacent state being compared against within the priority heap.
            
        Returns:
            bool: True if this state should be prioritized (popped first) ahead of the 'other' state.
        """
        # 1. Primary Priority Comparison
        # Floating point arithmetic can introduce micro-inaccuracies in Python (e.g., 0.1 + 0.2 != 0.30000000000000004). 
        # We use math.isclose to ensure that practically equal costs are treated as true ties.
        # If we used strict inequality (<) on close floats, it would arbitrarily break ties, 
        # which violates the strict secondary tie-breaking assignment requirements.
        if not math.isclose(self.priority_score, other.priority_score, rel_tol=1e-9):
            return self.priority_score < other.priority_score

        # 2. Secondary Tie-Breaker: Node Identifier (Ascending)
        # If priority scores are perfectly equal, the node with the smaller numerical ID is expanded first.
        if self.node_id != other.node_id:
            return self.node_id < other.node_id

        # 3. Tertiary Tie-Breaker: Chronological Order
        # If priority scores and node IDs are both equal (e.g., cyclic redundant paths), 
        # the node added to the frontier earliest (lowest timestamp) is expanded first.
        return self.timestamp < other.timestamp