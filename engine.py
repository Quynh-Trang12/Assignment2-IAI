# ---------------------------------------------------------------------------
# Imports & Dependencies
# ---------------------------------------------------------------------------
import heapq
from collections import deque
from typing import List, Tuple, Optional, Set, Any
from models import SearchState


# ---------------------------------------------------------------------------
# Core Class Definition
# ---------------------------------------------------------------------------
class SearchEngine:
    """
    SearchEngine is the core algorithmic orchestrator responsible for executing various 
    informed and uninformed pathfinding algorithms on a provided graph representation.
    
    Attributes:
        graph (Any): The instantiated mathematical problem space containing spatial node 
                     coordinates, adjacency lists, and routing parameters.
        total_nodes_created (int): A critical space-complexity metric tracking the absolute total 
                                   number of SearchState instances instantiated during a run.
        creation_timestamp (int): A strictly monotonically increasing counter. It guarantees the 
                                  tertiary assignment tie-breaking rule: if heuristic costs and 
                                  node IDs are identical, the node generated first chronologically 
                                  is expanded first.
    """

    # ---------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------
    def __init__(self, graph: Any) -> None:
        """
        Initializes the Search Engine and resets all internal benchmarking metrics.
        
        Args:
            graph (Any): The mathematical problem space to be traversed.
        """
        self.graph = graph
        self.total_nodes_created: int = 0
        self.creation_timestamp: int = 0

    # ---------------------------------------------------------------------------
    # State & Path Management
    # ---------------------------------------------------------------------------
    def _create_search_state(
        self, 
        node_identifier: int, 
        parent_state: Optional[SearchState], 
        cumulative_cost: float, 
        search_method: str
    ) -> SearchState:
        """
        A factory method that provisions a new SearchState object, computes its specific 
        heuristic estimation, and increments the global tracking metrics.
        
        Args:
            node_identifier (int): The unique numeric ID of the specific graph node.
            parent_state (Optional[SearchState]): The immediate predecessor state in the search tree. 
                                                  Passed as None for the origin node.
            cumulative_cost (float): The actual path cost accumulated from the origin to this node (g-value).
            search_method (str): The identifier of the algorithm dictating how the priority score is computed.
            
        Returns:
            SearchState: The newly instantiated, chronologically stamped node wrapper.
            
        Internal Variables:
            heuristic_cost (float): The estimated remaining cost to the goal (h-value), computed dynamically.
        """
        self.total_nodes_created += 1
        self.creation_timestamp += 1
        
        heuristic_cost = self.graph.heuristic(node_identifier)
        
        return SearchState(
            node_identifier, 
            parent_state, 
            cumulative_cost, 
            heuristic_cost, 
            search_method, 
            self.creation_timestamp
        )

    def _reconstruct_path(self, goal_state: SearchState) -> List[int]:
        """
        Backtracks through the parent pointers of the search tree from the destination 
        back to the origin to reconstruct the final sequential path.
        
        Args:
            goal_state (SearchState): The valid destination state that terminated the search.
            
        Returns:
            List[int]: An ordered list of node IDs from origin to destination.
            
        Internal Variables:
            path_sequence (List[int]): The temporary buffer holding the reversed path during backtracking.
            current_state (Optional[SearchState]): The active pointer walking up the ancestral chain.
        """
        path_sequence: List[int] = []
        current_state: Optional[SearchState] = goal_state
        
        while current_state is not None:
            path_sequence.append(current_state.node_id)
            current_state = current_state.parent
            
        # The path is constructed backwards (Goal -> Origin), so it must be reversed before returning
        return path_sequence[::-1]

    # ---------------------------------------------------------------------------
    # Execution Dispatcher
    # ---------------------------------------------------------------------------
    def solve(self, search_method: str) -> Optional[Tuple[int, int, List[int]]]:
        """
        The primary dispatcher method that normalizes user input and routes execution 
        to the appropriate internal traversal algorithm.
        
        Args:
            search_method (str): A string indicating the algorithm (e.g., 'dfs', 'as').
            
        Returns:
            Optional[Tuple[int, int, List[int]]]: A payload containing (Goal ID, Nodes Created, Path), 
                                                  or None if the entire space is exhausted without a solution.
        """
        normalized_method = search_method.lower()
        
        if normalized_method == "dfs":
            return self._execute_depth_first_search()
        if normalized_method == "bfs":
            return self._execute_breadth_first_search()
        if normalized_method == "gbfs":
            return self._execute_priority_search("gbfs")
        if normalized_method == "as":
            return self._execute_priority_search("as")
        if normalized_method == "cus1":
            return self._execute_priority_search("cus1")
        if normalized_method == "cus2":
            return self._execute_iterative_deepening_a_star()
            
        return None

    # ---------------------------------------------------------------------------
    # Uninformed Search Algorithms
    # ---------------------------------------------------------------------------
    def _execute_depth_first_search(self) -> Optional[Tuple[int, int, List[int]]]:
        """
        Executes a classic Depth-First Search (DFS) using a Graph Search approach.
        
        Architectural Note: 
        DFS explores as deeply as possible along each branch before backtracking. 
        We use a global `visited_nodes` set to prevent infinite loops in cyclic graphs.
        
        Internal Variables:
            stack (List[SearchState]): A Last-In-First-Out (LIFO) data structure governing the frontier.
            visited_nodes (Set[int]): A hash set ensuring nodes are expanded only once.
        """
        start_state = self._create_search_state(self.graph.origin, None, 0.0, "dfs")
        stack: List[SearchState] = [start_state]
        visited_nodes: Set[int] = {self.graph.origin} 

        while stack:
            current_state = stack.pop()

            if current_state.node_id in self.graph.destinations:
                return current_state.node_id, self.total_nodes_created, self._reconstruct_path(current_state)

            neighbors = self.graph.get_neighbors(current_state.node_id)
            
            # Tie-Breaking Justification (DFS): 
            # Because a stack pops the LAST element added, pushing neighbors in strictly DESCENDING 
            # order guarantees that the algorithm will pop and expand them in ASCENDING order.
            neighbors.sort(key=lambda neighbor: neighbor[0], reverse=True)

            for neighbor_identifier, edge_weight in neighbors:
                if neighbor_identifier not in visited_nodes:
                    visited_nodes.add(neighbor_identifier)
                    new_cumulative_cost = current_state.g + edge_weight
                    new_state = self._create_search_state(neighbor_identifier, current_state, new_cumulative_cost, "dfs")
                    stack.append(new_state)
                    
        return None

    def _execute_breadth_first_search(self) -> Optional[Tuple[int, int, List[int]]]:
        """
        Executes a Breadth-First Search (BFS) using a Graph Search approach.
        
        Architectural Note:
        BFS explores the graph strictly level-by-level, guaranteeing the shortest path 
        in terms of unweighted hops.
        
        Internal Variables:
            queue (deque[SearchState]): A First-In-First-Out (FIFO) queue governing the frontier.
            visited_nodes (Set[int]): A hash set ensuring nodes are expanded only once.
        """
        start_state = self._create_search_state(self.graph.origin, None, 0.0, "bfs")
        queue: deque[SearchState] = deque([start_state])
        visited_nodes: Set[int] = {self.graph.origin}

        while queue:
            current_state = queue.popleft()
            
            if current_state.node_id in self.graph.destinations:
                return current_state.node_id, self.total_nodes_created, self._reconstruct_path(current_state)

            neighbors = self.graph.get_neighbors(current_state.node_id)
            
            # Tie-Breaking Justification (BFS): 
            # Because a queue dequeues the FIRST element added, adding neighbors in strictly ASCENDING 
            # order guarantees they will be dequeued and expanded in that exact same ascending order.
            neighbors.sort(key=lambda neighbor: neighbor[0])

            for neighbor_identifier, edge_weight in neighbors:
                if neighbor_identifier not in visited_nodes:
                    visited_nodes.add(neighbor_identifier)
                    new_cumulative_cost = current_state.g + edge_weight
                    new_state = self._create_search_state(neighbor_identifier, current_state, new_cumulative_cost, "bfs")
                    queue.append(new_state)
                    
        return None

    # ---------------------------------------------------------------------------
    # Priority Queue Search Algorithms
    # ---------------------------------------------------------------------------
    def _execute_priority_search(self, search_method: str) -> Optional[Tuple[int, int, List[int]]]:
        """
        A unified execution engine for heuristic-driven priority algorithms: 
        Greedy Best-First Search (GBFS), A* Search (AS), and Uniform Cost Search (CUS1).
        
        Architectural Note:
        These algorithms evaluate nodes based on an f-cost. Instead of updating existing 
        nodes in the priority queue (which is O(N) in Python), we use "Lazy Deletion" by 
        ignoring nodes that have already been expanded via a cheaper path.
        
        Internal Variables:
            open_priority_queue (List[SearchState]): A binary min-heap frontier automatically 
                                                     sorted by f-cost, then ID, then timestamp.
            closed_set (Set[int]): Tracks nodes that have already been optimally expanded.
        """
        start_state = self._create_search_state(self.graph.origin, None, 0.0, search_method)
        open_priority_queue: List[SearchState] = [start_state]
        closed_set: Set[int] = set()

        while open_priority_queue:
            # heapq.heappop inherently respects the SearchState's __lt__ magic method tie-breakers
            current_state = heapq.heappop(open_priority_queue)

            if current_state.node_id in self.graph.destinations:
                return current_state.node_id, self.total_nodes_created, self._reconstruct_path(current_state)

            # Lazy Deletion: If this node was previously expanded, a shorter/better path 
            # already processed it. Skip redundant work.
            if current_state.node_id in closed_set:
                continue
                
            closed_set.add(current_state.node_id)

            for neighbor_identifier, edge_weight in self.graph.get_neighbors(current_state.node_id):
                if neighbor_identifier not in closed_set:
                    new_cumulative_cost = current_state.g + edge_weight
                    new_state = self._create_search_state(neighbor_identifier, current_state, new_cumulative_cost, search_method)
                    heapq.heappush(open_priority_queue, new_state)
                    
        return None

    # ---------------------------------------------------------------------------
    # Iterative Deepening A* (IDA*) Engine
    # ---------------------------------------------------------------------------
    def _execute_iterative_deepening_a_star(self) -> Optional[Tuple[int, int, List[int]]]:
        """
        Executes Iterative Deepening A* (IDA*), classified as Custom Method 2 (CUS2).
        
        Architectural Note:
        IDA* combines the space-efficiency of DFS with the optimality of A*. It performs 
        successive DFS passes, pruning branches that exceed a dynamically expanding f-cost threshold.
        
        Internal Variables:
            initial_heuristic (float): The starting threshold, which is the h-value of the origin.
            current_threshold (float): The maximum allowed total cost (f = g + h) for the active iteration.
            search_result (Any): The outcome of the recursive pass (either the Goal State, infinity, or a new threshold).
        """
        initial_heuristic = self.graph.heuristic(self.graph.origin)
        current_threshold = initial_heuristic

        while True:
            # Crucial Benchmark Requirement: Generate a fresh start state for EVERY deepening iteration. 
            # This ensures total_nodes_created accurately reflects the overlapping multi-pass nature of IDA*.
            start_state = self._create_search_state(self.graph.origin, None, 0.0, "cus2")

            search_result = self._iterative_deepening_recursive(start_state, current_threshold, [self.graph.origin])

            # Success: The goal was physically reached within the current cost threshold.
            if isinstance(search_result, SearchState):
                return search_result.node_id, self.total_nodes_created, self._reconstruct_path(search_result)

            # Exhaustion: The entire reachable graph was traversed and the threshold never increased.
            if search_result == float("inf"):
                return None

            # Deepen: Update the threshold to the smallest cost that exceeded the previous limit.
            current_threshold = float(search_result)

    def _iterative_deepening_recursive(
        self, 
        current_state: SearchState, 
        current_threshold: float, 
        current_path_identifiers: List[int]
    ) -> Any:
        """
        The recursive depth-first traversal engine powering IDA*.
        
        Architectural Note:
        Unlike standard DFS, IDA* evaluates nodes based on their estimated total cost (g + h). 
        To maintain linear space complexity, cycle checking is localized strictly to the current active 
        branch rather than using a global closed set.
        
        Args:
            current_state (SearchState): The specific node currently being evaluated in this recursive frame. 
                                         It encapsulates the node ID, the path cost so far (g), the heuristic (h), 
                                         and the parent pointer needed for backtracking if the goal is found.
            
            current_threshold (float): The strict upper bound for the f-cost (g + h) allowed in this specific 
                                       IDA* iteration. If `current_state`'s f-cost exceeds this limit, 
                                       the branch is aggressively pruned and the cost is bubbled up.
                                       
            current_path_identifiers (List[int]): A chronological sequence of node IDs representing the active 
                                                  branch from the origin down to the `current_state`. It is 
                                                  utilized exclusively for local cycle detection, ensuring the 
                                                  search does not traverse back up its own ancestral chain and 
                                                  fall into an infinite loop.
        
        Returns:
            SearchState: If the goal is successfully reached within the threshold, the goal state is returned.
            float: If the path is pruned, returns the minimum f-cost that exceeded the threshold, 
                   which is used to calculate the threshold limit for the next outer iteration.
                   
        Internal Variables:
            total_estimated_cost (float): The f-value (g + h) of the current node.
            minimum_exceeded_threshold (float): Tracks the smallest f-value among all pruned child branches.
            child_states (List[SearchState]): All valid, non-cyclic adjacent states slated for recursive exploration.
        """
        total_estimated_cost = current_state.g + current_state.h
        
        # Pruning condition: The path has become too expensive for this iteration
        if total_estimated_cost > current_threshold:
            return total_estimated_cost

        # Goal condition: We have successfully reached a valid destination
        if current_state.node_id in self.graph.destinations:
            return current_state

        minimum_exceeded_threshold = float("inf")
        neighbors = self.graph.get_neighbors(current_state.node_id)

        child_states: List[SearchState] = []
        for neighbor_identifier, edge_weight in neighbors:
            
            # Local Cycle Prevention: Ensures the current sequence doesn't loop back on itself, 
            # but allows other branches to visit the same node later if cheaper.
            if neighbor_identifier not in current_path_identifiers:  
                new_cumulative_cost = current_state.g + edge_weight
                new_child_state = self._create_search_state(neighbor_identifier, current_state, new_cumulative_cost, "cus2")
                child_states.append(new_child_state)

        # Tie-Breaking Justification (IDA*): 
        # The assignment dictates expanding nodes with the lowest f-cost first.
        # If f-costs are tied, we break the tie using an ASCENDING Node ID.
        child_states.sort(key=lambda state: (state.g + state.h, state.node_id))

        for child_state in child_states:
            extended_path = current_path_identifiers + [child_state.node_id]
            recursive_result = self._iterative_deepening_recursive(child_state, current_threshold, extended_path)
            
            # Bubble up the successful SearchState immediately to stop further traversal
            if isinstance(recursive_result, SearchState):
                return recursive_result
                
            # Track the lowest cost that crossed the threshold line
            if recursive_result < minimum_exceeded_threshold:
                minimum_exceeded_threshold = recursive_result

        return minimum_exceeded_threshold