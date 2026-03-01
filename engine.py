import heapq
from collections import deque
from typing import List, Tuple, Optional, Set, Any
from models import SearchState


class SearchEngine:
    """
    SearchEngine is responsible for executing various pathfinding and search algorithms 
    on a provided graph representation.
    
    Attributes:
        graph (Any): The graph object containing nodes, edges, heuristics, and origin/destination data.
        total_nodes_created (int): A metric tracking the total number of SearchState instances generated. 
                                   This is used to evaluate the space complexity and efficiency of the algorithm.
        creation_timestamp (int): A strictly increasing counter acting as a chronological timestamp. 
                                  It guarantees the secondary tie-breaking rule: if costs are equal, 
                                  nodes generated earlier are expanded first.
    """

    def __init__(self, graph: Any) -> None:
        """
        Initializes the Search Engine with the problem graph and resets tracking metrics.
        
        Args:
            graph: The problem space graph defining the Route Finding Problem.
        """
        self.graph = graph
        self.total_nodes_created: int = 0
        self.creation_timestamp: int = 0

    def _create_search_state(
        self, 
        node_identifier: int, 
        parent_state: Optional[SearchState], 
        cumulative_cost: float, 
        search_method: str
    ) -> SearchState:
        """
        Generates a new SearchState, calculates its heuristic cost, and increments tracking metrics.
        
        Args:
            node_identifier (int): The unique identifier of the current graph node.
            parent_state (Optional[SearchState]): The state object representing the predecessor node.
            cumulative_cost (float): The total path cost from the origin up to this node.
            search_method (str): The identifier of the algorithm currently being executed.
            
        Returns:
            SearchState: The newly instantiated and chronologically stamped search state.
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
        Backtracks from the reached goal state up to the origin to reconstruct the traversal path.
        
        Args:
            goal_state (SearchState): The state representing the reached destination.
            
        Returns:
            List[int]: A sequential list of node identifiers representing the path from origin to destination.
        """
        path_sequence: List[int] = []
        current_state: Optional[SearchState] = goal_state
        
        while current_state is not None:
            path_sequence.append(current_state.node_id)
            current_state = current_state.parent
            
        return path_sequence[::-1]

    def solve(self, search_method: str) -> Optional[Tuple[int, int, List[int]]]:
        """
        Dispatcher method that routes the execution to the appropriate search algorithm.
        
        Args:
            search_method (str): A string indicating the algorithm to execute (e.g., 'dfs', 'bfs', 'as').
            
        Returns:
            Optional[Tuple[int, int, List[int]]]: A tuple containing (Goal Node ID, Total Nodes Created, Path Sequence).
                                                  Returns None if all paths are exhausted and no solution exists.
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

    def _execute_depth_first_search(self) -> Optional[Tuple[int, int, List[int]]]:
        """
        Executes a Depth-First Search (DFS) traversal.
        
        Utilizes a global visited set (Graph Search approach) to efficiently prevent 
        infinite loops in cyclic graphs while guaranteeing a path is found if one exists.
        """
        start_state = self._create_search_state(self.graph.origin, None, 0.0, "dfs")
        stack: List[SearchState] = [start_state]
        visited_nodes: Set[int] = {self.graph.origin} 

        while stack:
            current_state = stack.pop()

            if current_state.node_id in self.graph.destinations:
                return current_state.node_id, self.total_nodes_created, self._reconstruct_path(current_state)

            neighbors = self.graph.get_neighbors(current_state.node_id)
            
            # DFS specific tie-breaking: Sort neighbors DESCENDING by identifier.
            # Because a stack is Last-In-First-Out (LIFO), pushing them in descending order 
            # ensures they are popped and expanded in strictly ASCENDING order.
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
        Executes a Breadth-First Search (BFS) traversal.
        """
        start_state = self._create_search_state(self.graph.origin, None, 0.0, "bfs")
        queue: deque[SearchState] = deque([start_state])
        visited_nodes: Set[int] = {self.graph.origin}

        while queue:
            current_state = queue.popleft()
            
            if current_state.node_id in self.graph.destinations:
                return current_state.node_id, self.total_nodes_created, self._reconstruct_path(current_state)

            neighbors = self.graph.get_neighbors(current_state.node_id)
            
            # BFS specific tie-breaking: Sort neighbors ASCENDING by identifier.
            # Because a queue is First-In-First-Out (FIFO), adding them in ascending order 
            # guarantees they are dequeued and expanded in the exact same ascending order.
            neighbors.sort(key=lambda neighbor: neighbor[0])

            for neighbor_identifier, edge_weight in neighbors:
                if neighbor_identifier not in visited_nodes:
                    visited_nodes.add(neighbor_identifier)
                    new_cumulative_cost = current_state.g + edge_weight
                    new_state = self._create_search_state(neighbor_identifier, current_state, new_cumulative_cost, "bfs")
                    queue.append(new_state)
                    
        return None

    def _execute_priority_search(self, search_method: str) -> Optional[Tuple[int, int, List[int]]]:
        """
        Executes priority-queue based algorithms: Greedy Best-First Search (GBFS), 
        A* Search (AS), and Uniform Cost Search (CUS1).
        
        Args:
            search_method (str): The identifier configuring how the evaluation function f(n) is computed.
        """
        start_state = self._create_search_state(self.graph.origin, None, 0.0, search_method)
        open_priority_queue: List[SearchState] = [start_state]
        closed_set: Set[int] = set()

        while open_priority_queue:
            current_state = heapq.heappop(open_priority_queue)

            if current_state.node_id in self.graph.destinations:
                return current_state.node_id, self.total_nodes_created, self._reconstruct_path(current_state)

            # Lazy deletion implementation: ignore states if the node was previously expanded 
            # via a shorter or equivalent path.
            if current_state.node_id in closed_set:
                continue
                
            closed_set.add(current_state.node_id)

            for neighbor_identifier, edge_weight in self.graph.get_neighbors(current_state.node_id):
                if neighbor_identifier not in closed_set:
                    new_cumulative_cost = current_state.g + edge_weight
                    new_state = self._create_search_state(neighbor_identifier, current_state, new_cumulative_cost, search_method)
                    
                    # Tie-breaking logic (Cost -> Chronological -> Identifier) is inherently 
                    # handled by Python's heapq combined with the SearchState's __lt__ magic method.
                    heapq.heappush(open_priority_queue, new_state)
                    
        return None

    def _execute_iterative_deepening_a_star(self) -> Optional[Tuple[int, int, List[int]]]:
        """
        Executes the Iterative Deepening A* (IDA*) search algorithm (CUS2 method).
        """
        initial_heuristic = self.graph.heuristic(self.graph.origin)
        current_threshold = initial_heuristic

        while True:
            # Generate a fresh start state for every deepening iteration. 
            # This ensures the total_nodes_created metric accurately reflects the multi-pass nature of IDA*.
            start_state = self._create_search_state(self.graph.origin, None, 0.0, "cus2")

            search_result = self._iterative_deepening_recursive(start_state, current_threshold, [self.graph.origin])

            # If the recursive function returns a SearchState, the goal was successfully reached.
            if isinstance(search_result, SearchState):
                return search_result.node_id, self.total_nodes_created, self._reconstruct_path(search_result)

            # If the threshold returned is infinity, all possible paths have been exhausted.
            if search_result == float("inf"):
                return None

            # Deepen the allowed cost threshold for the subsequent iteration.
            current_threshold = float(search_result)

    def _iterative_deepening_recursive(
        self, 
        current_state: SearchState, 
        current_threshold: float, 
        current_path_identifiers: List[int]
    ) -> Any:
        """
        Recursive depth-first traversal helper for Iterative Deepening A*.
        Evaluates nodes along a branch strictly up to the defined cost threshold.
        
        Args:
            current_state (SearchState): The state currently being evaluated.
            current_threshold (float): The maximum allowed total estimated cost (f = g + h) for this pass.
            current_path_identifiers (List[int]): Identifiers in the active path, used to prevent cycles.
            
        Returns:
            SearchState: If a valid destination is found.
            float: The minimum path cost that exceeded the threshold, which becomes the threshold for the next pass.
        """
        total_estimated_cost = current_state.g + current_state.h
        
        if total_estimated_cost > current_threshold:
            return total_estimated_cost

        if current_state.node_id in self.graph.destinations:
            return current_state

        minimum_exceeded_threshold = float("inf")
        neighbors = self.graph.get_neighbors(current_state.node_id)

        child_states: List[SearchState] = []
        for neighbor_identifier, edge_weight in neighbors:
            
            # Strict cycle prevention for the current active branch
            if neighbor_identifier not in current_path_identifiers:  
                new_cumulative_cost = current_state.g + edge_weight
                new_child_state = self._create_search_state(neighbor_identifier, current_state, new_cumulative_cost, "cus2")
                child_states.append(new_child_state)

        # Enforce exact IDA* tie-breaking logic: 
        # Primary: Ascending total estimated cost (f = g + h)
        # Secondary: Node identifier (Ascending)
        child_states.sort(key=lambda state: (state.g + state.h, state.node_id))

        for child_state in child_states:
            extended_path = current_path_identifiers + [child_state.node_id]
            recursive_result = self._iterative_deepening_recursive(child_state, current_threshold, extended_path)
            
            if isinstance(recursive_result, SearchState):
                return recursive_result
                
            if recursive_result < minimum_exceeded_threshold:
                minimum_exceeded_threshold = recursive_result

        return minimum_exceeded_threshold