import math
from typing import Dict, List, Tuple, Optional


class Graph:
    """
    Represents the problem space for the Route Finding Problem.
    
    This class manages the spatial coordinates of nodes, the weighted directed edges 
    connecting them, and the specific origin and destination parameters required for 
    the search algorithms.
    
    Attributes:
        node_coordinates (Dict[int, Tuple[float, float]]): A mapping of node identifiers to their 
                                                           spatial coordinates (X, Y) on a 2D plane. 
                                                           This is heavily utilized by the heuristic 
                                                           function to calculate spatial distances.
        adjacency_list (Dict[int, Dict[int, float]]): Maps a source node identifier to a dictionary of its 
                                                      target neighbors and corresponding edge weights.
        origin (Optional[int]): The starting node identifier for the search traversal.
        destinations (List[int]): A collection of valid target node identifiers.
    """

    def __init__(self) -> None:
        """
        Initializes an empty graph structure.
        """
        self.node_coordinates: Dict[int, Tuple[float, float]] = {}
        self.adjacency_list: Dict[int, Dict[int, float]] = {}
        self.origin: Optional[int] = None
        self.destinations: List[int] = []

    def load_from_file(self, filepath: str) -> None:
        """
        Parses a specifically formatted text file to populate the graph's structure.
        
        Args:
            filepath (str): The relative or absolute path to the graph configuration file.
            
        Raises:
            FileNotFoundError: If the specified file cannot be located.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as file_stream:
                # Strip whitespace and ignore completely blank lines to ensure robust parsing
                sanitized_lines = [line.strip() for line in file_stream if line.strip()]
        except FileNotFoundError:
            raise

        current_section: Optional[str] = None
        
        for line in sanitized_lines:
            # Detect section headers (e.g., "Nodes:", "Edges:")
            if line.endswith(":"):
                current_section = line[:-1]
                continue

            if current_section == "Nodes":
                # Expected Format: "1: (4,1)"
                parts = line.split(":")
                node_identifier = int(parts[0])
                coordinate_strings = parts[1].strip(" ()").split(",")
                
                self.node_coordinates[node_identifier] = (
                    float(coordinate_strings[0]), 
                    float(coordinate_strings[1])
                )
                
            elif current_section == "Edges":
                # Expected Format: "(2,1): 4"
                parts = line.split(":")
                source_node, target_node = map(int, parts[0].strip(" ()").split(","))
                edge_weight = float(parts[1])
                
                # Initialize the inner dictionary for a new source node
                if source_node not in self.adjacency_list:
                    self.adjacency_list[source_node] = {}
                    
                self.adjacency_list[source_node][target_node] = edge_weight
                
            elif current_section == "Origin":
                # Expected Format: "2"
                self.origin = int(line)
                
            elif current_section == "Destinations":
                # Expected Format: "5; 4"
                self.destinations = [int(destination_id) for destination_id in line.split(";")]

    def get_neighbors(self, node_identifier: int) -> List[Tuple[int, float]]:
        """
        Retrieves all reachable neighboring nodes and their step costs from a given node.
        
        Args:
            node_identifier (int): The identifier of the node currently being expanded.
            
        Returns:
            List[Tuple[int, float]]: A sorted list of tuples, where each tuple contains 
                                     (Neighbor Node ID, Edge Weight). 
        """
        if node_identifier not in self.adjacency_list:
            return []
            
        # Architectural Note: 
        # Returning sorted items guarantees that neighbors are processed in strictly ascending 
        # numerical order by their node IDs. This inherently satisfies the assignment's secondary 
        # tie-breaking rule at the expansion generation level.
        return sorted(self.adjacency_list[node_identifier].items())

    def heuristic(self, node_identifier: int) -> float:
        """
        Calculates the estimated cost (h-value) from the current node to the nearest destination.
        
        Architectural Note:
        This utilizes the Euclidean distance formula. For a 2D spatial routing problem, 
        Euclidean distance guarantees that the heuristic is both Admissible (it never 
        overestimates the true path cost) and Consistent (it satisfies the triangle inequality). 
        This is critical for ensuring A* (AS) yields optimal solutions.
        
        Args:
            node_identifier (int): The identifier of the current node being evaluated.
            
        Returns:
            float: The shortest Euclidean distance to any valid destination node. 
                   Returns 0.0 if no destinations are defined.
        """
        # If there are no target destinations, the estimated cost to reach a goal is effectively zero.
        if not self.destinations:
            return 0.0
            
        # Extract the specific 2D coordinates (X and Y) of the node currently being evaluated.
        # current_x represents the horizontal position, current_y represents the vertical position.
        current_x, current_y = self.node_coordinates[node_identifier]
        
        # Initialize the minimum distance to infinity. This acts as a baseline tracker 
        # so that any calculated valid distance will successfully overwrite it on the first iteration.
        minimum_heuristic_distance = float("inf")
        
        # Iterate through all valid destination nodes to find the one physically closest 
        # to our current node. (Required for multi-destination routing problems).
        for destination_identifier in self.destinations:
            
            # Extract the specific 2D coordinates (X and Y) of the target destination node.
            destination_x, destination_y = self.node_coordinates[destination_identifier]
            
            # Calculate the straight-line (Euclidean) distance between the current node and the target destination.
            # Formula: sqrt((x1 - x2)^2 + (y1 - y2)^2)
            # This represents the absolute shortest possible physical path (as the crow flies) 
            # between the two coordinates, fulfilling the heuristic's admissibility requirement.
            euclidean_distance = math.sqrt(
                (current_x - destination_x) ** 2 + 
                (current_y - destination_y) ** 2
            )
            
            # Update the tracker if the calculated distance to this specific destination 
            # is smaller than the smallest distance found so far.
            if euclidean_distance < minimum_heuristic_distance:
                minimum_heuristic_distance = euclidean_distance
                
        # Return the shortest calculated distance to represent the 'h(n)' value for this node.
        return minimum_heuristic_distance