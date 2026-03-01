# ---------------------------------------------------------------------------
# Imports & Dependencies
# ---------------------------------------------------------------------------
import math
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Core Problem Space Representation
# ---------------------------------------------------------------------------
class Graph:
    """
    Represents the mathematical state space for the Route Finding Problem.
    
    This architecture manages the mapping of spatial node coordinates, the weighted 
    directional edges connecting them, and the specific routing objectives (origin to destinations).
    
    Attributes:
        node_coordinates (Dict[int, Tuple[float, float]]): A strict mapping of node IDs to their (X, Y) 
                                                           spatial Euclidean coordinates. Utilized exclusively 
                                                           by the heuristic engine to calculate distance.
        adjacency_list (Dict[int, Dict[int, float]]): The core graph structure. Maps a source node ID to an 
                                                      inner dictionary of target neighbors and edge weights.
        origin (Optional[int]): The defined starting node ID for the traversal.
        destinations (List[int]): A collection of acceptable target node IDs. The agent must dynamically 
                                  seek the most optimal path to ANY of these valid goals.
    """

    # ---------------------------------------------------------------------------
    # Initialization & Parsing
    # ---------------------------------------------------------------------------
    def __init__(self) -> None:
        """Initializes a blank graph schema, ready to be populated by the file parser."""
        self.node_coordinates: Dict[int, Tuple[float, float]] = {}
        self.adjacency_list: Dict[int, Dict[int, float]] = {}
        self.origin: Optional[int] = None
        self.destinations: List[int] = []

    def load_from_file(self, filepath: str) -> None:
        """
        A robust parsing engine that ingests a custom-formatted text file and constructs 
        the in-memory mathematical graph representation.
        
        Args:
            filepath (str): The absolute or relative system path to the configuration file.
            
        Raises:
            FileNotFoundError: If the operating system cannot locate the target file.
            
        Internal Variables:
            file_stream (TextIO): The active read buffer.
            sanitized_lines (List[str]): The payload with all leading/trailing whitespace and blank lines stripped.
            current_section (Optional[str]): A state-tracker indicating which block of the file is actively being parsed.
            parts (List[str]): Temporary array holding split string tokens during line evaluation.
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
                # Expected Schema: "1: (4,1)"
                parts = line.split(":")
                node_identifier = int(parts[0])
                coordinate_strings = parts[1].strip(" ()").split(",")
                
                self.node_coordinates[node_identifier] = (
                    float(coordinate_strings[0]), 
                    float(coordinate_strings[1])
                )
                
            elif current_section == "Edges":
                # Expected Schema: "(2,1): 4"
                parts = line.split(":")
                source_node, target_node = map(int, parts[0].strip(" ()").split(","))
                edge_weight = float(parts[1])
                
                # Provision the inner dictionary lazily upon discovering a new source node
                if source_node not in self.adjacency_list:
                    self.adjacency_list[source_node] = {}
                    
                self.adjacency_list[source_node][target_node] = edge_weight
                
            elif current_section == "Origin":
                # Expected Schema: "2"
                self.origin = int(line)
                
            elif current_section == "Destinations":
                # Expected Schema: "5; 4"
                self.destinations = [int(destination_id) for destination_id in line.split(";")]

    # ---------------------------------------------------------------------------
    # Traversal & Heuristic Computations
    # ---------------------------------------------------------------------------
    def get_neighbors(self, node_identifier: int) -> List[Tuple[int, float]]:
        """
        Retrieves all valid outbound directional connections from a specific node.
        
        Architectural Note: 
        By returning a mathematically sorted list of neighbors, we guarantee that nodes are 
        always processed in strictly ascending numerical order by their node IDs. This natively 
        satisfies the assignment's secondary tie-breaking rule at the edge-generation level.
        
        Args:
            node_identifier (int): The ID of the node currently undergoing expansion.
            
        Returns:
            List[Tuple[int, float]]: A sorted list containing (Target Node ID, Edge Cost). 
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
        This utilizes the Euclidean distance formula. For a 2D spatial grid, Euclidean distance 
        represents the shortest possible physical path ("as the crow flies"). Because a real path 
        cannot be shorter than a straight line, this heuristic is mathematically guaranteed to be 
        both Admissible (never overestimates cost) and Consistent (satisfies the triangle inequality). 
        This is a strict requirement for ensuring A* Search yields perfectly optimal solutions.
        
        Args:
            node_identifier (int): The ID of the node currently being evaluated.
            
        Returns:
            float: The minimal straight-line distance to any defined destination.
            
        Internal Variables:
            current_x, current_y (float): The specific spatial coordinates of the node under evaluation.
            minimum_heuristic_distance (float): A tracker that continuously updates to hold the lowest 
                                                calculated distance across all potential goals.
            destination_x, destination_y (float): The coordinates of the target goal being compared against.
            euclidean_distance (float): The actual geometric straight-line distance computed using Pythagorean theorem.
        """
        # Base case: If no destinations exist, the heuristic cost to finish is zero.
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