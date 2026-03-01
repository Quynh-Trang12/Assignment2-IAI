# ---------------------------------------------------------------------------
# Imports & Dependencies
# ---------------------------------------------------------------------------
import logging
from typing import List, Dict, Tuple, NamedTuple, Union
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration & Telemetry Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("TestFactoryOrchestrator")


# ---------------------------------------------------------------------------
# Data Transfer Objects (DTOs)
# ---------------------------------------------------------------------------
class TestCaseDefinition(NamedTuple):
    """
    An immutable Data Transfer Object (DTO) representing the complete mathematical 
    topology and routing parameters of a single test case.
    
    Attributes:
        filename (str): The exact string identifier for the output file (e.g., 'T01_Standard.txt'). 
                        This dictates how the file will be saved and referenced by the runner.
        nodes (Dict[int, Tuple[Union[int, float], Union[int, float]]]): A spatial mapping dictionary. 
                                                Keys are integer node IDs. 
                                                Values are tuples representing (X, Y) Euclidean coordinates.
        adjacency_list (Dict[int, Dict[int, Union[int, float]]]): A nested dictionary representing directed, weighted edges.
                                                      The outer key is the source node ID. 
                                                      The inner key is the target neighbor node ID. 
                                                      The inner value is the traversal cost.
        origin (int): The starting node identifier where the pathfinding algorithm will begin its search.
        destinations (List[int]): A dynamic list of valid target node identifiers. The algorithm must 
                                  dynamically pathfind to whichever one is mathematically closest.
        architectural_purpose (str): A meta-data string explicitly defining exactly what edge-case or 
                                     algorithmic vulnerability this specific topology is designed to test.
    """
    filename: str
    nodes: Dict[int, Tuple[Union[int, float], Union[int, float]]]
    adjacency_list: Dict[int, Dict[int, Union[int, float]]]
    origin: int
    destinations: List[int]
    architectural_purpose: str


# ---------------------------------------------------------------------------
# Serialization Engine (Single Responsibility)
# ---------------------------------------------------------------------------
class GraphSerializer:
    """
    Responsible exclusively for serializing in-memory Graph objects (TestCaseDefinitions) 
    into the strict text-based format mandated by the Assignment 2A specification.
    
    Attributes:
        output_directory (Path): A robust, OS-agnostic Path object defining the absolute 
                                 or relative destination folder where the .txt files will be provisioned.
    """

    def __init__(self, output_directory: Path) -> None:
        """
        Initializes the serializer and guarantees the existence of the target directory.
        
        Args:
            output_directory (Path): The target file system destination.
        """
        self.output_directory = output_directory
        
        # exist_ok=True prevents crashes if the directory already exists from a previous run.
        # parents=True ensures any missing parent directories (e.g., 'tests/') are also created.
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def _format_number(self, value: Union[int, float]) -> Union[int, float]:
        """
        A smart-formatting helper utility ensuring strict compliance with the assignment's visual schema.
        
        Architectural Note:
        While python naturally processes edge weights as floats (e.g., 4.0), the provided sample
        'PathFinder-test.txt' uses whole integers. This function bridges that gap by dynamically 
        stripping the floating-point decimal if the number is mathematically a whole integer, 
        preventing potential string-matching failures during automated grading.
        
        Args:
            value (Union[int, float]): The mathematical value to be formatted.
            
        Returns:
            Union[int, float]: An integer if the float ends in .0, otherwise returns the original float.
        """
        return int(value) if float(value).is_integer() else value

    def write_to_disk(self, test_case: TestCaseDefinition) -> None:
        """
        Translates the mathematical graph topology into the assignment's custom syntax and writes it to disk.
        
        Args:
            test_case (TestCaseDefinition): The rigorously defined graph topology to serialize.
            
        Internal Variables:
            target_file_path (Path): The resolved, fully concatenated absolute path for the specific file.
            file_stream (TextIO): The active, open file buffer handling the I/O write operations.
            formatted_destinations (str): A stringified, semi-colon-separated representation of the target nodes.
        """
        target_file_path = self.output_directory / test_case.filename
        
        # Enforce UTF-8 encoding for cross-platform compatibility to prevent localized character errors
        with target_file_path.open(mode="w", encoding="utf-8") as file_stream:
            
            # 1. Serialize Spatial Node Coordinates
            file_stream.write("Nodes:\n")
            for node_identifier, coordinates in test_case.nodes.items():
                x_coordinate = self._format_number(coordinates[0])
                y_coordinate = self._format_number(coordinates[1])
                file_stream.write(f"{node_identifier}: ({x_coordinate},{y_coordinate})\n")

            # 2. Serialize Directed Edge Weights
            file_stream.write("Edges:\n")
            for source_node, neighbors in test_case.adjacency_list.items():
                for target_node, edge_weight in neighbors.items():
                    formatted_weight = self._format_number(edge_weight)
                    file_stream.write(f"({source_node},{target_node}): {formatted_weight}\n")

            # 3. Serialize Search Parameters
            file_stream.write("\nOrigin:\n")
            file_stream.write(f"{test_case.origin}\n")

            file_stream.write("Destinations:\n")
            # Enforce strict assignment requirement: multiple destinations MUST be semi-colon separated
            formatted_destinations = "; ".join(map(str, test_case.destinations))
            file_stream.write(f"{formatted_destinations}\n")
            
        logger.info(f"Successfully provisioned: {test_case.filename:<25} | Purpose: {test_case.architectural_purpose}")


# ---------------------------------------------------------------------------
# Test Generation Orchestrator
# ---------------------------------------------------------------------------
class TestFactoryOrchestrator:
    """
    Orchestrates the instantiation and generation of the 10 mathematically distinct 
    edge-case scenarios designed to stress-test the pathfinding algorithms.
    
    Attributes:
        serializer (GraphSerializer): The dependency-injected engine responsible for all disk I/O operations.
    """

    def __init__(self, target_directory: Path) -> None:
        """
        Initializes the test factory by instantiating the required serialization engine.
        
        Args:
            target_directory (Path): The root folder where the generated cases should be deposited.
        """
        self.serializer = GraphSerializer(target_directory)

    def _define_test_cases(self) -> List[TestCaseDefinition]:
        """
        Assembles the comprehensive suite of 10 assignment-specific topological maps.
        
        Returns:
            List[TestCaseDefinition]: A strictly typed list of DTOs, where each element 
                                      contains the full graph definition for one testing scenario.
        """
        return [
            TestCaseDefinition(
                filename="T01_Standard.txt",
                architectural_purpose="Baseline multi-path graph from assignment PDF",
                nodes={1: (4, 1), 2: (2, 2), 3: (4, 4), 4: (6, 3), 5: (5, 6), 6: (7, 5)},
                adjacency_list={
                    2: {1: 4, 3: 4},
                    3: {1: 5, 2: 5, 5: 6, 6: 7},
                    1: {3: 5, 4: 6},
                    4: {1: 6, 3: 5, 5: 7},
                    5: {3: 6, 4: 8},
                    6: {3: 7},
                },
                origin=2,
                destinations=[5, 4]
            ),
            TestCaseDefinition(
                filename="T02_Unreachable.txt",
                architectural_purpose="Validates complete space exhaustion without infinite loops",
                nodes={1: (0, 0), 2: (1, 1), 3: (10, 10)},
                adjacency_list={1: {2: 1}, 2: {1: 1}},
                origin=1,
                destinations=[3]
            ),
            TestCaseDefinition(
                filename="T03_Cycles.txt",
                architectural_purpose="Evaluates visited-set cycle prevention logic",
                nodes={1: (0, 0), 2: (1, 0), 3: (0, 1), 4: (1, 1)},
                adjacency_list={1: {2: 1}, 2: {3: 1, 4: 5}, 3: {1: 1}},
                origin=1,
                destinations=[4]
            ),
            TestCaseDefinition(
                filename="T04_TieBreak.txt",
                architectural_purpose="Forces strict ID-based ascending tie-breaking enforcement",
                nodes={1: (0, 0), 2: (1, 1), 3: (1, -1), 4: (2, 0)},
                adjacency_list={1: {2: 1, 3: 1}, 2: {4: 1}, 3: {4: 1}},
                origin=1,
                destinations=[4]
            ),
            TestCaseDefinition(
                filename="T05_Linear.txt",
                architectural_purpose="Zero branching factor for O(N) baseline verification",
                nodes={1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (3, 0), 5: (4, 0)},
                adjacency_list={1: {2: 1}, 2: {3: 1}, 3: {4: 1}, 4: {5: 1}},
                origin=1,
                destinations=[5]
            ),
            TestCaseDefinition(
                filename="T06_MultiDest.txt",
                architectural_purpose="Tests multi-goal dynamic heuristic tracking",
                nodes={1: (0, 0), 2: (0, 10), 3: (0, 20), 4: (5, 0), 5: (10, 0)},
                adjacency_list={1: {2: 10, 4: 1}, 2: {3: 10}, 4: {5: 1}},
                origin=1,
                destinations=[3, 5]
            ),
            TestCaseDefinition(
                filename="T07_HeuristicTrap.txt",
                architectural_purpose="Tricks GBFS heuristic to prove A* optimality superiority",
                nodes={1: (0, 0), 2: (10, 0), 3: (0, 10), 4: (11, 0)},
                adjacency_list={1: {2: 10, 3: 2}, 2: {4: 10}, 3: {4: 2}},
                origin=1,
                destinations=[4]
            ),
            TestCaseDefinition(
                filename="T08_CostVsHops.txt",
                architectural_purpose="Separates BFS (fewest hops) from UCS (lowest path cost)",
                nodes={1: (0, 0), 2: (2, 2), 3: (5, 5), 4: (1, -1), 5: (2, -2), 6: (3, -1)},
                adjacency_list={1: {2: 50, 4: 1}, 2: {3: 50}, 4: {5: 1}, 5: {6: 1}, 6: {3: 1}},
                origin=1,
                destinations=[3]
            ),
            TestCaseDefinition(
                filename="T09_DeadEnd.txt",
                architectural_purpose="Validates DFS and IDA* backtracking up the search tree",
                nodes={1: (0, 0), 2: (-5, 0), 3: (5, 0), 4: (10, 0)},
                adjacency_list={1: {2: 1, 3: 1}, 3: {4: 1}},
                origin=1,
                destinations=[4]
            ),
            TestCaseDefinition(
                filename="T10_ZeroCost.txt",
                architectural_purpose="Ensures algorithms do not divide-by-zero or prune incorrectly",
                nodes={1: (0, 0), 2: (1, 0), 3: (2, 0)},
                adjacency_list={1: {2: 0}, 2: {3: 0}},
                origin=1,
                destinations=[3]
            )
        ]

    def provision_all(self) -> None:
        """
        Executes the mass serialization pipeline to generate the testing environment.
        
        Internal Variables:
            test_cases (List[TestCaseDefinition]): The fully instantiated array of topologies from _define_test_cases.
            index (int): The current loop iteration number (used implicitly by enumerate, primarily for tracking).
            test_case (TestCaseDefinition): The specific iteration's data object currently being serialized to disk.
        """
        logger.info("Initiating comprehensive test case generation suite...")
        
        test_cases = self._define_test_cases()
        
        for index, test_case in enumerate(test_cases, 1):
            self.serializer.write_to_disk(test_case)
            
        logger.info(f"Suite provisioning complete. {len(test_cases)} topologies written to disk.")


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Define absolute systemic paths using pathlib for execution safety.
    # __file__ securely references the location of THIS specific python script,
    # ensuring the 'tests/cases' directory is built relative to the codebase root,
    # regardless of where the user's terminal is currently cd'd into.
    base_directory = Path(__file__).parent
    target_output_directory = base_directory / "tests" / "cases"

    # Instantiate via Dependency Injection and run
    factory_orchestrator = TestFactoryOrchestrator(target_directory=target_output_directory)
    factory_orchestrator.provision_all()