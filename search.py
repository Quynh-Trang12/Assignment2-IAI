# ---------------------------------------------------------------------------
# Imports & Dependencies
# ---------------------------------------------------------------------------
import sys
from typing import List, Tuple, Optional
from graph import Graph
from engine import SearchEngine


# ---------------------------------------------------------------------------
# Command Line Interface (CLI) Orchestrator
# ---------------------------------------------------------------------------
class SearchCLI:
    """
    SearchCLI acts as the primary execution entry point for the route finding application.
    
    Architectural Note:
    Following the Single Responsibility Principle, this class is strictly responsible for:
    1. Parsing and validating command-line arguments.
    2. Instantiating the mathematical problem space (Graph).
    3. Delegating the algorithmic execution to the SearchEngine.
    4. Formatting the standard output (stdout) to strictly comply with the 
       Assignment 2A automated testing specifications.
       
    Attributes:
        SUPPORTED_ALGORITHMS (List[str]): The authoritative registry of valid search methods.
    """
    
    SUPPORTED_ALGORITHMS: List[str] = ["dfs", "bfs", "gbfs", "as", "cus1", "cus2"]

    @classmethod
    def execute(cls) -> None:
        """
        The main execution pipeline. Handles the entire lifecycle of a single terminal command.
        
        Internal Variables:
            target_filepath (str): The relative or absolute path to the graph configuration file.
            target_method (str): The requested algorithmic strategy.
            problem_graph (Graph): The instantiated 2D spatial mapping and adjacency matrix.
            search_engine (SearchEngine): The configured algorithmic solver.
            search_result (Optional[Tuple]): The payload returned upon successful or exhausted traversal.
        """
        # 1. Input Validation: Ensure the user provided the correct number of CLI arguments.
        if len(sys.argv) < 3:
            print("Usage: python search.py <filepath> <method>")
            print(f"Supported methods: {', '.join(cls.SUPPORTED_ALGORITHMS)}")
            sys.exit(1)

        target_filepath: str = sys.argv[1]
        target_method: str = sys.argv[2].lower()

        # 2. Input Validation: Ensure the requested algorithm is mathematically supported.
        if target_method not in cls.SUPPORTED_ALGORITHMS:
            print(f"Error: Unknown search method '{target_method}'.")
            print(f"Supported methods: {', '.join(cls.SUPPORTED_ALGORITHMS)}")
            sys.exit(1)

        # 3. Environment Instantiation: Load the graph topology from disk into memory.
        problem_graph = Graph()
        try:
            problem_graph.load_from_file(target_filepath)
        except Exception as file_exception:
            # Catch file-not-found or parsing errors to prevent ugly stack traces for the end-user
            print(f"Critical System Error: Failed to load graph topology. Details: {file_exception}")
            sys.exit(1)

        # 4. Algorithmic Execution: Delegate the traversal to the SearchEngine.
        search_engine = SearchEngine(problem_graph)
        search_result = search_engine.solve(target_method)

        # 5. Output Formatting: Serialize the results strictly according to assignment requirements.
        cls._print_standardized_output(target_filepath, target_method, search_result)

    @staticmethod
    def _print_standardized_output(
        filepath: str, 
        method: str, 
        search_result: Optional[Tuple[int, int, List[int]]]
    ) -> None:
        """
        Formats and prints the execution payload to stdout.
        
        Architectural Note:
        The assignment requires an exact 3-line output structure. Any deviation 
        (like extra spaces, brackets, or missing lines) will cause automated grading scripts 
        to fail. This method rigidly enforces that schema.
        
        Args:
            filepath (str): The name/path of the tested graph file.
            method (str): The algorithm utilized.
            search_result (Optional[Tuple]): The result payload containing (Goal ID, Nodes Created, Path).
        """
        # Line 1 Specification: <filename> <method>
        print(f"{filepath} {method.upper()}")

        if search_result is not None:
            reached_goal_id, total_nodes_created, path_sequence = search_result

            # Line 2 Specification: <goal_node_identifier> <number_of_nodes_created>
            print(f"{reached_goal_id} {total_nodes_created}")

            # Line 3 Specification: <path_sequence_separated_by_single_spaces>
            # We map the integer node IDs to strings to securely join them without Python list brackets
            formatted_path_sequence = " ".join(map(str, path_sequence))
            print(formatted_path_sequence)
            
        else:
            # Fallback Specification: If the graph is entirely exhausted without reaching a valid goal.
            print("No solution found.")


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    SearchCLI.execute()