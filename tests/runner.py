# ---------------------------------------------------------------------------
# Imports & Dependencies
# ---------------------------------------------------------------------------
import subprocess
import csv
import sys
import time
import logging
from typing import NamedTuple, List
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration & Telemetry Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("BenchmarkOrchestrator")


# ---------------------------------------------------------------------------
# Data Transfer Objects (DTOs)
# ---------------------------------------------------------------------------
class SearchResult(NamedTuple):
    """
    An immutable Data Transfer Object (DTO) representing the benchmarking metrics 
    and outcome of a single algorithm execution on a specific graph topology.
    
    Attributes:
        test_case_filename (str): The isolated name of the text file evaluated (e.g., 'T01_Standard.txt').
        search_method (str): The search algorithm executed (e.g., 'dfs', 'as').
        reached_goal_id (str): The node ID of the goal reached, or a string flag if none was found.
        total_nodes_created (int): The empirical space-complexity metric indicating memory footprint.
        path_sequence (str): A space-separated string representing the chronological traversal path.
        execution_status (str): The operational health of the run ('SUCCESS', 'FAIL', 'TIMEOUT', 'No_Solution').
        execution_duration (float): The empirical time-complexity metric, measured in precise seconds.
    """
    test_case_filename: str
    search_method: str
    reached_goal_id: str
    total_nodes_created: int
    path_sequence: str
    execution_status: str
    execution_duration: float


# ---------------------------------------------------------------------------
# Reporting Engine
# ---------------------------------------------------------------------------
class CSVBenchmarkReporter:
    """
    Responsible exclusively for serializing the aggregated benchmarking telemetry 
    into a structured CSV format for subsequent statistical analysis.
    
    Attributes:
        output_file_path (Path): The resolved, absolute destination path for the CSV report.
        headers (List[str]): The rigid, ordered schema defining the columns of the telemetry report.
    """
    
    def __init__(self, output_file_path: Path) -> None:
        """
        Initializes the reporter with a target destination and a strict header schema.
        
        Args:
            output_file_path (Path): The OS-agnostic target location for the report.
        """
        self.output_file_path = output_file_path
        self.headers: List[str] = [
            "TestCase", "Method", "Goal", "NodesCreated", 
            "Path", "Status", "Duration"
        ]

    def generate_report(self, benchmark_results: List[SearchResult]) -> None:
        """
        Translates the in-memory array of SearchResult DTOs into a physical CSV file.
        
        Args:
            benchmark_results (List[SearchResult]): The complete matrix of execution outcomes.
            
        Internal Variables:
            csv_file_stream (TextIO): The active, safe file buffer handling disk writes.
            csv_writer (csv.writer): The standard library utility handling comma-separation and string quoting.
        """
        # Ensure the parent directory ('tests/') exists before attempting to write the file
        self.output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Enforce UTF-8 to prevent cross-platform character encoding issues
        with self.output_file_path.open(mode="w", newline="", encoding="utf-8") as csv_file_stream:
            csv_writer = csv.writer(csv_file_stream)
            csv_writer.writerow(self.headers)
            
            for result in benchmark_results:
                csv_writer.writerow([
                    result.test_case_filename,
                    result.search_method.upper(),
                    result.reached_goal_id,
                    result.total_nodes_created,
                    result.path_sequence,
                    result.execution_status,
                    f"{result.execution_duration:.4f}",  # Format to 4 decimal places for scientific precision
                ])
                
        logger.info(f"Telemetry report successfully generated at: {self.output_file_path.resolve()}")


# ---------------------------------------------------------------------------
# Execution Orchestrator
# ---------------------------------------------------------------------------
class BenchmarkOrchestrator:
    """
    Manages the isolated execution of pathfinding algorithms across a suite of test cases.
    
    Attributes:
        search_executable (Path): The absolute path to the main search.py script.
        test_cases_directory (Path): The folder containing the mathematically generated topological graphs.
        supported_methods (List[str]): The authoritative array of algorithms scheduled for evaluation.
        timeout_seconds (float): A rigid execution ceiling (in seconds) to prevent infinite loops 
                                 (e.g., from poorly implemented DFS cycle checking) from freezing the suite.
    """

    def __init__(
        self, 
        search_executable: Path, 
        test_cases_directory: Path, 
        supported_methods: List[str],
        timeout_seconds: float = 5.0
    ) -> None:
        self.search_executable = search_executable
        self.test_cases_directory = test_cases_directory
        self.supported_methods = supported_methods
        self.timeout_seconds = timeout_seconds

    def _execute_isolated_process(self, test_file_path: Path, search_method: str) -> SearchResult:
        """
        Spawns a highly isolated OS-level subprocess to execute the search algorithm.
        
        Architectural Note:
        Running algorithms via `subprocess` instead of direct Python imports guarantees 
        absolute isolation from global state memory leaks. Every single test run is provided 
        a completely clean heap, ensuring that time and space complexity metrics are 
        scientifically fair and untainted by previous executions.
        
        Args:
            test_file_path (Path): The targeted graph file.
            search_method (str): The targeted algorithmic method.
            
        Returns:
            SearchResult: The parsed benchmarking payload containing performance and correctness data.
            
        Internal Variables:
            start_time_counter (float): The high-resolution system clock baseline.
            process_result (CompletedProcess): The captured execution environment payload (including stdout).
            elapsed_duration (float): The total execution lifespan calculated post-termination.
        """
        start_time_counter = time.perf_counter()
        file_name = test_file_path.name
        
        try:
            # sys.executable securely targets the exact same Python binary currently running the suite
            process_result = subprocess.run(
                [sys.executable, str(self.search_executable), str(test_file_path), search_method],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            
            elapsed_duration = time.perf_counter() - start_time_counter

            # Handle hard application crashes (e.g., Syntax Errors, unhandled KeyErrors in search.py)
            if process_result.returncode != 0:
                logger.error(f"Process crashed for {search_method.upper()} on {file_name}")
                return SearchResult(file_name, search_method, "ERROR", 0, "Crash Detected", "FAIL", elapsed_duration)

            return self._parse_standard_output(process_result.stdout, file_name, search_method, elapsed_duration)

        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout exceeded ({self.timeout_seconds}s) for {search_method.upper()} on {file_name}")
            return SearchResult(file_name, search_method, "TIMEOUT", 0, "Execution Timeout", "FAIL", self.timeout_seconds)
            
        except Exception as unexpected_error:
            # Catch-all for profound OS-level or execution failures (e.g., out-of-memory killing by OS)
            logger.error(f"Critical execution failure: {unexpected_error}")
            return SearchResult(file_name, search_method, "ERROR", 0, str(unexpected_error), "FAIL", 0.0)

    def _parse_standard_output(
        self, 
        raw_output: str, 
        file_name: str, 
        search_method: str, 
        elapsed_duration: float
    ) -> SearchResult:
        """
        Strictly parses the stdout payload returned by the targeted search executable.
        
        Architectural Note:
        The assignment specifies a very rigid three-line output format. This parsing 
        engine acts as a validator, immediately flagging executions that violate the 
        expected structural schema.
        
        Args:
            raw_output (str): The raw standard output captured from the CLI execution.
            file_name (str): The test case identifier used for DTO construction.
            search_method (str): The algorithm identifier used for DTO construction.
            elapsed_duration (float): The calculated execution time used for DTO construction.
            
        Returns:
            SearchResult: The mapped and validated data object.
            
        Internal Variables:
            output_lines (List[str]): The payload split by standard newline characters.
            metadata_parts (List[str]): The parsed chunks from the second line containing Goal and Node Count.
        """
        output_lines = raw_output.strip().split("\n")
        
        # Validation: Ensure minimum structural compliance
        if len(output_lines) < 2:
            return SearchResult(file_name, search_method, "N/A", 0, "Insufficient Output", "FAIL", elapsed_duration)

        # Validation: Proper handling of isolated/unreachable graph configurations
        if "No solution found" in output_lines[1]:
            return SearchResult(file_name, search_method, "None", 0, "None", "No_Solution", elapsed_duration)

        metadata_parts = output_lines[1].strip().split()
        
        if len(metadata_parts) < 2:
            return SearchResult(file_name, search_method, "ERROR", 0, "Malformed Metadata", "FAIL", elapsed_duration)

        reached_goal = metadata_parts[0]
        nodes_created = int(metadata_parts[1])
        
        # Extrapolate path sequence safely; default to empty brackets if missing
        path_sequence = output_lines[2].strip() if len(output_lines) > 2 else "[]"

        return SearchResult(
            file_name, search_method, reached_goal, nodes_created, path_sequence, "SUCCESS", elapsed_duration
        )

    def execute_suite(self) -> List[SearchResult]:
        """
        Orchestrates the Cartesian product execution matrix: iterating over every 
        available test file and running every supported search algorithm against it.
        
        Returns:
            List[SearchResult]: The complete collection of operational metrics and paths.
            
        Internal Variables:
            available_test_files (List[Path]): A dynamically resolved array of all text files in the test directory.
            aggregated_results (List[SearchResult]): The cumulative buffer holding the DTOs as they are generated.
        """
        # pathlib.Path.glob allows for elegant, OS-agnostic wildcard file discovery
        available_test_files = sorted(self.test_cases_directory.glob("*.txt"))
        aggregated_results: List[SearchResult] = []

        logger.info(f"Discovered {len(available_test_files)} test files. Initiating benchmarking matrix...")

        # Nested loop architecture (Cartesian Matrix execution)
        for test_file_path in available_test_files:
            for search_method in self.supported_methods:
                
                execution_result = self._execute_isolated_process(test_file_path, search_method)
                aggregated_results.append(execution_result)
                
                logger.info(
                    f"Executed {search_method.upper():<5} on {test_file_path.name:<20} "
                    f"Status: {execution_result.execution_status:<10} "
                    f"Nodes: {execution_result.total_nodes_created}"
                )

        return aggregated_results


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # base_directory is currently the 'tests/' folder
    base_directory = Path(__file__).parent
    
    # The project root is one level up from the 'tests/' folder
    project_root = base_directory.parent
    
    # Correctly resolve the absolute paths
    target_executable = project_root / "search.py"
    target_test_cases_dir = base_directory / "cases"
    target_report_file = base_directory / "results.csv"
    
    algorithms_to_evaluate = ["dfs", "bfs", "gbfs", "as", "cus1", "cus2"]

    # Instantiate the Orchestrator via Dependency Injection
    benchmark_orchestrator = BenchmarkOrchestrator(
        search_executable=target_executable,
        test_cases_directory=target_test_cases_dir,
        supported_methods=algorithms_to_evaluate,
        timeout_seconds=5.0
    )

    # Instantiate the Reporting Engine via Dependency Injection
    csv_reporter = CSVBenchmarkReporter(output_file_path=target_report_file)

    # Execute the entire automated testing and reporting pipeline
    final_metrics = benchmark_orchestrator.execute_suite()
    csv_reporter.generate_report(final_metrics)