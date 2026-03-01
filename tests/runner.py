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
    """
    test_case_filename: str
    search_method: str
    reached_goal_id: str
    total_nodes_created: int
    path_sequence: str
    execution_status: str
    execution_duration: float


# ---------------------------------------------------------------------------
# Reporting Engine (High Cohesion / Single Responsibility)
# ---------------------------------------------------------------------------
class CSVBenchmarkReporter:
    """
    Responsible exclusively for serializing benchmarking telemetry into a 
    structured CSV format for statistical analysis.
    """
    
    def __init__(self, output_file_path: Path) -> None:
        self.output_file_path = output_file_path
        self.headers = [
            "TestCase", "Method", "Goal", "NodesCreated", 
            "Path", "Status", "Duration"
        ]

    def generate_report(self, benchmark_results: List[SearchResult]) -> None:
        """Writes the aggregated SearchResult objects to the configured CSV file."""
        # Ensure the parent directory exists before attempting to write the file
        self.output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
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
                    f"{result.execution_duration:.4f}",
                ])
                
        logger.info(f"Telemetry report successfully generated at: {self.output_file_path.resolve()}")


# ---------------------------------------------------------------------------
# Execution Orchestrator
# ---------------------------------------------------------------------------
class BenchmarkOrchestrator:
    """
    Manages the isolated execution of pathfinding algorithms across a suite of test cases.
    
    Attributes:
        search_executable (Path): The target Python script to execute.
        test_cases_directory (Path): The directory containing the topological graph configurations.
        supported_methods (List[str]): The algorithms scheduled for evaluation.
        timeout_seconds (float): The maximum allowed execution time per algorithm before termination.
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
        Spawns a highly isolated subprocess to execute the search algorithm.
        This guarantees zero state-leakage or memory contamination between runs.
        """
        start_time_counter = time.perf_counter()
        file_name = test_file_path.name
        
        try:
            # sys.executable ensures the subprocess uses the exact same Python environment
            process_result = subprocess.run(
                [sys.executable, str(self.search_executable), str(test_file_path), search_method],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            
            elapsed_duration = time.perf_counter() - start_time_counter

            if process_result.returncode != 0:
                logger.error(f"Process crashed for {search_method.upper()} on {file_name}")
                return SearchResult(file_name, search_method, "ERROR", 0, "Crash Detected", "FAIL", elapsed_duration)

            return self._parse_standard_output(process_result.stdout, file_name, search_method, elapsed_duration)

        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout exceeded ({self.timeout_seconds}s) for {search_method.upper()} on {file_name}")
            return SearchResult(file_name, search_method, "TIMEOUT", 0, "Execution Timeout", "FAIL", self.timeout_seconds)
            
        except Exception as unexpected_error:
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
        Strictly parses the stdout payload returned by the search executable.
        """
        output_lines = raw_output.strip().split("\n")
        
        if len(output_lines) < 2:
            return SearchResult(file_name, search_method, "N/A", 0, "Insufficient Output", "FAIL", elapsed_duration)

        if "No solution found" in output_lines[1]:
            return SearchResult(file_name, search_method, "None", 0, "None", "No_Solution", elapsed_duration)

        metadata_parts = output_lines[1].strip().split()
        if len(metadata_parts) < 2:
            return SearchResult(file_name, search_method, "ERROR", 0, "Malformed Metadata", "FAIL", elapsed_duration)

        reached_goal = metadata_parts[0]
        nodes_created = int(metadata_parts[1])
        path_sequence = output_lines[2].strip() if len(output_lines) > 2 else "[]"

        return SearchResult(file_name, search_method, reached_goal, nodes_created, path_sequence, "SUCCESS", elapsed_duration)

    def execute_suite(self) -> List[SearchResult]:
        """
        Orchestrates the Cartesian product execution of all test files against all algorithms.
        """
        # pathlib.Path.glob allows for elegant, OS-agnostic file discovery
        available_test_files = sorted(self.test_cases_directory.glob("*.txt"))
        aggregated_results: List[SearchResult] = []

        logger.info(f"Discovered {len(available_test_files)} test files. Initiating benchmarking matrix...")

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
    # Define system dependencies using pathlib for absolute structural safety
    base_directory = Path(__file__).parent
    target_executable = base_directory / "search.py"
    target_test_cases_dir = base_directory / "tests" / "cases"
    target_report_file = base_directory / "tests" / "results.csv"
    
    algorithms_to_evaluate = ["dfs", "bfs", "gbfs", "as", "cus1", "cus2"]

    # Instantiate the Orchestrator (Dependency Injection)
    benchmark_orchestrator = BenchmarkOrchestrator(
        search_executable=target_executable,
        test_cases_directory=target_test_cases_dir,
        supported_methods=algorithms_to_evaluate,
        timeout_seconds=5.0
    )

    # Instantiate the Reporter
    csv_reporter = CSVBenchmarkReporter(output_file_path=target_report_file)

    # Execute the pipeline
    final_metrics = benchmark_orchestrator.execute_suite()
    csv_reporter.generate_report(final_metrics)