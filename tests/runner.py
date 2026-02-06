import subprocess
import glob
import re
import csv
import sys
import time
import os
from typing import NamedTuple, Optional, List


class SearchResult(NamedTuple):
    case: str
    method: str
    goal: str
    nodes_created: int
    path: str
    status: str
    duration: float


class TestRunner:
    """
    Executes search.py across all test cases and methods.
    Parses CLI output and generates a comprehensive CSV report.
    """

    METHODS = ["dfs", "bfs", "gbfs", "as", "cus1", "cus2"]
    SEARCH_SCRIPT = "search.py"
    RESULTS_FILE = "tests/results.csv"

    @staticmethod
    def run_single(filepath: str, method: str) -> SearchResult:
        start_time = time.perf_counter()
        try:
            # subprocess.run guarantees isolation from global state
            result = subprocess.run(
                [sys.executable, TestRunner.SEARCH_SCRIPT, filepath, method],
                capture_output=True,
                text=True,
                timeout=5,
            )
            duration = time.perf_counter() - start_time

            if result.returncode != 0:
                return SearchResult(
                    filepath, method, "ERR", 0, "Crash", "FAIL", duration
                )

            # Strict Output Parsing based on Spec:
            # Line 1: filename method
            # Line 2: goal num_nodes (OR "No solution found.")
            # Line 3: path
            lines = result.stdout.strip().split("\n")
            if len(lines) < 2:
                return SearchResult(
                    filepath, method, "N/A", 0, "No Output", "FAIL", duration
                )

            header = lines[0].strip()  # Check if needed?

            if "No solution found" in lines[1]:
                return SearchResult(
                    filepath, method, "None", 0, "None", "NoSol", duration
                )

            # Parse Goal and Nodes
            meta_parts = lines[1].strip().split()
            if len(meta_parts) < 2:
                return SearchResult(
                    filepath, method, "ERR", 0, "ParseErr", "FAIL", duration
                )

            goal = meta_parts[0]
            nodes = int(meta_parts[1])
            path = lines[2].strip() if len(lines) > 2 else "[]"

            return SearchResult(
                filepath, method, goal, nodes, path, "SUCCESS", duration
            )

        except subprocess.TimeoutExpired:
            return SearchResult(filepath, method, "TIMEOUT", 0, "Timeout", "FAIL", 5.0)
        except Exception as e:
            return SearchResult(filepath, method, "ERR", 0, str(e), "FAIL", 0.0)

    @classmethod
    def execute_suite(cls):
        test_files = sorted(glob.glob("tests/cases/*.txt"))
        results = []

        print(f"[Runner] Found {len(test_files)} test files. Starting execution...")

        # Matrix Execution: Files x Methods
        for fpath in test_files:
            fname = os.path.basename(fpath)
            for method in cls.METHODS:
                print(
                    f"  -> Running {method.upper()} on {fname}...", end="", flush=True
                )
                res = cls.run_single(fpath, method)
                results.append(res)
                print(f" {res.status} ({res.nodes_created} nodes)")

        cls.save_report(results)

    @staticmethod
    def save_report(results: List[SearchResult]):
        with open(TestRunner.RESULTS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "TestCase",
                    "Method",
                    "Goal",
                    "NodesCreated",
                    "Path",
                    "Status",
                    "Duration",
                ]
            )
            for r in results:
                writer.writerow(
                    [
                        os.path.basename(r.case),
                        r.method.upper(),
                        r.goal,
                        r.nodes_created,
                        r.path,
                        r.status,
                        f"{r.duration:.4f}",
                    ]
                )
        print(f"\n[Runner] Report generated at {TestRunner.RESULTS_FILE}")


if __name__ == "__main__":
    TestRunner.execute_suite()
