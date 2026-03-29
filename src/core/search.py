# ---------------------------------------------------------------------------
# Imports & Dependencies
# ---------------------------------------------------------------------------
import sys
import sqlite3
import re
from pathlib import Path
from typing import List, Tuple, Optional, Any

# Add project root to sys.path to resolve 'src' module imports natively
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import Phase 4/5 Utilities & Configuration
from src.utils.speed_time_converter import calculate_travel_time
from src.utils.logger_setup import get_colored_logger
from src.utils.config import APP_CONFIG

from src.core.graph import Graph
from src.core.engine import SearchEngine

# Initialize the centralized color-coded logger
logger = get_colored_logger(__name__)


# ---------------------------------------------------------------------------
# Command Line Interface (CLI) Orchestrator
# ---------------------------------------------------------------------------
class SearchCLI:
    """
    SearchCLI acts as the primary execution entry point for the route finding application.
    Integrated with Assignment 2B Top-K paths and Phase 5 ML Traffic Predictions.
    """

    @classmethod
    def _apply_traffic_kinematics(cls, graph: Graph) -> None:
        """
        Connects to the SQLite database and mathematically overrides the physical
        distance edge weights with ML-predicted travel times (hours).
        """
        graph.is_time_based = True

        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent
        base_db_path = project_root / "data" / "database"

        # Dynamically hunt for the default model specified in config
        default_model = APP_CONFIG["GUI_Defaults"]["default_model"]
        db_path_model = base_db_path / f"traffic_state_{default_model}.db"
        db_path_general = base_db_path / "traffic_state.db"

        active_db = db_path_model if db_path_model.exists() else db_path_general

        if not active_db.exists():
            logger.error("CRITICAL: No traffic database found. Run ML pipeline first.")
            sys.exit(1)

        flow_predictions = {}
        with sqlite3.connect(active_db) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT scats_id, predicted_flow FROM predictions")
                for row in cursor.fetchall():
                    match = re.search(r"\d+", str(row[0]))
                    if match:
                        flow_predictions[int(match.group())] = float(row[1])
            except sqlite3.OperationalError as e:
                logger.error(f"Database schema error: {e}")
                sys.exit(1)

        # Apply Assignment 2B Configuration Rules
        delay_hrs = APP_CONFIG["TBRGS_Settings"]["intersection_delay_seconds"] / 3600.0
        fallback_flow = APP_CONFIG["ML_Settings"]["unmonitored_flow_assumption"]

        # Overwrite physical distances in the adjacency list with travel time costs
        for source_node, neighbors in graph.adjacency_list.items():
            for neighbor_id, physical_distance in neighbors.items():
                predicted_flow = flow_predictions.get(neighbor_id, fallback_flow)
                travel_time_hours = (
                    calculate_travel_time(predicted_flow, physical_distance) + delay_hrs
                )
                graph.adjacency_list[source_node][neighbor_id] = travel_time_hours

    @classmethod
    def execute(cls) -> None:
        """
        The main execution pipeline.
        """
        if len(sys.argv) < 3:
            logger.warning("Usage: python search.py <filepath> <method> [--traffic]")
            logger.info(
                "Methods: dfs, bfs, gbfs, as, cus1, cus2, topk-<n> (e.g., topk-5)"
            )
            sys.exit(1)

        target_filepath: str = sys.argv[1]
        target_method: str = sys.argv[2].lower()
        use_traffic_ml = "--traffic" in sys.argv

        # Validate that the requested method is supported (Legacy or Top-K)
        is_legacy = target_method in ["dfs", "bfs", "gbfs", "as", "cus1", "cus2"]
        is_topk = target_method.startswith("topk-")

        if not (is_legacy or is_topk):
            logger.error(f"Unknown search method '{target_method}'.")
            sys.exit(1)

        problem_graph = Graph()
        try:
            problem_graph.load_from_file(target_filepath)
        except Exception as file_exception:
            logger.critical(f"Failed to load graph topology. Details: {file_exception}")
            sys.exit(1)

        # ---------------------------------------------------------
        # PHASE 5 & TBRGS CONFIGURATION INJECTION
        # ---------------------------------------------------------
        # Requirement 4: Base state converts physical distances to travel time
        # using the 60km/h baseline + intersection delay.
        problem_graph.is_time_based = True
        base_speed = APP_CONFIG["TBRGS_Settings"]["speed_limit_kmh"]
        delay_hrs = APP_CONFIG["TBRGS_Settings"]["intersection_delay_seconds"] / 3600.0

        for src, neighbors in problem_graph.adjacency_list.items():
            for dst, physical_dist in neighbors.items():
                problem_graph.adjacency_list[src][dst] = (
                    physical_dist / base_speed
                ) + delay_hrs

        if use_traffic_ml:
            logger.info("Phase 5: Engaging ML Traffic Kinematics override...")
            cls._apply_traffic_kinematics(problem_graph)

        # Execute the search engine
        search_engine = SearchEngine(problem_graph)
        search_result = search_engine.solve(target_method)

        cls._print_standardized_output(target_filepath, target_method, search_result)

    @staticmethod
    def _print_standardized_output(
        filepath: str, method: str, search_result: Any
    ) -> None:
        """
        Dynamically formats stdout. Maintains the strict 3-line format for Assignment 2A
        algorithms, but expands into a multi-path format for Yen's Top-K.
        """
        print(f"{filepath} {method.upper()}")

        if search_result:
            if isinstance(search_result, list):
                # Yen's Top-K Output Format
                for i, (
                    reached_goal_id,
                    total_nodes_created,
                    path_sequence,
                ) in enumerate(search_result):
                    print(
                        f"Path {i+1} --- Goal: {reached_goal_id} | Nodes Explored: {total_nodes_created}"
                    )
                    print(" ".join(map(str, path_sequence)))
            else:
                # Legacy Assignment 2A Output Format (Tuple)
                reached_goal_id, total_nodes_created, path_sequence = search_result
                print(f"{reached_goal_id} {total_nodes_created}")
                print(" ".join(map(str, path_sequence)))
        else:
            print("No solution found.")


if __name__ == "__main__":
    SearchCLI.execute()
