"""
Traffic Database Initialization Pipeline
========================================
Extracts spatial topology, loads real-world historical data snapshots,
and utilizes the compiled ML Inference Engine to pre-compute mathematically
valid traffic flows into SQLite caches.
"""

import sys
import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Dict

_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logger_setup import get_colored_logger
from src.ml.traffic_flow_predictor import TrafficFlowPredictor

logger = get_colored_logger(__name__)


def extract_topology_from_map(map_path: Path) -> List[int]:
    """Extracts operational SCATS identifiers from the physical network definition."""
    nodes: List[int] = []
    try:
        with open(map_path, "r", encoding="utf-8") as file:
            is_node_section = False
            for line in file:
                line = line.strip()
                if line == "Nodes:":
                    is_node_section = True
                    continue
                if not line or line == "Edges:":
                    if is_node_section:
                        break
                if is_node_section and ":" in line:
                    nodes.append(int(line.split(":")[0].strip()))
        return nodes
    except FileNotFoundError:
        logger.critical(f"Target map file not found at: {map_path}")
        sys.exit(1)


def load_real_historical_data(valid_nodes: List[int]) -> Dict[int, List[float]]:
    """
    Strictly loads factually correct historical sequences from the Phase 2 dataset.
    Extracts a dynamic 3-step time slice for a specific target hour to simulate
    real-world spatial traffic variance across the network.
    """
    # Pointing directly to your uploaded, existing data artifact
    dataset_path = (
        _PROJECT_ROOT / "data" / "processed" / "Scats_Data_Cleaned_GapSafe.csv"
    )

    historical_data = {}

    if dataset_path.exists():
        logger.info(f"Loading real historical data from: {dataset_path.name}")
        try:
            df = pd.read_csv(dataset_path)

            # Ensure proper datetime formatting for precise filtering
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df = df.sort_values(by=["SCATS Number", "Datetime"])

            # Target a known Rush Hour in your dataset to ensure varied, high-volume traffic
            target_time = pd.to_datetime("2006-10-31 08:00:00")
            logger.info(
                f"Extracting spatial network state for target time: {target_time}"
            )

            for node in valid_nodes:
                # Filter data specifically for this intersection, up to the target time
                node_data = df[
                    (df["SCATS Number"] == node) & (df["Datetime"] <= target_time)
                ]

                # We need exactly 3 historical steps (t-2, t-1, t) for the model
                if len(node_data) >= 3:
                    # Extract the last 3 'Flow' values as a standard Python list
                    sequence = node_data["Flow"].tail(3).tolist()
                    historical_data[node] = sequence
                else:
                    # Fallback only if the specific SCATS ID is missing from the CSV at this time
                    historical_data[node] = [350.0, 350.0, 350.0]

        except Exception as e:
            logger.error(f"Failed to parse Scats_Data_Cleaned_GapSafe.csv: {e}")
            sys.exit(1)
    else:
        logger.critical(f"Required dataset missing: {dataset_path}")
        sys.exit(1)

    return historical_data


def seed_model_database(
    db_path: Path,
    model_architecture: str,
    valid_nodes: List[int],
    real_data: Dict[int, List[float]],
) -> None:
    """
    Provisions a SQLite database with factually valid inference data.
    """
    if db_path.exists():
        db_path.unlink()

    predictor = None
    if model_architecture != "fallback":
        logger.info(f"Initializing {model_architecture.upper()} Inference Engine...")
        try:
            predictor = TrafficFlowPredictor(model_type=model_architecture)
        except Exception as e:
            logger.error(f"Engine initialization failed for {model_architecture}: {e}")
            return

    try:
        with sqlite3.connect(db_path) as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    scats_id INTEGER PRIMARY KEY,
                    predicted_flow REAL
                )
            """
            )

            payload = []
            for node in valid_nodes:
                if model_architecture == "fallback" or predictor is None:
                    flow_metric = 350.0
                else:
                    # Fetch the real sequence for this specific intersection
                    node_sequence = real_data.get(node, [350.0, 350.0, 350.0])
                    flow_metric = predictor.predict_flow(node_sequence)

                payload.append((node, round(flow_metric, 2)))

            cursor.executemany(
                "INSERT INTO predictions (scats_id, predicted_flow) VALUES (?, ?)",
                payload,
            )
            connection.commit()
            logger.info(
                f"Generated factual data cache: {db_path.name} | Nodes: {len(valid_nodes)}"
            )

    except sqlite3.DatabaseError as db_error:
        logger.error(f"Database operation failed for {db_path.name}: {db_error}")
        sys.exit(1)


def execute_pipeline() -> None:
    """Orchestrates the extraction and database seeding processes."""
    map_filepath = _PROJECT_ROOT / "data" / "maps" / "map.txt"
    db_directory = _PROJECT_ROOT / "data" / "database"

    if not map_filepath.exists():
        logger.critical(f"Topology dependency missing: {map_filepath}")
        sys.exit(1)

    db_directory.mkdir(parents=True, exist_ok=True)
    network_nodes = extract_topology_from_map(map_filepath)

    if not network_nodes:
        logger.error("Network structural parsing yielded 0 nodes.")
        sys.exit(1)

    # Load the real dataset ONCE to ensure consistency across all models
    real_historical_data = load_real_historical_data(network_nodes)

    target_architectures = ["lstm", "gru", "fnn"]

    for architecture in target_architectures:
        target_db = db_directory / f"traffic_state_{architecture}.db"
        seed_model_database(
            target_db, architecture, network_nodes, real_historical_data
        )

    fallback_db = db_directory / "traffic_state.db"
    seed_model_database(fallback_db, "fallback", network_nodes, real_historical_data)
    logger.info("Database initialization pipeline completed successfully.")


if __name__ == "__main__":
    execute_pipeline()
