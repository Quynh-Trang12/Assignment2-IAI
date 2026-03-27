# ---------------------------------------------------------------------------
# Imports & Dependencies
# ---------------------------------------------------------------------------
import sys
import os
import pandas as pd
from typing import List, Tuple, Optional

from graph import Graph
from engine import SearchEngine
from predictor import FlowPredictor
from sequence_store import SequenceStore


# ---------------------------------------------------------------------------
# Command Line Interface (CLI) Orchestrator
# ---------------------------------------------------------------------------
class SearchCLI:
    """
    Command-line entry point for the route finding system.
    """

    SUPPORTED_ALGORITHMS: List[str] = ["dfs", "bfs", "gbfs", "as", "cus1", "cus2"]
    SUPPORTED_MODELS: List[str] = ["mock", "baseline", "lstm", "gru"]

    @classmethod
    def execute(cls) -> None:
        """
        Run the search program from command line.
        """
        if len(sys.argv) < 3:
            print("Usage: python search.py <filepath> <method> [model_type]")
            print(f"Supported methods: {', '.join(cls.SUPPORTED_ALGORITHMS)}")
            print(f"Supported model types: {', '.join(cls.SUPPORTED_MODELS)}")
            sys.exit(1)

        target_filepath: str = sys.argv[1]
        target_method: str = sys.argv[2].lower()
        target_model_type: str = sys.argv[3].lower() if len(sys.argv) >= 4 else "mock"

        if target_method not in cls.SUPPORTED_ALGORITHMS:
            print(f"Error: Unknown search method '{target_method}'.")
            print(f"Supported methods: {', '.join(cls.SUPPORTED_ALGORITHMS)}")
            sys.exit(1)

        if target_model_type not in cls.SUPPORTED_MODELS:
            print(f"Error: Unknown model type '{target_model_type}'.")
            print(f"Supported model types: {', '.join(cls.SUPPORTED_MODELS)}")
            sys.exit(1)

        try:
            predictor = cls._build_predictor(target_model_type)
            problem_graph = Graph(predictor=predictor)
            problem_graph.load_from_file(target_filepath)

        except Exception as file_exception:
            print(f"Critical System Error: Failed to initialize system. Details: {file_exception}")
            sys.exit(1)

        search_engine = SearchEngine(problem_graph)
        search_result = search_engine.solve(target_method)

        cls._print_standardized_output(target_filepath, target_method, search_result)

    @staticmethod
    def _build_predictor(model_type: str) -> FlowPredictor:
        """
        Build the correct predictor for the selected model type.
        LSTM/GRU require SequenceStore; mock/baseline do not.
        """
        if model_type in {"mock", "baseline"}:
            return FlowPredictor(model_type=model_type)

        # For LSTM/GRU, load the processed time-series dataset
        ts_df = SearchCLI._load_ts_dataframe()
        store = SequenceStore(ts_df)

        return FlowPredictor(
            model_type=model_type,
            sequence_store=store,
            default_time_index=100
        )

    @staticmethod
    def _load_ts_dataframe() -> pd.DataFrame:
        """
        Load and preprocess the SCATS dataset into:
        SCATS, timestamp, flow

        This matches the notebook preprocessing pipeline.
        """
        base_dir = os.path.dirname(__file__)
        excel_path = os.path.join(base_dir, "Scats Data October 2006.xls")

        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Traffic dataset not found: {excel_path}")

        # Read the same sheet used in the notebook
        df = pd.read_excel(excel_path, sheet_name="Data")

        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]

        # Match notebook renaming
        df = df.rename(columns={
            "Start Time": "Date",
            "Unnamed: 0": "SCATS",
            "SCATS Number": "SCATS",   # harmless fallback if present
        })

        if "SCATS" not in df.columns or "Date" not in df.columns:
            raise ValueError("Dataset format error: missing SCATS or Date column.")

        # Flow columns are the time columns like 00:00:00, 00:15:00, ...
        volume_cols = [col for col in df.columns if ":" in str(col)]

        if not volume_cols:
            raise ValueError("Dataset format error: no flow/time columns found.")

        # Keep only the needed columns
        df = df[["SCATS", "Date"] + volume_cols].copy()

        # Wide -> long
        df_long = df.melt(
            id_vars=["SCATS", "Date"],
            value_vars=volume_cols,
            var_name="time",
            value_name="flow"
        )

        # Clean flow
        df_long["flow"] = pd.to_numeric(df_long["flow"], errors="coerce")
        df_long = df_long.dropna(subset=["flow"])

        # Ensure Date is datetime before using .dt
        df_long["Date"] = pd.to_datetime(df_long["Date"], errors="coerce")

        # Build timestamp exactly like notebook
        df_long["timestamp"] = pd.to_datetime(
            df_long["Date"].dt.strftime("%Y-%m-%d") + " " + df_long["time"],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce"
        )

        df_long = df_long.dropna(subset=["SCATS", "timestamp", "flow"])

        # Make SCATS consistent
        df_long["SCATS"] = df_long["SCATS"].astype(str).str.strip()

        # Keep only required columns
        ts_df = df_long[["SCATS", "timestamp", "flow"]].copy()

        # Aggregate duplicate rows
        ts_df = (
            ts_df.groupby(["SCATS", "timestamp"], as_index=False)["flow"]
            .sum()
        )

        # Sort for time series
        ts_df = ts_df.sort_values(["SCATS", "timestamp"]).reset_index(drop=True)

        return ts_df

    @staticmethod
    def _print_standardized_output(
        filepath: str,
        method: str,
        search_result: Optional[Tuple[int, int, List[int]]]
    ) -> None:
        """
        Print results in assignment format.
        """
        print(f"{filepath} {method.upper()}")

        if search_result is not None:
            reached_goal_id, total_nodes_created, path_sequence = search_result
            print(f"{reached_goal_id} {total_nodes_created}")
            print(" ".join(map(str, path_sequence)))
        else:
            print("No solution found.")


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    SearchCLI.execute()