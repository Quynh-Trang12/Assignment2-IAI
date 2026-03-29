import sys
from src.utils.logger_setup import get_colored_logger
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

# Instantiate the centralized, color-coded logger
logger = get_colored_logger(__name__)
warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# ETL Pipeline
# ---------------------------------------------------------
def clean_and_interpolate_traffic_data(
    input_csv_path: Path, output_csv_path: Path
) -> None:
    logger.info("Initiating data load sequence from: %s", input_csv_path)

    if not input_csv_path.exists():
        logger.error("Source file not found: %s", input_csv_path)
        sys.exit(1)

    # 1. Dynamic File Loading
    try:
        if input_csv_path.suffix.lower() == ".xls":
            logger.info(
                "Legacy Excel binary (.xls) detected. Engaging xlrd engine for the 'Data' sheet..."
            )
            df = pd.read_excel(input_csv_path, sheet_name="Data", engine="xlrd")
        elif input_csv_path.suffix.lower() == ".xlsx":
            logger.info(
                "Modern Excel binary (.xlsx) detected. Engaging openpyxl engine..."
            )
            df = pd.read_excel(input_csv_path, sheet_name="Data", engine="openpyxl")
        else:
            logger.info("Text-based CSV detected. Engaging standard parser...")
            df = pd.read_csv(input_csv_path)
    except Exception as e:
        logger.error("Failed to parse the data file: %s", e)
        sys.exit(1)

    # 2. Matrix Melting & Noise Filtering
    if "Datetime" not in df.columns:
        logger.info(
            "Raw matrix format detected. Initiating bulletproof ETL and positional mapping..."
        )

        try:
            df = df.copy()

            # Identify structural boundaries
            time_cols = df.columns[-96:].tolist()
            date_col = df.columns[-97]
            scats_col = df.columns[0]

            # Strict Schema Enforcement (Vaporize headers/metadata)
            df["Strict_SCATS"] = pd.to_numeric(df[scats_col], errors="coerce")
            df = df.dropna(subset=["Strict_SCATS"]).copy()
            df[scats_col] = df["Strict_SCATS"].astype(int)
            df = df.drop(columns=["Strict_SCATS"])

            # Temporal Parsing (Date only)
            if pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df["BaseDate"] = df[date_col].dt.floor("D")
            else:
                numeric_dates = pd.to_numeric(df[date_col], errors="coerce")
                is_numeric = numeric_dates.notna()
                df["BaseDate"] = pd.NaT

                if is_numeric.any():
                    df.loc[is_numeric, "BaseDate"] = pd.to_datetime(
                        numeric_dates[is_numeric], origin="1899-12-30", unit="D"
                    ).dt.floor("D")
                if (~is_numeric).any():
                    df.loc[~is_numeric, "BaseDate"] = pd.to_datetime(
                        df.loc[~is_numeric, date_col], errors="coerce", dayfirst=True
                    ).dt.floor("D")

                df = df.dropna(subset=["BaseDate"]).copy()

            # POSITIONAL TIME MAPPING
            interval_mapping = {col: i for i, col in enumerate(time_cols)}
            df = df.rename(columns=interval_mapping)
            clean_time_cols = list(range(96))

            # Matrix Melting
            df = pd.melt(
                df,
                id_vars=[scats_col, "BaseDate"],
                value_vars=clean_time_cols,
                var_name="IntervalIndex",
                value_name="Flow",
            )
            df = df.rename(columns={scats_col: "SCATS Number"})

            # Force the Flow column to be strict numbers so any garbage text (like "-" or " ") will instantly vaporize into NaN.
            df["Flow"] = pd.to_numeric(df["Flow"], errors="coerce")

            # Chronological Reconstruction using integer math (15 min per interval)
            df["TimeDelta"] = pd.to_timedelta(df["IntervalIndex"] * 15, unit="m")
            df["Datetime"] = df["BaseDate"] + df["TimeDelta"]

            # Final Cleanup
            df = df.dropna(subset=["Datetime", "Flow"]).copy()
            df = df.drop(
                columns=["BaseDate", "IntervalIndex", "TimeDelta"], errors="ignore"
            )

            # ---------------------------------------------------------
            # SPATIAL AGGREGATION (Duplicate Timestamp Resolution)
            # ---------------------------------------------------------
            # VicRoads often logs multiple rows per site for different approaches/lanes.
            # Therefore, group them by Site and Time, summing the flow to get the total intersection volume.
            initial_length = len(df)
            df = df.groupby(["SCATS Number", "Datetime"])["Flow"].sum().reset_index()

            if len(df) < initial_length:
                logger.info(
                    "Spatial aggregation successfully merged %d duplicate lane/detector records.",
                    initial_length - len(df),
                )

            logger.info(
                "ETL complete. Matrix successfully converted to a continuous vertical time-series."
            )

        except Exception as e:
            logger.error("Critical failure during ETL matrix melting: %s", e)
            sys.exit(1)

    if df.empty:
        logger.error("The ETL pipeline stripped all data. Check the raw file format.")
        sys.exit(1)

    # 3. Time-Series Interpolation
    df = df.sort_values(by=["SCATS Number", "Datetime"])
    df = df.set_index("Datetime")

    cleaned_dataframes = []
    unique_sites = df["SCATS Number"].unique()
    logger.info(
        "Processing temporal matrices for %d unique SCATS sites.", len(unique_sites)
    )

    for site in unique_sites:
        site_data = df[df["SCATS Number"] == site].copy()

        # Failsafe: Skip if a site somehow has no data
        if site_data.empty:
            continue

        full_time_range = pd.date_range(
            start=site_data.index.min(), end=site_data.index.max(), freq="15min"
        )

        site_data = site_data.reindex(full_time_range)
        site_data["SCATS Number"] = site

        site_data["Flow"] = site_data["Flow"].interpolate(
            method="linear", limit=12, limit_direction="forward"
        )
        cleaned_dataframes.append(site_data)

    logger.info("Concatenating processed matrices and preparing file write.")
    final_cleaned_df = pd.concat(cleaned_dataframes)
    final_cleaned_df = final_cleaned_df.reset_index().rename(
        columns={"index": "Datetime"}
    )

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        final_cleaned_df.to_csv(output_csv_path, index=False)
        logger.info(
            "Data pipeline execution complete. Artifact generated at: %s",
            output_csv_path,
        )
    except Exception as e:
        logger.error("Failed to write output artifact: %s", e)
        sys.exit(1)


# ---------------------------------------------------------
# Execution Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name == "src" else script_dir

    default_input_path = project_root / "data" / "raw" / "Scats Data October 2006.xls"
    default_output_path = (
        project_root / "data" / "processed" / "Scats_Data_Cleaned_GapSafe.csv"
    )

    parser = argparse.ArgumentParser(
        description="SCATS Traffic Data ETL and Interpolation Pipeline"
    )
    parser.add_argument("--input", type=Path, default=default_input_path)
    parser.add_argument("--output", type=Path, default=default_output_path)
    args = parser.parse_args()

    clean_and_interpolate_traffic_data(args.input, args.output)
