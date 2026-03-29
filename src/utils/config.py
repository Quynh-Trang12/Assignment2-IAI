import json
import sys
from pathlib import Path
from src.utils.logger_setup import get_colored_logger

logger = get_colored_logger(__name__)


def load_config() -> dict:
    """
    Locates and loads the root config.json file as mandated by Assignment 2B.
    """
    # Resolve the path to the root directory
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    config_path = project_root / "config.json"

    if not config_path.exists():
        logger.critical(f"Missing configuration file at {config_path}")
        sys.exit(1)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.critical(f"Invalid JSON format in config.json: {e}")
        sys.exit(1)


# Expose a global configuration dictionary that any file can import
APP_CONFIG = load_config()
