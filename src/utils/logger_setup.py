import logging
import sys


class ColorFormatter(logging.Formatter):
    """
    Injects ANSI color codes into the logging pipeline based on the log level.
    """

    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"

    FORMAT_STR = "%(asctime)s - %(levelname)s - %(message)s"
    COLORS = {
        logging.DEBUG: BLUE,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        color_format = f"{color}{self.FORMAT_STR}{self.RESET}"
        formatter = logging.Formatter(color_format, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def get_colored_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance with the ColorFormatter applied.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if called multiple times
    if not logger.hasHandlers():
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColorFormatter())
        logger.addHandler(console_handler)

    # Prevent log messages from propagating up to the root logger (avoids double printing)
    logger.propagate = False

    return logger
