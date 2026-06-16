import logging
import sys
from pathlib import Path


def setup_logging(name=None, log_file=None, level=logging.INFO):
    """Configure logging with console and optional file output.

    Args:
        name: logger name (uses root logger if None)
        log_file: path to log file (no file logging if None)
        level: logging level
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(path), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def get_logger(name):
    """Get a logger for the calling module.

    Usage in each file:
        from logging_config import get_logger
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)