"""Logging setup shared by every CLI entry point.

A single configuration function so that all experiments produce identically formatted
logs, and so that each run's log is captured alongside its result files.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """Configure the root logger for a run.

    Args:
        level: Threshold for the console handler.
        log_file: Optional path; when given, a second handler writes the full
            DEBUG-level stream there so a finished run carries its own log.

    Returns:
        The configured root logger.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Re-running a CLI in the same interpreter (tests, notebooks) must not stack handlers.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root.addHandler(file_handler)

    # These libraries are chatty at INFO and drown out the experiment's own output.
    for noisy in ("urllib3", "filelock", "matplotlib", "datasets", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)
