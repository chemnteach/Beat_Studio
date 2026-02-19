"""Structured logging for Beat Studio.

All loggers live under the "beat_studio" namespace so they can be controlled
with a single root level.
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

_ROOT = "beat_studio"
_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 100 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """Configure the beat_studio root logger.

    Args:
        level: Log level string ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        log_file: Optional path to a rotating file log.
        max_bytes: Max size before rotation (default 100 MB).
        backup_count: Number of backup files to keep.

    Raises:
        ValueError: If level is not a valid log level string.
    """
    upper = level.upper()
    if upper not in _VALID_LEVELS:
        raise ValueError(f"Invalid log level: {level!r}. Must be one of {_VALID_LEVELS}")

    numeric = getattr(logging, upper)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger(_ROOT)
    root.setLevel(numeric)

    # Remove existing handlers to avoid duplication on repeated calls
    root.handlers.clear()

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(numeric)
    console.setFormatter(fmt)
    root.addHandler(console)

    # Optional rotating file handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the beat_studio namespace.

    Args:
        name: Sub-namespace, e.g. "audio.analyzer" → "beat_studio.audio.analyzer".
              If the name already starts with "beat_studio.", it is used as-is.
    """
    if name.startswith(_ROOT):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT}.{name}")
