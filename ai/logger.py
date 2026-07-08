"""
Centralized logging configuration for the IDBI AI module.

Every module should obtain its logger via:
    from ai.logger import get_logger
    logger = get_logger(__name__)

Logs are written to both stdout (INFO+) and a rotating file (DEBUG+).
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ai.config import LOG_DIR, LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Handlers are added only once per logger (idempotent).

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` instance with console + file handlers.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger is already configured
    if logger.handlers:
        return logger

    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(logging.DEBUG)  # capture all; handlers filter by level
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)

    # ── Console handler (INFO and above) ──────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── Rotating file handler (DEBUG and above, max 5 MB × 3 backups) ────
    log_file: Path = LOG_DIR / "ai_module.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
