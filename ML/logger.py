"""
logger.py — AIGOFIN ML Package
================================
Centralized logging factory for the entire AI trading system.

Every module should obtain its logger via:

    from ML.logger import get_logger
    log = get_logger(__name__)

Features
--------
- Console handler  : INFO level, colour-coded by level
- File handler     : DEBUG level, appended to ML/trading.log
- Automatic dedup  : calling get_logger() with the same name returns
                     the existing logger (standard Python behaviour)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# ── resolve log file path relative to this file ──────────────────────────────
_LOG_FILE_DEFAULT = Path(__file__).resolve().parent / "trading.log"


# ════════════════════════════════════════════════════════════════════════════
# ANSI colour formatter (console only)
# ════════════════════════════════════════════════════════════════════════════

_COLOURS = {
    logging.DEBUG:    "\033[36m",   # cyan
    logging.INFO:     "\033[32m",   # green
    logging.WARNING:  "\033[33m",   # yellow
    logging.ERROR:    "\033[31m",   # red
    logging.CRITICAL: "\033[35m",   # magenta
}
_RESET = "\033[0m"
_BOLD  = "\033[1m"


class _ColourFormatter(logging.Formatter):
    """Console formatter with ANSI colours and module name truncation."""

    FMT = "{colour}{bold}[{levelname:8s}]{reset}  {colour}{name:30s}{reset}  {message}"

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        colour = _COLOURS.get(record.levelno, "")
        record.colour = colour
        record.reset  = _RESET
        record.bold   = _BOLD
        formatter = logging.Formatter(self.FMT, style="{")
        return formatter.format(record)


# ════════════════════════════════════════════════════════════════════════════
# Internal setup — runs once per process
# ════════════════════════════════════════════════════════════════════════════

_ROOT_LOGGER_NAME = "aigofin"
_setup_done = False


def _setup_root_logger(log_file: Optional[str] = None) -> None:
    global _setup_done
    if _setup_done:
        return

    log_path = Path(log_file) if log_file else _LOG_FILE_DEFAULT

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(logging.DEBUG)
    root.propagate = False

    # ── Console handler ──────────────────────────────────────────────────────
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    # Only use colour if we are attached to a real terminal
    if sys.stdout.isatty():
        stdout_handler.setFormatter(_ColourFormatter())
    else:
        stdout_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root.addHandler(stdout_handler)

    # ── File handler ─────────────────────────────────────────────────────────
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(file_handler)
    except (OSError, PermissionError) as exc:
        root.warning(f"Could not open log file {log_path}: {exc}. File logging disabled.")

    _setup_done = True


# ════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Return a logger namespaced under 'aigofin.<name>'.

    Parameters
    ----------
    name     : typically ``__name__`` of the calling module.
    log_file : override the default log file path (optional).

    Returns
    -------
    logging.Logger  — ready to use with .debug / .info / .warning / .error
    """
    _setup_root_logger(log_file)
    # Strip the package prefix so logger names are short and readable
    short_name = name.split(".")[-1].replace("__main__", "root")
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{short_name}")


# ── Module-level logger (useful when logger.py itself needs to log) ──────────
log = get_logger(__name__)
log.debug("Logger system initialised.")
