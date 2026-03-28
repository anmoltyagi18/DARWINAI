"""
config.py — AIGOFIN ML Package
================================
Central configuration loader for the entire AI trading system.

Loads environment variables from the project-root .env file using
python-dotenv, and exposes them as a typed singleton so every ML
module imports from one consistent location.

Usage
-----
    from ML.config import config

    key = config.FINNHUB_API_KEY
    port = config.PORT
"""

from __future__ import annotations

import os
from pathlib import Path

# ── python-dotenv (graceful fallback if not installed) ───────────────────────
try:
    from dotenv import load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False

# ── locate the project root (.env lives there, not inside ML/) ───────────────
_ROOT = Path(__file__).resolve().parent.parent   # AIGOFIN/
_ENV_FILE = _ROOT / ".env"

if _DOTENV_AVAILABLE:
    load_dotenv(dotenv_path=_ENV_FILE, override=False)
else:
    # Manually parse `.env` if python-dotenv is missing
    if _ENV_FILE.exists():
        with open(_ENV_FILE) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip())


# ════════════════════════════════════════════════════════════════════════════
# Config dataclass
# ════════════════════════════════════════════════════════════════════════════

class _Config:
    """
    Typed configuration object.
    All values are read once at import time from environment variables.
    """

    # ── API keys ─────────────────────────────────────────────────────────────
    FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "")

    # ── Server settings ───────────────────────────────────────────────────────
    PORT: int = int(os.getenv("PORT", "0"))          # 0 = auto-detect free port
    HOST: str = os.getenv("HOST", "127.0.0.1")
    ENV:  str = os.getenv("ENV",  "development")

    # ── Data settings ─────────────────────────────────────────────────────────
    DEFAULT_PERIOD:   str = os.getenv("DEFAULT_PERIOD",   "1y")
    DEFAULT_INTERVAL: str = os.getenv("DEFAULT_INTERVAL", "1d")

    # ── RL model settings ─────────────────────────────────────────────────────
    RL_ALGORITHM:        str = os.getenv("RL_ALGORITHM",       "PPO")
    RL_TOTAL_TIMESTEPS:  int = int(os.getenv("RL_TOTAL_TIMESTEPS", "50000"))
    RL_INITIAL_CAPITAL:  float = float(os.getenv("RL_INITIAL_CAPITAL", "100000"))

    # ── Strategy evolution settings ───────────────────────────────────────────
    EVOLVER_GENERATIONS:     int = int(os.getenv("EVOLVER_GENERATIONS",    "10"))
    EVOLVER_POPULATION_SIZE: int = int(os.getenv("EVOLVER_POPULATION_SIZE","30"))

    # ── Paths ─────────────────────────────────────────────────────────────────
    ROOT_DIR:       Path = _ROOT
    ML_DIR:         Path = _ROOT / "ML"
    DATA_DIR:       Path = _ROOT / "data"
    STRATEGIES_DB:  str  = str(_ROOT / "ML" / "strategies_db.json")
    LOG_FILE:       str  = str(_ROOT / "ML" / "trading.log")

    # ── Go gateway ────────────────────────────────────────────────────────────
    GO_GATEWAY_URL: str = os.getenv("GO_GATEWAY_URL", "http://localhost:8080")

    def is_production(self) -> bool:
        return self.ENV.lower() == "production"

    def has_finnhub_key(self) -> bool:
        return bool(self.FINNHUB_API_KEY)

    def __repr__(self) -> str:
        key_preview = (self.FINNHUB_API_KEY[:6] + "…") if self.FINNHUB_API_KEY else "NOT SET"
        return (
            f"Config(env={self.ENV}, host={self.HOST}, port={self.PORT},"
            f" finnhub_key={key_preview})"
        )


# ── Public singleton ──────────────────────────────────────────────────────────
config = _Config()
