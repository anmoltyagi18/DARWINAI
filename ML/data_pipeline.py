"""
data_pipeline.py — AIGOFIN ML Package
=======================================
Central controller that fetches, cleans, enriches, and routes stock data
through the AIGOFIN ML stack.

Pipeline steps
--------------
1. Fetch    — pull raw OHLCV bars via stock_fetcher.fetch_ohlcv()
2. Clean    — normalise column names, drop NaN/zero-volume rows, sort index
3. Enrich   — compute all technical indicators via IndicatorsEngine
4. Signal   — pass the latest feature row to StrategyEngine.generate_signal()

Public API
----------
    from ML.data_pipeline import run_data_pipeline

    result = run_data_pipeline("AAPL")
    print(result["signal"])        # {"signal": 1, "confidence": 0.0, ...}
    print(result["enriched_df"])   # full DataFrame with indicator columns

Dependencies
------------
    pip install yfinance pandas numpy
"""

from __future__ import annotations

import sys
import os
from typing import Optional

import pandas as pd

# ── Project-root on sys.path so `stock_fetcher` is importable ────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Modular imports (no code duplication) ────────────────────────────────────
from stock_fetcher import fetch_ohlcv           # root-level utility
from .indicators_engine import IndicatorsEngine
from .strategy_engine import StrategyEngine


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Fetch
# ─────────────────────────────────────────────────────────────────────────────

def _fetch(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """
    Pull raw OHLCV data.  Delegates entirely to stock_fetcher — no duplication.
    """
    df = fetch_ohlcv(symbol=symbol, period=period, interval=interval)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Clean
# ─────────────────────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pandas pipeline that sanitises the raw OHLCV DataFrame:

    - Lower-case all column names (IndicatorsEngine expects lowercase)
    - Ensure DatetimeIndex, sorted ascending
    - Drop rows with any NaN in OHLCV columns
    - Drop rows where volume == 0 (non-trading days / data gaps)
    - Reset index so Date becomes a plain column (avoids index ambiguity)
    """
    return (
        df
        .rename(columns=str.lower)                          # Open→open, etc.
        .pipe(lambda d: d.sort_index())                     # chronological order
        .pipe(lambda d: d.dropna(subset=["open", "high", "low", "close", "volume"]))
        .pipe(lambda d: d[d["volume"] > 0])                 # remove zero-vol bars
        .pipe(lambda d: d.copy())                           # avoid SettingWithCopy
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Enrich (compute indicators)
# ─────────────────────────────────────────────────────────────────────────────

def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all technical indicators via IndicatorsEngine.calculate_all().
    Returns the enriched DataFrame (EMA_20, EMA_50, RSI, MACD, BB, VWAP, …).
    """
    engine = IndicatorsEngine(df)
    enriched = engine.calculate_all()
    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Signal generation
# ─────────────────────────────────────────────────────────────────────────────

def _extract_features(df: pd.DataFrame) -> dict:
    """
    Pull the latest bar's indicator values into a flat feature dict that
    StrategyEngine.generate_signal() understands.
    """
    last = df.iloc[-1]

    def _get(col: str, default: float = 0.0) -> float:
        val = last.get(col, default)
        return float(val) if pd.notna(val) else default

    return {
        # Price
        "close":          _get("close"),
        "open":           _get("open"),
        "high":           _get("high"),
        "low":            _get("low"),
        "volume":         _get("volume"),
        # Trend
        "ema_20":         _get("EMA_20"),
        "ema_50":         _get("EMA_50"),
        # Momentum
        "rsi":            _get("RSI"),
        "macd":           _get("MACD"),
        "macd_signal":    _get("MACD_signal"),
        "macd_hist":      _get("MACD_hist"),
        # Volatility
        "bb_upper":       _get("BB_upper"),
        "bb_middle":      _get("BB_middle"),
        "bb_lower":       _get("BB_lower"),
        # Volume
        "vwap":           _get("VWAP"),
        "volume_ma_20":   _get("Volume_MA_20"),
    }


def _generate_signal(features: dict, config: Optional[dict] = None) -> dict:
    """
    Pass extracted features to StrategyEngine and return the signal dict.
    """
    engine = StrategyEngine(config=config)
    return engine.generate_signal(features)


# ─────────────────────────────────────────────────────────────────────────────
# Public controller
# ─────────────────────────────────────────────────────────────────────────────

def run_data_pipeline(
    symbol:          str,
    period:          str            = "3mo",
    interval:        str            = "1d",
    strategy_config: Optional[dict] = None,
    verbose:         bool           = False,
) -> dict:
    """
    End-to-end data pipeline: fetch → clean → indicators → signal.

    Parameters
    ----------
    symbol          : Ticker symbol, e.g. "AAPL", "TSLA", "^NSEI"
    period          : yfinance period string (default "3mo").
                      Valid: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max
    interval        : yfinance interval string (default "1d").
                      Valid: 1m, 5m, 15m, 1h, 1d, 1wk, 1mo
    strategy_config : Optional config dict forwarded to StrategyEngine.
    verbose         : If True, prints pipeline progress logs.

    Returns
    -------
    dict with keys:
        symbol       : str   — the ticker that was processed
        raw_df       : pd.DataFrame — raw OHLCV from yfinance
        enriched_df  : pd.DataFrame — OHLCV + all indicator columns
        features     : dict  — latest-bar feature values passed to strategy
        signal       : dict  — StrategyEngine output
                               {signal, confidence, reason}
        bars         : int   — number of bars after cleaning
        period       : str
        interval     : str

    Raises
    ------
    ValueError : if no data is returned for the symbol.
    """

    def _log(msg: str) -> None:
        if verbose:
            print(f"[pipeline:{symbol}] {msg}")

    # ── Step 1: Fetch ─────────────────────────────────────────────────────
    _log(f"Fetching {period}/{interval} data …")
    raw_df = _fetch(symbol, period, interval)
    _log(f"  → {len(raw_df)} raw bars")

    # ── Step 2: Clean ─────────────────────────────────────────────────────
    _log("Cleaning dataframe …")
    clean_df = _clean(raw_df)
    _log(f"  → {len(clean_df)} bars after cleaning")

    if clean_df.empty:
        raise ValueError(
            f"No usable bars remain for '{symbol}' after cleaning. "
            "Try a longer period or check the symbol."
        )

    # ── Step 3: Enrich ────────────────────────────────────────────────────
    _log("Computing indicators …")
    enriched_df = _enrich(clean_df)
    _log(f"  → columns: {list(enriched_df.columns)}")

    # ── Step 4: Signal ────────────────────────────────────────────────────
    _log("Extracting features and generating signal …")
    features = _extract_features(enriched_df)
    signal   = _generate_signal(features, config=strategy_config)
    _log(f"  → signal={signal['signal']}  confidence={signal['confidence']}")

    return {
        "symbol":      symbol,
        "raw_df":      raw_df,
        "enriched_df": enriched_df,
        "features":    features,
        "signal":      signal,
        "bars":        len(enriched_df),
        "period":      period,
        "interval":    interval,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI / demo entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AIGOFIN Data Pipeline — fetch → clean → indicators → signal"
    )
    parser.add_argument("symbol", type=str, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--period",   default="3mo", help="yfinance period (default: 3mo)")
    parser.add_argument("--interval", default="1d",  help="yfinance interval (default: 1d)")
    args = parser.parse_args()

    print(f"\nRunning AIGOFIN data pipeline for: {args.symbol.upper()}\n")

    result = run_data_pipeline(
        symbol   = args.symbol,
        period   = args.period,
        interval = args.interval,
        verbose  = True,
    )

    print(f"\n{'='*55}")
    print(f"  Pipeline complete — {result['symbol'].upper()}")
    print(f"{'='*55}")
    print(f"  Bars processed : {result['bars']}")
    print(f"  Period / Int   : {result['period']} / {result['interval']}")
    print(f"\n  Latest features:")
    for k, v in result["features"].items():
        print(f"    {k:<16} : {v:.4f}")
    print(f"\n  Strategy signal:")
    sig = result["signal"]
    print(f"    signal     = {sig['signal']}")
    print(f"    confidence = {sig['confidence']}")
    print(f"    reason     = {sig['reason']}")
    print(f"\n  Enriched DataFrame tail (5 rows):")
    print(result["enriched_df"].tail(5).to_string())
    print(f"{'='*55}\n")
