"""
signals.py — Combine indicator DataFrames into a final BUY / SELL / HOLD
signal with a confidence score and a human-readable rationale list.
"""

from __future__ import annotations

import json
from typing import Any
import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Scoring weights  (must sum to 1.0)
# ──────────────────────────────────────────────────────────────────────────────
WEIGHTS = {
    "rsi":      0.25,
    "macd":     0.30,
    "ma":       0.25,
    "bb":       0.20,
}

# Score map: positive → bullish pressure, negative → bearish pressure
RSI_SCORES   = {"oversold": +1, "neutral": 0, "overbought": -1}
MACD_SCORES  = {"bullish": +1, "none": 0, "bearish": -1}
MA_TREND_SCORES   = {"golden_cross": +1, "neutral": 0, "death_cross": -1}
PRICE_VS_SMA_SCORES = {"above": +0.5, "below": -0.5}
BB_SCORES    = {"squeeze": 0, "neutral": 0, "breakout_up": +1, "breakout_down": -1}


class SignalGenerator:
    """
    Combine a set of indicator DataFrames for a single ticker and produce
    row-level signals.

    Parameters
    ----------
    indicators_df : pd.DataFrame
        Wide DataFrame where each column is an indicator output.
        Typically produced by ``IndicatorEngine.run()``.
    buy_threshold : float
        Weighted score above which a BUY is triggered  (default 0.30).
    sell_threshold : float
        Weighted score below which a SELL is triggered (default -0.30).
    """

    def __init__(
        self,
        indicators_df: pd.DataFrame,
        buy_threshold: float = 0.30,
        sell_threshold: float = -0.30,
    ) -> None:
        self.df = indicators_df.copy()
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def generate(self) -> list[dict[str, Any]]:
        """Return a list of signal dicts (one per bar), JSON-serialisable."""
        records: list[dict] = []
        for date, row in self.df.iterrows():
            record = self._score_row(date, row)
            records.append(record)
        return records

    def generate_json(self, indent: int = 2) -> str:
        """Return JSON string of all signals."""
        return json.dumps(self.generate(), indent=indent, default=str)

    def latest(self) -> dict[str, Any]:
        """Return the signal dict for the most recent bar only."""
        last_row = self.df.iloc[-1]
        return self._score_row(self.df.index[-1], last_row)

    def latest_json(self, indent: int = 2) -> str:
        """Return JSON string for the latest signal only."""
        return json.dumps(self.latest(), indent=indent, default=str)

    def to_dataframe(self) -> pd.DataFrame:
        """Return signals as a DataFrame (indexed by date)."""
        records = self.generate()
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _score_row(self, date: Any, row: pd.Series) -> dict[str, Any]:
        score = 0.0
        rationale: list[str] = []
        missing: list[str] = []

        def safe(col: str, default=None):
            v = row.get(col, default)
            return default if (isinstance(v, float) and np.isnan(v)) else v

        # ── RSI ──────────────────────────────────────────────────────────────
        rsi_val = safe("rsi")
        rsi_sig = safe("rsi_signal", "neutral")
        if rsi_val is not None:
            s = RSI_SCORES.get(rsi_sig, 0) * WEIGHTS["rsi"]
            score += s
            rationale.append(f"RSI={rsi_val:.1f} ({rsi_sig})")
        else:
            missing.append("rsi")

        # ── MACD ─────────────────────────────────────────────────────────────
        macd_cross = safe("macd_crossover", "none")
        macd_hist  = safe("macd_hist")
        if macd_hist is not None:
            # crossover event carries more weight than histogram direction alone
            s = MACD_SCORES.get(macd_cross, 0) * WEIGHTS["macd"]
            # histogram direction as a secondary nudge (½ weight)
            hist_direction = +0.5 if macd_hist > 0 else -0.5
            s += hist_direction * WEIGHTS["macd"] * 0.5
            score += s
            rationale.append(f"MACD hist={macd_hist:.4f}, crossover={macd_cross}")
        else:
            missing.append("macd")

        # ── Moving Average ────────────────────────────────────────────────────
        ma_trend   = safe("ma_trend", "neutral")
        price_pos  = safe("price_vs_sma", "below")
        if ma_trend is not None:
            s  = MA_TREND_SCORES.get(ma_trend, 0) * WEIGHTS["ma"]
            s += PRICE_VS_SMA_SCORES.get(price_pos, 0) * WEIGHTS["ma"]
            score += s
            rationale.append(f"MA trend={ma_trend}, price {price_pos} SMA")
        else:
            missing.append("ma")

        # ── Bollinger Bands ───────────────────────────────────────────────────
        bb_sig  = safe("bb_signal", "neutral")
        pct_b   = safe("bb_pct_b")
        if pct_b is not None:
            s = BB_SCORES.get(bb_sig, 0) * WEIGHTS["bb"]
            score += s
            rationale.append(f"BB signal={bb_sig}, %B={pct_b:.2f}")
        else:
            missing.append("bb")

        # ── Final decision ────────────────────────────────────────────────────
        if score >= self.buy_threshold:
            action = "BUY"
        elif score <= self.sell_threshold:
            action = "SELL"
        else:
            action = "HOLD"

        confidence = round(min(abs(score) / max(self.buy_threshold, abs(self.sell_threshold)), 1.0), 4)

        record: dict[str, Any] = {
            "date":        str(date)[:10],
            "action":      action,
            "score":       round(score, 4),
            "confidence":  confidence,
            "rationale":   rationale,
        }

        # Attach raw indicator values for traceability
        _attach_if(record, row, "close",             "close")
        _attach_if(record, row, "rsi",               "rsi")
        _attach_if(record, row, "macd",              "macd")
        _attach_if(record, row, "macd_hist",         "macd_hist")
        _attach_if(record, row, "bb_pct_b",          "bb_pct_b")
        _attach_if(record, row, "bb_width",          "bb_width")

        if missing:
            record["missing_indicators"] = missing

        return record


def _attach_if(record: dict, row: pd.Series, key: str, alias: str) -> None:
    """Attach a rounded float value to record if present and not NaN."""
    v = row.get(key)
    if v is not None and not (isinstance(v, float) and np.isnan(v)):
        record[alias] = round(float(v), 4)
