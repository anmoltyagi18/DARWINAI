"""
indicators.py — Pure-pandas implementations of RSI, MACD, Moving Average,
and Bollinger Bands.  Each class accepts a Close price Series and exposes
a .compute() method that returns a named DataFrame ready for merging.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Literal


# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────

class BaseIndicator:
    """Shared interface for all indicators."""

    def __init__(self, close: pd.Series) -> None:
        if not isinstance(close, pd.Series):
            raise TypeError("close must be a pandas Series.")
        self.close = close.astype(float)

    def compute(self) -> pd.DataFrame:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# RSI
# ──────────────────────────────────────────────────────────────────────────────

class RSI(BaseIndicator):
    """
    Relative Strength Index (Wilder's smoothed method).

    Columns produced:
        rsi          — RSI value  [0, 100]
        rsi_signal   — "overbought" | "oversold" | "neutral"
    """

    def __init__(
        self,
        close: pd.Series,
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
    ) -> None:
        super().__init__(close)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def compute(self) -> pd.DataFrame:
        delta = self.close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Wilder's smoothing (equivalent to EMA with alpha = 1/period)
        avg_gain = gain.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        signal = pd.Series("neutral", index=rsi.index, name="rsi_signal")
        signal[rsi >= self.overbought] = "overbought"
        signal[rsi <= self.oversold] = "oversold"

        return pd.DataFrame({"rsi": rsi, "rsi_signal": signal})


# ──────────────────────────────────────────────────────────────────────────────
# MACD
# ──────────────────────────────────────────────────────────────────────────────

class MACD(BaseIndicator):
    """
    Moving Average Convergence Divergence.

    Columns produced:
        macd          — MACD line  (fast_ema − slow_ema)
        macd_signal   — Signal line (EMA of MACD)
        macd_hist     — Histogram  (macd − signal)
        macd_crossover — "bullish" | "bearish" | "none"
    """

    def __init__(
        self,
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> None:
        super().__init__(close)
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    def compute(self) -> pd.DataFrame:
        ema_fast = self.close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = self.close.ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        # Detect crossovers
        prev_diff = (macd_line - signal_line).shift(1)
        curr_diff = macd_line - signal_line
        crossover = pd.Series("none", index=macd_line.index, name="macd_crossover")
        crossover[(prev_diff < 0) & (curr_diff >= 0)] = "bullish"
        crossover[(prev_diff > 0) & (curr_diff <= 0)] = "bearish"

        return pd.DataFrame(
            {
                "macd": macd_line,
                "macd_signal_line": signal_line,
                "macd_hist": histogram,
                "macd_crossover": crossover,
            }
        )


# ──────────────────────────────────────────────────────────────────────────────
# Moving Average
# ──────────────────────────────────────────────────────────────────────────────

class MovingAverage(BaseIndicator):
    """
    Simple (SMA) and Exponential (EMA) Moving Averages with a
    short/long golden-cross / death-cross signal.

    Columns produced:
        sma_{short}    — Short SMA
        sma_{long}     — Long SMA
        ema_{short}    — Short EMA
        ema_{long}     — Long EMA
        ma_trend       — "golden_cross" | "death_cross" | "neutral"
        price_vs_sma   — "above" | "below"
    """

    def __init__(
        self,
        close: pd.Series,
        short: int = 20,
        long: int = 50,
        kind: Literal["sma", "ema", "both"] = "both",
    ) -> None:
        super().__init__(close)
        self.short = short
        self.long = long
        self.kind = kind

    def compute(self) -> pd.DataFrame:
        result: dict[str, pd.Series] = {}

        sma_s = self.close.rolling(self.short).mean()
        sma_l = self.close.rolling(self.long).mean()
        ema_s = self.close.ewm(span=self.short, adjust=False).mean()
        ema_l = self.close.ewm(span=self.long, adjust=False).mean()

        if self.kind in ("sma", "both"):
            result[f"sma_{self.short}"] = sma_s
            result[f"sma_{self.long}"] = sma_l

        if self.kind in ("ema", "both"):
            result[f"ema_{self.short}"] = ema_s
            result[f"ema_{self.long}"] = ema_l

        # Trend signal based on SMA crossover
        prev_diff = (sma_s - sma_l).shift(1)
        curr_diff = sma_s - sma_l
        trend = pd.Series("neutral", index=self.close.index, name="ma_trend")
        trend[(prev_diff < 0) & (curr_diff >= 0)] = "golden_cross"
        trend[(prev_diff > 0) & (curr_diff <= 0)] = "death_cross"
        result["ma_trend"] = trend

        # Price vs short SMA
        pv = pd.Series("below", index=self.close.index, name="price_vs_sma")
        pv[self.close >= sma_s] = "above"
        result["price_vs_sma"] = pv

        return pd.DataFrame(result)


# ──────────────────────────────────────────────────────────────────────────────
# Bollinger Bands
# ──────────────────────────────────────────────────────────────────────────────

class BollingerBands(BaseIndicator):
    """
    Bollinger Bands (SMA ± k × rolling std).

    Columns produced:
        bb_upper     — Upper band
        bb_middle    — Middle band (SMA)
        bb_lower     — Lower band
        bb_width     — Band width  ((upper − lower) / middle)
        bb_pct_b     — %B  ((close − lower) / (upper − lower))
        bb_signal    — "squeeze" | "breakout_up" | "breakout_down" | "neutral"
    """

    def __init__(
        self,
        close: pd.Series,
        period: int = 20,
        num_std: float = 2.0,
    ) -> None:
        super().__init__(close)
        self.period = period
        self.num_std = num_std

    def compute(self) -> pd.DataFrame:
        sma = self.close.rolling(self.period).mean()
        std = self.close.rolling(self.period).std()

        upper = sma + self.num_std * std
        lower = sma - self.num_std * std
        width = (upper - lower) / sma
        pct_b = (self.close - lower) / (upper - lower)

        # Signal logic
        squeeze_threshold = width.rolling(50, min_periods=20).quantile(0.20)
        signal = pd.Series("neutral", index=self.close.index, name="bb_signal")
        signal[width <= squeeze_threshold] = "squeeze"
        signal[self.close > upper] = "breakout_up"
        signal[self.close < lower] = "breakout_down"

        return pd.DataFrame(
            {
                "bb_upper": upper,
                "bb_middle": sma,
                "bb_lower": lower,
                "bb_width": width,
                "bb_pct_b": pct_b,
                "bb_signal": signal,
            }
        )
