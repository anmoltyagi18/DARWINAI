"""
engine.py — IndicatorEngine: the single entry-point for the whole pipeline.

Usage
-----
from indicator_engine import IndicatorEngine

engine = IndicatorEngine("AAPL")
signals_json = engine.run_json()          # all bars
latest_json  = engine.run_latest_json()  # only the most recent bar
df           = engine.run_dataframe()    # pandas DataFrame
"""

from __future__ import annotations

import json
from typing import Any, Optional

import pandas as pd

from .data import fetch_ohlcv
from .indicators import RSI, MACD, MovingAverage, BollingerBands
from .signals import SignalGenerator


class IndicatorEngine:
    """
    End-to-end pipeline: fetch OHLCV → compute indicators → generate signals.

    Parameters
    ----------
    ticker : str
        Yahoo Finance symbol, e.g. "AAPL", "TSLA", "BTC-USD", "^NSEI".
    period : str
        Data range (e.g. "6mo", "1y", "2y"). Ignored if start/end given.
    interval : str
        Bar size ("1d", "1wk", "1h", …).
    start / end : str, optional
        ISO date strings "YYYY-MM-DD" to override period.
    rsi_period : int
        RSI look-back window (default 14).
    macd_fast / macd_slow / macd_signal : int
        MACD parameters (default 12, 26, 9).
    ma_short / ma_long : int
        Moving-average windows (default 20, 50).
    bb_period / bb_std : int / float
        Bollinger Band parameters (default 20, 2.0).
    buy_threshold / sell_threshold : float
        Composite score thresholds for BUY / SELL (default ±0.30).
    """

    def __init__(
        self,
        ticker: str,
        period: str = "6mo",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        # Indicator params
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        ma_short: int = 20,
        ma_long: int = 50,
        bb_period: int = 20,
        bb_std: float = 2.0,
        # Signal thresholds
        buy_threshold: float = 0.30,
        sell_threshold: float = -0.30,
    ) -> None:
        self.ticker = ticker.upper()
        self.period = period
        self.interval = interval
        self.start = start
        self.end = end

        self._rsi_period = rsi_period
        self._macd_fast = macd_fast
        self._macd_slow = macd_slow
        self._macd_signal = macd_signal
        self._ma_short = ma_short
        self._ma_long = ma_long
        self._bb_period = bb_period
        self._bb_std = bb_std
        self._buy_threshold = buy_threshold
        self._sell_threshold = sell_threshold

        self._ohlcv: Optional[pd.DataFrame] = None
        self._combined: Optional[pd.DataFrame] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> list[dict[str, Any]]:
        """Run the full pipeline and return a list of signal dicts."""
        return self._build_signal_generator().generate()

    def run_json(self, indent: int = 2) -> str:
        """Run the full pipeline and return a JSON string (all bars)."""
        signals = self.run()
        meta = self._build_meta(len(signals))
        output = {"meta": meta, "signals": signals}
        return json.dumps(output, indent=indent, default=str)

    def run_latest(self) -> dict[str, Any]:
        """Return only the signal for the most recent bar."""
        return self._build_signal_generator().latest()

    def run_latest_json(self, indent: int = 2) -> str:
        """Return JSON string for the most recent bar only."""
        signal = self.run_latest()
        meta = self._build_meta(bars=1)
        output = {"meta": meta, "signal": signal}
        return json.dumps(output, indent=indent, default=str)

    def run_dataframe(self) -> pd.DataFrame:
        """Return signals as a DatetimeIndex DataFrame."""
        return self._build_signal_generator().to_dataframe()

    @property
    def ohlcv(self) -> pd.DataFrame:
        """Raw OHLCV DataFrame (fetched lazily and cached)."""
        if self._ohlcv is None:
            self._ohlcv = fetch_ohlcv(
                self.ticker, self.period, self.interval, self.start, self.end
            )
        return self._ohlcv

    @property
    def indicators(self) -> pd.DataFrame:
        """Wide DataFrame of OHLCV + all indicator columns (cached)."""
        if self._combined is None:
            self._combined = self._compute_indicators()
        return self._combined

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_indicators(self) -> pd.DataFrame:
        close = self.ohlcv["Close"]
        df = self.ohlcv.copy()

        frames = [
            RSI(close, period=self._rsi_period).compute(),
            MACD(close, fast=self._macd_fast, slow=self._macd_slow,
                 signal=self._macd_signal).compute(),
            MovingAverage(close, short=self._ma_short, long=self._ma_long).compute(),
            BollingerBands(close, period=self._bb_period, num_std=self._bb_std).compute(),
        ]

        for frame in frames:
            df = df.join(frame, how="left")

        # Rename Close for signal attachment
        df = df.rename(columns={"Close": "close"})
        return df

    def _build_signal_generator(self) -> SignalGenerator:
        return SignalGenerator(
            self.indicators,
            buy_threshold=self._buy_threshold,
            sell_threshold=self._sell_threshold,
        )

    def _build_meta(self, bars: int) -> dict[str, Any]:
        ohlcv = self.ohlcv
        return {
            "ticker":    self.ticker,
            "interval":  self.interval,
            "bars":      bars,
            "from":      str(ohlcv.index[0])[:10],
            "to":        str(ohlcv.index[-1])[:10],
            "parameters": {
                "rsi_period":   self._rsi_period,
                "macd":         f"{self._macd_fast}/{self._macd_slow}/{self._macd_signal}",
                "ma":           f"SMA{self._ma_short}/SMA{self._ma_long}",
                "bb":           f"BB({self._bb_period},{self._bb_std})",
                "buy_threshold":  self._buy_threshold,
                "sell_threshold": self._sell_threshold,
            },
        }
