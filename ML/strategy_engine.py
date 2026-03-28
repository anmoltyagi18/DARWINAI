"""
strategy_engine.py — AIGOFIN ML Package
=========================================
Strategy Engine: generates trading signals by combining inputs from
technical indicators, market regime, and sentiment data.
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd


class StrategyEngine:
    """
    Generates trading signals by combining:
      - RSI momentum signal
      - MACD crossover signal
      - SMA trend signal
      - Regime / sentiment context (when provided)

    Returns: signal ∈ {-1 (sell), 0 (hold), 1 (buy)} and confidence score.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        # Weights for each sub-signal
        self._w_rsi       = self.config.get("w_rsi",     0.35)
        self._w_macd      = self.config.get("w_macd",    0.35)
        self._w_trend     = self.config.get("w_trend",   0.20)
        self._w_sentiment = self.config.get("w_sentiment", 0.10)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _sma(arr: np.ndarray, period: int) -> float:
        if len(arr) < period:
            return float(arr.mean()) if len(arr) > 0 else 0.0
        return float(arr[-period:].mean())

    @staticmethod
    def _ema(arr: np.ndarray, period: int) -> np.ndarray:
        out = np.full(len(arr), np.nan)
        alpha = 2.0 / (period + 1)
        start = min(period - 1, len(arr) - 1)
        out[start] = arr[:start + 1].mean()
        for i in range(start + 1, len(arr)):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
        return out

    @staticmethod
    def _rsi(arr: np.ndarray, period: int = 14) -> float:
        if len(arr) < period + 1:
            return 50.0
        delta = np.diff(arr[-(period + 1):])
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        avg_gain = gains.mean() + 1e-9
        avg_loss = losses.mean() + 1e-9
        rs = avg_gain / avg_loss
        return float(100 - 100 / (1 + rs))

    def _rsi_signal(self, rsi: float) -> float:
        """Returns a score in [-1, 1] based on RSI level."""
        if rsi < 30:
            return 0.8
        elif rsi < 40:
            return 0.4
        elif rsi > 70:
            return -0.8
        elif rsi > 60:
            return -0.4
        else:
            return 0.0

    def _macd_signal(self, closes: np.ndarray) -> float:
        """MACD histogram direction → score in [-1, 1]."""
        if len(closes) < 35:
            return 0.0
        ema12 = self._ema(closes, 12)
        ema26 = self._ema(closes, 26)
        macd_line = ema12 - ema26
        # remove NaNs before computing signal line
        valid = macd_line[~np.isnan(macd_line)]
        if len(valid) < 9:
            return 0.0
        signal_line = self._ema(valid, 9)
        histogram = valid - signal_line
        if len(histogram) < 2:
            return 0.0
        # Direction of histogram change
        if histogram[-1] > 0 and histogram[-1] > histogram[-2]:
            return 0.8
        elif histogram[-1] > 0:
            return 0.4
        elif histogram[-1] < 0 and histogram[-1] < histogram[-2]:
            return -0.8
        elif histogram[-1] < 0:
            return -0.4
        return 0.0

    def _trend_signal(self, closes: np.ndarray) -> float:
        """Price vs SMA20 and SMA50 → score in [-1, 1]."""
        price = float(closes[-1])
        sma20 = self._sma(closes, 20)
        sma50 = self._sma(closes, 50)
        score = 0.0
        if price > sma20:
            score += 0.5
        else:
            score -= 0.5
        if price > sma50:
            score += 0.5
        else:
            score -= 0.5
        return float(np.clip(score, -1, 1))

    @staticmethod
    def compute_stop_loss(closes: np.ndarray, atr_multiplier: float = 2.0, period: int = 14) -> float:
        """Volatility-based stop distance (% of price). Proxy for ATR when only closes are available."""
        if len(closes) < period + 1:
            return 0.05  # fallback 5%
        
        returns = np.abs(np.diff(closes[-(period+1):])) / closes[-(period+1):-1]
        avg_vol = float(np.mean(returns))
        # Cap between 1% and 15%
        return float(min(max(avg_vol * atr_multiplier, 0.01), 0.15))

    @staticmethod
    def kelly_position_size(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Fractional (Half) Kelly Criterion for position sizing."""
        if win_rate <= 0 or avg_win <= 0 or avg_loss <= 0:
            return 0.0
        r = avg_win / avg_loss
        kelly_pct = win_rate - ((1.0 - win_rate) / r)
        # Half-Kelly for safety, capped at 20%
        return float(max(min(kelly_pct * 0.5, 0.20), 0.0))

    # ── public API ────────────────────────────────────────────────────────────

    def generate_signal(self, features: dict) -> dict:
        """
        Generate a trading signal from a feature dictionary.

        The features dict may contain either:
          a) Pre-computed indicators: 'rsi', 'macd', 'sma_20' etc.
          b) Raw OHLCV fields: 'open', 'high', 'low', 'close', 'volume'
          c) A 'closes' key with a list/array of close prices

        Returns:
            dict with keys:
              - 'signal'     : int  (-1, 0, 1)
              - 'confidence' : float (0.0 – 1.0)
              - 'reason'     : str
        """
        # ── Extract close prices (for indicator computation) ─────────────────
        closes: Optional[np.ndarray] = None
        if "closes" in features:
            closes = np.asarray(features["closes"], dtype=float)
        elif "close" in features and isinstance(features["close"], (list, np.ndarray)):
            closes = np.asarray(features["close"], dtype=float)

        # ── Compute or use pre-computed indicators ────────────────────────────
        if "rsi" in features:
            rsi = float(features["rsi"])
        elif closes is not None:
            rsi = self._rsi(closes)
        else:
            rsi = 50.0

        if "macd_histogram" in features:
            macd_hist = float(features["macd_histogram"])
            macd_score = 0.8 if macd_hist > 0 else (-0.8 if macd_hist < 0 else 0.0)
        elif closes is not None:
            macd_score = self._macd_signal(closes)
        else:
            macd_score = 0.0

        if closes is not None:
            trend_score = self._trend_signal(closes)
        elif "sma_20" in features and "sma_50" in features:
            price = float(features.get("close", features.get("sma_20", 0)))
            trend_score = 0.0
            if price > float(features["sma_20"]):
                trend_score += 0.5
            else:
                trend_score -= 0.5
            if price > float(features["sma_50"]):
                trend_score += 0.5
            else:
                trend_score -= 0.5
        else:
            trend_score = 0.0

        sentiment_score = float(features.get("sentiment_score", 0.0))

        # ── Weighted composite ────────────────────────────────────────────────
        rsi_score = self._rsi_signal(rsi)
        composite = (
            self._w_rsi       * rsi_score
            + self._w_macd    * macd_score
            + self._w_trend   * trend_score
            + self._w_sentiment * sentiment_score
        )
        composite = float(np.clip(composite, -1, 1))
        confidence = round(abs(composite), 3)

        # ── Discretise ────────────────────────────────────────────────────────
        if composite >= 0.20:
            signal_int = 1
            signal_str = "BUY"
        elif composite <= -0.20:
            signal_int = -1
            signal_str = "SELL"
        else:
            signal_int = 0
            signal_str = "HOLD"

        reason = (
            f"RSI={rsi:.1f}({rsi_score:+.2f}), "
            f"MACD_score={macd_score:+.2f}, "
            f"Trend_score={trend_score:+.2f}, "
            f"Composite={composite:+.3f} → {signal_str}"
        )

        # ── Risk metrics ──────────────────────────────────────────────────────
        stop_loss_pct = 0.05
        if closes is not None and len(closes) > 0:
            stop_loss_pct = self.compute_stop_loss(closes)
        
        # Simple proxy for Kelly based on confidence
        w = 0.40 + (confidence * 0.20) # 40% to 60% win rate assumption
        pos_size = self.kelly_position_size(win_rate=w, avg_win=0.03, avg_loss=0.015)
        if pos_size == 0.0:
            pos_size = 0.02 # fallback 2%

        return {
            "signal":     signal_int,
            "signal_str": signal_str,
            "confidence": confidence,
            "reason":     reason,
            "stop_loss_pct": round(stop_loss_pct, 4),
            "suggested_position_size": round(pos_size, 4),
        }

    def get_signal(self, market_data) -> dict:
        """
        Convenience wrapper accepting a pandas DataFrame (OHLCV).
        Passes close prices to generate_signal.
        """
        if hasattr(market_data, "iloc"):
            df = market_data
            closes_col = [c for c in df.columns if c.lower() == "close"]
            if closes_col:
                closes = df[closes_col[0]].dropna().values
                return self.generate_signal({"closes": closes})
        return self.generate_signal({})

    def __repr__(self) -> str:
        return f"StrategyEngine(config={self.config})"
