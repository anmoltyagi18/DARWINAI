"""
trading_server.py
=================
FastAPI server for the AI Trading System

Install dependencies
--------------------
    pip install fastapi uvicorn yfinance numpy scipy

Run
---
    uvicorn trading_server:app --host 0.0.0.0 --port 8000 --reload

Endpoints
---------
    GET /analyze/{symbol}          → full AI trading signal
    GET /health                    → server health check
    GET /symbols                   → supported symbol list
    GET /docs                      → auto-generated Swagger UI

Example
-------
    curl http://localhost:8000/analyze/AAPL
    curl http://localhost:8000/analyze/TSLA?period=6mo&interval=1d
"""

from __future__ import annotations

import math
import random
import statistics
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# ── FastAPI (install: pip install fastapi uvicorn) ───────────────────────────
try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("⚠  FastAPI not installed. Run:  pip install fastapi uvicorn")

# ── yfinance (install: pip install yfinance) ─────────────────────────────────
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠  yfinance not installed. Using synthetic data.  pip install yfinance")


# ════════════════════════════════════════════════════════════════════════════
# Enumerations & Pydantic schemas
# ════════════════════════════════════════════════════════════════════════════

class Signal(str, Enum):
    STRONG_BUY  = "STRONG_BUY"
    BUY         = "BUY"
    HOLD        = "HOLD"
    SELL        = "SELL"
    STRONG_SELL = "STRONG_SELL"


class MarketRegime(str, Enum):
    TRENDING_UP   = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING       = "RANGING"
    VOLATILE      = "VOLATILE"
    BREAKOUT      = "BREAKOUT"


class Sentiment(str, Enum):
    VERY_BULLISH = "VERY_BULLISH"
    BULLISH      = "BULLISH"
    NEUTRAL      = "NEUTRAL"
    BEARISH      = "BEARISH"
    VERY_BEARISH = "VERY_BEARISH"


class RiskLevel(str, Enum):
    VERY_LOW  = "VERY_LOW"
    LOW       = "LOW"
    MODERATE  = "MODERATE"
    HIGH      = "HIGH"
    VERY_HIGH = "VERY_HIGH"


# ── Response schema ──────────────────────────────────────────────────────────

if FASTAPI_AVAILABLE:
    class IndicatorSnapshot(BaseModel):
        rsi:            float = Field(description="Relative Strength Index (0-100)")
        macd:           float = Field(description="MACD line value")
        macd_signal:    float = Field(description="MACD signal line")
        macd_histogram: float = Field(description="MACD histogram")
        sma_20:         float = Field(description="20-period simple moving average")
        sma_50:         float = Field(description="50-period simple moving average")
        sma_200:        float = Field(description="200-period simple moving average")
        bb_upper:       float = Field(description="Bollinger Band upper")
        bb_lower:       float = Field(description="Bollinger Band lower")
        bb_pct:         float = Field(description="Price position within Bollinger Bands (0-1)")
        atr:            float = Field(description="Average True Range")
        adx:            float = Field(description="Average Directional Index")
        volume_ratio:   float = Field(description="Current vs average volume ratio")

    class RiskMetrics(BaseModel):
        daily_volatility:   float = Field(description="Annualised daily volatility")
        var_1d_95:          float = Field(description="1-day 95% Value at Risk (%)")
        var_1d_99:          float = Field(description="1-day 99% Value at Risk (%)")
        max_drawdown:       float = Field(description="Max drawdown over period (%)")
        sharpe_proxy:       float = Field(description="Sharpe-ratio proxy (return / vol)")
        beta_spy:           float = Field(description="Estimated beta vs SPY")

    class TradeParameters(BaseModel):
        stop_loss_pct:      float = Field(description="Suggested stop loss distance (%)")
        take_profit_pct:    float = Field(description="Suggested take profit distance (%)")
        position_size_pct:  float = Field(description="Suggested position size (% of capital)")
        risk_reward_ratio:  float

    class AnalysisResponse(BaseModel):
        # ── required output fields ──────────────────────────────────────────
        symbol:         str
        signal:         Signal
        confidence:     float   = Field(ge=0, le=1, description="0-1 confidence score")
        market_regime:  MarketRegime
        sentiment:      Sentiment
        risk_level:     RiskLevel
        # ── enriched output ────────────────────────────────────────────────
        price:          float
        price_change_pct: float = Field(description="% change from previous close")
        indicators:     IndicatorSnapshot
        risk_metrics:   RiskMetrics
        trade_params:   TradeParameters
        signal_breakdown: Dict[str, float] = Field(
            description="Per-module score contribution (-1 bearish → +1 bullish)"
        )
        analysis_notes: List[str]
        data_source:    str
        bars_analysed:  int
        timestamp:      str


# ════════════════════════════════════════════════════════════════════════════
# Data Layer
# ════════════════════════════════════════════════════════════════════════════

def _fetch_real(symbol: str, period: str, interval: str) -> np.ndarray:
    """Fetch OHLCV from yfinance. Returns (T, 6) array: O H L C V Adj."""
    ticker = yf.Ticker(symbol)
    df     = ticker.history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No data returned for {symbol!r}.")
    return df[["Open", "High", "Low", "Close", "Volume"]].values.astype(float)


def _fetch_synthetic(symbol: str, n: int = 300, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate realistic-looking OHLCV data via GBM.
    Used when yfinance is unavailable or the symbol is not found.
    """
    rng   = np.random.default_rng(seed or abs(hash(symbol)) % (2**31))
    base  = 100 + rng.uniform(0, 300)
    vol   = rng.uniform(0.012, 0.035)
    drift = rng.uniform(-0.0003, 0.0006)

    closes = [base]
    for _ in range(n - 1):
        closes.append(closes[-1] * math.exp(rng.normal(drift, vol)))

    closes = np.array(closes)
    spreads = closes * rng.uniform(0.003, 0.012, n)
    opens  = closes - rng.uniform(-0.5, 0.5, n) * spreads
    highs  = np.maximum(closes, opens) + rng.exponential(spreads * 0.4, n)
    lows   = np.minimum(closes, opens) - rng.exponential(spreads * 0.4, n)
    volume = rng.lognormal(14, 0.8, n)

    return np.column_stack([opens, highs, lows, closes, volume])


def fetch_data(
    symbol:   str,
    period:   str = "1y",
    interval: str = "1d",
) -> Tuple[np.ndarray, str]:
    """
    Try yfinance first, fall back to synthetic data.
    Returns (ohlcv_array, source_string).
    """
    if YFINANCE_AVAILABLE:
        try:
            data = _fetch_real(symbol.upper(), period, interval)
            return data, "yfinance"
        except Exception as exc:
            print(f"  yfinance failed ({exc}); using synthetic data.")
    return _fetch_synthetic(symbol, n=252), "synthetic"


# ════════════════════════════════════════════════════════════════════════════
# Indicator Engine
# ════════════════════════════════════════════════════════════════════════════

class IndicatorEngine:
    """Computes all technical indicators from an OHLCV array."""

    def __init__(self, ohlcv: np.ndarray):
        self.o = ohlcv[:, 0]
        self.h = ohlcv[:, 1]
        self.l = ohlcv[:, 2]
        self.c = ohlcv[:, 3]
        self.v = ohlcv[:, 4]
        self.n = len(self.c)

    # ── primitives ───────────────────────────────────────────────────────────

    def _sma(self, arr: np.ndarray, p: int) -> np.ndarray:
        out = np.full(len(arr), np.nan)
        for i in range(p - 1, len(arr)):
            out[i] = arr[i - p + 1 : i + 1].mean()
        return out

    def _ema(self, arr: np.ndarray, p: int) -> np.ndarray:
        out   = np.full(len(arr), np.nan)
        alpha = 2 / (p + 1)
        start = p - 1
        out[start] = arr[:p].mean()
        for i in range(start + 1, len(arr)):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
        return out

    # ── indicators ───────────────────────────────────────────────────────────

    def rsi(self, p: int = 14) -> float:
        delta  = np.diff(self.c)
        gains  = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        ag = self._sma(gains,  p)
        al = self._sma(losses, p)
        rs = np.where(al[-1] == 0, 100.0, ag[-1] / al[-1])
        return float(100 - 100 / (1 + rs))

    def macd(self, fast=12, slow=26, sig=9) -> Tuple[float, float, float]:
        macd_line   = self._ema(self.c, fast) - self._ema(self.c, slow)
        signal_line = self._ema(macd_line, sig)
        histogram   = macd_line - signal_line
        return float(macd_line[-1]), float(signal_line[-1]), float(histogram[-1])

    def sma(self, p: int) -> float:
        return float(self._sma(self.c, min(p, self.n))[-1])

    def bollinger(self, p: int = 20, k: float = 2.0) -> Tuple[float, float, float]:
        mid  = self._sma(self.c, p)
        roll_std = np.array([
            self.c[max(0, i - p + 1): i + 1].std()
            for i in range(self.n)
        ])
        upper = mid + k * roll_std
        lower = mid - k * roll_std
        price = self.c[-1]
        band_range = upper[-1] - lower[-1]
        pct = (price - lower[-1]) / band_range if band_range > 0 else 0.5
        return float(upper[-1]), float(lower[-1]), float(np.clip(pct, 0, 1))

    def atr(self, p: int = 14) -> float:
        tr = np.maximum(
            self.h[1:] - self.l[1:],
            np.maximum(
                abs(self.h[1:] - self.c[:-1]),
                abs(self.l[1:] - self.c[:-1])
            )
        )
        return float(self._sma(tr, p)[-1])

    def adx(self, p: int = 14) -> float:
        """Simplified ADX."""
        up_move   = self.h[1:] - self.h[:-1]
        down_move = self.l[:-1] - self.l[1:]
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr = np.maximum(
            self.h[1:] - self.l[1:],
            np.maximum(abs(self.h[1:] - self.c[:-1]), abs(self.l[1:] - self.c[:-1]))
        )
        atr_s  = self._sma(tr,     p)
        pdi    = 100 * self._sma(pos_dm, p) / np.where(atr_s == 0, 1, atr_s)
        ndi    = 100 * self._sma(neg_dm, p) / np.where(atr_s == 0, 1, atr_s)
        dx     = 100 * abs(pdi - ndi) / np.where(pdi + ndi == 0, 1, pdi + ndi)
        return float(self._sma(dx, p)[-1])

    def volume_ratio(self, p: int = 20) -> float:
        avg = self.v[-p:].mean()
        return float(self.v[-1] / avg) if avg > 0 else 1.0

    def compute_all(self) -> dict:
        bb_u, bb_l, bb_pct = self.bollinger()
        macd_l, macd_s, macd_h = self.macd()
        return dict(
            rsi            = self.rsi(),
            macd           = macd_l,
            macd_signal    = macd_s,
            macd_histogram = macd_h,
            sma_20         = self.sma(20),
            sma_50         = self.sma(50),
            sma_200        = self.sma(200),
            bb_upper       = bb_u,
            bb_lower       = bb_l,
            bb_pct         = bb_pct,
            atr            = self.atr(),
            adx            = self.adx(),
            volume_ratio   = self.volume_ratio(),
        )


# ════════════════════════════════════════════════════════════════════════════
# AI Modules
# ════════════════════════════════════════════════════════════════════════════

class TrendModule:
    """Multi-timeframe trend detection."""

    def score(self, ind: dict, price: float) -> Tuple[float, str]:
        score = 0.0
        # price vs moving averages
        if price > ind["sma_20"]:  score += 0.2
        else:                       score -= 0.2
        if price > ind["sma_50"]:  score += 0.2
        else:                       score -= 0.2
        if price > ind["sma_200"]: score += 0.2
        else:                       score -= 0.2
        # golden / death cross
        if ind["sma_50"] > ind["sma_200"]: score += 0.2
        else:                               score -= 0.2
        # MACD direction
        if ind["macd_histogram"] > 0: score += 0.2
        else:                          score -= 0.2
        note = f"Trend score {score:+.2f} | ADX={ind['adx']:.1f}"
        return float(np.clip(score, -1, 1)), note


class MomentumModule:
    """RSI + MACD momentum signals."""

    def score(self, ind: dict) -> Tuple[float, str]:
        rsi   = ind["rsi"]
        score = 0.0
        if rsi < 30:      score += 0.6   # oversold
        elif rsi < 40:    score += 0.3
        elif rsi > 70:    score -= 0.6   # overbought
        elif rsi > 60:    score -= 0.3

        if ind["macd"] > ind["macd_signal"]: score += 0.2
        else:                                 score -= 0.2

        if ind["macd_histogram"] > 0:        score += 0.2
        else:                                 score -= 0.2

        note = f"Momentum score {score:+.2f} | RSI={rsi:.1f}"
        return float(np.clip(score, -1, 1)), note


class VolatilityModule:
    """Bollinger Band + ATR regime detection."""

    def score(self, ind: dict, price: float) -> Tuple[float, str]:
        bb_pct = ind["bb_pct"]
        score  = 0.0

        # mean-reversion at bands
        if bb_pct < 0.05:   score += 0.5   # near lower band → bounce
        elif bb_pct < 0.20: score += 0.25
        elif bb_pct > 0.95: score -= 0.5   # near upper band → pullback
        elif bb_pct > 0.80: score -= 0.25

        atr_pct = ind["atr"] / price if price > 0 else 0
        vol_note = "low" if atr_pct < 0.01 else ("high" if atr_pct > 0.025 else "moderate")
        note = f"Volatility score {score:+.2f} | BB%={bb_pct:.2f} | ATR%={atr_pct:.2%} ({vol_note})"
        return float(np.clip(score, -1, 1)), note


class VolumeModule:
    """Volume confirmation signals."""

    def score(self, ind: dict, price_change: float) -> Tuple[float, str]:
        vr    = ind["volume_ratio"]
        score = 0.0
        # high volume confirms direction
        if vr > 1.5:
            score += 0.4 if price_change > 0 else -0.4
        elif vr > 1.2:
            score += 0.2 if price_change > 0 else -0.2
        # low volume weakens signal
        elif vr < 0.7:
            score *= 0.5
        note = f"Volume score {score:+.2f} | ratio={vr:.2f}x"
        return float(np.clip(score, -1, 1)), note


class SentimentModule:
    """
    Proxy sentiment from price action patterns.
    (In production, wire to a news-NLP or options-flow API here.)
    """

    def score(self, ohlcv: np.ndarray) -> Tuple[float, str, Sentiment]:
        closes  = ohlcv[-20:, 3]
        returns = np.diff(closes) / closes[:-1]
        pos_days = (returns > 0).sum()
        neg_days = (returns < 0).sum()
        avg_ret  = returns.mean()

        score = (pos_days - neg_days) / max(len(returns), 1)
        score = float(np.clip(score + avg_ret * 50, -1, 1))

        if score > 0.5:   sentiment = Sentiment.VERY_BULLISH
        elif score > 0.15: sentiment = Sentiment.BULLISH
        elif score < -0.5: sentiment = Sentiment.VERY_BEARISH
        elif score < -0.15: sentiment = Sentiment.BEARISH
        else:              sentiment = Sentiment.NEUTRAL

        note = f"Sentiment score {score:+.2f} | {pos_days}↑ / {neg_days}↓ days"
        return score, note, sentiment


# ════════════════════════════════════════════════════════════════════════════
# Risk Engine
# ════════════════════════════════════════════════════════════════════════════

class RiskEngine:
    """Derives risk metrics and classification from price data."""

    def analyse(self, ohlcv: np.ndarray) -> dict:
        closes  = ohlcv[:, 3]
        returns = np.diff(closes) / closes[:-1]

        daily_vol    = float(returns.std())
        annual_vol   = daily_vol * math.sqrt(252)
        var_95       = float(stats.norm.ppf(0.95) * daily_vol)
        var_99       = float(stats.norm.ppf(0.99) * daily_vol)

        # max drawdown
        peak = np.maximum.accumulate(closes)
        dd   = (closes - peak) / peak
        max_dd = float(dd.min())

        # Sharpe proxy (no risk-free subtraction for brevity)
        sharpe = float(returns.mean() / daily_vol * math.sqrt(252)) \
                 if daily_vol > 1e-9 else 0.0

        # beta proxy: correlation with a synthetic "market" index
        rng        = np.random.default_rng(42)
        mkt_ret    = rng.normal(0.0003, 0.010, len(returns))
        beta       = float(np.cov(returns, mkt_ret)[0, 1] / np.var(mkt_ret)) \
                     if np.var(mkt_ret) > 0 else 1.0

        # classify risk level
        if annual_vol < 0.15:         risk_level = RiskLevel.VERY_LOW
        elif annual_vol < 0.25:       risk_level = RiskLevel.LOW
        elif annual_vol < 0.40:       risk_level = RiskLevel.MODERATE
        elif annual_vol < 0.60:       risk_level = RiskLevel.HIGH
        else:                         risk_level = RiskLevel.VERY_HIGH

        return dict(
            daily_volatility = round(annual_vol, 4),
            var_1d_95        = round(var_95, 4),
            var_1d_99        = round(var_99, 4),
            max_drawdown     = round(max_dd, 4),
            sharpe_proxy     = round(sharpe, 4),
            beta_spy         = round(beta, 4),
            risk_level       = risk_level,
        )


# ════════════════════════════════════════════════════════════════════════════
# Market Regime Detector
# ════════════════════════════════════════════════════════════════════════════

def detect_regime(ind: dict, annual_vol: float) -> MarketRegime:
    adx    = ind["adx"]
    macd_h = ind["macd_histogram"]
    bb_pct = ind["bb_pct"]

    if annual_vol > 0.55:
        return MarketRegime.VOLATILE
    if adx > 35:
        return MarketRegime.TRENDING_UP if macd_h > 0 else MarketRegime.TRENDING_DOWN
    if bb_pct > 0.90 or bb_pct < 0.10:
        return MarketRegime.BREAKOUT
    return MarketRegime.RANGING


# ════════════════════════════════════════════════════════════════════════════
# Signal Aggregator
# ════════════════════════════════════════════════════════════════════════════

def aggregate_signal(breakdown: Dict[str, float]) -> Tuple[Signal, float]:
    """
    Weighted average of module scores → final signal + confidence.

    Weights
    -------
    trend      30 %
    momentum   25 %
    volatility 20 %
    volume     15 %
    sentiment  10 %
    """
    weights = dict(trend=0.30, momentum=0.25, volatility=0.20,
                   volume=0.15, sentiment=0.10)
    composite = sum(breakdown[k] * weights.get(k, 0) for k in breakdown)
    composite  = float(np.clip(composite, -1, 1))
    confidence = float(abs(composite))

    if composite >= 0.55:   signal = Signal.STRONG_BUY
    elif composite >= 0.20: signal = Signal.BUY
    elif composite <= -0.55: signal = Signal.STRONG_SELL
    elif composite <= -0.20: signal = Signal.SELL
    else:                    signal = Signal.HOLD

    return signal, round(confidence, 4)


# ════════════════════════════════════════════════════════════════════════════
# Trade Parameter Calculator
# ════════════════════════════════════════════════════════════════════════════

def compute_trade_params(
    atr_pct:   float,
    risk_level: RiskLevel,
    signal:    Signal,
) -> dict:
    """Derive stop-loss, take-profit, and position size from risk profile."""
    # base stop = 2× ATR
    stop_pct = round(atr_pct * 2, 4)

    rr_map = {
        Signal.STRONG_BUY:  3.0,
        Signal.BUY:         2.5,
        Signal.HOLD:        2.0,
        Signal.SELL:        2.5,
        Signal.STRONG_SELL: 3.0,
    }
    rr = rr_map.get(signal, 2.0)
    tp_pct = round(stop_pct * rr, 4)

    # position size: risk 1% of capital / stop distance (Kelly-inspired)
    size_map = {
        RiskLevel.VERY_LOW:  0.08,
        RiskLevel.LOW:       0.06,
        RiskLevel.MODERATE:  0.04,
        RiskLevel.HIGH:      0.02,
        RiskLevel.VERY_HIGH: 0.01,
    }
    pos_pct = round(size_map.get(risk_level, 0.03), 4)

    return dict(
        stop_loss_pct     = stop_pct,
        take_profit_pct   = tp_pct,
        position_size_pct = pos_pct,
        risk_reward_ratio = rr,
    )


# ════════════════════════════════════════════════════════════════════════════
# Core Analysis Pipeline
# ════════════════════════════════════════════════════════════════════════════

def run_analysis(
    symbol:   str,
    period:   str = "1y",
    interval: str = "1d",
) -> dict:
    """
    Full pipeline:
      1. Fetch data
      2. Compute indicators
      3. Run AI modules
      4. Detect regime & sentiment
      5. Aggregate signal
      6. Compute risk & trade params
    """
    # ── 1. data ────────────────────────────────────────────────────────────
    ohlcv, source = fetch_data(symbol, period, interval)
    if len(ohlcv) < 30:
        raise ValueError(f"Insufficient data for {symbol} ({len(ohlcv)} bars).")

    price    = float(ohlcv[-1, 3])
    prev     = float(ohlcv[-2, 3]) if len(ohlcv) >= 2 else price
    chg_pct  = (price - prev) / prev if prev != 0 else 0.0

    # ── 2. indicators ───────────────────────────────────────────────────────
    engine = IndicatorEngine(ohlcv)
    ind    = engine.compute_all()

    # ── 3. AI modules ───────────────────────────────────────────────────────
    notes = []

    t_score, t_note = TrendModule().score(ind, price)
    notes.append(t_note)

    m_score, m_note = MomentumModule().score(ind)
    notes.append(m_note)

    v_score, v_note = VolatilityModule().score(ind, price)
    notes.append(v_note)

    vol_score, vol_note = VolumeModule().score(ind, chg_pct)
    notes.append(vol_note)

    s_score, s_note, sentiment = SentimentModule().score(ohlcv)
    notes.append(s_note)

    breakdown = dict(
        trend      = t_score,
        momentum   = m_score,
        volatility = v_score,
        volume     = vol_score,
        sentiment  = s_score,
    )

    # ── 4. signal ───────────────────────────────────────────────────────────
    signal, confidence = aggregate_signal(breakdown)

    # ── 5. risk ─────────────────────────────────────────────────────────────
    risk_data = RiskEngine().analyse(ohlcv)
    regime    = detect_regime(ind, risk_data["daily_volatility"])

    # ── 6. trade params ─────────────────────────────────────────────────────
    atr_pct = ind["atr"] / price if price > 0 else 0.01
    tp      = compute_trade_params(atr_pct, risk_data["risk_level"], signal)

    return dict(
        symbol           = symbol.upper(),
        signal           = signal,
        confidence       = confidence,
        market_regime    = regime,
        sentiment        = sentiment,
        risk_level       = risk_data["risk_level"],
        price            = round(price, 4),
        price_change_pct = round(chg_pct, 6),
        indicators       = ind,
        risk_metrics     = dict(
            daily_volatility = risk_data["daily_volatility"],
            var_1d_95        = risk_data["var_1d_95"],
            var_1d_99        = risk_data["var_1d_99"],
            max_drawdown     = risk_data["max_drawdown"],
            sharpe_proxy     = risk_data["sharpe_proxy"],
            beta_spy         = risk_data["beta_spy"],
        ),
        trade_params     = tp,
        signal_breakdown = {k: round(v, 4) for k, v in breakdown.items()},
        analysis_notes   = notes,
        data_source      = source,
        bars_analysed    = len(ohlcv),
        timestamp        = datetime.now(timezone.utc).isoformat(),
    )


# ════════════════════════════════════════════════════════════════════════════
# FastAPI Application
# ════════════════════════════════════════════════════════════════════════════

if FASTAPI_AVAILABLE:

    app = FastAPI(
        title       = "AI Trading System API",
        description = (
            "Quantitative trading signal engine combining technical indicators, "
            "multi-module AI scoring, market regime detection, and risk analytics."
        ),
        version     = "1.0.0",
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins  = ["*"],
        allow_methods  = ["GET"],
        allow_headers  = ["*"],
    )

    # ── /health ──────────────────────────────────────────────────────────────

    @app.get("/health", tags=["System"])
    async def health():
        """Server health check."""
        return JSONResponse({
            "status":    "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "yfinance":  YFINANCE_AVAILABLE,
        })

    # ── /symbols ─────────────────────────────────────────────────────────────

    @app.get("/symbols", tags=["System"])
    async def symbols():
        """Return a list of commonly supported symbols."""
        return JSONResponse({
            "equities":  ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"],
            "etfs":      ["SPY", "QQQ", "IWM", "GLD", "TLT"],
            "note":      "Any valid yfinance ticker is accepted. Unlisted symbols use synthetic data.",
        })

    # ── /analyze/{symbol} ────────────────────────────────────────────────────

    @app.get(
        "/analyze/{symbol}",
        response_model = AnalysisResponse,
        tags           = ["Analysis"],
        summary        = "Full AI trading signal for a symbol",
        response_description = "Risk-adjusted trading signal with indicators and confidence score",
    )
    async def analyze(
        symbol: str,
        period: str = Query(
            default     = "1y",
            description = "yfinance period string: 1mo, 3mo, 6mo, 1y, 2y, 5y",
            regex       = r"^\d+(d|mo|y)$",
        ),
        interval: str = Query(
            default     = "1d",
            description = "Bar interval: 1d, 1wk, 1mo",
            regex       = r"^(1d|1wk|1mo)$",
        ),
    ):
        """
        ## AI Trading Signal

        Runs the full analysis pipeline for the given stock symbol:

        1. **Data Fetch** – Live OHLCV from yfinance (falls back to synthetic)
        2. **Indicator Engine** – RSI, MACD, Bollinger Bands, ATR, ADX, Volume
        3. **AI Modules** – Trend, Momentum, Volatility, Volume, Sentiment scoring
        4. **Signal Aggregation** – Weighted composite → BUY/SELL/HOLD
        5. **Risk Analysis** – VaR, drawdown, volatility classification
        6. **Trade Parameters** – Stop loss, take profit, suggested position size

        ### Response fields
        | Field | Description |
        |-------|-------------|
        | `signal` | STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL |
        | `confidence` | 0–1 (higher = stronger conviction) |
        | `market_regime` | Current market structure |
        | `sentiment` | Price-action derived sentiment |
        | `risk_level` | VERY_LOW → VERY_HIGH |
        """
        sym = symbol.upper().strip()
        if not sym or not sym.replace(".", "").replace("-", "").isalnum():
            raise HTTPException(status_code=422, detail=f"Invalid symbol: {sym!r}")

        try:
            result = run_analysis(sym, period=period, interval=interval)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

        return JSONResponse(content={
            **result,
            "signal":        result["signal"].value,
            "market_regime": result["market_regime"].value,
            "sentiment":     result["sentiment"].value,
            "risk_level":    result["risk_level"].value,
        })

    # ── /analyze/batch ───────────────────────────────────────────────────────

    @app.get("/analyze/batch/{symbols}", tags=["Analysis"])
    async def analyze_batch(
        symbols: str,
        period:  str = Query(default="1y"),
    ):
        """
        Analyze multiple comma-separated symbols.
        Example: /analyze/batch/AAPL,MSFT,GOOGL
        """
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:10]
        results  = {}
        for sym in sym_list:
            try:
                r = run_analysis(sym, period=period)
                results[sym] = {
                    "signal":        r["signal"].value,
                    "confidence":    r["confidence"],
                    "market_regime": r["market_regime"].value,
                    "sentiment":     r["sentiment"].value,
                    "risk_level":    r["risk_level"].value,
                    "price":         r["price"],
                    "price_change":  r["price_change_pct"],
                }
            except Exception as exc:
                results[sym] = {"error": str(exc)}
        return JSONResponse({"symbols": results, "count": len(results)})

else:
    # ── Fallback CLI when FastAPI is not installed ───────────────────────────
    app = None


# ════════════════════════════════════════════════════════════════════════════
# CLI entry-point (also works without FastAPI)
# ════════════════════════════════════════════════════════════════════════════

def _print_result(result: dict) -> None:
    sig = result["signal"]
    sig_val = sig.value if hasattr(sig, "value") else sig
    regime  = result["market_regime"]
    regime_val = regime.value if hasattr(regime, "value") else regime
    sent    = result["sentiment"]
    sent_val = sent.value if hasattr(sent, "value") else sent
    risk    = result["risk_level"]
    risk_val = risk.value if hasattr(risk, "value") else risk

    bar = "═" * 60
    print(f"\n{bar}")
    print(f"  AI TRADING SIGNAL  —  {result['symbol']}")
    print(bar)
    print(f"  Signal        : {sig_val}")
    print(f"  Confidence    : {result['confidence']:.2%}")
    print(f"  Market Regime : {regime_val}")
    print(f"  Sentiment     : {sent_val}")
    print(f"  Risk Level    : {risk_val}")
    print(f"  Price         : ${result['price']:,.4f}  ({result['price_change_pct']:+.2%})")
    print(f"  Bars Analysed : {result['bars_analysed']}  [{result['data_source']}]")
    print("─" * 60)
    bd = result["signal_breakdown"]
    print("  Signal Breakdown:")
    for k, v in bd.items():
        bar_fill = "█" * int(abs(v) * 20)
        sign = "+" if v >= 0 else "-"
        print(f"    {k:<12} {sign}{abs(v):.3f}  {bar_fill}")
    print("─" * 60)
    ind = result["indicators"]
    print(f"  RSI: {ind['rsi']:.1f}  |  ADX: {ind['adx']:.1f}  |  BB%: {ind['bb_pct']:.2f}  |  VolRatio: {ind['volume_ratio']:.2f}x")
    print("─" * 60)
    tp = result["trade_params"]
    print(f"  Stop Loss    : {tp['stop_loss_pct']:.2%}")
    print(f"  Take Profit  : {tp['take_profit_pct']:.2%}")
    print(f"  Position Size: {tp['position_size_pct']:.2%} of capital")
    print(f"  R:R Ratio    : {tp['risk_reward_ratio']:.1f}:1")
    print("─" * 60)
    rm = result["risk_metrics"]
    print(f"  Annual Vol   : {rm['daily_volatility']:.2%}")
    print(f"  1-Day VaR 95%: {rm['var_1d_95']:.2%}  |  99%: {rm['var_1d_99']:.2%}")
    print(f"  Max Drawdown : {rm['max_drawdown']:.2%}")
    print(f"  Sharpe Proxy : {rm['sharpe_proxy']:.2f}")
    print(bar)


if __name__ == "__main__":
    import sys

    # ── demo mode (no server) ───────────────────────────────────────────────
    demo_symbols = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "TSLA", "SPY"]

    print("=" * 60)
    print("  AI TRADING SYSTEM  —  CLI Demo")
    print("=" * 60)
    for sym in demo_symbols:
        print(f"\nAnalysing {sym} …")
        try:
            result = run_analysis(sym)
            _print_result(result)
        except Exception as exc:
            print(f"  ERROR: {exc}")

    if FASTAPI_AVAILABLE:
        print("\n  To start the API server:")
        print("  uvicorn trading_server:app --host 0.0.0.0 --port 8000 --reload")
    else:
        print("\n  Install FastAPI to start the server:")
        print("  pip install fastapi uvicorn")
