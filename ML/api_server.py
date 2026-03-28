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

def normalize_symbol(symbol: str) -> str:
    """Normalize common ticker variations (e.g., .NSE to .NS for yfinance)."""
    sym = symbol.upper().strip()
    if sym.endswith(".NSE"):
        return sym.replace(".NSE", ".NS")
    return sym

def _fetch_real(symbol: str, period: str, interval: str) -> np.ndarray:
    """Fetch OHLCV from yfinance (with caching). Returns (T, 5) array: O H L C V."""
    sym = normalize_symbol(symbol)
    
    # Check cache first
    cached_df = _market_data_cache.get_data(sym, period, interval)
    if cached_df is not None:
        return cached_df[["Open", "High", "Low", "Close", "Volume"]].values.astype(float)
    
    # Fetch from yfinance - disable auto_adjust to match market terminals (LTP)
    ticker = yf.Ticker(sym)
    df     = ticker.history(period=period, interval=interval, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data returned for {sym!r}.")
    
    # Store in cache
    _market_data_cache.set_data(sym, period, interval, df)
    
    return df[["Open", "High", "Low", "Close", "Volume"]].values.astype(float)


def _get_live_price(symbol: str) -> float:
    """Get the absolute latest price (un-cached) for accuracy."""
    try:
        sym = normalize_symbol(symbol)
        ticker = yf.Ticker(sym)
        # Try fast_info first (fastest)
        try:
            return float(ticker.fast_info['lastPrice'])
        except:
            # Fallback to last row of high-resolution history
            df = ticker.history(period="1d", interval="1m")
            if not df.empty:
                return float(df['Close'].iloc[-1])
            return 0.0
    except:
        return 0.0


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

    # Use live price for analysis if available (fresher than daily close)
    live_price = _get_live_price(symbol)
    price    = live_price if live_price > 0 else float(ohlcv[-1, 3])
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

class _AnalysisCache:
    def __init__(self, ttl: int = 60):
        self.ttl = ttl
        self.cache = {}

    def get(self, key: str):
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["data"]
            else:
                del self.cache[key]
        return None

    def set(self, key: str, data: dict):
        self.cache[key] = {
            "timestamp": time.time(),
            "data": data
        }

_signal_cache = _AnalysisCache(ttl=60)


class MarketDataCache:
    """Thread-safe cache for raw yfinance DataFrames to prevent redundant network I/O."""
    def __init__(self, ttl: int = 300):
        self.ttl = ttl
        self.cache = {}

    def get_data(self, symbol: str, period: str, interval: str):
        key = f"{symbol}_{period}_{interval}"
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["df"]
            else:
                del self.cache[key]
        return None

    def set_data(self, symbol: str, period: str, interval: str, df):
        key = f"{symbol}_{period}_{interval}"
        self.cache[key] = {
            "timestamp": time.time(),
            "df": df
        }

_market_data_cache = MarketDataCache(ttl=300)


def _sanitize(obj):
    """Recursively replace NaN/Infinity with JSON-safe values."""
    try:
        from fastapi.encoders import jsonable_encoder
        obj = jsonable_encoder(obj)
    except ImportError:
        pass

    def _clean(x):
        if isinstance(x, dict):
            return {k: _clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_clean(v) for v in x]
        if isinstance(x, float):
            return 0.0 if (math.isnan(x) or math.isinf(x)) else x
        if hasattr(x, 'item'):          # numpy scalar fallback
            try:
                val = float(x.item())
                return 0.0 if (math.isnan(val) or math.isinf(val)) else val
            except:
                pass
        return x

    return _clean(obj)


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
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── /health ──────────────────────────────────────────────────────────────
    # ── /chart/{symbol} ──────────────────────────────────────────────────────

    @app.get("/chart/{symbol}", tags=["Data"])
    async def get_chart(
        symbol: str, 
        period: str = "1mo", 
        interval: str = "1d"
    ):
        """Historical candlestick data (with caching)."""
        try:
            sym = symbol.upper().strip()
            sym = normalize_symbol(symbol)
            # Reuse MarketDataCache
            df = _market_data_cache.get_data(sym, period, interval)
            if df is None:
                import yfinance as yf
                df = yf.Ticker(sym).history(period=period, interval=interval)
                _market_data_cache.set_data(sym, period, interval, df)
            
            if df.empty:
                return []
                
            data = []
            for timestamp, row in df.iterrows():
                data.append({
                    "time": int(timestamp.timestamp()),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"])
                })
            return data
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/health", tags=["System"])
    async def health():
        """Server health check."""
        return JSONResponse({
            "status":    "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "yfinance":  YFINANCE_AVAILABLE,
        })

    # ── /healthz ─────────────────────────────────────────────────────────────

    @app.get("/healthz", tags=["System"])
    async def healthz():
        """Deep health check using diagnostic module."""
        try:
            from .health_check import run_diagnostics
            passed = run_diagnostics()
            return JSONResponse(
                content={"status": "healthy" if passed else "degraded"},
                status_code=200 if passed else 503
            )
        except Exception as e:
            return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)
    @app.get("/history/{symbol}", tags=["Data"])
    async def get_history(symbol: str, days: int = 10):
        """Fetch historical OHLCV data (with caching)."""
        try:
            sym = normalize_symbol(symbol)
            # Check cache for 1mo/1d resolution which is what history needs
            df = _market_data_cache.get_data(sym, "1mo", "1d")
            if df is None:
                import yfinance as yf
                df = yf.Ticker(sym).history(period="1mo", interval="1d")
                _market_data_cache.set_data(sym, "1mo", "1d", df)
            
            if df.empty:
                return JSONResponse({"data": []})
            
            df_slice = df.tail(days)
            data = []
            for date, row in df_slice.iterrows():
                data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"])
                })
            return JSONResponse({"data": data[::-1]})  # newest first
        except Exception as e:
            return JSONResponse({"data": [], "error": str(e)}, status_code=500)

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
            description = "Bar interval: 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo",
            regex       = r"^(1m|5m|15m|30m|1h|1d|1wk|1mo)$",
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
        sym = normalize_symbol(symbol)
        if not sym or not sym.replace(".", "").replace("-", "").isalnum():
            raise HTTPException(status_code=422, detail=f"Invalid symbol: {sym!r}")

        cache_key = f"{sym}_{period}_{interval}"
        cached = _signal_cache.get(cache_key)
        if cached:
            return JSONResponse(content=cached)

        try:
            result = run_analysis(sym, period=period, interval=interval)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

        resp_content = _sanitize({
            **result,
            "signal":        result["signal"].value,
            "market_regime": result["market_regime"].value,
            "sentiment":     result["sentiment"].value,
            "risk_level":    result["risk_level"].value,
        })
        _signal_cache.set(cache_key, resp_content)

        return JSONResponse(content=resp_content)

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
        sym_list = [normalize_symbol(s.strip()) for s in symbols.split(",") if s.strip()][:10]
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

    # /backtest/{symbol} moved to lower section for better consolidation

    # ── /live/{symbol} ───────────────────────────────────────────────────────

    from .live_data_stream import get_live_stock_price
    from .ai_brain import AIBrain

    class LiveResponse(BaseModel):
        symbol: str
        price: float
        signal: str
        confidence: float
        market_regime: str
        risk_level: float

    @app.get(
        "/live/{symbol}",
        response_model=LiveResponse,
        tags=["Live"],
        summary="Real-time AI trading signal via Finnhub",
        response_description="Risk-adjusted trading signal based on Finnhub live data"
    )
    async def live_analyze(symbol: str):
        """
        Fetches live stock price via Finnhub, computes technical indicators,
        and runs the AI evaluation pipeline to return a live trading signal.
        """
        sym = normalize_symbol(symbol)
        if not sym or not sym.replace(".", "").replace("-", "").isalnum():
            raise HTTPException(status_code=422, detail=f"Invalid symbol: {sym!r}")

        # 1. Fetch live stock price
        live_data = get_live_stock_price(sym)
        live_price = live_data.get("price", 0.0)
        
        if live_price == 0.0:
            raise HTTPException(status_code=503, detail=f"Could not fetch live price for {sym}")
        
        # 2. Compute indicators
        try:
            ohlcv, _ = fetch_data(sym, period="1y", interval="1d")
            
            # Update the latest bar with the live Finnhub price
            if len(ohlcv) > 0:
                ohlcv[-1, 3] = live_price  # Update close
                
            engine = IndicatorEngine(ohlcv)
            ind = engine.compute_all()
            
            price = live_price
            prev = float(ohlcv[-2, 3]) if len(ohlcv) >= 2 else price
            chg_pct = (price - prev) / prev if prev != 0 else 0.0
            
            t_score, _ = TrendModule().score(ind, price)
            m_score, _ = MomentumModule().score(ind)
            v_score, _ = VolatilityModule().score(ind, price)
            vol_score, _ = VolumeModule().score(ind, chg_pct)
            s_score, _, sentiment = SentimentModule().score(ohlcv)
            
            risk_data = RiskEngine().analyse(ohlcv)
            regime = detect_regime(ind, risk_data["daily_volatility"])
            
            module_outputs = {
                "indicators_engine": {
                    "signal": "BUY" if m_score > 0 else ("SELL" if m_score < 0 else "HOLD"),
                    "confidence": abs(m_score),
                    "indicators": ind
                },
                "strategy_engine": {
                    "signal": "BUY" if t_score > 0 else ("SELL" if t_score < 0 else "HOLD"),
                    "confidence": abs(t_score)
                },
                "market_regime_detector": {
                    "regime": regime.value,
                    "signal": "BUY" if regime in [MarketRegime.TRENDING_UP, MarketRegime.BREAKOUT] else "HOLD",
                    "confidence": 0.8
                },
                "sentiment_engine": {
                    "sentiment": sentiment.value,
                    "signal": "BUY" if s_score > 0 else ("SELL" if s_score < 0 else "HOLD"),
                    "confidence": abs(s_score)
                },
                "risk_manager": {
                    "risk_level": risk_data.get("daily_volatility", 0.5),
                    "signal": "HOLD",
                    "confidence": 1.0
                }
            }
            
            # 3. Run AI Brain
            brain = AIBrain()
            decision = brain.evaluate(module_outputs, symbol=sym)
            
            # 4. Return trading signal
            return JSONResponse({
                "symbol": sym,
                "price": live_price,
                "signal": decision.get("signal", "HOLD"),
                "confidence": decision.get("confidence", 0.0),
                "market_regime": regime.value,
                "risk_level": risk_data.get("daily_volatility", 0.5)
            })
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Live analysis failed: {e}")

    # ── /backtest/{symbol} ───────────────────────────────────────────────────

    from .backtest_engine import run_backtest, load_historical_data

    @app.get(
        "/backtest/{symbol}",
        tags=["Backtest"],
        summary="Run Backtrader backtest for a symbol",
        response_description="Backtest performance metrics: profit, drawdown, win_rate, sharpe",
    )
    async def backtest(
        symbol: str,
        period: str = Query(default="1y", description="yfinance period (e.g. 1y, 6mo)"),
        initial_cash: float = Query(default=100_000.0, description="Starting capital"),
        commission: float = Query(default=0.001, description="Commission per trade"),
    ):
        """
        Runs a full Backtrader backtest using the StrategyEngine signals on historical data.

        Returns profit, max_drawdown, win_rate, and sharpe_ratio.
        """
        sym = normalize_symbol(symbol)
        try:
            from .strategy_engine import StrategyEngine

            # Fetch historical OHLCV data
            ohlcv, source = fetch_data(sym, period=period, interval="1d")
            if len(ohlcv) < 60:
                raise HTTPException(status_code=404, detail=f"Insufficient data for {sym}")

            import pandas as pd
            import numpy as np

            # Convert numpy array to DataFrame for backtest engine
            dates = pd.date_range(end=pd.Timestamp.today(), periods=len(ohlcv), freq="B")
            df_hist = pd.DataFrame(
                ohlcv, index=dates,
                columns=["open", "high", "low", "close", "volume"]
            )

            engine = StrategyEngine()

            # Provide closes to the strategy engine
            closes = ohlcv[:, 3]

            class _ClosesAdapter:
                """Thin adapter passing pre-computed close prices into StrategyEngine."""
                def __init__(self, closes_arr, engine_obj):
                    self._closes = closes_arr
                    self._engine = engine_obj
                    self._idx = 0

                def generate_signal(self, features):
                    idx = min(self._idx, len(self._closes) - 1)
                    self._idx += 1
                    subset = self._closes[: idx + 1]
                    return self._engine.generate_signal({"closes": subset})

            adapter = _ClosesAdapter(closes, engine)

            result = run_backtest(df_hist, adapter, initial_cash=initial_cash, commission=commission)

            if "error" in result:
                # Backtrader not installed — return lightweight internal backtest
                signals_arr = np.sign(
                    np.diff(closes, prepend=closes[0])
                ).astype(int)
                cash = initial_cash
                position = 0
                entry = 0.0
                wins = trades = 0
                peak = cash
                max_dd = 0.0
                equity_curve = []
                for i, sig in enumerate(signals_arr):
                    price = closes[i]
                    if position == 0 and sig == 1:
                        position = 1
                        entry = price
                    elif position == 1 and sig == -1:
                        ret = (price - entry) / entry - commission
                        cash *= 1 + ret
                        if ret > 0:
                            wins += 1
                        trades += 1
                        position = 0
                    equity = cash if position == 0 else cash * (price / entry)
                    equity_curve.append(equity)
                    peak = max(peak, equity)
                    dd = (equity - peak) / peak
                    max_dd = min(max_dd, dd)
                profit = cash - initial_cash
                win_rate = (wins / trades * 100) if trades else 0.0
                result = {
                    "starting_value": round(initial_cash, 2),
                    "final_value": round(cash, 2),
                    "profit": round(profit, 2),
                    "profit_pct": round((profit / initial_cash) * 100, 2),
                    "max_drawdown": round(abs(max_dd) * 100, 2),
                    "win_rate": round(win_rate, 2),
                    "sharpe_ratio": 0.0,
                    "total_trades": trades,
                    "data_source": source,
                    "note": "backtrader not installed — internal backtest used",
                }

            result["symbol"] = sym
            result["data_source"] = source
            result["bars"] = len(ohlcv)
            result["timestamp"] = datetime.now(timezone.utc).isoformat()
            return JSONResponse(result)

        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}")

    # ── /strategy/{symbol} ──────────────────────────────────────────────────

    @app.get(
        "/strategy/{symbol}",
        tags=["Strategy"],
        summary="Evolve and return best trading strategy for a symbol",
        response_description="Best strategy genes, fitness, and performance metrics",
    )
    async def strategy(
        symbol: str,
        generations: int = Query(default=5, description="Number of GA generations"),
        population: int = Query(default=20, description="Strategy population size"),
    ):
        """
        Runs a quick genetic algorithm to evolve the best trading strategy
        for the given symbol using historical data.

        Returns the best strategy's genes, fitness, return, and Sharpe ratio.
        """
        sym = normalize_symbol(symbol)
        try:
            from .strategy_evolver import evolve, generate_price_data

            ohlcv, source = fetch_data(sym, period="1y", interval="1d")
            if len(ohlcv) < 60:
                raise HTTPException(status_code=404, detail=f"Insufficient data for {sym}")

            prices = ohlcv[:, 3].astype(float)

            best, history = evolve(
                prices,
                generations=generations,
                population_size=population,
                verbose=False,
            )

            genes_list = [
                {
                    "indicator": g.indicator,
                    "period":    g.period,
                    "period2":   g.period2,
                    "threshold": round(g.threshold, 2),
                    "condition": g.condition,
                    "action":    g.action,
                }
                for g in best.genes
            ]

            return JSONResponse({
                "symbol":       sym,
                "generations":  generations,
                "population":   population,
                "best_strategy": {
                    "fitness":      round(best.fitness, 4),
                    "total_return": round(best.total_return, 4),
                    "sharpe_ratio": round(best.sharpe_ratio, 4),
                    "max_drawdown": round(best.max_drawdown, 4),
                    "win_rate":     round(best.win_rate, 4),
                    "num_trades":   best.num_trades,
                    "num_genes":    len(best.genes),
                    "genes":        genes_list,
                },
                "history_last": history[-1] if history else {},
                "data_source":  source,
                "bars":         len(prices),
                "timestamp":    datetime.now(timezone.utc).isoformat(),
            })

        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Strategy evolution failed: {exc}")

    # ── /explain/{symbol} ───────────────────────────────────────────────────

    @app.get(
        "/explain/{symbol}",
        tags=["Explain"],
        summary="Generate human-readable trade explanation for a symbol",
        response_description="Plain-text reasoning for the AI trading signal",
    )
    async def explain(
        symbol: str,
        period: str = Query(default="1y"),
    ):
        """
        Runs the full analysis pipeline and then calls the TradeExplainer
        to produce a human-readable plain-text explanation of the trading signal.
        """
        sym = normalize_symbol(symbol)
        try:
            from .trade_explainer import TradeExplainer, TradeContext

            # Run the standard analysis first to get signal context
            result = run_analysis(sym, period=period)

            signal_val = result["signal"]
            signal_str = signal_val.value if hasattr(signal_val, "value") else str(signal_val)
            confidence  = result["confidence"]

            # Map confidence → string signal strength
            if confidence > 0.75 and signal_str in ("BUY", "STRONG_BUY"):
                explain_signal = "STRONG_BUY"
            elif confidence > 0.75 and signal_str in ("SELL", "STRONG_SELL"):
                explain_signal = "STRONG_SELL"
            else:
                explain_signal = signal_str

            regime_val = result["market_regime"]
            regime_str = regime_val.value if hasattr(regime_val, "value") else str(regime_val)
            sent_val   = result["sentiment"]
            sent_str   = sent_val.value if hasattr(sent_val, "value") else str(sent_val)
            risk_val   = result["risk_level"]
            risk_str   = risk_val.value if hasattr(risk_val, "value") else str(risk_val)

            ctx = TradeContext(
                symbol        = sym,
                signal        = explain_signal,
                confidence    = confidence,
                market_regime = regime_str,
                sentiment     = sent_str,
                risk_level    = risk_str,
                indicators    = result["indicators"],
                signal_breakdown = result["signal_breakdown"],
            )

            explainer = TradeExplainer()
            report    = explainer.explain(ctx)

            return JSONResponse({
                "symbol":          sym,
                "signal":          explain_signal,
                "confidence":      confidence,
                "market_regime":   regime_str,
                "sentiment":       sent_str,
                "risk_level":      risk_str,
                "price":           result["price"],
                "reasoning":       report.plain_text if hasattr(report, "plain_text") else str(report),
                "summary":         report.summary    if hasattr(report, "summary")    else "",
                "indicators":      result["indicators"],
                "signal_breakdown": result["signal_breakdown"],
                "timestamp":       datetime.now(timezone.utc).isoformat(),
            })

        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Explain failed: {exc}")

    # ── /opportunity/{symbol} ────────────────────────────────────────────────

    @app.get(
        "/opportunity/{symbol}",
        tags=["Opportunity"],
        summary="Detect best trade opportunity (buy/sell points + strategy) for a symbol",
        response_description="Optimal entry/exit with profit %, strategy type, and chart markers",
    )
    async def opportunity(
        symbol: str,
        period: str = Query(default="1y", description="yfinance period (e.g. 1y, 6mo)"),
        interval: str = Query(default="1d"),
    ):
        """
        Runs the full Trade Opportunity Analyzer + Strategy Classifier pipeline.

        Returns the best historical buy and sell points detected by four algorithms
        (RSI reversal, MACD crossover, momentum breakout, MA crossover),
        the implied profit percentage, and the recommended strategy type.
        """
        sym = normalize_symbol(symbol)
        try:
            from .trade_opportunity_analyzer import analyze_opportunities
            from .strategy_classifier import classify_strategy

            # Fetch OHLCV data
            ohlcv, source = fetch_data(sym, period=period, interval=interval)
            if len(ohlcv) < 60:
                raise HTTPException(status_code=404, detail=f"Insufficient data for {sym}")

            import pandas as pd
            import numpy as np

            dates = pd.date_range(end=pd.Timestamp.today(), periods=len(ohlcv), freq="B")
            df = pd.DataFrame(
                ohlcv, index=dates,
                columns=["open", "high", "low", "close", "volume"]
            )

            # Run opportunity analysis
            opp = analyze_opportunities(df, symbol=sym)

            # Run analysis pipeline to get indicator context
            analysis = run_analysis(sym, period=period, interval=interval)
            indicators = analysis.get("indicators", {})
            indicators["price"] = analysis.get("price", 0)

            # Classify the strategy
            buy_signal_meta = {
                "signal_type": opp.get("buy_signal_type", ""),
                "profit_pct":  opp.get("profit_pct", 0),
            }
            strategy_info = classify_strategy(
                indicators=indicators,
                signals=buy_signal_meta,
                holding_days=opp.get("holding_days", 0),
            )

            # Build response
            return JSONResponse({
                "symbol":           sym,
                "best_buy_date":    opp["best_buy_date"],
                "best_buy_price":   opp["best_buy_price"],
                "best_sell_date":   opp["best_sell_date"],
                "best_sell_price":  opp["best_sell_price"],
                "profit_pct":       opp["profit_pct"],
                "buy_signal_type":  opp["buy_signal_type"],
                "sell_signal_type": opp["sell_signal_type"],
                "holding_days":     opp["holding_days"],
                "strategy":         strategy_info["strategy"],
                "trade_type":       strategy_info["trade_type"],
                "holding_period":   strategy_info["holding_period"],
                "strategy_confidence": strategy_info["confidence"],
                "strategy_rationale":  strategy_info["rationale"],
                "all_scores":       strategy_info["all_scores"],
                "chart_markers": {
                    "buy_signals":  opp["all_buy_signals"][:5],
                    "sell_signals": opp["all_sell_signals"][:5],
                },
                "current_signal":   analysis.get("signal", "").value
                                    if hasattr(analysis.get("signal", ""), "value")
                                    else str(analysis.get("signal", "HOLD")),
                "current_confidence": analysis.get("confidence", 0),
                "market_regime":    analysis.get("market_regime", "").value
                                    if hasattr(analysis.get("market_regime", ""), "value")
                                    else str(analysis.get("market_regime", "")),
                "data_source":      source,
                "bars":             len(ohlcv),
                "timestamp":        datetime.now(timezone.utc).isoformat(),
            })

        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Opportunity analysis failed: {exc}")

    # ── /chart/{symbol} ──────────────────────────────────────────────────────

    @app.get(
        "/chart/{symbol}",
        tags=["Chart"],
        summary="Fetch OHLCV candle data for charting",
        response_description="List of OHLCV candles formatted for lightweight-charts",
    )
    async def get_chart_data(
        symbol: str,
        period: str = Query(default="1mo", description="yfinance period (1d, 5d, 1mo, 3mo, 1y)"),
        interval: str = Query(default="1h", description="Bar interval (1m, 5m, 15m, 1h, 1d)"),
    ):
        """
        Returns historical candle data for the given symbol, formatted for 
        TradingView Lightweight Charts.
        """
        sym = normalize_symbol(symbol)
        print(f"  [DEBUG] Fetching chart data for {symbol} -> {sym}")
        try:
            # Use yfinance directly for higher resolution if possible
            ticker = yf.Ticker(sym)
            df = ticker.history(period=period, interval=interval)
            print(f"  [DEBUG] yf.history returned {len(df)} rows")
            
            if df.empty:
                # Fallback to internal fetch_data (synthetic if needed)
                ohlcv, _ = fetch_data(sym, period=period, interval=interval)
                # Convert back to list of dicts with dummy timestamps
                base_time = int(time.time()) - (len(ohlcv) * 3600)
                candles = [
                    {
                        "time": base_time + (i * 3600),
                        "open": float(row[0]),
                        "high": float(row[1]),
                        "low": float(row[2]),
                        "close": float(row[3]),
                        "volume": float(row[4]),
                    }
                    for i, row in enumerate(ohlcv)
                ]
                return JSONResponse(candles)

            # Format yfinance dataframe
            df = df.reset_index()
            # Lightweight-charts needs 'time' as Unix timestamp (seconds) or string 'YYYY-MM-DD'
            # We'll use Unix timestamp for intra-day support
            candles = []
            for _, row in df.iterrows():
                # Handle different index types (Datetime or just Date)
                dt = row['Date'] if 'Date' in row else row['Datetime']
                ts = int(dt.timestamp())
                
                candles.append({
                    "time": ts,
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"]),
                })
            
            return JSONResponse(candles)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Chart data fetch failed: {e}")

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
    print(f"  Price         : ₹{result['price']:,.2f}  ({result['price_change_pct']:+.2%})")
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
