"""
strategy_classifier.py — AIGOFIN ML Package
=============================================
Determines which trading strategy class generated a signal,
based on the indicator values and signal metadata.

Strategy categories
-------------------
  intraday        — short-lived, high volatility, low ADX, small ATR window
  swing_trading   — multi-day moves, moderate trend, RSI reversal or MACD
  momentum        — strong directional move, high ADX, price above MA
  mean_reversion  — extreme RSI, overextended move, BB contraction expected
  trend_following — sustained trend, golden cross, price above SMA200

Trade types
-----------
  Scalp Trade   (< 1 day)
  Intraday      (same day, short timeframe signal)
  Swing Trade   (2–10 days)
  Position      (> 10 days)
  Breakout      (momentum-based catalyst)

Public function
---------------
  classify_strategy(indicators, signals) -> dict

Output example
--------------
  {
    "strategy":        "Momentum Breakout",
    "trade_type":      "Swing Trade",
    "holding_period":  "3-7 days",
    "confidence":      0.82,
    "rationale":       "Strong ADX with RSI above 55 ..."
  }
"""

from __future__ import annotations

from typing import Dict, Any, Optional


# ════════════════════════════════════════════════════════════════════════════
# Strategy scoring helpers
# ════════════════════════════════════════════════════════════════════════════

def _score_intraday(ind: dict, sig: dict) -> float:
    """High-frequency signals: volatile, low ADX, tight BB."""
    score = 0.0
    adx = ind.get("adx", 20)
    bb_pct = ind.get("bb_pct", 0.5)
    volume_ratio = ind.get("volume_ratio", 1.0)

    if adx < 20:
        score += 0.3
    if 0.1 < bb_pct < 0.9:
        score += 0.2
    if volume_ratio > 1.5:
        score += 0.3      # intraday spike
    if sig.get("signal_type", "").lower() in ("rsi reversal",):
        score += 0.2

    return min(score, 1.0)


def _score_swing(ind: dict, sig: dict) -> float:
    """Multi-day directional move using RSI reversal or MACD crossover."""
    score = 0.0
    rsi = ind.get("rsi", 50)
    macd_hist = ind.get("macd_histogram", 0)
    buy_signal = sig.get("signal_type", "").lower()

    if rsi < 35 or rsi > 65:
        score += 0.35      # strong reversal or continuation candidate
    if abs(macd_hist) > 0.5:
        score += 0.25
    if "macd" in buy_signal:
        score += 0.25
    if "rsi" in buy_signal:
        score += 0.15

    return min(score, 1.0)


def _score_momentum(ind: dict, sig: dict) -> float:
    """Strong trend move; high ADX, price above SMAs."""
    score = 0.0
    adx = ind.get("adx", 20)
    sma20 = ind.get("sma_20", 0)
    sma50 = ind.get("sma_50", 0)
    price = ind.get("price", 0)
    buy_signal = sig.get("signal_type", "").lower()

    if adx > 25:
        score += 0.3
    if adx > 40:
        score += 0.2      # extra for very strong trend
    if price and sma20 and price > sma20:
        score += 0.15
    if price and sma50 and price > sma50:
        score += 0.15
    if "momentum" in buy_signal or "breakout" in buy_signal:
        score += 0.2

    return min(score, 1.0)


def _score_mean_reversion(ind: dict, sig: dict) -> float:
    """Extreme RSI or price outside Bollinger Bands snapping back."""
    score = 0.0
    rsi = ind.get("rsi", 50)
    bb_pct = ind.get("bb_pct", 0.5)
    buy_signal = sig.get("signal_type", "").lower()

    if rsi < 25:
        score += 0.5      # deeply oversold
    elif rsi > 75:
        score += 0.5      # deeply overbought on sell side
    elif rsi < 30 or rsi > 70:
        score += 0.25

    if bb_pct < 0.05 or bb_pct > 0.95:
        score += 0.35     # outside or at BB extremes
    elif bb_pct < 0.10 or bb_pct > 0.90:
        score += 0.15

    if "rsi" in buy_signal and (rsi < 32 or rsi > 68):
        score += 0.15

    return min(score, 1.0)


def _score_trend_following(ind: dict, sig: dict) -> float:
    """Sustained trend riding using MA crossover / golden cross."""
    score = 0.0
    sma20 = ind.get("sma_20", 0)
    sma50 = ind.get("sma_50", 0)
    sma200 = ind.get("sma_200", 0)
    price = ind.get("price", 0)
    buy_signal = sig.get("signal_type", "").lower()
    adx = ind.get("adx", 20)

    if sma20 and sma50 and sma20 > sma50:
        score += 0.25     # golden cross territory
    if price and sma200 and price > sma200:
        score += 0.2      # long-term uptrend
    if adx > 20:
        score += 0.15
    if "ma crossover" in buy_signal or "golden" in buy_signal:
        score += 0.4

    return min(score, 1.0)


# ════════════════════════════════════════════════════════════════════════════
# Trade type inference
# ════════════════════════════════════════════════════════════════════════════

def _infer_trade_type(strategy_name: str, ind: dict, holding_days: int = 0) -> Dict:
    """Map strategy → trade type with suggested holding period."""
    s = strategy_name.lower()

    if holding_days > 0:
        if holding_days <= 1:
            return {"trade_type": "Intraday", "holding_period": "same day"}
        elif holding_days <= 7:
            return {"trade_type": "Swing Trade", "holding_period": f"{holding_days} days"}
        elif holding_days <= 30:
            return {"trade_type": "Position", "holding_period": f"{holding_days} days"}
        else:
            return {"trade_type": "Long-term Position", "holding_period": f"{holding_days} days"}

    # Fallback by strategy name if holding_days not provided
    if "intraday" in s or "scalp" in s:
        return {"trade_type": "Intraday", "holding_period": "< 1 day"}
    elif "momentum" in s or "breakout" in s:
        return {"trade_type": "Swing Trade", "holding_period": "3–7 days"}
    elif "mean reversion" in s:
        return {"trade_type": "Swing Trade", "holding_period": "2–5 days"}
    elif "trend" in s:
        return {"trade_type": "Position", "holding_period": "2–6 weeks"}
    elif "swing" in s:
        return {"trade_type": "Swing Trade", "holding_period": "3–10 days"}
    else:
        return {"trade_type": "Swing Trade", "holding_period": "3–7 days"}


def _build_rationale(strategy: str, ind: dict, sig: dict) -> str:
    """Build a one-line human-readable rationale for the chosen strategy."""
    rsi = ind.get("rsi", 50)
    adx = ind.get("adx", 20)
    bb = ind.get("bb_pct", 0.5)
    sig_type = sig.get("signal_type", "technical signal")

    parts = []
    if strategy == "Mean Reversion":
        parts.append(f"RSI={rsi:.0f} is at an extreme")
    elif strategy == "Momentum Breakout":
        parts.append(f"ADX={adx:.0f} shows strong directional momentum")
    elif strategy == "Trend Following":
        parts.append("Price is above both SMA20 and SMA50")
    elif strategy == "Swing Trading":
        parts.append(f"{sig_type} detected on intermediate timeframe")
    else:
        parts.append(f"{sig_type} identified as primary trigger")

    if bb < 0.15:
        parts.append("BB shows price near lower band (potential reversal)")
    elif bb > 0.85:
        parts.append("BB shows price near upper band (potential reversal)")

    return ". ".join(parts) + "."


# ════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════

def classify_strategy(
    indicators: Dict[str, Any],
    signals: Dict[str, Any],
    holding_days: int = 0,
) -> Dict:
    """
    Determine which trading strategy class best matches the provided
    indicator state and signal metadata.

    Parameters
    ----------
    indicators  : dict — raw indicator values from IndicatorsEngine
                  Expected keys (all optional with defaults):
                    rsi, adx, macd_histogram, sma_20, sma_50, sma_200,
                    bb_pct, volume_ratio, price
    signals     : dict — signal metadata; at minimum {"signal_type": "..."}
    holding_days: int  — if known, informs the trade_type output

    Returns
    -------
    dict with:
      strategy, trade_type, holding_period, confidence, rationale
    """
    ind = indicators or {}
    sig = signals or {}

    scores = {
        "Intraday":          _score_intraday(ind, sig),
        "Swing Trading":     _score_swing(ind, sig),
        "Momentum Breakout": _score_momentum(ind, sig),
        "Mean Reversion":    _score_mean_reversion(ind, sig),
        "Trend Following":   _score_trend_following(ind, sig),
    }

    # Primary strategy = highest score
    best_strategy = max(scores, key=lambda k: scores[k])
    best_score    = scores[best_strategy]

    # Guard: if all scores are near zero, default to Swing Trading
    if best_score < 0.05:
        best_strategy = "Swing Trading"
        best_score    = 0.3

    trade_info = _infer_trade_type(best_strategy, ind, holding_days)
    rationale  = _build_rationale(best_strategy, ind, sig)

    return {
        "strategy":       best_strategy,
        "trade_type":     trade_info["trade_type"],
        "holding_period": trade_info["holding_period"],
        "confidence":     round(best_score, 3),
        "rationale":      rationale,
        "all_scores":     {k: round(v, 3) for k, v in scores.items()},
    }
