"""
trade_opportunity_analyzer.py — AIGOFIN ML Package
====================================================
Detects optimal historical buy and sell points using multiple
technical algorithms:

  1. RSI reversal detection (oversold/overbought extremes)
  2. MACD crossover detection (bullish/bearish signal line crosses)
  3. Momentum breakout detection (price breaking above N-bar high/low)
  4. Moving average crossover (SMA20 / SMA50)

Public functions
----------------
  find_best_buy_point(data)  → dict
  find_best_sell_point(data) → dict
  analyze_opportunities(data) → dict  (combined result for the API endpoint)

Input
-----
  data : pd.DataFrame with columns open, high, low, close, volume
         (DatetimeIndex or integer index)

Output (example)
----------------
  {
    "best_buy_date":     "2024-03-14",
    "best_buy_price":    176.20,
    "best_sell_date":    "2024-03-18",
    "best_sell_price":   182.40,
    "profit_pct":        3.4,
    "buy_signal_type":   "RSI Reversal",
    "sell_signal_type":  "MACD Crossover",
    "holding_days":      4,
    "all_buy_signals":   [...],
    "all_sell_signals":  [...]
  }
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any


# ════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ════════════════════════════════════════════════════════════════════════════

def _ensure_df(data: Any) -> pd.DataFrame:
    """Normalise input to a DataFrame with lowercase OHLCV columns."""
    if isinstance(data, np.ndarray):
        cols = ["open", "high", "low", "close", "volume"]
        df = pd.DataFrame(data, columns=cols[:data.shape[1]])
    else:
        df = data.copy()
    df.columns = [c.lower() for c in df.columns]
    return df.reset_index(drop=True)


def _ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


def _macd(close: pd.Series):
    """Returns (macd_line, signal_line, histogram)."""
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    line = ema12 - ema26
    signal = _ema(line, 9)
    return line, signal, line - signal


def _fmt_date(df: pd.DataFrame, idx: int) -> str:
    """Return ISO date string for DataFrame row."""
    if hasattr(df.index, "strftime"):
        return df.index[idx].strftime("%Y-%m-%d")
    return str(idx)


# ════════════════════════════════════════════════════════════════════════════
# Signal detection functions
# ════════════════════════════════════════════════════════════════════════════

def _rsi_signals(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """Detect RSI reversal points (oversold → buy, overbought → sell)."""
    rsi = _rsi(df["close"])
    buys, sells = [], []

    for i in range(1, len(df)):
        if pd.isna(rsi.iloc[i]):
            continue
        # Oversold cross back above 30  → BUY
        if rsi.iloc[i - 1] < 30 <= rsi.iloc[i]:
            buys.append({
                "idx": i,
                "date": _fmt_date(df, i),
                "price": float(df["close"].iloc[i]),
                "score": float(30 - min(rsi.iloc[i], 30)) / 30,  # deeper=stronger
                "signal_type": "RSI Reversal",
            })
        # Overbought cross back below 70 → SELL
        if rsi.iloc[i - 1] > 70 >= rsi.iloc[i]:
            sells.append({
                "idx": i,
                "date": _fmt_date(df, i),
                "price": float(df["close"].iloc[i]),
                "score": float(max(rsi.iloc[i], 70) - 70) / 30,
                "signal_type": "RSI Reversal",
            })

    return {"buys": buys, "sells": sells}


def _macd_signals(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """Detect MACD crossover signals."""
    line, signal, hist = _macd(df["close"])
    buys, sells = [], []

    for i in range(1, len(df)):
        if pd.isna(hist.iloc[i]) or pd.isna(hist.iloc[i - 1]):
            continue
        # Bullish crossover (histogram turns positive)
        if hist.iloc[i - 1] < 0 <= hist.iloc[i]:
            buys.append({
                "idx": i,
                "date": _fmt_date(df, i),
                "price": float(df["close"].iloc[i]),
                "score": float(min(abs(hist.iloc[i]), 2)) / 2,
                "signal_type": "MACD Crossover",
            })
        # Bearish crossover (histogram turns negative)
        if hist.iloc[i - 1] > 0 >= hist.iloc[i]:
            sells.append({
                "idx": i,
                "date": _fmt_date(df, i),
                "price": float(df["close"].iloc[i]),
                "score": float(min(abs(hist.iloc[i]), 2)) / 2,
                "signal_type": "MACD Crossover",
            })

    return {"buys": buys, "sells": sells}


def _momentum_signals(df: pd.DataFrame, window: int = 20) -> Dict[str, List[Dict]]:
    """Detect momentum breakout: price breaks N-bar high (buy) or low (sell)."""
    highs = df["high"].rolling(window).max().shift(1)
    lows = df["low"].rolling(window).min().shift(1)
    close = df["close"]
    buys, sells = [], []

    for i in range(window, len(df)):
        if pd.isna(highs.iloc[i]):
            continue
        # Breakout above rolling high → BUY
        if close.iloc[i] > highs.iloc[i] and close.iloc[i - 1] <= highs.iloc[i - 1]:
            pct = (close.iloc[i] - highs.iloc[i]) / highs.iloc[i]
            buys.append({
                "idx": i,
                "date": _fmt_date(df, i),
                "price": float(close.iloc[i]),
                "score": float(min(pct * 20, 1.0)),
                "signal_type": "Momentum Breakout",
            })
        # Breakdown below rolling low → SELL
        if close.iloc[i] < lows.iloc[i] and close.iloc[i - 1] >= lows.iloc[i - 1]:
            pct = (lows.iloc[i] - close.iloc[i]) / lows.iloc[i]
            sells.append({
                "idx": i,
                "date": _fmt_date(df, i),
                "price": float(close.iloc[i]),
                "score": float(min(pct * 20, 1.0)),
                "signal_type": "Momentum Breakout",
            })

    return {"buys": buys, "sells": sells}


def _ma_crossover_signals(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """Detect SMA20 / SMA50 golden and death crosses."""
    close = df["close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    buys, sells = [], []

    for i in range(1, len(df)):
        if pd.isna(sma50.iloc[i]) or pd.isna(sma50.iloc[i - 1]):
            continue
        diff_now = sma20.iloc[i] - sma50.iloc[i]
        diff_prev = sma20.iloc[i - 1] - sma50.iloc[i - 1]
        # Golden cross: SMA20 crosses above SMA50
        if diff_prev < 0 <= diff_now:
            buys.append({
                "idx": i,
                "date": _fmt_date(df, i),
                "price": float(close.iloc[i]),
                "score": float(min(abs(diff_now) / close.iloc[i] * 100, 1.0)),
                "signal_type": "MA Crossover (Golden Cross)",
            })
        # Death cross: SMA20 crosses below SMA50
        if diff_prev > 0 >= diff_now:
            sells.append({
                "idx": i,
                "date": _fmt_date(df, i),
                "price": float(close.iloc[i]),
                "score": float(min(abs(diff_now) / close.iloc[i] * 100, 1.0)),
                "signal_type": "MA Crossover (Death Cross)",
            })

    return {"buys": buys, "sells": sells}


# ════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════

def find_best_buy_point(data: Any) -> Dict:
    """
    Scan all four algorithms and return the single highest-conviction
    buy signal from the historical data.

    Returns
    -------
    dict with keys: date, price, signal_type, score
    """
    df = _ensure_df(data)
    all_buys: List[Dict] = []

    for fn in [_rsi_signals, _macd_signals, _momentum_signals, _ma_crossover_signals]:
        result = fn(df)
        all_buys.extend(result["buys"])

    if not all_buys:
        # Fallback: lowest RSI point
        rsi = _rsi(df["close"])
        idx_min = int(rsi.dropna().idxmin())
        return {
            "date":        _fmt_date(df, idx_min),
            "price":       float(df["close"].iloc[idx_min]),
            "signal_type": "RSI Oversold (extreme)",
            "score":       0.5,
        }

    best = max(all_buys, key=lambda x: x["score"])
    return {
        "date":        best["date"],
        "price":       round(best["price"], 2),
        "signal_type": best["signal_type"],
        "score":       round(best["score"], 4),
    }


def find_best_sell_point(data: Any, after_buy_idx: int = 0) -> Dict:
    """
    Return the single highest-conviction sell signal occurring
    after `after_buy_idx` in the historical data.

    Returns
    -------
    dict with keys: date, price, signal_type, score
    """
    df = _ensure_df(data)
    all_sells: List[Dict] = []

    for fn in [_rsi_signals, _macd_signals, _momentum_signals, _ma_crossover_signals]:
        result = fn(df)
        sells = [s for s in result["sells"] if s["idx"] > after_buy_idx]
        all_sells.extend(sells)

    if not all_sells:
        # Fallback: highest close after buy
        sub = df["close"].iloc[after_buy_idx + 1:]
        if len(sub) == 0:
            sub = df["close"]
        idx_max = int(sub.idxmax())
        return {
            "date":        _fmt_date(df, idx_max),
            "price":       float(df["close"].iloc[idx_max]),
            "signal_type": "Peak Price",
            "score":       0.5,
        }

    best = max(all_sells, key=lambda x: x["score"])
    return {
        "date":        best["date"],
        "price":       round(best["price"], 2),
        "signal_type": best["signal_type"],
        "score":       round(best["score"], 4),
    }


def analyze_opportunities(data: Any, symbol: str = "") -> Dict:
    """
    Full opportunity analysis — detects the best buy and sell points
    and computes the implied profit.

    Parameters
    ----------
    data   : OHLCV DataFrame or numpy array
    symbol : optional ticker label

    Returns
    -------
    dict matching the /opportunity/{symbol} API response schema
    """
    df = _ensure_df(data)

    # Collect every buy signal (for chart markers)
    all_buys: List[Dict] = []
    all_sells_raw: List[Dict] = []
    for fn in [_rsi_signals, _macd_signals, _momentum_signals, _ma_crossover_signals]:
        r = fn(df)
        all_buys.extend(r["buys"])
        all_sells_raw.extend(r["sells"])

    # Best buy first
    buy = find_best_buy_point(df)
    buy_idx_in_all = next(
        (b["idx"] for b in all_buys if b["date"] == buy["date"]),
        0
    )

    # Best sell after the buy
    sell = find_best_sell_point(df, after_buy_idx=buy_idx_in_all)
    sell_idx_in_all = next(
        (s["idx"] for s in all_sells_raw if s["date"] == sell["date"]),
        len(df) - 1
    )

    # Profit calculation
    buy_price = buy["price"]
    sell_price = sell["price"]
    profit_pct = round((sell_price - buy_price) / buy_price * 100, 2)

    # Holding period
    try:
        buy_date = pd.Timestamp(buy["date"])
        sell_date = pd.Timestamp(sell["date"])
        holding_days = max((sell_date - buy_date).days, 1)
    except Exception:
        holding_days = sell_idx_in_all - buy_idx_in_all

    # Trim all signals list for the chart (max 10 of each)
    all_buys_sorted = sorted(all_buys, key=lambda x: x["score"], reverse=True)[:10]
    all_sells_sorted = sorted(all_sells_raw, key=lambda x: x["score"], reverse=True)[:10]

    return {
        "symbol":           symbol.upper() if symbol else "",
        "best_buy_date":    buy["date"],
        "best_buy_price":   buy_price,
        "best_sell_date":   sell["date"],
        "best_sell_price":  sell_price,
        "profit_pct":       profit_pct,
        "buy_signal_type":  buy["signal_type"],
        "sell_signal_type": sell["signal_type"],
        "holding_days":     holding_days,
        "all_buy_signals":  all_buys_sorted,
        "all_sell_signals": all_sells_sorted,
    }
