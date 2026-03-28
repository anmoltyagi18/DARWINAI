"""
simulator.py
------------
Trading strategy simulation and backtesting module.

Features:
  - Historical backtesting engine
  - Profit / loss tracking (per-trade and cumulative)
  - Drawdown analysis (max drawdown, drawdown duration)
  - Win rate calculation and trade statistics
  - Comprehensive performance metrics

Built-in strategies:
  - MovingAverageCrossover   : classic SMA/EMA dual-MA crossover
  - RSIMeanReversion         : oversold/overbought RSI entries
  - BollingerBandBreakout    : volatility breakout on BB squeeze
  - MomentumStrategy         : rate-of-change momentum

Extend by subclassing BaseStrategy.

Dependencies:
    pip install numpy pandas
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# ---------------------------------------------------------------------------
# Enums & core data structures
# ---------------------------------------------------------------------------

class Direction(str, Enum):
    LONG  = "long"
    SHORT = "short"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT  = "limit"


@dataclass
class Trade:
    entry_time:  pd.Timestamp
    exit_time:   pd.Timestamp
    direction:   Direction
    entry_price: float
    exit_price:  float
    shares:      float
    pnl:         float          # net dollar P&L after commission
    pnl_pct:     float          # percentage return on this trade
    commission:  float
    mae:         float = 0.0    # maximum adverse excursion (worst intra-trade move)
    mfe:         float = 0.0    # maximum favourable excursion (best intra-trade move)

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

    @property
    def duration(self) -> pd.Timedelta:
        return self.exit_time - self.entry_time

    def __repr__(self) -> str:
        sign = "+" if self.pnl >= 0 else ""
        return (
            f"Trade({self.direction.value} | "
            f"entry={self.entry_price:.4f} exit={self.exit_price:.4f} | "
            f"PnL={sign}{self.pnl:.2f} [{sign}{self.pnl_pct*100:.2f}%])"
        )


@dataclass
class PerformanceMetrics:
    # --- identity ---
    strategy_name:        str
    ticker:               str
    start_date:           pd.Timestamp
    end_date:             pd.Timestamp
    n_bars:               int

    # --- capital ---
    initial_capital:      float
    final_equity:         float
    total_return_pct:     float        # (final - initial) / initial
    annualised_return:    float
    cagr:                 float

    # --- trade statistics ---
    total_trades:         int
    winning_trades:       int
    losing_trades:        int
    win_rate:             float        # 0–1
    avg_win:              float
    avg_loss:             float
    profit_factor:        float        # gross profit / gross loss
    expectancy:           float        # expected $ per trade

    # --- risk / drawdown ---
    max_drawdown_pct:     float        # peak-to-trough as fraction
    max_drawdown_dollar:  float
    avg_drawdown_pct:     float
    max_drawdown_duration: Optional[pd.Timedelta]
    recovery_factor:      float        # total return / max drawdown

    # --- risk-adjusted ---
    sharpe_ratio:         float
    sortino_ratio:        float
    calmar_ratio:         float        # annualised return / max drawdown

    # --- trade timing ---
    avg_trade_duration:   Optional[pd.Timedelta]
    avg_bars_in_trade:    float

    # --- per-bar P&L ---
    equity_curve:         pd.Series    # DatetimeIndex -> cumulative equity

    def summary(self) -> str:
        dd = (
            str(self.max_drawdown_duration)
            if self.max_drawdown_duration is not None
            else "N/A"
        )
        return (
            f"\n{'='*65}\n"
            f"  BACKTEST RESULTS — {self.strategy_name} on {self.ticker}\n"
            f"{'='*65}\n"
            f"  Period         : {self.start_date.date()} -> {self.end_date.date()} ({self.n_bars} bars)\n"
            f"  Capital        : ${self.initial_capital:,.2f} -> ${self.final_equity:,.2f}\n"
            f"  Total return   : {self.total_return_pct*100:+.2f}%\n"
            f"  CAGR           : {self.cagr*100:.2f}%\n"
            f"{'-'*65}\n"
            f"  Trades         : {self.total_trades}  (W:{self.winning_trades} / L:{self.losing_trades})\n"
            f"  Win rate       : {self.win_rate*100:.1f}%\n"
            f"  Avg win        : ${self.avg_win:,.2f}   Avg loss: ${self.avg_loss:,.2f}\n"
            f"  Profit factor  : {self.profit_factor:.2f}\n"
            f"  Expectancy     : ${self.expectancy:,.2f} per trade\n"
            f"{'-'*65}\n"
            f"  Max drawdown   : {self.max_drawdown_pct*100:.2f}%  (${self.max_drawdown_dollar:,.2f})\n"
            f"  DD duration    : {dd}\n"
            f"  Recovery factor: {self.recovery_factor:.2f}\n"
            f"{'-'*65}\n"
            f"  Sharpe ratio   : {self.sharpe_ratio:.3f}\n"
            f"  Sortino ratio  : {self.sortino_ratio:.3f}\n"
            f"  Calmar ratio   : {self.calmar_ratio:.3f}\n"
            f"  Avg trade dur  : {self.avg_trade_duration}\n"
            f"{'='*65}\n"
        )

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "equity_curve"}
        for k, v in d.items():
            if isinstance(v, pd.Timestamp):
                d[k] = str(v)
            elif isinstance(v, pd.Timedelta):
                d[k] = str(v)
        return d


# ---------------------------------------------------------------------------
# Strategy base class
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    """
    Subclass this and implement `generate_signals`.

    generate_signals must return a pd.Series of float signals
    aligned with the input df index:
      +1  = go long (or stay long)
      -1  = go short (or stay short)   [if allow_short=True]
       0  = flat / close position
    """

    name: str = "BaseStrategy"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------

class MovingAverageCrossover(BaseStrategy):
    """
    Enter long when fast MA crosses above slow MA; exit (or go short)
    when fast crosses below slow.
    """

    name = "MA Crossover"

    def __init__(
        self,
        fast: int  = 10,
        slow: int  = 50,
        ma_type: str = "ema",   # "sma" or "ema"
    ):
        self.fast    = fast
        self.slow    = slow
        self.ma_type = ma_type.lower()

    def _ma(self, series: pd.Series, period: int) -> pd.Series:
        if self.ma_type == "ema":
            return series.ewm(span=period, adjust=False).mean()
        return series.rolling(period).mean()

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close  = df["close"]
        fast_m = self._ma(close, self.fast)
        slow_m = self._ma(close, self.slow)
        signal = pd.Series(0.0, index=df.index)
        signal[fast_m > slow_m] = 1.0
        signal[fast_m < slow_m] = -1.0
        return signal


class RSIMeanReversion(BaseStrategy):
    """
    Buy when RSI < oversold threshold; sell when RSI > overbought threshold.
    """

    name = "RSI Mean Reversion"

    def __init__(
        self,
        period:     int   = 14,
        oversold:   float = 30.0,
        overbought: float = 70.0,
    ):
        self.period     = period
        self.oversold   = oversold
        self.overbought = overbought

    def _rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(self.period).mean()
        loss  = (-delta.clip(upper=0)).rolling(self.period).mean()
        rs    = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rsi    = self._rsi(df["close"])
        signal = pd.Series(0.0, index=df.index)
        # enter long from oversold; flat from overbought
        position = 0.0
        for i in range(len(rsi)):
            r = rsi.iloc[i]
            if math.isnan(r):
                signal.iloc[i] = 0.0
                continue
            if r < self.oversold:
                position = 1.0
            elif r > self.overbought:
                position = 0.0
            signal.iloc[i] = position
        return signal


class BollingerBandBreakout(BaseStrategy):
    """
    Go long on close above upper band; go short on close below lower band.
    """

    name = "Bollinger Breakout"

    def __init__(self, period: int = 20, n_std: float = 2.0):
        self.period = period
        self.n_std  = n_std

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close  = df["close"]
        mid    = close.rolling(self.period).mean()
        std    = close.rolling(self.period).std()
        upper  = mid + self.n_std * std
        lower  = mid - self.n_std * std
        signal = pd.Series(0.0, index=df.index)
        signal[close > upper] =  1.0
        signal[close < lower] = -1.0
        return signal


class MomentumStrategy(BaseStrategy):
    """
    Long when rate-of-change over `lookback` bars is positive (and above
    a minimum threshold); short when negative.
    """

    name = "Momentum ROC"

    def __init__(self, lookback: int = 20, threshold: float = 0.01):
        self.lookback  = lookback
        self.threshold = threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        roc    = df["close"].pct_change(self.lookback)
        signal = pd.Series(0.0, index=df.index)
        signal[roc >  self.threshold] =  1.0
        signal[roc < -self.threshold] = -1.0
        return signal


# ---------------------------------------------------------------------------
# Backtesting engine
# ---------------------------------------------------------------------------

class Simulator:
    """
    Backtests any BaseStrategy subclass on OHLCV data.

    Parameters
    ----------
    initial_capital : starting cash in dollars
    commission_pct  : round-trip commission as a fraction of trade value
                      e.g. 0.001 = 0.1 %
    slippage_pct    : one-way slippage fraction applied to entry/exit price
    allow_short     : whether to take -1 signals as short positions
    position_size   : fraction of equity to risk per trade (0–1)
    risk_free_rate  : annualised rate for Sharpe / Sortino calculation

    Usage
    -----
    >>> sim = Simulator(initial_capital=100_000)
    >>> metrics = sim.run(df, MovingAverageCrossover(10, 50), ticker="AAPL")
    >>> print(metrics.summary())
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_pct:  float = 0.001,
        slippage_pct:    float = 0.0005,
        allow_short:     bool  = False,
        position_size:   float = 1.0,
        risk_free_rate:  float = 0.04,
    ):
        self.initial_capital = initial_capital
        self.commission_pct  = commission_pct
        self.slippage_pct    = slippage_pct
        self.allow_short     = allow_short
        self.position_size   = position_size
        self.risk_free_rate  = risk_free_rate

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        ticker: str = "UNKNOWN",
    ) -> PerformanceMetrics:
        """
        Run a full backtest.

        Parameters
        ----------
        df       : OHLCV DataFrame with DatetimeIndex
        strategy : any BaseStrategy instance
        ticker   : symbol label used in the output

        Returns
        -------
        PerformanceMetrics
        """
        df = self._prepare(df)
        signals = strategy.generate_signals(df)
        signals = signals.reindex(df.index).fillna(0)

        if not self.allow_short:
            signals = signals.clip(lower=0)

        trades, equity_curve = self._simulate(df, signals)
        metrics = self._compute_metrics(
            trades        = trades,
            equity_curve  = equity_curve,
            df            = df,
            strategy_name = strategy.name,
            ticker        = ticker,
        )
        return metrics

    def run_multiple(
        self,
        df: pd.DataFrame,
        strategies: list[BaseStrategy],
        ticker: str = "UNKNOWN",
    ) -> dict[str, PerformanceMetrics]:
        """Run a list of strategies and return {strategy_name: metrics}."""
        return {s.name: self.run(df, s, ticker=ticker) for s in strategies}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        required = {"open", "high", "low", "close", "volume"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df.dropna(subset=["open", "high", "low", "close"])

    def _fill_price(self, bar_price: float, side: int) -> float:
        """Apply slippage: longs fill slightly higher, shorts slightly lower."""
        slip = self.slippage_pct * bar_price
        return bar_price + slip if side == 1 else bar_price - slip

    def _simulate(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
    ) -> tuple[list[Trade], pd.Series]:
        """
        Walk forward bar by bar executing trades on the NEXT open
        after a signal change (to avoid look-ahead bias).
        """
        equity       = self.initial_capital
        position     = 0           # current position direction: 1, -1, 0
        entry_price  = 0.0
        entry_time   = None
        shares       = 0.0
        entry_high   = -np.inf     # track MFE/MAE
        entry_low    =  np.inf

        trades: list[Trade] = []
        equity_series = pd.Series(dtype=float, name="equity")
        equity_series[df.index[0]] = equity

        closes = df["close"].values
        opens  = df["open"].values
        highs  = df["high"].values
        lows   = df["low"].values
        sig    = signals.values
        idx    = df.index

        for i in range(1, len(df)):
            desired  = int(sig[i - 1])
            bar_open = opens[i]
            bar_high = highs[i]
            bar_low  = lows[i]

            # --- update intra-trade excursion tracking ---
            if position != 0:
                entry_high = max(entry_high, bar_high)
                entry_low  = min(entry_low,  bar_low)

            # --- close or reverse ---
            if position != 0 and desired != position:
                fill = self._fill_price(bar_open, -position)
                comm = abs(shares * fill) * self.commission_pct

                if position == 1:
                    pnl     = shares * (fill - entry_price) - comm
                    pnl_pct = (fill - entry_price) / entry_price
                    mae     = (entry_low  - entry_price) / entry_price
                    mfe     = (entry_high - entry_price) / entry_price
                else:
                    pnl     = shares * (entry_price - fill) - comm
                    pnl_pct = (entry_price - fill) / entry_price
                    mae     = (entry_price - entry_high) / entry_price
                    mfe     = (entry_price - entry_low)  / entry_price

                equity += pnl
                trades.append(Trade(
                    entry_time  = entry_time,
                    exit_time   = idx[i],
                    direction   = Direction.LONG if position == 1 else Direction.SHORT,
                    entry_price = entry_price,
                    exit_price  = fill,
                    shares      = shares,
                    pnl         = pnl,
                    pnl_pct     = pnl_pct,
                    commission  = comm,
                    mae         = mae,
                    mfe         = mfe,
                ))
                position   = 0
                entry_high = -np.inf
                entry_low  =  np.inf

            # --- open new position ---
            if desired != 0 and position == 0:
                fill        = self._fill_price(bar_open, desired)
                trade_value = equity * self.position_size
                shares      = trade_value / fill
                comm        = trade_value * self.commission_pct
                equity     -= comm
                entry_price = fill
                entry_time  = idx[i]
                position    = desired
                entry_high  = bar_high
                entry_low   = bar_low

            # mark-to-market equity (open position valued at close)
            if position == 1:
                mtm_equity = equity + shares * (closes[i] - entry_price)
            elif position == -1:
                mtm_equity = equity + shares * (entry_price - closes[i])
            else:
                mtm_equity = equity

            equity_series[idx[i]] = max(mtm_equity, 0)

        # force-close any open position at the last close
        if position != 0:
            last_close = closes[-1]
            fill       = self._fill_price(last_close, -position)
            comm       = abs(shares * fill) * self.commission_pct
            if position == 1:
                pnl     = shares * (fill - entry_price) - comm
                pnl_pct = (fill - entry_price) / entry_price
                mae     = (entry_low  - entry_price) / entry_price
                mfe     = (entry_high - entry_price) / entry_price
            else:
                pnl     = shares * (entry_price - fill) - comm
                pnl_pct = (entry_price - fill) / entry_price
                mae     = (entry_price - entry_high) / entry_price
                mfe     = (entry_price - entry_low)  / entry_price
            equity += pnl
            trades.append(Trade(
                entry_time  = entry_time,
                exit_time   = idx[-1],
                direction   = Direction.LONG if position == 1 else Direction.SHORT,
                entry_price = entry_price,
                exit_price  = fill,
                shares      = shares,
                pnl         = pnl,
                pnl_pct     = pnl_pct,
                commission  = comm,
                mae         = mae,
                mfe         = mfe,
            ))
            equity_series[idx[-1]] = equity

        return trades, equity_series

    # ------------------------------------------------------------------
    # Metrics calculation
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        trades:        list[Trade],
        equity_curve:  pd.Series,
        df:            pd.DataFrame,
        strategy_name: str,
        ticker:        str,
    ) -> PerformanceMetrics:

        equity_curve = equity_curve.sort_index()

        # --- capital ---
        initial     = self.initial_capital
        final       = float(equity_curve.iloc[-1])
        total_ret   = (final - initial) / initial

        start = equity_curve.index[0]
        end   = equity_curve.index[-1]
        years = max((end - start).days / 365.25, 1 / 365.25)
        cagr  = (final / initial) ** (1 / years) - 1

        # --- trade stats ---
        n = len(trades)
        if n == 0:
            winners = losers = 0
            avg_win = avg_loss = profit_factor = expectancy = 0.0
            win_rate = 0.0
        else:
            wins      = [t.pnl for t in trades if t.is_winner]
            losses    = [t.pnl for t in trades if not t.is_winner]
            winners   = len(wins)
            losers    = len(losses)
            win_rate  = winners / n
            avg_win   = float(np.mean(wins))   if wins   else 0.0
            avg_loss  = float(np.mean(losses)) if losses else 0.0
            gross_p   = sum(wins)
            gross_l   = abs(sum(losses))
            profit_factor = gross_p / gross_l if gross_l else float("inf")
            expectancy    = float(np.mean([t.pnl for t in trades]))

        # --- drawdown ---
        (
            max_dd_pct, max_dd_dollar, avg_dd_pct,
            max_dd_dur, dd_series
        ) = self._drawdown_analysis(equity_curve)

        recovery_factor = abs(total_ret / max_dd_pct) if max_dd_pct != 0 else float("inf")

        # --- risk-adjusted (daily returns from equity curve) ---
        daily_eq   = equity_curve.resample("D").last().dropna()
        daily_rets = daily_eq.pct_change().dropna()
        sharpe     = self._sharpe(daily_rets, self.risk_free_rate)
        sortino    = self._sortino(daily_rets, self.risk_free_rate)
        calmar     = cagr / abs(max_dd_pct) if max_dd_pct != 0 else float("inf")

        # --- trade timing ---
        if trades:
            durations     = [t.duration for t in trades]
            avg_dur       = pd.to_timedelta(np.mean([d.total_seconds() for d in durations]), unit="s")
            bar_freq      = (df.index[1] - df.index[0]).total_seconds()
            avg_bars      = avg_dur.total_seconds() / bar_freq if bar_freq else 0.0
        else:
            avg_dur  = None
            avg_bars = 0.0

        return PerformanceMetrics(
            strategy_name         = strategy_name,
            ticker                = ticker,
            start_date            = start,
            end_date              = end,
            n_bars                = len(df),
            initial_capital       = initial,
            final_equity          = round(final, 2),
            total_return_pct      = round(total_ret, 6),
            annualised_return     = round(cagr, 6),
            cagr                  = round(cagr, 6),
            total_trades          = n,
            winning_trades        = winners,
            losing_trades         = losers,
            win_rate              = round(win_rate, 4),
            avg_win               = round(avg_win, 2),
            avg_loss              = round(avg_loss, 2),
            profit_factor         = round(profit_factor, 4),
            expectancy            = round(expectancy, 2),
            max_drawdown_pct      = round(max_dd_pct, 6),
            max_drawdown_dollar   = round(max_dd_dollar, 2),
            avg_drawdown_pct      = round(avg_dd_pct, 6),
            max_drawdown_duration = max_dd_dur,
            recovery_factor       = round(recovery_factor, 4),
            sharpe_ratio          = round(sharpe, 4),
            sortino_ratio         = round(sortino, 4),
            calmar_ratio          = round(calmar, 4),
            avg_trade_duration    = avg_dur,
            avg_bars_in_trade     = round(avg_bars, 2),
            equity_curve          = equity_curve,
        )

    # ------------------------------------------------------------------
    # Drawdown analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _drawdown_analysis(
        equity: pd.Series,
    ) -> tuple[float, float, float, Optional[pd.Timedelta], pd.Series]:
        """
        Returns:
            max_dd_pct      : max peak-to-trough fraction (negative)
            max_dd_dollar   : dollar value of max drawdown
            avg_dd_pct      : average drawdown fraction across all DD periods
            max_dd_duration : longest drawdown duration (from peak to recovery)
            dd_series       : per-bar drawdown fraction series
        """
        peak    = equity.cummax()
        dd      = (equity - peak) / peak       # 0 or negative
        dd_pct  = dd.min()
        peak_eq = peak[dd.idxmin()]
        dd_dollar = float(peak_eq * abs(dd_pct))

        # average drawdown (only bars actually in drawdown)
        in_dd      = dd[dd < 0]
        avg_dd_pct = float(in_dd.mean()) if len(in_dd) else 0.0

        # max drawdown duration: longest stretch from a peak to full recovery
        max_dur: Optional[pd.Timedelta] = None
        peak_date = None
        for date, val in equity.items():
            if peak_date is None or val >= float(equity[:date].max()):
                if peak_date is not None and val >= float(equity[peak_date]):
                    dur = date - peak_date
                    if max_dur is None or dur > max_dur:
                        max_dur = dur
                peak_date = date

        return dd_pct, dd_dollar, avg_dd_pct, max_dur, dd

    # ------------------------------------------------------------------
    # Risk-adjusted metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _sharpe(daily_rets: pd.Series, rf: float) -> float:
        if daily_rets.std() == 0 or len(daily_rets) < 2:
            return 0.0
        excess = daily_rets - rf / 252
        return float((excess.mean() / excess.std()) * math.sqrt(252))

    @staticmethod
    def _sortino(daily_rets: pd.Series, rf: float) -> float:
        excess     = daily_rets - rf / 252
        downside   = excess[excess < 0]
        if len(downside) < 2 or downside.std() == 0:
            return 0.0
        return float((excess.mean() / downside.std()) * math.sqrt(252))


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def backtest(
    df: pd.DataFrame,
    strategy: BaseStrategy,
    ticker: str    = "UNKNOWN",
    capital: float = 100_000.0,
    **kwargs,
) -> PerformanceMetrics:
    """
    One-call shortcut for running a single backtest.

    Parameters
    ----------
    df       : OHLCV DataFrame with DatetimeIndex
    strategy : any BaseStrategy instance
    ticker   : symbol label
    capital  : starting capital in dollars
    **kwargs : forwarded to Simulator.__init__

    Returns
    -------
    PerformanceMetrics
    """
    return Simulator(initial_capital=capital, **kwargs).run(df, strategy, ticker=ticker)


def compare_strategies(
    df: pd.DataFrame,
    strategies: list[BaseStrategy],
    ticker: str    = "UNKNOWN",
    capital: float = 100_000.0,
    **kwargs,
) -> pd.DataFrame:
    """
    Run multiple strategies and return a comparison DataFrame.

    Returns
    -------
    pd.DataFrame  — one row per strategy, columns = key metrics
    """
    sim     = Simulator(initial_capital=capital, **kwargs)
    results = sim.run_multiple(df, strategies, ticker=ticker)
    rows    = []
    for name, m in results.items():
        rows.append({
            "strategy":           name,
            "total_return_%":     round(m.total_return_pct * 100, 2),
            "cagr_%":             round(m.cagr * 100, 2),
            "sharpe":             m.sharpe_ratio,
            "sortino":            m.sortino_ratio,
            "calmar":             m.calmar_ratio,
            "max_drawdown_%":     round(m.max_drawdown_pct * 100, 2),
            "win_rate_%":         round(m.win_rate * 100, 1),
            "profit_factor":      m.profit_factor,
            "total_trades":       m.total_trades,
            "final_equity":       m.final_equity,
        })
    return pd.DataFrame(rows).set_index("strategy")


# ---------------------------------------------------------------------------
# Demo data generator
# ---------------------------------------------------------------------------

def _generate_demo_data(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Synthetic OHLCV series with a mild upward drift and volatility clusters."""
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="1D")
    close = 100.0
    opens, highs, lows, closes, volumes = [], [], [], [], []
    vol_regime = 0.01

    for i in range(n):
        if i % 100 == 0:
            vol_regime = rng.choice([0.008, 0.012, 0.020])
        ret   = rng.normal(0.0003, vol_regime)
        o     = close
        close = max(0.1, close * (1 + ret))
        h     = max(o, close) * (1 + abs(rng.normal(0, vol_regime * 0.5)))
        l     = min(o, close) * (1 - abs(rng.normal(0, vol_regime * 0.5)))
        v     = abs(rng.normal(1_000_000, 200_000))
        opens.append(o); closes.append(close)
        highs.append(h); lows.append(l); volumes.append(v)

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running simulator demo...\n")
    df = _generate_demo_data(n=1000)

    strategies = [
        MovingAverageCrossover(fast=10, slow=50),
        RSIMeanReversion(period=14, oversold=35, overbought=65),
        BollingerBandBreakout(period=20, n_std=2.0),
        MomentumStrategy(lookback=20, threshold=0.01),
    ]

    sim = Simulator(
        initial_capital = 100_000,
        commission_pct  = 0.001,
        slippage_pct    = 0.0005,
        allow_short     = False,
    )

    # individual report
    result = sim.run(df, strategies[0], ticker="DEMO")
    print(result.summary())

    # comparison table
    print("\nStrategy comparison:")
    table = compare_strategies(df, strategies, ticker="DEMO", capital=100_000)
    print(table.to_string())
    print()
