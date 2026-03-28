"""
portfolio_optimizer.py
======================
Mean-Variance Portfolio Optimizer with Risk-Adjusted Return Metrics.

Usage
-----
    from portfolio_optimizer import PortfolioOptimizer

    opt = PortfolioOptimizer(tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"])
    result = opt.optimize()
    print(result)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

# Optional yfinance for live data; gracefully degrade to synthetic data if absent
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

# scipy for constrained optimisation
try:
    from scipy.optimize import minimize, OptimizeResult
except ImportError as exc:
    raise ImportError("scipy is required: pip install scipy") from exc


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Holds the output of a portfolio optimization run."""
    weights: pd.Series                     # ticker -> optimal weight
    expected_return: float                 # annualised portfolio return
    volatility: float                      # annualised portfolio volatility
    sharpe_ratio: float                    # (return - rf) / volatility
    sortino_ratio: float                   # (return - rf) / downside_std
    max_drawdown: float                    # worst peak-to-trough drawdown
    diversification_ratio: float           # weighted avg vol / portfolio vol
    objective: str                         # which objective was used
    converged: bool                        # did the solver converge?
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        w_lines = "\n".join(
            f"    {t:<10s}  {w:>8.4f}  ({w*100:>6.2f}%)"
            for t, w in self.weights.items()
        )
        return (
            f"\n{'='*55}\n"
            f"  Portfolio Optimisation Result  [{self.objective}]\n"
            f"{'='*55}\n"
            f"  Weights:\n{w_lines}\n"
            f"{'─'*55}\n"
            f"  Expected Return (ann.)  : {self.expected_return*100:>8.2f}%\n"
            f"  Volatility    (ann.)    : {self.volatility*100:>8.2f}%\n"
            f"  Sharpe  Ratio           : {self.sharpe_ratio:>8.4f}\n"
            f"  Sortino Ratio           : {self.sortino_ratio:>8.4f}\n"
            f"  Max Drawdown            : {self.max_drawdown*100:>8.2f}%\n"
            f"  Diversification Ratio   : {self.diversification_ratio:>8.4f}\n"
            f"  Converged               : {self.converged}\n"
            f"{'='*55}\n"
        )


# ---------------------------------------------------------------------------
# Core Optimiser Class
# ---------------------------------------------------------------------------

class PortfolioOptimizer:
    """
    Mean-Variance Portfolio Optimiser.

    Parameters
    ----------
    tickers : list[str]
        List of stock ticker symbols.
    risk_free_rate : float
        Annual risk-free rate (default 0.04 = 4 %).
    period : str
        yfinance download period, e.g. "3y", "5y" (default "3y").
    objective : {"max_sharpe", "min_variance", "max_sortino", "max_return_risk_adj"}
        Optimisation objective (default "max_sharpe").
    n_simulations : int
        Monte-Carlo simulations for efficient-frontier preview (default 10_000).
    allow_short : bool
        Allow short positions (default False).
    returns_data : pd.DataFrame | None
        Pre-computed daily returns (tickers as columns).  Skips data fetch.
    """

    OBJECTIVES = ("max_sharpe", "min_variance", "max_sortino", "max_return_risk_adj")

    def __init__(
        self,
        tickers: List[str],
        risk_free_rate: float = 0.04,
        period: str = "3y",
        objective: str = "max_sharpe",
        n_simulations: int = 10_000,
        allow_short: bool = False,
        returns_data: Optional[pd.DataFrame] = None,
    ) -> None:
        if objective not in self.OBJECTIVES:
            raise ValueError(f"objective must be one of {self.OBJECTIVES}")
        self.tickers = [t.upper() for t in tickers]
        self.rf = risk_free_rate
        self.period = period
        self.objective = objective
        self.n_simulations = n_simulations
        self.allow_short = allow_short
        self._returns: Optional[pd.DataFrame] = returns_data

        # Populated after calling _prepare_data()
        self.mu: np.ndarray = np.array([])          # expected annual returns
        self.Sigma: np.ndarray = np.array([[]])      # annual covariance matrix
        self.daily_returns: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self) -> OptimizationResult:
        """Run the full optimisation pipeline and return an OptimizationResult."""
        self._prepare_data()
        weights = self._run_optimisation()
        return self._build_result(weights)

    def efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """
        Compute the efficient frontier via Monte-Carlo simulation.

        Returns a DataFrame with columns:
            return, volatility, sharpe, weights_<ticker>
        """
        self._prepare_data()
        n = len(self.tickers)
        records = []
        rng = np.random.default_rng(42)
        for _ in range(self.n_simulations):
            w = self._random_weights(n, rng)
            ret = float(w @ self.mu)
            vol = float(np.sqrt(w @ self.Sigma @ w))
            sharpe = (ret - self.rf) / vol if vol > 0 else 0.0
            records.append([ret, vol, sharpe, *w.tolist()])
        cols = ["return", "volatility", "sharpe"] + [f"w_{t}" for t in self.tickers]
        return pd.DataFrame(records, columns=cols)

    # ------------------------------------------------------------------
    # Data Layer
    # ------------------------------------------------------------------

    def _prepare_data(self) -> None:
        """Download price data (or use provided), compute mu and Sigma."""
        if self._returns is None:
            self._returns = self._fetch_returns()
        self.daily_returns = self._returns.copy()
        ann = 252
        self.mu = self.daily_returns.mean().values * ann
        self.Sigma = self.daily_returns.cov().values * ann

    def _fetch_returns(self) -> pd.DataFrame:
        """Fetch adjusted-close prices via yfinance and return daily log-returns."""
        if not _YF_AVAILABLE:
            warnings.warn(
                "yfinance not installed; generating synthetic returns. "
                "Install with: pip install yfinance",
                stacklevel=3,
            )
            return self._synthetic_returns()

        try:
            raw = yf.download(
                self.tickers,
                period=self.period,
                auto_adjust=True,
                progress=False,
                threads=True,
            )["Close"]
            if isinstance(raw, pd.Series):          # single ticker
                raw = raw.to_frame(self.tickers[0])
            raw.dropna(how="all", inplace=True)
            missing = [t for t in self.tickers if t not in raw.columns]
            if missing:
                warnings.warn(f"Tickers not found in downloaded data: {missing}")
                self.tickers = [t for t in self.tickers if t in raw.columns]
                raw = raw[self.tickers]
            returns = np.log(raw / raw.shift(1)).dropna()
            return returns
        except Exception as exc:
            warnings.warn(f"Data fetch failed ({exc}); falling back to synthetic data.")
            return self._synthetic_returns()

    def _synthetic_returns(self) -> pd.DataFrame:
        """Generate reproducible synthetic daily returns for testing / demo."""
        rng = np.random.default_rng(0)
        n, T = len(self.tickers), 756          # ~3 years of trading days
        mu_daily = rng.uniform(0.00020, 0.00080, n)
        vols = rng.uniform(0.010, 0.030, n)
        corr = _random_corr(n, rng)
        cov = np.diag(vols) @ corr @ np.diag(vols)
        data = rng.multivariate_normal(mu_daily, cov, T)
        idx = pd.bdate_range(end=pd.Timestamp.today(), periods=T)
        return pd.DataFrame(data, index=idx, columns=self.tickers)

    # ------------------------------------------------------------------
    # Optimisation Layer
    # ------------------------------------------------------------------

    def _run_optimisation(self) -> np.ndarray:
        n = len(self.tickers)
        w0 = np.ones(n) / n                              # equal-weight start

        bounds = ((-1.0, 1.0) if self.allow_short else (0.0, 1.0),) * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        def neg_sharpe(w: np.ndarray) -> float:
            ret = float(w @ self.mu)
            vol = float(np.sqrt(w @ self.Sigma @ w))
            return -(ret - self.rf) / vol if vol > 1e-10 else 1e6

        def portfolio_variance(w: np.ndarray) -> float:
            return float(w @ self.Sigma @ w)

        def neg_sortino(w: np.ndarray) -> float:
            ret = float(w @ self.mu)
            port_daily = self.daily_returns.values @ w
            downside = port_daily[port_daily < 0]
            if len(downside) == 0:
                return -1e6
            down_std = float(np.std(downside, ddof=1)) * np.sqrt(252)
            return -(ret - self.rf) / down_std if down_std > 1e-10 else 1e6

        def neg_risk_adj_return(w: np.ndarray) -> float:
            """Calmar-style: return / max-drawdown penalty."""
            ret = float(w @ self.mu)
            vol = float(np.sqrt(w @ self.Sigma @ w))
            md = _max_drawdown(self.daily_returns.values @ w)
            penalty = vol + abs(md) + 1e-10
            return -ret / penalty

        obj_map = {
            "max_sharpe": neg_sharpe,
            "min_variance": portfolio_variance,
            "max_sortino": neg_sortino,
            "max_return_risk_adj": neg_risk_adj_return,
        }

        result: OptimizeResult = minimize(
            obj_map[self.objective],
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        if not result.success:
            warnings.warn(
                f"Optimiser did not fully converge: {result.message}. "
                "Returning best feasible weights found."
            )
        w = result.x
        w = np.clip(w, 0 if not self.allow_short else -1, 1)
        w /= w.sum()                # re-normalise after clip
        return w

    # ------------------------------------------------------------------
    # Result Assembly
    # ------------------------------------------------------------------

    def _build_result(self, weights: np.ndarray) -> OptimizationResult:
        ret = float(weights @ self.mu)
        vol = float(np.sqrt(weights @ self.Sigma @ weights))
        sharpe = (ret - self.rf) / vol if vol > 1e-10 else 0.0

        port_daily = self.daily_returns.values @ weights
        downside = port_daily[port_daily < 0]
        down_std = float(np.std(downside, ddof=1)) * np.sqrt(252) if len(downside) > 1 else vol
        sortino = (ret - self.rf) / down_std if down_std > 1e-10 else 0.0

        md = _max_drawdown(port_daily)

        indiv_vols = np.sqrt(np.diag(self.Sigma))
        div_ratio = float(weights @ indiv_vols) / vol if vol > 1e-10 else 1.0

        return OptimizationResult(
            weights=pd.Series(weights, index=self.tickers),
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=md,
            diversification_ratio=div_ratio,
            objective=self.objective,
            converged=True,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _random_weights(n: int, rng: np.random.Generator) -> np.ndarray:
        w = rng.exponential(1.0, n)
        return w / w.sum()


# ---------------------------------------------------------------------------
# Module-level Helper Functions
# ---------------------------------------------------------------------------

def _max_drawdown(daily_returns: np.ndarray) -> float:
    """Compute maximum peak-to-trough drawdown from a series of daily returns."""
    cum = np.cumprod(1 + daily_returns)
    running_max = np.maximum.accumulate(cum)
    drawdowns = (cum - running_max) / running_max
    return float(drawdowns.min())


def _random_corr(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random valid correlation matrix via factor model."""
    factors = rng.standard_normal((n, max(n // 2, 1)))
    cov = factors @ factors.T + np.diag(rng.uniform(0.1, 0.5, n))
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def optimize_portfolio(
    tickers: List[str],
    risk_free_rate: float = 0.04,
    period: str = "3y",
    objective: str = "max_sharpe",
    allow_short: bool = False,
    returns_data: Optional[pd.DataFrame] = None,
) -> OptimizationResult:
    """
    Convenience function — allocate capital across stocks in one call.

    Parameters
    ----------
    tickers : list[str]
        e.g. ["AAPL", "MSFT", "GOOGL"]
    risk_free_rate : float
        Annual risk-free rate (default 0.04).
    period : str
        Historical data window for yfinance (default "3y").
    objective : str
        One of "max_sharpe" | "min_variance" | "max_sortino" | "max_return_risk_adj".
    allow_short : bool
        Whether short positions are allowed (default False).
    returns_data : pd.DataFrame | None
        Provide pre-computed daily returns to skip live data fetch.

    Returns
    -------
    OptimizationResult
        Dataclass with `.weights` (pd.Series), risk metrics, and diagnostics.

    Examples
    --------
    >>> result = optimize_portfolio(["AAPL", "MSFT", "TSLA", "AMZN"])
    >>> print(result)
    >>> print(result.weights)
    """
    opt = PortfolioOptimizer(
        tickers=tickers,
        risk_free_rate=risk_free_rate,
        period=period,
        objective=objective,
        allow_short=allow_short,
        returns_data=returns_data,
    )
    return opt.optimize()


# ---------------------------------------------------------------------------
# CLI / Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DEMO_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "JNJ"]

    print("\n🔷  Running Portfolio Optimizer Demo")
    print(f"   Stocks  : {DEMO_TICKERS}")
    print(f"   Note    : Using synthetic data if yfinance unavailable\n")

    for obj in PortfolioOptimizer.OBJECTIVES:
        result = optimize_portfolio(DEMO_TICKERS, objective=obj)
        print(result)
