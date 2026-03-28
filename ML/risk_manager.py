"""
risk_manager.py
===============
Quantitative Risk Management Module for Trading

Calculates
----------
- Stop Loss          : price levels to exit a losing trade
- Position Size      : units to trade given risk tolerance
- Max Risk Per Trade : dollar / % cap on a single trade
- Value at Risk      : statistical worst-case loss over a horizon

Quick start
-----------
    from risk_manager import RiskManager

    rm = RiskManager(account_balance=50_000, risk_pct=0.01)
    params = rm.full_analysis(
        entry_price  = 150.0,
        volatility   = 0.022,   # daily σ as a decimal (e.g. 2.2 %)
        direction    = "long",
    )
    rm.print_report(params)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

import numpy as np
from scipy import stats   # pip install scipy


# ────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskParameters:
    """All risk-adjusted trade parameters returned by RiskManager."""

    # ── inputs ──────────────────────────────────────────────────────────────
    account_balance:    float
    entry_price:        float
    daily_volatility:   float          # σ as a decimal
    direction:          str            # "long" | "short"
    risk_pct:           float          # fraction of account risked per trade

    # ── stop loss ───────────────────────────────────────────────────────────
    stop_loss_price:        float = 0.0
    stop_loss_pct:          float = 0.0   # % distance from entry
    stop_loss_atr_mult:     float = 0.0   # ATR multiples used

    # ── position sizing ─────────────────────────────────────────────────────
    position_size_units:    float = 0.0   # number of shares / contracts
    position_size_dollars:  float = 0.0   # notional value
    position_size_pct:      float = 0.0   # % of account

    # ── max risk per trade ───────────────────────────────────────────────────
    max_risk_dollars:       float = 0.0
    max_risk_pct:           float = 0.0
    risk_reward_ratio:      float = 0.0   # requires take_profit kwarg
    take_profit_price:      float = 0.0

    # ── value at risk ────────────────────────────────────────────────────────
    var_1d_95:              float = 0.0   # 1-day 95 % VaR (dollars)
    var_1d_99:              float = 0.0   # 1-day 99 % VaR (dollars)
    var_5d_95:              float = 0.0   # 5-day 95 % VaR (dollars)
    var_5d_99:              float = 0.0   # 5-day 99 % VaR (dollars)
    cvar_1d_95:             float = 0.0   # 1-day 95 % CVaR / Expected Shortfall
    cvar_1d_99:             float = 0.0

    # ── position-level VaR ───────────────────────────────────────────────────
    position_var_1d_95:     float = 0.0
    position_var_1d_99:     float = 0.0

    # ── portfolio heat ───────────────────────────────────────────────────────
    portfolio_heat:         float = 0.0   # % of balance at risk on this trade
    margin_of_safety:       float = 0.0   # buffer before account drawdown limit

    # ── Kelly criterion ──────────────────────────────────────────────────────
    kelly_fraction:         float = 0.0
    half_kelly_units:       float = 0.0

    # ── extra metadata ───────────────────────────────────────────────────────
    notes: list = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────────────
# Core module
# ────────────────────────────────────────────────────────────────────────────

class RiskManager:
    """
    Stateful risk manager tied to a trading account.

    Parameters
    ----------
    account_balance   : total capital in the account (dollars)
    risk_pct          : fraction of account to risk per trade (default 1 %)
    max_portfolio_heat: max fraction of account risked across all open trades
    atr_multiplier    : ATR multiples for stop-loss placement
    rr_ratio          : default reward-to-risk ratio for take-profit targets
    commission        : round-trip commission per unit (dollars)
    """

    def __init__(
        self,
        account_balance:    float = 10_000.0,
        risk_pct:           float = 0.01,       # 1 %
        max_portfolio_heat: float = 0.06,       # 6 % of account at risk simultaneously
        atr_multiplier:     float = 2.0,        # stop = entry ± 2×ATR
        rr_ratio:           float = 2.0,        # take-profit = entry + 2×risk
        commission:         float = 0.0,        # per-share commission
    ):
        if account_balance <= 0:
            raise ValueError("account_balance must be positive.")
        if not (0 < risk_pct <= 1):
            raise ValueError("risk_pct must be in (0, 1].")

        self.account_balance    = account_balance
        self.risk_pct           = risk_pct
        self.max_portfolio_heat = max_portfolio_heat
        self.atr_multiplier     = atr_multiplier
        self.rr_ratio           = rr_ratio
        self.commission         = commission

    # ── 1. Stop Loss ─────────────────────────────────────────────────────────

    def calculate_stop_loss(
        self,
        entry_price:   float,
        daily_vol:     float,
        direction:     Literal["long", "short"] = "long",
        method:        Literal["atr", "pct", "support"] = "atr",
        pct_stop:      float = 0.02,       # for method="pct"
        support_price: Optional[float] = None,  # for method="support"
    ) -> Dict:
        """
        Compute stop-loss price using one of three methods.

        Methods
        -------
        atr      : stop = entry ± atr_multiplier × daily_vol × entry_price
        pct      : stop = entry ± pct_stop × entry_price
        support  : stop = support_price (caller supplies key level)

        Returns
        -------
        dict with stop_price, stop_pct, dollar_risk_per_unit
        """
        if method == "atr":
            atr_dollar   = self.atr_multiplier * daily_vol * entry_price
            stop_distance = atr_dollar
        elif method == "pct":
            stop_distance = pct_stop * entry_price
        elif method == "support":
            if support_price is None:
                raise ValueError("support_price required for method='support'.")
            stop_distance = abs(entry_price - support_price)
        else:
            raise ValueError(f"Unknown stop method: {method!r}")

        if direction == "long":
            stop_price = entry_price - stop_distance
        else:
            stop_price = entry_price + stop_distance

        stop_pct            = stop_distance / entry_price
        dollar_risk_per_unit = stop_distance + self.commission

        return dict(
            stop_price           = round(stop_price, 4),
            stop_pct             = stop_pct,
            stop_distance_dollar = round(stop_distance, 4),
            dollar_risk_per_unit = round(dollar_risk_per_unit, 4),
            atr_mult_used        = self.atr_multiplier if method == "atr" else None,
        )

    # ── 2. Position Size ──────────────────────────────────────────────────────

    def calculate_position_size(
        self,
        entry_price:         float,
        dollar_risk_per_unit: float,
        max_risk_dollars:    Optional[float] = None,
    ) -> Dict:
        """
        Fixed-fractional position sizing.

        Position Size = Max Risk $ / Dollar Risk Per Unit

        Parameters
        ----------
        entry_price          : trade entry price
        dollar_risk_per_unit : loss per share if stop is hit
        max_risk_dollars     : override; defaults to risk_pct × balance

        Returns
        -------
        dict with units, notional_value, pct_of_account
        """
        if dollar_risk_per_unit <= 0:
            raise ValueError("dollar_risk_per_unit must be positive.")

        max_risk = max_risk_dollars if max_risk_dollars else self.account_balance * self.risk_pct

        units            = max_risk / dollar_risk_per_unit
        notional_value   = units * entry_price
        pct_of_account   = notional_value / self.account_balance

        # warn if position > 25 % of account (concentration risk)
        notes = []
        if pct_of_account > 0.25:
            notes.append(
                f"⚠ Position is {pct_of_account:.1%} of account — consider reducing size."
            )

        return dict(
            units          = round(units, 4),
            units_whole    = int(units),       # floor to whole shares
            notional_value = round(notional_value, 2),
            pct_of_account = pct_of_account,
            max_risk_used  = round(max_risk, 2),
            notes          = notes,
        )

    # ── 3. Maximum Risk Per Trade ─────────────────────────────────────────────

    def calculate_max_risk(
        self,
        custom_risk_pct: Optional[float] = None,
    ) -> Dict:
        """
        Compute the maximum dollar amount and % risked on one trade,
        and derive a take-profit target given the reward-to-risk ratio.

        Returns
        -------
        dict with max_risk_dollars, max_risk_pct, and portfolio heat check
        """
        pct            = custom_risk_pct if custom_risk_pct else self.risk_pct
        max_risk_dollars = self.account_balance * pct
        heat_check       = pct <= self.max_portfolio_heat

        notes = []
        if not heat_check:
            notes.append(
                f"⚠ Risk {pct:.2%} exceeds portfolio heat limit {self.max_portfolio_heat:.2%}."
            )

        return dict(
            max_risk_dollars     = round(max_risk_dollars, 2),
            max_risk_pct         = pct,
            rr_ratio             = self.rr_ratio,
            within_heat_limit    = heat_check,
            portfolio_heat_limit = self.max_portfolio_heat,
            notes                = notes,
        )

    def calculate_take_profit(
        self,
        entry_price:   float,
        stop_distance: float,
        direction:     Literal["long", "short"] = "long",
        rr_ratio:      Optional[float] = None,
    ) -> float:
        """Return take-profit price for a given R:R ratio."""
        rr     = rr_ratio if rr_ratio else self.rr_ratio
        target = stop_distance * rr
        if direction == "long":
            return round(entry_price + target, 4)
        return round(entry_price - target, 4)

    # ── 4. Value at Risk ──────────────────────────────────────────────────────

    def calculate_var(
        self,
        position_value: float,
        daily_vol:      float,
        horizons:       Tuple[int, ...] = (1, 5),
        confidence_levels: Tuple[float, ...] = (0.95, 0.99),
        method:         Literal["parametric", "historical", "monte_carlo"] = "parametric",
        returns_history: Optional[np.ndarray] = None,
        n_simulations:  int = 10_000,
        seed:           Optional[int] = None,
    ) -> Dict:
        """
        Value at Risk (VaR) and Conditional VaR (CVaR / Expected Shortfall).

        Methods
        -------
        parametric   : Gaussian assumption  VaR = z × σ × √t × V
        historical   : empirical quantile from returns_history
        monte_carlo  : GBM simulation

        Parameters
        ----------
        position_value : notional dollar value of the position
        daily_vol      : daily standard deviation (decimal)
        horizons       : holding periods in days
        confidence_levels: e.g. (0.95, 0.99)

        Returns
        -------
        Nested dict: results[horizon][confidence] = {var, cvar}
        """
        rng     = np.random.default_rng(seed)
        results = {}

        for t in horizons:
            results[t] = {}
            for cl in confidence_levels:
                alpha = 1 - cl

                if method == "parametric":
                    z   = stats.norm.ppf(cl)
                    var = z * daily_vol * math.sqrt(t) * position_value
                    # CVaR = E[loss | loss > VaR] under normality
                    cvar = (stats.norm.pdf(stats.norm.ppf(cl)) / alpha) \
                           * daily_vol * math.sqrt(t) * position_value

                elif method == "historical":
                    if returns_history is None:
                        raise ValueError("returns_history required for method='historical'.")
                    scaled = returns_history * math.sqrt(t)
                    losses = -scaled * position_value
                    var    = float(np.quantile(losses, cl))
                    cvar   = float(losses[losses >= var].mean()) if (losses >= var).any() else var

                elif method == "monte_carlo":
                    sim_returns = rng.normal(0, daily_vol * math.sqrt(t), n_simulations)
                    sim_losses  = -sim_returns * position_value
                    var         = float(np.quantile(sim_losses, cl))
                    cvar        = float(sim_losses[sim_losses >= var].mean()) \
                                  if (sim_losses >= var).any() else var

                else:
                    raise ValueError(f"Unknown VaR method: {method!r}")

                results[t][cl] = dict(
                    var  = round(max(var, 0), 2),
                    cvar = round(max(cvar, 0), 2),
                )

        return results

    # ── 5. Kelly Criterion ────────────────────────────────────────────────────

    def calculate_kelly(
        self,
        win_rate:       float,
        avg_win:        float,
        avg_loss:       float,
        entry_price:    float,
    ) -> Dict:
        """
        Kelly Criterion optimal bet fraction.

        f* = (p × b - q) / b
        where b = avg_win / avg_loss, p = win_rate, q = 1 - p

        Returns full-Kelly and half-Kelly position sizes.
        """
        if avg_loss <= 0:
            raise ValueError("avg_loss must be positive.")
        b = avg_win / avg_loss
        q = 1 - win_rate
        kelly_fraction = (win_rate * b - q) / b
        kelly_fraction = max(0.0, kelly_fraction)   # never negative

        kelly_dollars      = kelly_fraction * self.account_balance
        half_kelly_dollars = kelly_fraction / 2 * self.account_balance

        units_full = kelly_dollars / entry_price
        units_half = half_kelly_dollars / entry_price

        notes = []
        if kelly_fraction > 0.25:
            notes.append(
                f"⚠ Full Kelly ({kelly_fraction:.1%}) is aggressive; half-Kelly recommended."
            )
        if kelly_fraction == 0:
            notes.append("Kelly = 0: expected-value negative — do not trade this setup.")

        return dict(
            kelly_fraction       = round(kelly_fraction, 4),
            kelly_dollars        = round(kelly_dollars, 2),
            half_kelly_dollars   = round(half_kelly_dollars, 2),
            units_full_kelly     = round(units_full, 4),
            units_half_kelly     = round(units_half, 4),
            win_rate             = win_rate,
            avg_win              = avg_win,
            avg_loss             = avg_loss,
            rr_implied           = round(b, 4),
            notes                = notes,
        )

    # ── 6. Full Analysis ──────────────────────────────────────────────────────

    def full_analysis(
        self,
        entry_price:        float,
        volatility:         float,
        direction:          Literal["long", "short"] = "long",
        stop_method:        Literal["atr", "pct", "support"] = "atr",
        pct_stop:           float = 0.02,
        support_price:      Optional[float] = None,
        var_method:         Literal["parametric", "historical", "monte_carlo"] = "parametric",
        returns_history:    Optional[np.ndarray] = None,
        win_rate:           Optional[float] = None,
        avg_win:            Optional[float] = None,
        avg_loss:           Optional[float] = None,
        custom_risk_pct:    Optional[float] = None,
        rr_ratio:           Optional[float] = None,
        seed:               Optional[int] = None,
    ) -> RiskParameters:
        """
        Run the full risk pipeline and return a populated RiskParameters object.

        Parameters
        ----------
        entry_price     : intended trade entry price
        volatility      : daily volatility (σ) as a decimal fraction
        direction       : "long" or "short"
        stop_method     : "atr", "pct", or "support"
        pct_stop        : % stop distance (for method="pct")
        support_price   : key support/resistance level (for method="support")
        var_method      : VaR estimation method
        returns_history : array of daily returns (for historical VaR)
        win_rate        : historical win-rate (for Kelly)
        avg_win / avg_loss: dollar P&L averages (for Kelly)
        custom_risk_pct : override default risk_pct
        rr_ratio        : override default rr_ratio
        seed            : RNG seed for Monte Carlo

        Returns
        -------
        RiskParameters dataclass with all fields populated
        """
        risk_pct = custom_risk_pct if custom_risk_pct else self.risk_pct
        rr       = rr_ratio if rr_ratio else self.rr_ratio

        # ── stop loss ──────────────────────────────────────────────────────
        sl = self.calculate_stop_loss(
            entry_price   = entry_price,
            daily_vol     = volatility,
            direction     = direction,
            method        = stop_method,
            pct_stop      = pct_stop,
            support_price = support_price,
        )

        # ── max risk ───────────────────────────────────────────────────────
        mr = self.calculate_max_risk(custom_risk_pct=risk_pct)

        # ── position size ──────────────────────────────────────────────────
        ps = self.calculate_position_size(
            entry_price           = entry_price,
            dollar_risk_per_unit  = sl["dollar_risk_per_unit"],
            max_risk_dollars      = mr["max_risk_dollars"],
        )

        # ── take profit ────────────────────────────────────────────────────
        tp_price = self.calculate_take_profit(
            entry_price   = entry_price,
            stop_distance = sl["stop_distance_dollar"],
            direction     = direction,
            rr_ratio      = rr,
        )

        # ── VaR ────────────────────────────────────────────────────────────
        var_results = self.calculate_var(
            position_value   = ps["notional_value"],
            daily_vol        = volatility,
            horizons         = (1, 5),
            confidence_levels= (0.95, 0.99),
            method           = var_method,
            returns_history  = returns_history,
            seed             = seed,
        )

        # ── account-level VaR (full balance) ───────────────────────────────
        acct_var = self.calculate_var(
            position_value   = self.account_balance,
            daily_vol        = volatility,
            horizons         = (1,),
            confidence_levels= (0.95, 0.99),
            method           = var_method,
            returns_history  = returns_history,
            seed             = seed,
        )

        # ── Kelly (optional) ───────────────────────────────────────────────
        kelly_fraction = 0.0
        half_kelly_units = 0.0
        if win_rate is not None and avg_win is not None and avg_loss is not None:
            kelly = self.calculate_kelly(win_rate, avg_win, avg_loss, entry_price)
            kelly_fraction   = kelly["kelly_fraction"]
            half_kelly_units = kelly["units_half_kelly"]

        # ── portfolio heat ─────────────────────────────────────────────────
        portfolio_heat  = (mr["max_risk_dollars"] / self.account_balance)
        margin_of_safety = self.max_portfolio_heat - portfolio_heat

        # ── assemble ───────────────────────────────────────────────────────
        notes = sl.get("notes", []) + ps["notes"] + mr["notes"]

        p = RiskParameters(
            account_balance       = self.account_balance,
            entry_price           = entry_price,
            daily_volatility      = volatility,
            direction             = direction,
            risk_pct              = risk_pct,
            # stop loss
            stop_loss_price       = sl["stop_price"],
            stop_loss_pct         = sl["stop_pct"],
            stop_loss_atr_mult    = sl["atr_mult_used"] or 0.0,
            # position size
            position_size_units   = ps["units"],
            position_size_dollars = ps["notional_value"],
            position_size_pct     = ps["pct_of_account"],
            # max risk
            max_risk_dollars      = mr["max_risk_dollars"],
            max_risk_pct          = mr["max_risk_pct"],
            risk_reward_ratio     = rr,
            take_profit_price     = tp_price,
            # VaR (position-level)
            var_1d_95             = var_results[1][0.95]["var"],
            var_1d_99             = var_results[1][0.99]["var"],
            var_5d_95             = var_results[5][0.95]["var"],
            var_5d_99             = var_results[5][0.99]["var"],
            cvar_1d_95            = var_results[1][0.95]["cvar"],
            cvar_1d_99            = var_results[1][0.99]["cvar"],
            # VaR (account-level, used for portfolio risk)
            position_var_1d_95    = acct_var[1][0.95]["var"],
            position_var_1d_99    = acct_var[1][0.99]["var"],
            # portfolio
            portfolio_heat        = portfolio_heat,
            margin_of_safety      = margin_of_safety,
            # kelly
            kelly_fraction        = kelly_fraction,
            half_kelly_units      = half_kelly_units,
            notes                 = notes,
        )
        return p

    # ── Pretty Printer ────────────────────────────────────────────────────────

    @staticmethod
    def print_report(p: RiskParameters) -> None:
        """Print a formatted risk report to stdout."""
        bar  = "═" * 64
        dash = "─" * 64

        def row(label: str, value: str) -> str:
            return f"  {label:<32} {value}"

        dir_arrow = "▲ LONG" if p.direction == "long" else "▼ SHORT"

        print(f"\n{bar}")
        print(f"  RISK MANAGEMENT REPORT          {dir_arrow}")
        print(bar)
        print(row("Account Balance",   f"${p.account_balance:>12,.2f}"))
        print(row("Entry Price",        f"${p.entry_price:>12,.4f}"))
        print(row("Daily Volatility",   f"{p.daily_volatility:>12.2%}"))
        print(row("Risk Per Trade",     f"{p.risk_pct:>12.2%}"))
        print(dash)

        print("  STOP LOSS")
        print(row("  Stop Price",       f"${p.stop_loss_price:>12,.4f}"))
        print(row("  Stop Distance",    f"{p.stop_loss_pct:>12.2%}"))
        if p.stop_loss_atr_mult:
            print(row("  ATR Multiplier",  f"{p.stop_loss_atr_mult:>12.1f}×"))
        print(dash)

        print("  POSITION SIZING")
        print(row("  Units (fractional)",f"{p.position_size_units:>12,.4f}"))
        print(row("  Units (whole)",     f"{int(p.position_size_units):>12,}"))
        print(row("  Notional Value",    f"${p.position_size_dollars:>12,.2f}"))
        print(row("  % of Account",      f"{p.position_size_pct:>12.2%}"))
        print(dash)

        print("  MAX RISK PER TRADE")
        print(row("  Max Risk ($)",      f"${p.max_risk_dollars:>12,.2f}"))
        print(row("  Max Risk (%)",      f"{p.max_risk_pct:>12.2%}"))
        print(row("  Take Profit Price", f"${p.take_profit_price:>12,.4f}"))
        print(row("  Reward : Risk",     f"{p.risk_reward_ratio:>12.1f} : 1"))
        print(row("  Portfolio Heat",    f"{p.portfolio_heat:>12.2%}"))
        print(row("  Heat Margin Left",  f"{p.margin_of_safety:>12.2%}"))
        print(dash)

        print("  VALUE AT RISK  (position notional)")
        print(row("  1-Day VaR  95%",   f"${p.var_1d_95:>12,.2f}"))
        print(row("  1-Day VaR  99%",   f"${p.var_1d_99:>12,.2f}"))
        print(row("  5-Day VaR  95%",   f"${p.var_5d_95:>12,.2f}"))
        print(row("  5-Day VaR  99%",   f"${p.var_5d_99:>12,.2f}"))
        print(row("  1-Day CVaR 95%",   f"${p.cvar_1d_95:>12,.2f}"))
        print(row("  1-Day CVaR 99%",   f"${p.cvar_1d_99:>12,.2f}"))
        print(dash)

        if p.kelly_fraction > 0:
            print("  KELLY CRITERION")
            print(row("  Kelly Fraction",    f"{p.kelly_fraction:>12.2%}"))
            print(row("  Half-Kelly Units",  f"{p.half_kelly_units:>12,.4f}"))
            print(dash)

        if p.notes:
            print("  NOTES")
            for note in p.notes:
                print(f"  {note}")
            print(dash)

        print()


# ────────────────────────────────────────────────────────────────────────────
# Convenience function (module-level)
# ────────────────────────────────────────────────────────────────────────────

def analyze(
    account_balance: float,
    entry_price:     float,
    volatility:      float,
    direction:       Literal["long", "short"] = "long",
    risk_pct:        float = 0.01,
    **kwargs,
) -> RiskParameters:
    """
    One-liner risk analysis.

    Example
    -------
        from risk_manager import analyze
        p = analyze(50_000, 150.0, 0.022)
    """
    rm = RiskManager(account_balance=account_balance, risk_pct=risk_pct)
    return rm.full_analysis(entry_price=entry_price, volatility=volatility,
                            direction=direction, **kwargs)


# ────────────────────────────────────────────────────────────────────────────
# CLI demo
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 64)
    print("  RISK MANAGER — DEMO")
    print("=" * 64)

    # ── scenario 1: standard long trade ────────────────────────────────────
    print("\n[Scenario 1] Standard long trade — ATR stop")
    rm = RiskManager(
        account_balance    = 50_000,
        risk_pct           = 0.01,
        atr_multiplier     = 2.0,
        rr_ratio           = 2.5,
    )
    p1 = rm.full_analysis(
        entry_price = 150.0,
        volatility  = 0.022,
        direction   = "long",
        win_rate    = 0.55,
        avg_win     = 300.0,
        avg_loss    = 150.0,
    )
    rm.print_report(p1)

    # ── scenario 2: short trade with % stop ────────────────────────────────
    print("[Scenario 2] Short trade — percentage stop")
    rm2 = RiskManager(
        account_balance = 100_000,
        risk_pct        = 0.005,
        rr_ratio        = 3.0,
    )
    p2 = rm2.full_analysis(
        entry_price  = 220.0,
        volatility   = 0.018,
        direction    = "short",
        stop_method  = "pct",
        pct_stop     = 0.025,
    )
    rm2.print_report(p2)

    # ── scenario 3: historical VaR ─────────────────────────────────────────
    print("[Scenario 3] Monte Carlo VaR")
    rng      = np.random.default_rng(42)
    fake_ret = rng.normal(0.0005, 0.02, 1000)
    rm3 = RiskManager(account_balance=25_000, risk_pct=0.02)
    p3  = rm3.full_analysis(
        entry_price     = 80.0,
        volatility      = 0.02,
        direction       = "long",
        var_method      = "monte_carlo",
        returns_history = fake_ret,
        seed            = 0,
    )
    rm3.print_report(p3)
