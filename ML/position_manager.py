# =============================================================================
# MODULE: ML/position_manager.py
# PROJECT: AIGOFIN - AI Quant Trading Platform
#
# PURPOSE:
#   Professional trade risk management module.
#   Given a trade signal and account state, calculates the correct
#   position size, stop-loss, take-profit, and risk/reward metrics.
#
# DESIGN PRINCIPLES:
#   - Risk per trade is capped at a configurable % of account balance.
#   - Stop-loss distance is derived from current volatility (ATR-style).
#   - Take-profit targets a minimum 1:2 risk/reward ratio by default.
#   - All calculations are deterministic and unit-testable.
#
# INTEGRATES WITH:
#   strategy_engine.py, risk_manager.py, rl_trader.py
#
# AUTHOR: AIGOFIN System
# =============================================================================

import logging
from dataclasses import dataclass, field
from typing import Literal, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TradeParameters:
    """
    Immutable result returned by PositionManager.evaluate_trade().
    Contains everything needed to place and manage a trade.
    """
    signal: Literal["buy", "sell", "hold"]
    entry_price: float
    position_size: float          # Units / contracts / shares
    stop_loss: float              # Absolute price level
    take_profit: float            # Absolute price level
    stop_loss_distance: float     # Distance in price units from entry
    take_profit_distance: float   # Distance in price units from entry
    risk_amount: float            # Dollar amount at risk
    risk_reward: float            # Ratio e.g. 2.0 means 1:2
    account_balance: float
    risk_pct: float               # Fraction of balance at risk (e.g. 0.01)
    is_valid: bool = True         # False if trade does not meet minimum criteria
    rejection_reason: str = ""    # Populated when is_valid = False


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PositionManager:
    """
    Calculates professional-grade trade parameters using volatility-adjusted
    position sizing and fixed risk-reward discipline.

    Usage
    -----
    pm = PositionManager(risk_per_trade=0.01, risk_reward_ratio=2.0)
    params = pm.evaluate_trade(
        signal="buy",
        account_balance=100_000,
        entry_price=150.0,
        volatility=0.015,   # e.g. 1.5% daily volatility
    )
    print(params.position_size, params.stop_loss, params.take_profit)
    """

    # Default ATR multiplier for stop-loss distance
    ATR_MULTIPLIER: float = 1.5

    # Minimum stop distance as fraction of price (safety floor)
    MIN_STOP_DISTANCE_PCT: float = 0.002   # 0.2%

    # Maximum position size as fraction of balance (sanity cap)
    MAX_POSITION_SIZE_PCT: float = 0.20    # 20% of account

    def __init__(
        self,
        risk_per_trade: float = 0.01,
        risk_reward_ratio: float = 2.0,
        atr_multiplier: float = 1.5,
    ) -> None:
        """
        Parameters
        ----------
        risk_per_trade : float
            Fraction of account balance to risk per trade. Default = 0.01 (1%).
        risk_reward_ratio : float
            Minimum take-profit to stop-loss distance ratio. Default = 2.0.
        atr_multiplier : float
            Multiplier applied to volatility to derive stop-loss distance.
        """
        if not (0 < risk_per_trade <= 0.10):
            raise ValueError("risk_per_trade must be between 0 and 0.10 (10%).")
        if risk_reward_ratio < 1.0:
            raise ValueError("risk_reward_ratio must be >= 1.0.")

        self.risk_per_trade = risk_per_trade
        self.risk_reward_ratio = risk_reward_ratio
        self.ATR_MULTIPLIER = atr_multiplier

        logger.info(
            f"PositionManager ready — risk={risk_per_trade*100:.1f}%, "
            f"RR={risk_reward_ratio}:1, ATR×{atr_multiplier}"
        )

    # ------------------------------------------------------------------
    # Core calculation methods
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_distance: float,
    ) -> float:
        """
        Compute position size using the fixed-fractional method.

        Formula:
            risk_amount       = account_balance × risk_per_trade
            position_size     = risk_amount / stop_loss_distance

        The result is capped at MAX_POSITION_SIZE_PCT × balance / entry_price
        to prevent over-concentration.

        Parameters
        ----------
        account_balance : float
            Total available account equity.
        entry_price : float
            Anticipated trade entry price.
        stop_loss_distance : float
            Absolute price distance from entry to stop-loss.

        Returns
        -------
        float
            Number of units to trade (may be fractional for crypto/FX).
        """
        if stop_loss_distance <= 0:
            raise ValueError("stop_loss_distance must be positive.")

        risk_amount = account_balance * self.risk_per_trade

        # Core sizing formula
        position_size = risk_amount / stop_loss_distance

        # Apply maximum position cap
        max_units = (account_balance * self.MAX_POSITION_SIZE_PCT) / entry_price
        position_size = min(position_size, max_units)

        logger.debug(
            f"calculate_position_size: risk_amount={risk_amount:.2f}, "
            f"position_size={position_size:.4f}"
        )
        return round(position_size, 4)

    def calculate_stop_loss(
        self,
        entry_price: float,
        volatility: float,
        signal: Literal["buy", "sell"],
    ) -> tuple[float, float]:
        """
        Derive stop-loss level from volatility (ATR-equivalent).

        stop_distance = max(
            entry_price × volatility × ATR_MULTIPLIER,
            entry_price × MIN_STOP_DISTANCE_PCT
        )

        For a BUY  → stop = entry_price - stop_distance
        For a SELL → stop = entry_price + stop_distance

        Returns
        -------
        tuple[float, float]
            (stop_loss_price, stop_loss_distance)
        """
        raw_distance = entry_price * volatility * self.ATR_MULTIPLIER
        min_distance = entry_price * self.MIN_STOP_DISTANCE_PCT
        stop_distance = max(raw_distance, min_distance)

        if signal == "buy":
            stop_price = entry_price - stop_distance
        elif signal == "sell":
            stop_price = entry_price + stop_distance
        else:
            raise ValueError(f"Signal must be 'buy' or 'sell', got: {signal}")

        logger.debug(
            f"calculate_stop_loss: entry={entry_price}, "
            f"stop_distance={stop_distance:.4f}, stop_price={stop_price:.4f}"
        )
        return round(stop_price, 4), round(stop_distance, 4)

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss_distance: float,
        signal: Literal["buy", "sell"],
    ) -> tuple[float, float]:
        """
        Set take-profit at risk_reward_ratio × stop_loss_distance from entry.

        For a BUY  → take_profit = entry_price + (stop_distance × RR)
        For a SELL → take_profit = entry_price - (stop_distance × RR)

        Returns
        -------
        tuple[float, float]
            (take_profit_price, take_profit_distance)
        """
        tp_distance = stop_loss_distance * self.risk_reward_ratio

        if signal == "buy":
            tp_price = entry_price + tp_distance
        else:
            tp_price = entry_price - tp_distance

        logger.debug(
            f"calculate_take_profit: entry={entry_price}, "
            f"tp_distance={tp_distance:.4f}, tp_price={tp_price:.4f}"
        )
        return round(tp_price, 4), round(tp_distance, 4)

    # ------------------------------------------------------------------
    # Master evaluation
    # ------------------------------------------------------------------

    def evaluate_trade(
        self,
        signal: Literal["buy", "sell", "hold"],
        account_balance: float,
        entry_price: float,
        volatility: float,
        risk_per_trade: float = None,
    ) -> TradeParameters:
        """
        Full trade evaluation pipeline.

        Computes all trade parameters and returns a TradeParameters object.
        If signal = 'hold', returns an invalid TradeParameters immediately.

        Parameters
        ----------
        signal : str
            Trade direction: 'buy', 'sell', or 'hold'.
        account_balance : float
            Current account equity.
        entry_price : float
            Expected fill price.
        volatility : float
            Recent realised or implied volatility (e.g. 0.015 = 1.5%).
        risk_per_trade : float, optional
            Override instance-level risk_per_trade for this trade only.

        Returns
        -------
        TradeParameters
            Complete trade sizing result.
        """
        # --- Hold signal: no trade ---
        if signal == "hold":
            return TradeParameters(
                signal="hold",
                entry_price=entry_price,
                position_size=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                stop_loss_distance=0.0,
                take_profit_distance=0.0,
                risk_amount=0.0,
                risk_reward=0.0,
                account_balance=account_balance,
                risk_pct=0.0,
                is_valid=False,
                rejection_reason="Signal is HOLD — no position opened.",
            )

        # --- Input validation ---
        if account_balance <= 0:
            return self._rejected(signal, entry_price, account_balance,
                                  "account_balance must be positive.")
        if entry_price <= 0:
            return self._rejected(signal, entry_price, account_balance,
                                  "entry_price must be positive.")
        if volatility <= 0:
            return self._rejected(signal, entry_price, account_balance,
                                  "volatility must be positive.")

        # Use override risk if provided
        effective_risk = risk_per_trade if risk_per_trade else self.risk_per_trade

        # --- Step 1: Stop-loss ---
        stop_loss_price, stop_distance = self.calculate_stop_loss(
            entry_price, volatility, signal
        )

        # --- Step 2: Take-profit ---
        take_profit_price, tp_distance = self.calculate_take_profit(
            entry_price, stop_distance, signal
        )

        # --- Step 3: Risk amount ---
        risk_amount = account_balance * effective_risk

        # --- Step 4: Position size ---
        position_size = self.calculate_position_size(
            account_balance, entry_price, stop_distance
        )

        # --- Step 5: Actual risk/reward ratio (should equal configured RR) ---
        actual_rr = tp_distance / stop_distance if stop_distance > 0 else 0.0

        params = TradeParameters(
            signal=signal,
            entry_price=entry_price,
            position_size=position_size,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
            stop_loss_distance=stop_distance,
            take_profit_distance=tp_distance,
            risk_amount=round(risk_amount, 2),
            risk_reward=round(actual_rr, 2),
            account_balance=account_balance,
            risk_pct=effective_risk,
            is_valid=True,
        )

        logger.info(
            f"evaluate_trade [{signal.upper()}]: "
            f"entry={entry_price}, SL={stop_loss_price}, TP={take_profit_price}, "
            f"size={position_size}, risk=${risk_amount:.2f}, RR={actual_rr:.2f}"
        )
        return params

    def to_dict(self, params: TradeParameters) -> Dict[str, Any]:
        """
        Serialise TradeParameters to a plain dictionary.
        Useful for JSON responses in api_server.py.
        """
        return {
            "signal": params.signal,
            "entry_price": params.entry_price,
            "position_size": params.position_size,
            "stop_loss": params.stop_loss,
            "take_profit": params.take_profit,
            "stop_loss_distance": params.stop_loss_distance,
            "take_profit_distance": params.take_profit_distance,
            "risk_amount": params.risk_amount,
            "risk_reward": params.risk_reward,
            "account_balance": params.account_balance,
            "risk_pct": params.risk_pct,
            "is_valid": params.is_valid,
            "rejection_reason": params.rejection_reason,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rejected(
        signal: str,
        entry_price: float,
        account_balance: float,
        reason: str,
    ) -> TradeParameters:
        """Return an invalid TradeParameters with a rejection reason."""
        logger.warning(f"Trade rejected: {reason}")
        return TradeParameters(
            signal=signal,
            entry_price=entry_price,
            position_size=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            stop_loss_distance=0.0,
            take_profit_distance=0.0,
            risk_amount=0.0,
            risk_reward=0.0,
            account_balance=account_balance,
            risk_pct=0.0,
            is_valid=False,
            rejection_reason=reason,
        )
