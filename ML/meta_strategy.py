# =============================================================================
# MODULE: ML/meta_strategy.py
# PROJECT: AIGOFIN - AI Quant Trading Platform
#
# PURPOSE:
#   A rule-based + scoring meta-layer that selects the optimal trading
#   strategy for the current market regime.
#
#   Instead of using a single strategy for all conditions, MetaStrategyAI
#   reads market regime signals and returns the best-fit strategy with a
#   confidence score and human-readable reasoning.
#
# INPUTS:
#   market_regime, volatility, trend_strength, volume, sentiment_score
#
# OUTPUTS:
#   selected_strategy, confidence_score, reasoning
#
# SUPPORTED REGIMES:
#   bull_market, bear_market, sideways_market, high_volatility
#
# AVAILABLE STRATEGIES:
#   momentum_strategy, mean_reversion_strategy, trend_following_strategy,
#   breakout_strategy, scalping_strategy
#
# INTEGRATES WITH:
#   market_regime_detector.py, strategy_engine.py, sentiment_engine.py
#
# AUTHOR: AIGOFIN System
# =============================================================================

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

MarketRegime = Literal[
    "bull_market",
    "bear_market",
    "sideways_market",
    "high_volatility",
    "unknown",
]

StrategyName = Literal[
    "momentum_strategy",
    "mean_reversion_strategy",
    "trend_following_strategy",
    "breakout_strategy",
    "scalping_strategy",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MarketCondition:
    """
    Snapshot of current market state fed into MetaStrategyAI.
    All fields are normalised to consistent scales.
    """
    market_regime: MarketRegime
    volatility: float          # 0.0 – 1.0+  (e.g. 0.015 = 1.5% daily vol)
    trend_strength: float      # -1.0 (strong down) to +1.0 (strong up)
    volume: float              # Volume ratio vs average (e.g. 1.3 = 30% above avg)
    sentiment_score: float     # -1.0 (very bearish) to +1.0 (very bullish)


@dataclass
class StrategySelection:
    """
    Result returned by MetaStrategyAI.select_best_strategy().
    """
    selected_strategy: StrategyName
    confidence_score: float            # 0.0 – 1.0
    reasoning: str                     # Human-readable explanation
    ranked_strategies: List[Tuple[StrategyName, float]] = field(
        default_factory=list
    )  # All strategies with their scores, sorted desc


# ---------------------------------------------------------------------------
# MetaStrategyAI
# ---------------------------------------------------------------------------

class MetaStrategyAI:
    """
    Regime-aware strategy selector.

    Each strategy has a base affinity score per market regime. Additional
    market signals (volatility, trend strength, volume, sentiment) provide
    dynamic adjustments to the base scores.

    The strategy with the highest aggregate score is selected.

    Design philosophy
    -----------------
    * Deterministic and auditable — every score delta is logged.
    * No ML model required; purely rule-based scoring ensures reliability
      during live trading while remaining extensible to learned weights.
    * Confidence is the normalised score of the winner vs the second-best.
    """

    # ------------------------------------------------------------------
    # Base affinity matrix: regime × strategy → score (0–10)
    # Higher = more suitable for that regime
    # ------------------------------------------------------------------
    REGIME_AFFINITY: Dict[MarketRegime, Dict[StrategyName, float]] = {
        "bull_market": {
            "momentum_strategy": 9.0,
            "trend_following_strategy": 8.0,
            "breakout_strategy": 7.0,
            "mean_reversion_strategy": 3.0,
            "scalping_strategy": 4.0,
        },
        "bear_market": {
            "momentum_strategy": 6.0,        # Short-side momentum
            "trend_following_strategy": 7.0,  # Downtrend following
            "breakout_strategy": 5.0,
            "mean_reversion_strategy": 4.0,
            "scalping_strategy": 5.0,
        },
        "sideways_market": {
            "momentum_strategy": 2.0,
            "trend_following_strategy": 2.0,
            "breakout_strategy": 5.0,
            "mean_reversion_strategy": 9.0,
            "scalping_strategy": 8.0,
        },
        "high_volatility": {
            "momentum_strategy": 4.0,
            "trend_following_strategy": 3.0,
            "breakout_strategy": 8.0,
            "mean_reversion_strategy": 2.0,
            "scalping_strategy": 7.0,
        },
        "unknown": {
            "momentum_strategy": 5.0,
            "trend_following_strategy": 5.0,
            "breakout_strategy": 5.0,
            "mean_reversion_strategy": 5.0,
            "scalping_strategy": 5.0,
        },
    }

    # Normalised volatility threshold above which we consider it "high"
    HIGH_VOL_THRESHOLD: float = 0.03   # 3% daily vol

    # Trend strength threshold above which momentum / trend strategies excel
    STRONG_TREND_THRESHOLD: float = 0.3

    # Volume spike threshold for breakout confirmation
    VOLUME_SPIKE_THRESHOLD: float = 1.5

    def __init__(self) -> None:
        self._all_strategies: List[StrategyName] = list(
            self.REGIME_AFFINITY["bull_market"].keys()
        )
        logger.info("MetaStrategyAI initialised.")

    # ------------------------------------------------------------------
    # Step 1: Detect / refine market condition
    # ------------------------------------------------------------------

    def detect_market_condition(
        self,
        market_regime: MarketRegime,
        volatility: float,
        trend_strength: float,
        volume: float,
        sentiment_score: float,
    ) -> MarketCondition:
        """
        Bundle raw inputs into a MarketCondition and validate ranges.

        Also applies a secondary regime override: if the caller provides
        a non-high_volatility regime but the volatility reading is extreme,
        this method upgrades the regime to 'high_volatility' automatically.

        Parameters
        ----------
        market_regime : str
            Primary regime label from market_regime_detector.py.
        volatility : float
            Realised/implied volatility as a decimal (e.g. 0.015).
        trend_strength : float
            EMA-gap derived trend signal in [-1, 1].
        volume : float
            Volume ratio vs rolling average.
        sentiment_score : float
            Aggregated sentiment in [-1, 1].

        Returns
        -------
        MarketCondition
        """
        # Clamp to valid ranges
        trend_strength = float(np.clip(trend_strength, -1.0, 1.0))
        sentiment_score = float(np.clip(sentiment_score, -1.0, 1.0))
        volatility = max(0.0, float(volatility))
        volume = max(0.0, float(volume))

        # Auto-upgrade regime if volatility is extreme
        effective_regime: MarketRegime = market_regime
        if (
            market_regime not in ("high_volatility", "unknown")
            and volatility >= self.HIGH_VOL_THRESHOLD * 1.5
        ):
            effective_regime = "high_volatility"
            logger.info(
                f"detect_market_condition: regime overridden to "
                f"'high_volatility' due to extreme vol ({volatility:.4f})."
            )

        condition = MarketCondition(
            market_regime=effective_regime,
            volatility=volatility,
            trend_strength=trend_strength,
            volume=volume,
            sentiment_score=sentiment_score,
        )
        logger.debug(f"MarketCondition: {condition}")
        return condition

    # ------------------------------------------------------------------
    # Step 2: Score all strategies for a given condition
    # ------------------------------------------------------------------

    def score_strategies(
        self, condition: MarketCondition
    ) -> Dict[StrategyName, float]:
        """
        Compute a composite score for each strategy given the market
        condition. Combines regime affinity with dynamic signal adjustments.

        Parameters
        ----------
        condition : MarketCondition

        Returns
        -------
        dict
            {strategy_name: score (0.0 – 10.0+)}
        """
        scores: Dict[StrategyName, float] = {}

        # --- Base scores from regime affinity table ---
        regime = condition.market_regime
        if regime not in self.REGIME_AFFINITY:
            regime = "unknown"
        base = self.REGIME_AFFINITY[regime]

        for strategy in self._all_strategies:
            score = base.get(strategy, 5.0)
            adjustments: List[str] = []

            # ── Volatility adjustments ─────────────────────────────────
            if condition.volatility >= self.HIGH_VOL_THRESHOLD:
                # High vol → favours breakout and scalping
                if strategy in ("breakout_strategy", "scalping_strategy"):
                    delta = min(2.0, (condition.volatility / self.HIGH_VOL_THRESHOLD) * 1.0)
                    score += delta
                    adjustments.append(f"high_vol +{delta:.2f}")
                # High vol → penalises trend following and mean reversion
                elif strategy in ("trend_following_strategy", "mean_reversion_strategy"):
                    score -= 1.0
                    adjustments.append("high_vol -1.00")

            # ── Trend strength adjustments ─────────────────────────────
            abs_trend = abs(condition.trend_strength)
            if abs_trend >= self.STRONG_TREND_THRESHOLD:
                if strategy in ("momentum_strategy", "trend_following_strategy"):
                    delta = abs_trend * 2.0
                    score += delta
                    adjustments.append(f"strong_trend +{delta:.2f}")
                elif strategy == "mean_reversion_strategy":
                    score -= abs_trend * 2.0
                    adjustments.append(f"strong_trend -{abs_trend*2:.2f}")

            # ── Volume spike adjustments ────────────────────────────────
            if condition.volume >= self.VOLUME_SPIKE_THRESHOLD:
                if strategy in ("breakout_strategy", "momentum_strategy"):
                    delta = min(1.5, (condition.volume - 1.0) * 0.8)
                    score += delta
                    adjustments.append(f"volume_spike +{delta:.2f}")

            # ── Sentiment adjustments ───────────────────────────────────
            # Positive sentiment boosts momentum; negative boosts short strategies
            if condition.sentiment_score > 0.3:
                if strategy == "momentum_strategy":
                    score += condition.sentiment_score * 1.5
                    adjustments.append(
                        f"bullish_sentiment +{condition.sentiment_score*1.5:.2f}"
                    )
            elif condition.sentiment_score < -0.3:
                if strategy == "mean_reversion_strategy":
                    score += abs(condition.sentiment_score) * 0.8
                    adjustments.append(
                        f"bearish_sentiment +{abs(condition.sentiment_score)*0.8:.2f}"
                    )

            # Clamp final score to a sensible range
            scores[strategy] = round(max(0.0, min(15.0, score)), 3)
            logger.debug(
                f"  {strategy}: {scores[strategy]} "
                f"(base={base.get(strategy, 5)}, adj={adjustments})"
            )

        return scores

    # ------------------------------------------------------------------
    # Step 3: Select the best strategy
    # ------------------------------------------------------------------

    def select_best_strategy(
        self, condition: MarketCondition
    ) -> StrategySelection:
        """
        Run scoring and return a StrategySelection with the winning
        strategy, confidence, and reasoning.

        Parameters
        ----------
        condition : MarketCondition

        Returns
        -------
        StrategySelection
        """
        scores = self.score_strategies(condition)

        # Rank strategies highest to lowest
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_name, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        # Confidence = normalised margin between 1st and 2nd
        score_range = max(best_score - second_score, 0.0)
        confidence = round(min(score_range / 5.0, 1.0), 3)  # Cap at 1.0

        reasoning = self._build_reasoning(
            best_name, best_score, condition, ranked
        )

        selection = StrategySelection(
            selected_strategy=best_name,
            confidence_score=confidence,
            reasoning=reasoning,
            ranked_strategies=[(name, score) for name, score in ranked],
        )

        logger.info(
            f"select_best_strategy → {best_name} "
            f"(score={best_score}, confidence={confidence})"
        )
        return selection

    # ------------------------------------------------------------------
    # Convenience: single-call API
    # ------------------------------------------------------------------

    def run(
        self,
        market_regime: MarketRegime,
        volatility: float,
        trend_strength: float,
        volume: float,
        sentiment_score: float,
    ) -> StrategySelection:
        """
        End-to-end call: detect condition → score → select.

        This is the primary method called by strategy_engine.py.

        Returns
        -------
        StrategySelection
        """
        condition = self.detect_market_condition(
            market_regime=market_regime,
            volatility=volatility,
            trend_strength=trend_strength,
            volume=volume,
            sentiment_score=sentiment_score,
        )
        return self.select_best_strategy(condition)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_reasoning(
        winner: StrategyName,
        winner_score: float,
        condition: MarketCondition,
        ranked: List[Tuple[StrategyName, float]],
    ) -> str:
        """
        Compose a human-readable reasoning string for the selection.
        Designed for display in trade_explainer.py output.
        """
        lines = [
            f"Regime: {condition.market_regime}",
            f"Volatility: {condition.volatility*100:.2f}%  |  "
            f"Trend: {condition.trend_strength:+.3f}  |  "
            f"Volume ratio: {condition.volume:.2f}x  |  "
            f"Sentiment: {condition.sentiment_score:+.3f}",
            "",
            f"Selected: {winner} (score={winner_score:.3f})",
            "",
            "All strategies ranked:",
        ]
        for name, score in ranked:
            marker = " ← SELECTED" if name == winner else ""
            lines.append(f"  {name}: {score:.3f}{marker}")

        return "\n".join(lines)
