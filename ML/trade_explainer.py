"""
trade_explainer.py
==================
Human-Readable Trading Decision Explainer

Transforms raw quantitative signals into clear, structured narratives
that explain WHY the AI made a particular trading decision.

Features
--------
- Natural language generation for each indicator
- Confidence-weighted explanation depth
- Market regime context
- Sentiment narrative
- Risk-adjusted commentary
- Executive summary + detailed breakdown
- Multiple output formats (plain text, markdown, JSON)

Quick start
-----------
    from trade_explainer import TradeExplainer, TradeContext

    ctx = TradeContext(
        symbol        = "AAPL",
        signal        = "BUY",
        confidence    = 0.72,
        market_regime = "TRENDING_UP",
        sentiment     = "BULLISH",
        risk_level    = "MODERATE",
        indicators    = {
            "rsi": 42.5, "macd": 1.2, "macd_signal": 0.8,
            "macd_histogram": 0.4, "sma_20": 148.0, "sma_50": 145.0,
            "sma_200": 138.0, "bb_upper": 155.0, "bb_lower": 140.0,
            "bb_pct": 0.35, "atr": 2.8, "adx": 34.0, "volume_ratio": 1.4,
        },
        price         = 149.50,
        price_change_pct = 0.012,
        signal_breakdown = {
            "trend": 0.60, "momentum": 0.40, "volatility": 0.25,
            "volume": 0.30, "sentiment": 0.20,
        },
        stop_loss_pct    = 0.032,
        take_profit_pct  = 0.080,
        position_size_pct= 0.04,
        risk_reward_ratio= 2.5,
    )

    explainer = TradeExplainer()
    report    = explainer.explain(ctx)
    print(report.plain_text)
"""

from __future__ import annotations

import json
import math
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ════════════════════════════════════════════════════════════════════════════
# Enumerations
# ════════════════════════════════════════════════════════════════════════════

class SignalStrength(Enum):
    STRONG_BUY  = "STRONG_BUY"
    BUY         = "BUY"
    HOLD        = "HOLD"
    SELL        = "SELL"
    STRONG_SELL = "STRONG_SELL"


class ExplainDepth(Enum):
    BRIEF    = "brief"      # 2-3 sentences
    STANDARD = "standard"  # full breakdown (default)
    VERBOSE  = "verbose"   # everything including edge cases


# ════════════════════════════════════════════════════════════════════════════
# Input / Output dataclasses
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeContext:
    """All inputs needed to generate a trade explanation."""

    # ── identity ─────────────────────────────────────────────────────────────
    symbol:           str
    signal:           str          # "STRONG_BUY" | "BUY" | "HOLD" | "SELL" | "STRONG_SELL"
    confidence:       float        # 0-1
    market_regime:    str          # "TRENDING_UP" | "TRENDING_DOWN" | "RANGING" | "VOLATILE" | "BREAKOUT"
    sentiment:        str          # "VERY_BULLISH" | "BULLISH" | "NEUTRAL" | "BEARISH" | "VERY_BEARISH"
    risk_level:       str          # "VERY_LOW" | "LOW" | "MODERATE" | "HIGH" | "VERY_HIGH"

    # ── price ────────────────────────────────────────────────────────────────
    price:            float        = 0.0
    price_change_pct: float        = 0.0   # today's % change

    # ── indicators ───────────────────────────────────────────────────────────
    indicators: Dict[str, float]   = field(default_factory=dict)
    # expected keys: rsi, macd, macd_signal, macd_histogram,
    #   sma_20, sma_50, sma_200, bb_upper, bb_lower, bb_pct,
    #   atr, adx, volume_ratio

    # ── module scores (-1 to +1) ──────────────────────────────────────────────
    signal_breakdown: Dict[str, float] = field(default_factory=dict)

    # ── trade parameters ─────────────────────────────────────────────────────
    stop_loss_pct:    float        = 0.0
    take_profit_pct:  float        = 0.0
    position_size_pct: float       = 0.0
    risk_reward_ratio: float       = 2.0

    # ── optional enrichment ──────────────────────────────────────────────────
    analyst_notes:    List[str]    = field(default_factory=list)
    timestamp:        Optional[str] = None


@dataclass
class ExplanationSection:
    """One logical section of the explanation."""
    title:     str
    body:      str
    emoji:     str = ""
    score:     Optional[float] = None   # module score if applicable


@dataclass
class TradeExplanation:
    """Complete explanation output in multiple formats."""
    symbol:        str
    signal:        str
    summary:       str                       # one-line TL;DR
    sections:      List[ExplanationSection]  # structured sections
    plain_text:    str                       # formatted plain text
    markdown:      str                       # GitHub-flavoured Markdown
    json_report:   Dict                      # machine-readable dict
    timestamp:     str


# ════════════════════════════════════════════════════════════════════════════
# Language Templates
# ════════════════════════════════════════════════════════════════════════════

# ── signal copy ──────────────────────────────────────────────────────────────
_SIGNAL_VERB: Dict[str, str] = {
    "STRONG_BUY":  "strongly recommends entering a long position",
    "BUY":         "recommends entering a long position",
    "HOLD":        "recommends holding the current position",
    "SELL":        "recommends exiting or reducing the position",
    "STRONG_SELL": "strongly recommends exiting or initiating a short position",
}

_SIGNAL_EMOJI: Dict[str, str] = {
    "STRONG_BUY":  "🚀",
    "BUY":         "📈",
    "HOLD":        "⏸️",
    "SELL":        "📉",
    "STRONG_SELL": "🔴",
}

_CONFIDENCE_LABEL: Dict[Tuple, str] = {
    (0.00, 0.30): "low-conviction",
    (0.30, 0.55): "moderate-conviction",
    (0.55, 0.75): "solid",
    (0.75, 0.90): "high-conviction",
    (0.90, 1.01): "very high-conviction",
}

# ── regime copy ──────────────────────────────────────────────────────────────
_REGIME_CONTEXT: Dict[str, str] = {
    "TRENDING_UP":
        "The market is in a confirmed uptrend, with price making higher highs "
        "and moving averages fanning upward. Trend-following strategies carry "
        "a structural tailwind in this environment.",
    "TRENDING_DOWN":
        "The market is in a confirmed downtrend. Price action is below key "
        "moving averages and momentum is declining. Short bias or cash "
        "preservation is prudent.",
    "RANGING":
        "The market is in a sideways consolidation phase, oscillating between "
        "support and resistance without a clear directional trend. Mean-reversion "
        "strategies work best here; breakout trades carry higher failure risk.",
    "VOLATILE":
        "Elevated volatility dominates price action. Large intra-day swings "
        "and unpredictable reversals are common. Tighter position sizing and "
        "wider stops are necessary; signals carry lower reliability.",
    "BREAKOUT":
        "Price is pressing against a Bollinger Band extreme, suggesting a "
        "potential breakout or breakdown. Volume confirmation is critical — "
        "high volume validates the move, low volume suggests a false break.",
}

# ── sentiment copy ───────────────────────────────────────────────────────────
_SENTIMENT_CONTEXT: Dict[str, str] = {
    "VERY_BULLISH":
        "Price action has been decisively positive. The majority of recent "
        "sessions closed higher, with above-average gains, reflecting strong "
        "buying pressure and broad market participation.",
    "BULLISH":
        "Recent sessions lean bullish, with more advancing than declining days. "
        "The overall tone of price action supports further upside.",
    "NEUTRAL":
        "Price action shows no strong directional bias. Gains and losses are "
        "roughly balanced, indicating indecision or a wait-and-see posture "
        "among market participants.",
    "BEARISH":
        "Recent sessions lean bearish, with more declining than advancing days. "
        "Selling pressure is present, though not yet decisive.",
    "VERY_BEARISH":
        "Price action has been decisively negative, with persistent selling, "
        "large down-days, and a deteriorating trend in daily returns.",
}

# ── risk copy ────────────────────────────────────────────────────────────────
_RISK_CONTEXT: Dict[str, str] = {
    "VERY_LOW":
        "This asset exhibits exceptionally low volatility, well below typical "
        "equity levels. Price movements are predictable and risk of large losses "
        "is minimal in the near term.",
    "LOW":
        "Volatility is subdued, suggesting stable price action. This supports "
        "larger position sizes relative to higher-volatility alternatives.",
    "MODERATE":
        "Volatility is within normal equity ranges. The risk-to-reward profile "
        "is balanced, and standard position-sizing rules apply.",
    "HIGH":
        "Elevated volatility warrants caution. Unexpected price swings are "
        "common; keep position sizes conservative and honour stop-loss levels.",
    "VERY_HIGH":
        "Extreme volatility is present. This asset can move sharply against "
        "any position. Only experienced traders with clearly defined risk limits "
        "should trade this setup.",
}


def _confidence_label(confidence: float) -> str:
    for (lo, hi), label in _CONFIDENCE_LABEL.items():
        if lo <= confidence < hi:
            return label
    return "uncertain"


# ════════════════════════════════════════════════════════════════════════════
# Indicator Interpreters
# ════════════════════════════════════════════════════════════════════════════

class IndicatorInterpreter:
    """Translates raw indicator values into plain-language sentences."""

    # ── RSI ──────────────────────────────────────────────────────────────────
    @staticmethod
    def rsi(rsi: float) -> Tuple[str, str]:
        """Returns (interpretation, implication)."""
        if rsi < 20:
            return (
                f"RSI is extremely oversold at {rsi:.1f}",
                "The asset has been sold aggressively and is statistically "
                "due for a mean-reversion bounce. This is a strong contrarian "
                "buy signal, though momentum can remain negative short-term."
            )
        elif rsi < 30:
            return (
                f"RSI sits in oversold territory at {rsi:.1f}",
                "Selling pressure has been intense, bringing the RSI below the "
                "key 30 threshold. Historically, readings at this level precede "
                "recoveries, supporting a bullish lean."
            )
        elif rsi < 40:
            return (
                f"RSI is mildly weak at {rsi:.1f}",
                "Recent price weakness has pushed RSI below the neutral zone, "
                "but conditions are not yet oversold. There may be further "
                "downside before a durable low forms."
            )
        elif rsi < 60:
            return (
                f"RSI is neutral at {rsi:.1f}",
                "Momentum is balanced, with neither buyers nor sellers "
                "dominating. RSI alone provides no directional edge here."
            )
        elif rsi < 70:
            return (
                f"RSI is firm at {rsi:.1f}",
                "Buying momentum is healthy without being overextended. "
                "The asset has room to continue higher before reaching "
                "overbought territory."
            )
        elif rsi < 80:
            return (
                f"RSI is overbought at {rsi:.1f}",
                "The asset has rallied sharply, pushing RSI above 70. "
                "Short-term profit-taking risk is elevated; new long positions "
                "carry higher timing risk."
            )
        else:
            return (
                f"RSI is extremely overbought at {rsi:.1f}",
                "Buying momentum has reached an extreme rarely sustained for "
                "long. A pullback or consolidation is the most likely near-term "
                "outcome, even in strong uptrends."
            )

    # ── MACD ─────────────────────────────────────────────────────────────────
    @staticmethod
    def macd(
        macd_val: float,
        signal: float,
        histogram: float,
    ) -> Tuple[str, str]:
        cross = "above" if macd_val > signal else "below"
        hist_dir = "expanding" if abs(histogram) > abs(macd_val - signal) * 0.5 else "contracting"
        if macd_val > signal and histogram > 0:
            stance = "bullish"
            impl = (
                "The MACD line has crossed above its signal line and the "
                "histogram is positive, confirming upward momentum. This is "
                "a classic bullish signal that tends to precede sustained gains."
            )
        elif macd_val < signal and histogram < 0:
            stance = "bearish"
            impl = (
                "The MACD line is below its signal line with a negative "
                "histogram, confirming downward momentum. Sellers are in "
                "control and the path of least resistance is lower."
            )
        elif macd_val > signal and histogram < 0:
            stance = "weakening bullish"
            impl = (
                "Although the MACD remains above the signal line, the shrinking "
                "histogram warns that bullish momentum is fading. Watch for a "
                "potential bearish crossover if histogram continues to narrow."
            )
        else:
            stance = "weakening bearish"
            impl = (
                "The MACD is below the signal line but the histogram is shrinking, "
                "suggesting bearish momentum may be exhausting. A bullish crossover "
                "could be forming, though confirmation is still needed."
            )
        reading = (
            f"MACD is {stance} (line={macd_val:+.3f}, signal={signal:+.3f}, "
            f"histogram={histogram:+.3f}, {hist_dir})"
        )
        return reading, impl

    # ── Moving Averages ───────────────────────────────────────────────────────
    @staticmethod
    def moving_averages(
        price: float,
        sma20: float,
        sma50: float,
        sma200: float,
    ) -> Tuple[str, str]:
        above = [p for p, v in [("20-day", sma20), ("50-day", sma50), ("200-day", sma200)] if price > v]
        below = [p for p, v in [("20-day", sma20), ("50-day", sma50), ("200-day", sma200)] if price <= v]

        if len(above) == 3:
            posture = "fully above all key moving averages (20, 50, 200-day)"
            impl = (
                "Price trading above all three major moving averages is the "
                "textbook definition of a healthy uptrend. The structure "
                "provides a layered support floor beneath current price."
            )
        elif len(above) == 0:
            posture = "below all key moving averages (20, 50, 200-day)"
            impl = (
                "Price trading below all major moving averages signals a "
                "broad-based downtrend. Each moving average acts as overhead "
                "resistance, making rallies likely to stall."
            )
        elif "200-day" in above:
            posture = f"above the 200-day MA but below the {' and '.join(below)}"
            impl = (
                "The long-term trend remains intact (price above 200-day), "
                "but short-term momentum has softened. This is often a "
                "healthy pullback within a larger uptrend."
            )
        else:
            posture = f"below the 200-day MA (above: {', '.join(above) if above else 'none'})"
            impl = (
                "Price has fallen below the critical 200-day moving average, "
                "which many institutional traders treat as the dividing line "
                "between bull and bear phases."
            )

        # golden / death cross
        if sma50 > sma200:
            cross_note = "The 50-day MA is above the 200-day (golden cross formation), a structural bullish signal."
        else:
            cross_note = "The 50-day MA is below the 200-day (death cross formation), a structural bearish signal."

        return f"Price is {posture}", f"{impl} {cross_note}"

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    @staticmethod
    def bollinger(
        price: float,
        bb_upper: float,
        bb_lower: float,
        bb_pct: float,
    ) -> Tuple[str, str]:
        band_width = (bb_upper - bb_lower) / ((bb_upper + bb_lower) / 2)
        width_desc = (
            "narrow (low-volatility squeeze)"
            if band_width < 0.04 else
            "wide (elevated volatility)" if band_width > 0.10 else "moderate"
        )
        if bb_pct < 0.05:
            position = "touching the lower band"
            impl = (
                "Price is pressed against the lower Bollinger Band, a region "
                "associated with short-term oversold conditions. A snap-back "
                "toward the middle band (~20-day SMA) is statistically likely."
            )
        elif bb_pct < 0.25:
            position = "in the lower quartile of the bands"
            impl = (
                "Price is in the lower portion of the Bollinger Bands, "
                "suggesting mild oversold pressure with potential for "
                "mean-reversion toward the mid-band."
            )
        elif bb_pct > 0.95:
            position = "pressing the upper band"
            impl = (
                "Price is at the upper Bollinger Band, historically a zone "
                "of elevated short-term reversal risk. Strong trends can 'walk "
                "the band', but profit-taking risk is elevated."
            )
        elif bb_pct > 0.75:
            position = "in the upper quartile of the bands"
            impl = (
                "Price is in the upper portion of the bands, reflecting "
                "short-term bullish momentum. The asset is not yet at an "
                "extreme but upside room to the upper band is narrowing."
            )
        else:
            position = f"mid-band ({bb_pct:.0%} of the range)"
            impl = (
                "Price is in the middle of the Bollinger Bands, suggesting "
                "balanced conditions with no extreme to exploit."
            )
        return (
            f"Bollinger Bands are {width_desc}; price is {position}",
            impl
        )

    # ── ADX / Trend Strength ──────────────────────────────────────────────────
    @staticmethod
    def adx(adx_val: float) -> Tuple[str, str]:
        if adx_val < 15:
            strength = "very weak — no discernible trend"
            impl = (
                "An ADX below 15 indicates the market is drifting without "
                "directional conviction. Trend-following strategies underperform "
                "in this environment; range-bound tactics are preferred."
            )
        elif adx_val < 25:
            strength = "weak trend developing"
            impl = (
                "ADX is below 25, signalling a weak or nascent trend. "
                "Trend signals should be treated with caution until ADX "
                "confirms momentum above the 25 threshold."
            )
        elif adx_val < 40:
            strength = "moderate, established trend"
            impl = (
                "ADX between 25-40 is the sweet spot for trend-following: "
                "strong enough to provide reliable directional signals without "
                "the exhaustion risk of extreme readings."
            )
        elif adx_val < 60:
            strength = "strong trend in force"
            impl = (
                "A high ADX reading confirms the current trend is robust. "
                "Pullbacks within this trend are likely buying (or selling) "
                "opportunities rather than trend reversals."
            )
        else:
            strength = "extreme — potential trend exhaustion"
            impl = (
                "Very high ADX readings often precede trend exhaustion and "
                "volatile reversals. While the trend is dominant, risk of "
                "sudden mean-reversion increases significantly at these levels."
            )
        return f"ADX at {adx_val:.1f} — {strength}", impl

    # ── Volume ────────────────────────────────────────────────────────────────
    @staticmethod
    def volume(volume_ratio: float, price_change: float) -> Tuple[str, str]:
        direction = "higher" if price_change > 0 else "lower"
        if volume_ratio > 2.0:
            desc = f"surging ({volume_ratio:.1f}x average)"
            impl = (
                f"Volume is more than double its recent average on a "
                f"{direction} price move. This is a high-conviction signal — "
                f"institutional participation is likely behind this move."
            )
        elif volume_ratio > 1.3:
            desc = f"above average ({volume_ratio:.1f}x)"
            impl = (
                f"Above-average volume on a {direction} day provides "
                f"meaningful confirmation of the price move, increasing "
                f"signal reliability."
            )
        elif volume_ratio > 0.8:
            desc = f"normal ({volume_ratio:.1f}x average)"
            impl = (
                "Volume is within its typical range, offering no strong "
                "confirming or contradicting evidence for the price move."
            )
        else:
            desc = f"light ({volume_ratio:.1f}x average)"
            impl = (
                f"Low-volume price moves are less reliable. The {direction} "
                f"session lacks broad participation, making this signal "
                f"easier to reverse."
            )
        return f"Volume is {desc}", impl


# ════════════════════════════════════════════════════════════════════════════
# Module Score Narrator
# ════════════════════════════════════════════════════════════════════════════

class ModuleNarrator:
    """Converts per-module scores into narrative sentences."""

    @staticmethod
    def narrate(module: str, score: float) -> str:
        label_pos = {
            (0.7, 1.01): "strongly bullish",
            (0.3, 0.70): "moderately bullish",
            (0.1, 0.30): "mildly bullish",
        }
        label_neg = {
            (-0.10, 0.10):  "neutral",
            (-0.30, -0.10): "mildly bearish",
            (-0.70, -0.30): "moderately bearish",
            (-1.01, -0.70): "strongly bearish",
        }
        all_labels = {**label_pos, **label_neg}
        stance = "neutral"
        for (lo, hi), lbl in all_labels.items():
            if lo <= score < hi:
                stance = lbl
                break

        narratives = {
            "trend": (
                f"The **trend module** is {stance} ({score:+.2f}). "
                + (
                    "Price structure, moving-average alignment, and MACD all "
                    "point in the same direction, producing a strong, coherent trend signal."
                    if abs(score) > 0.5 else
                    "Moving averages and MACD are providing mixed signals, "
                    "suggesting a transition or indecisive phase."
                )
            ),
            "momentum": (
                f"The **momentum module** is {stance} ({score:+.2f}). "
                + (
                    "RSI and MACD momentum indicators reinforce each other, "
                    "creating a high-confidence momentum reading."
                    if abs(score) > 0.5 else
                    "Momentum indicators show mixed or neutral readings, "
                    "offering limited directional guidance."
                )
            ),
            "volatility": (
                f"The **volatility module** is {stance} ({score:+.2f}). "
                + (
                    "Bollinger Band positioning suggests an attractive mean-reversion "
                    "or continuation opportunity relative to current band structure."
                    if abs(score) > 0.3 else
                    "Price is near the mid-band, offering no strong volatility-based signal."
                )
            ),
            "volume": (
                f"The **volume module** is {stance} ({score:+.2f}). "
                + (
                    "Volume is confirming the price move with above-average participation."
                    if abs(score) > 0.3 else
                    "Volume is near average and provides neither confirmation nor contradiction."
                )
            ),
            "sentiment": (
                f"The **sentiment module** is {stance} ({score:+.2f}). "
                + (
                    "The recent pattern of advancing and declining days paints a "
                    "clear directional bias in market sentiment."
                    if abs(score) > 0.3 else
                    "Sentiment is balanced, with no clear bias emerging from recent price action."
                )
            ),
        }
        return narratives.get(module, f"Module '{module}' scored {score:+.2f} ({stance}).")


# ════════════════════════════════════════════════════════════════════════════
# Risk Explainer
# ════════════════════════════════════════════════════════════════════════════

class RiskExplainer:
    """Converts trade parameters into plain-language risk instructions."""

    @staticmethod
    def explain(
        stop_loss_pct:    float,
        take_profit_pct:  float,
        position_size_pct: float,
        risk_reward:      float,
        risk_level:       str,
        confidence:       float,
    ) -> str:
        sl_pct   = stop_loss_pct * 100
        tp_pct   = take_profit_pct * 100
        pos_pct  = position_size_pct * 100

        risk_adj = (
            "Given the elevated volatility classification, the suggested "
            "position size is intentionally conservative. "
            if risk_level in ("HIGH", "VERY_HIGH") else
            "The asset's low-to-moderate volatility supports a standard position size. "
            if risk_level in ("VERY_LOW", "LOW") else
            ""
        )

        conf_adj = (
            "Signal confidence is strong, so the full recommended position "
            "size is appropriate. "
            if confidence > 0.65 else
            "Signal confidence is moderate; consider entering at half-size "
            "and scaling in if the trade moves in your favour. "
            if confidence > 0.35 else
            "Signal confidence is low. Reduce position size further or wait "
            "for confirming price action before entering. "
        )

        return (
            f"Place the stop loss {sl_pct:.1f}% from entry (approximately 2× ATR), "
            f"protecting the trade from normal price noise while avoiding premature "
            f"exits. The take-profit target is set {tp_pct:.1f}% from entry, "
            f"delivering a {risk_reward:.1f}:1 reward-to-risk ratio. "
            f"Allocate {pos_pct:.1f}% of your trading capital to this position. "
            f"{risk_adj}{conf_adj}"
            f"Never risk more than you are prepared to lose entirely on this trade."
        )


# ════════════════════════════════════════════════════════════════════════════
# Core Explainer
# ════════════════════════════════════════════════════════════════════════════

class TradeExplainer:
    """
    Generates human-readable trade explanations from quantitative inputs.

    Parameters
    ----------
    depth    : ExplainDepth controlling explanation verbosity
    wrap     : line-wrap width for plain text output (0 = no wrap)
    """

    def __init__(
        self,
        depth: ExplainDepth = ExplainDepth.STANDARD,
        wrap:  int = 80,
    ):
        self.depth = depth
        self.wrap  = wrap
        self._interpreter = IndicatorInterpreter()
        self._narrator    = ModuleNarrator()
        self._risk_exp    = RiskExplainer()

    # ── public entry point ────────────────────────────────────────────────────

    def explain(self, ctx: TradeContext) -> TradeExplanation:
        """
        Generate a complete trade explanation for the given context.

        Returns
        -------
        TradeExplanation with plain_text, markdown, and json_report fields.
        """
        sections = []

        # ── 1. Executive Summary ─────────────────────────────────────────────
        sections.append(self._build_summary(ctx))

        # ── 2. Market Regime ─────────────────────────────────────────────────
        sections.append(self._build_regime(ctx))

        # ── 3. Indicator Analysis ────────────────────────────────────────────
        if self.depth != ExplainDepth.BRIEF:
            sections.extend(self._build_indicators(ctx))

        # ── 4. AI Module Scorecard ───────────────────────────────────────────
        if self.depth != ExplainDepth.BRIEF and ctx.signal_breakdown:
            sections.append(self._build_scorecard(ctx))

        # ── 5. Sentiment ─────────────────────────────────────────────────────
        sections.append(self._build_sentiment(ctx))

        # ── 6. Risk Assessment ───────────────────────────────────────────────
        sections.append(self._build_risk(ctx))

        # ── 7. Trade Execution ───────────────────────────────────────────────
        if ctx.stop_loss_pct or ctx.take_profit_pct:
            sections.append(self._build_execution(ctx))

        # ── 8. Caveats (verbose only) ────────────────────────────────────────
        if self.depth == ExplainDepth.VERBOSE:
            sections.append(self._build_caveats(ctx))

        summary    = sections[0].body.split(".")[0] + "."
        plain_text = self._render_plain(ctx, sections)
        markdown   = self._render_markdown(ctx, sections)
        json_rep   = self._render_json(ctx, sections)

        return TradeExplanation(
            symbol      = ctx.symbol,
            signal      = ctx.signal,
            summary     = summary,
            sections    = sections,
            plain_text  = plain_text,
            markdown    = markdown,
            json_report = json_rep,
            timestamp   = ctx.timestamp or datetime.now(timezone.utc).isoformat(),
        )

    # ── Section builders ─────────────────────────────────────────────────────

    def _build_summary(self, ctx: TradeContext) -> ExplanationSection:
        verb       = _SIGNAL_VERB.get(ctx.signal, "has generated a signal for")
        emoji      = _SIGNAL_EMOJI.get(ctx.signal, "📊")
        conf_label = _confidence_label(ctx.confidence)
        chg_desc   = (
            f"up {abs(ctx.price_change_pct):.2%} today"
            if ctx.price_change_pct > 0 else
            f"down {abs(ctx.price_change_pct):.2%} today"
            if ctx.price_change_pct < 0 else
            "unchanged today"
        )
        price_str  = f" at ${ctx.price:,.2f} ({chg_desc})" if ctx.price else ""

        body = (
            f"The AI system {verb} in {ctx.symbol}{price_str}. "
            f"This is a {conf_label} signal with {ctx.confidence:.0%} confidence, "
            f"generated within a {ctx.market_regime.replace('_', ' ').lower()} "
            f"market regime. "
        )

        # add primary reason
        if ctx.signal_breakdown:
            top_module = max(ctx.signal_breakdown, key=lambda k: abs(ctx.signal_breakdown[k]))
            top_score  = ctx.signal_breakdown[top_module]
            direction  = "bullish" if top_score > 0 else "bearish"
            body += (
                f"The dominant factor driving this decision is the "
                f"{top_module} analysis, which produced a strongly "
                f"{direction} reading ({top_score:+.2f})."
            )

        return ExplanationSection(
            title = "Executive Summary",
            body  = body,
            emoji = emoji,
        )

    def _build_regime(self, ctx: TradeContext) -> ExplanationSection:
        context = _REGIME_CONTEXT.get(ctx.market_regime, "The market regime is undefined.")
        body = (
            f"Current regime: {ctx.market_regime.replace('_', ' ')}. "
            f"{context}"
        )
        return ExplanationSection(
            title = "Market Regime",
            body  = body,
            emoji = "🗺️",
        )

    def _build_indicators(self, ctx: TradeContext) -> List[ExplanationSection]:
        ind  = ctx.indicators
        secs = []

        # RSI
        if "rsi" in ind:
            reading, impl = self._interpreter.rsi(ind["rsi"])
            secs.append(ExplanationSection(
                title = "RSI — Relative Strength Index",
                body  = f"{reading}. {impl}",
                emoji = "🔋",
            ))

        # MACD
        if all(k in ind for k in ("macd", "macd_signal", "macd_histogram")):
            reading, impl = self._interpreter.macd(
                ind["macd"], ind["macd_signal"], ind["macd_histogram"]
            )
            secs.append(ExplanationSection(
                title = "MACD — Moving Average Convergence Divergence",
                body  = f"{reading}. {impl}",
                emoji = "〰️",
            ))

        # Moving Averages
        if all(k in ind for k in ("sma_20", "sma_50", "sma_200")) and ctx.price:
            reading, impl = self._interpreter.moving_averages(
                ctx.price, ind["sma_20"], ind["sma_50"], ind["sma_200"]
            )
            secs.append(ExplanationSection(
                title = "Moving Averages",
                body  = f"{reading}. {impl}",
                emoji = "📏",
            ))

        # Bollinger Bands
        if all(k in ind for k in ("bb_upper", "bb_lower", "bb_pct")) and ctx.price:
            reading, impl = self._interpreter.bollinger(
                ctx.price, ind["bb_upper"], ind["bb_lower"], ind["bb_pct"]
            )
            secs.append(ExplanationSection(
                title = "Bollinger Bands",
                body  = f"{reading}. {impl}",
                emoji = "📡",
            ))

        # ADX
        if "adx" in ind:
            reading, impl = self._interpreter.adx(ind["adx"])
            secs.append(ExplanationSection(
                title = "ADX — Average Directional Index",
                body  = f"{reading}. {impl}",
                emoji = "📐",
            ))

        # Volume
        if "volume_ratio" in ind:
            reading, impl = self._interpreter.volume(
                ind["volume_ratio"], ctx.price_change_pct
            )
            secs.append(ExplanationSection(
                title = "Volume Analysis",
                body  = f"{reading}. {impl}",
                emoji = "📊",
            ))

        return secs

    def _build_scorecard(self, ctx: TradeContext) -> ExplanationSection:
        lines  = []
        weights = dict(trend=0.30, momentum=0.25, volatility=0.20,
                       volume=0.15, sentiment=0.10)
        total  = 0.0

        for module, score in ctx.signal_breakdown.items():
            w         = weights.get(module, 0.0)
            weighted  = score * w
            total    += weighted
            bar_len   = int(abs(score) * 12)
            bar       = ("█" * bar_len).ljust(12)
            direction = "▲" if score > 0 else ("▼" if score < 0 else "─")
            lines.append(
                f"{module.capitalize():<12} {direction}  [{bar}]  "
                f"{score:+.3f}  (weight {w:.0%})"
            )
            lines.append(f"  └─ {self._narrator.narrate(module, score)}")

        composite_dir = "bullish" if total > 0 else "bearish"
        lines.append("")
        lines.append(
            f"Composite score: {total:+.3f}  →  {composite_dir.upper()} bias "
            f"(confidence {ctx.confidence:.0%})"
        )

        return ExplanationSection(
            title = "AI Module Scorecard",
            body  = "\n".join(lines),
            emoji = "🤖",
        )

    def _build_sentiment(self, ctx: TradeContext) -> ExplanationSection:
        context = _SENTIMENT_CONTEXT.get(ctx.sentiment, "Sentiment is undefined.")
        body = (
            f"Sentiment classification: {ctx.sentiment.replace('_', ' ')}. "
            f"{context}"
        )
        return ExplanationSection(
            title = "Market Sentiment",
            body  = body,
            emoji = "💬",
        )

    def _build_risk(self, ctx: TradeContext) -> ExplanationSection:
        context = _RISK_CONTEXT.get(ctx.risk_level, "Risk is undefined.")
        body = (
            f"Risk classification: {ctx.risk_level.replace('_', ' ')}. "
            f"{context}"
        )
        return ExplanationSection(
            title = "Risk Assessment",
            body  = body,
            emoji = "🛡️",
        )

    def _build_execution(self, ctx: TradeContext) -> ExplanationSection:
        body = self._risk_exp.explain(
            ctx.stop_loss_pct,
            ctx.take_profit_pct,
            ctx.position_size_pct,
            ctx.risk_reward_ratio,
            ctx.risk_level,
            ctx.confidence,
        )
        return ExplanationSection(
            title = "Trade Execution Guide",
            body  = body,
            emoji = "🎯",
        )

    def _build_caveats(self, ctx: TradeContext) -> ExplanationSection:
        caveats = [
            "This analysis is generated by a quantitative model and should not "
            "be treated as personalised financial advice.",
            "Past signal performance does not guarantee future results. Markets "
            "can and do behave in ways not captured by historical patterns.",
            "Always perform your own due diligence before executing any trade, "
            "including reviewing fundamental factors not addressed here.",
            "In volatile or low-liquidity conditions, slippage may cause actual "
            "fill prices to differ materially from quoted levels.",
        ]
        if ctx.risk_level in ("HIGH", "VERY_HIGH"):
            caveats.append(
                "⚠  This asset is classified as HIGH risk. Position sizing and "
                "stop-loss discipline are especially critical here."
            )
        if ctx.confidence < 0.35:
            caveats.append(
                "⚠  Signal confidence is below 35%. Consider waiting for "
                "additional confirmation before committing capital."
            )
        return ExplanationSection(
            title = "Important Disclaimers",
            body  = "\n".join(f"• {c}" for c in caveats),
            emoji = "⚠️",
        )

    # ── Renderers ─────────────────────────────────────────────────────────────

    def _wrap(self, text: str) -> str:
        if self.wrap <= 0:
            return text
        lines = []
        for paragraph in text.split("\n"):
            if paragraph.strip() == "":
                lines.append("")
            elif paragraph.startswith("  ") or paragraph.startswith("•") or paragraph.startswith("└"):
                lines.append(paragraph)   # preserve indented / list lines
            else:
                lines.append(textwrap.fill(paragraph, width=self.wrap))
        return "\n".join(lines)

    def _render_plain(self, ctx: TradeContext, sections: List[ExplanationSection]) -> str:
        sig_emoji = _SIGNAL_EMOJI.get(ctx.signal, "📊")
        divider   = "═" * 66
        thin      = "─" * 66
        lines     = [
            divider,
            f"  TRADE EXPLANATION  {sig_emoji}  {ctx.symbol}  →  {ctx.signal}",
            divider,
        ]
        for sec in sections:
            lines.append(f"\n  {sec.emoji}  {sec.title.upper()}")
            lines.append(thin)
            lines.append(self._wrap(sec.body))
        lines.append("")
        lines.append(divider)
        ts = ctx.timestamp or datetime.now(timezone.utc).isoformat()
        lines.append(f"  Generated: {ts}")
        lines.append(divider)
        return "\n".join(lines)

    def _render_markdown(self, ctx: TradeContext, sections: List[ExplanationSection]) -> str:
        sig_emoji = _SIGNAL_EMOJI.get(ctx.signal, "📊")
        lines     = [
            f"# {sig_emoji} Trade Explanation — {ctx.symbol}",
            f"**Signal:** `{ctx.signal}` &nbsp; **Confidence:** {ctx.confidence:.0%} &nbsp; "
            f"**Regime:** {ctx.market_regime} &nbsp; **Risk:** {ctx.risk_level}",
            "",
            "---",
            "",
        ]
        for sec in sections:
            lines.append(f"## {sec.emoji} {sec.title}")
            lines.append("")
            # convert bullet-style lines to md
            for line in sec.body.split("\n"):
                if line.startswith("• "):
                    lines.append(f"- {line[2:]}")
                else:
                    lines.append(line)
            lines.append("")
        ts = ctx.timestamp or datetime.now(timezone.utc).isoformat()
        lines.append(f"---\n*Generated: {ts}*")
        return "\n".join(lines)

    def _render_json(self, ctx: TradeContext, sections: List[ExplanationSection]) -> Dict:
        return {
            "symbol":       ctx.symbol,
            "signal":       ctx.signal,
            "confidence":   ctx.confidence,
            "market_regime":ctx.market_regime,
            "sentiment":    ctx.sentiment,
            "risk_level":   ctx.risk_level,
            "summary":      sections[0].body if sections else "",
            "sections":     [
                {"title": s.title, "body": s.body, "emoji": s.emoji}
                for s in sections
            ],
            "signal_breakdown": ctx.signal_breakdown,
            "trade_params": {
                "stop_loss_pct":     ctx.stop_loss_pct,
                "take_profit_pct":   ctx.take_profit_pct,
                "position_size_pct": ctx.position_size_pct,
                "risk_reward_ratio": ctx.risk_reward_ratio,
            },
            "timestamp": ctx.timestamp or datetime.now(timezone.utc).isoformat(),
        }


# ════════════════════════════════════════════════════════════════════════════
# Convenience function
# ════════════════════════════════════════════════════════════════════════════

def explain_trade(
    symbol:           str,
    signal:           str,
    confidence:       float,
    market_regime:    str,
    sentiment:        str,
    risk_level:       str,
    indicators:       Dict[str, float],
    signal_breakdown: Optional[Dict[str, float]] = None,
    price:            float = 0.0,
    price_change_pct: float = 0.0,
    stop_loss_pct:    float = 0.0,
    take_profit_pct:  float = 0.0,
    position_size_pct: float = 0.0,
    risk_reward_ratio: float = 2.0,
    depth:            ExplainDepth = ExplainDepth.STANDARD,
    output:           str = "plain",    # "plain" | "markdown" | "json"
):
    """
    Convenience wrapper — build context and return chosen output format.

    Example
    -------
        text = explain_trade(
            symbol="NVDA", signal="BUY", confidence=0.68,
            market_regime="TRENDING_UP", sentiment="BULLISH",
            risk_level="MODERATE",
            indicators={"rsi": 48, "adx": 32, "bb_pct": 0.4,
                        "macd": 0.5, "macd_signal": 0.3, "macd_histogram": 0.2,
                        "sma_20": 410, "sma_50": 395, "sma_200": 370,
                        "bb_upper": 440, "bb_lower": 385, "atr": 8, "volume_ratio": 1.3},
            price=418.0, price_change_pct=0.018,
            stop_loss_pct=0.038, take_profit_pct=0.095, position_size_pct=0.04,
            risk_reward_ratio=2.5,
        )
        print(text)
    """
    ctx = TradeContext(
        symbol            = symbol,
        signal            = signal,
        confidence        = confidence,
        market_regime     = market_regime,
        sentiment         = sentiment,
        risk_level        = risk_level,
        indicators        = indicators,
        signal_breakdown  = signal_breakdown or {},
        price             = price,
        price_change_pct  = price_change_pct,
        stop_loss_pct     = stop_loss_pct,
        take_profit_pct   = take_profit_pct,
        position_size_pct = position_size_pct,
        risk_reward_ratio = risk_reward_ratio,
    )
    exp = TradeExplainer(depth=depth).explain(ctx)
    if output == "markdown":
        return exp.markdown
    if output == "json":
        return exp.json_report
    return exp.plain_text


# ════════════════════════════════════════════════════════════════════════════
# CLI Demo
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    DEMO_CASES = [
        dict(
            symbol        = "AAPL",
            signal        = "BUY",
            confidence    = 0.72,
            market_regime = "TRENDING_UP",
            sentiment     = "BULLISH",
            risk_level    = "LOW",
            price         = 189.45,
            price_change_pct = 0.0142,
            indicators    = {
                "rsi": 44.8, "macd": 1.14, "macd_signal": 0.76,
                "macd_histogram": 0.38, "sma_20": 185.20, "sma_50": 181.40,
                "sma_200": 172.00, "bb_upper": 196.0, "bb_lower": 174.4,
                "bb_pct": 0.71, "atr": 3.20, "adx": 36.5, "volume_ratio": 1.45,
            },
            signal_breakdown = {
                "trend": 0.60, "momentum": 0.40, "volatility": 0.25,
                "volume": 0.30, "sentiment": 0.22,
            },
            stop_loss_pct    = 0.034,
            take_profit_pct  = 0.085,
            position_size_pct= 0.06,
            risk_reward_ratio= 2.5,
        ),
        dict(
            symbol        = "TSLA",
            signal        = "STRONG_SELL",
            confidence    = 0.81,
            market_regime = "VOLATILE",
            sentiment     = "VERY_BEARISH",
            risk_level    = "VERY_HIGH",
            price         = 178.20,
            price_change_pct = -0.0510,
            indicators    = {
                "rsi": 78.2, "macd": -0.88, "macd_signal": 0.14,
                "macd_histogram": -1.02, "sma_20": 201.30, "sma_50": 215.60,
                "sma_200": 224.80, "bb_upper": 218.0, "bb_lower": 162.0,
                "bb_pct": 0.29, "atr": 12.4, "adx": 55.2, "volume_ratio": 2.30,
            },
            signal_breakdown = {
                "trend": -0.80, "momentum": -0.60, "volatility": -0.20,
                "volume": -0.55, "sentiment": -0.70,
            },
            stop_loss_pct    = 0.052,
            take_profit_pct  = 0.156,
            position_size_pct= 0.01,
            risk_reward_ratio= 3.0,
        ),
        dict(
            symbol        = "SPY",
            signal        = "HOLD",
            confidence    = 0.22,
            market_regime = "RANGING",
            sentiment     = "NEUTRAL",
            risk_level    = "MODERATE",
            price         = 452.80,
            price_change_pct = 0.0008,
            indicators    = {
                "rsi": 52.1, "macd": 0.10, "macd_signal": 0.08,
                "macd_histogram": 0.02, "sma_20": 450.10, "sma_50": 448.30,
                "sma_200": 430.00, "bb_upper": 462.0, "bb_lower": 438.0,
                "bb_pct": 0.62, "atr": 5.80, "adx": 18.4, "volume_ratio": 0.91,
            },
            signal_breakdown = {
                "trend": 0.10, "momentum": 0.05, "volatility": -0.05,
                "volume": 0.00, "sentiment": 0.00,
            },
            stop_loss_pct    = 0.026,
            take_profit_pct  = 0.052,
            position_size_pct= 0.04,
            risk_reward_ratio= 2.0,
        ),
    ]

    explainer = TradeExplainer(depth=ExplainDepth.STANDARD, wrap=72)

    for demo in DEMO_CASES:
        ctx = TradeContext(**demo)
        exp = explainer.explain(ctx)
        print(exp.plain_text)
        print()
