import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Import all required ML modules
from . import indicators_engine as ind_engine
from . import market_regime_detector as regime_detector
from . import anomaly_detector as anom_detector
from . import sentiment_engine as sent_engine
from . import strategy_engine as strat_engine
from . import strategy_evolver as strat_evolver
from . import rl_trader as rl_trader
from . import portfolio_optimizer as port_optimizer
from . import risk_manager as risk_manager
from .trade_explainer import TradeExplainer, TradeContext

# Newly migrated LLM modules
from .meta_strategy import MetaStrategyAI
from .ai_strategy_discovery import discover_strategies
from .strategy_genome import StrategyGenome
from .model_registry import ModelRegistry
from .position_manager import PositionManager
from .feature_engineering import FeatureEngineer

class AIBrain:
    """
    AI decision pipeline that connects all AI modules.
    Responsibilities:
    - collect signals from all modules
    - score each signal
    - combine them using weighted scoring
    - produce final trading decision
    """
    
    def __init__(self):
        # Initialize default weights for each AI module
        self.weights = {
            "indicators_engine": 0.10,
            "market_regime_detector": 0.15,
            "anomaly_detector": 0.05,
            "sentiment_engine": 0.10,
            "strategy_engine": 0.15,
            "strategy_evolver": 0.10,
            "rl_trader": 0.15,
            "portfolio_optimizer": 0.10,
            "risk_manager": 0.10,
            "meta_strategy": 0.05,
            "ai_strategy_discovery": 0.05
        }
        
    def _signal_to_score(self, signal: str) -> float:
        """Convert string signal to numeric score."""
        mapping = {
            "BUY": 1.0,
            "SELL": -1.0,
            "HOLD": 0.0
        }
        return mapping.get(str(signal).upper(), 0.0)

    def _score_to_signal(self, score: float, threshold: float = 0.15) -> str:
        """Convert numeric score back to discrete signal."""
        if score >= threshold:
            return "BUY"
        elif score <= -threshold:
            return "SELL"
        else:
            return "HOLD"
            
    def _collect_signals(self, market_data: pd.DataFrame, context: Dict[str, Any] = None) -> Dict[str, Dict[str, Any]]:
        """
        Collects signals from all integrated modules.
        This provides a mock or interface integration assuming each module exposes 
        a standard method like `get_signal` or similar prediction endpoint.
        """
        Context = context or {}
        signals = {}
        
        # Note: In a real integration, the specific classes/methods of each module 
        # would be instantiated and invoked according to their exact API.
        # This implementation delegates the responsibility to a wrapper or expects 
        # standard dictionaries with 'signal', 'confidence', etc.
        
        # Below is an abstraction of the collection process:
        # 1. indicators_engine
        # signals["indicators_engine"] = ind_engine.get_signal(market_data)
        
        # 2. market_regime_detector
        # signals["market_regime_detector"] = regime_detector.get_signal(market_data)
        
        # 3. anomaly_detector
        # signals["anomaly_detector"] = anom_detector.get_signal(market_data)
        
        # 4. sentiment_engine
        # signals["sentiment_engine"] = sent_engine.get_signal(Context.get("news_data", []))
        
        # 5. strategy_engine
        # signals["strategy_engine"] = strat_engine.get_signal(market_data)
        
        # 6. strategy_evolver
        # signals["strategy_evolver"] = strat_evolver.get_signal(market_data)
        
        # 7. rl_trader
        # signals["rl_trader"] = rl_trader.get_signal(market_data)
        
        # 8. portfolio_optimizer
        # signals["portfolio_optimizer"] = port_optimizer.get_signal(market_data)
        
        # 9. risk_manager
        # signals["risk_manager"] = risk_manager.get_risk_assessment(market_data)

        # 10. MetaStrategyAI (Safely wrapped)
        try:
            meta = MetaStrategyAI()
            # Assuming detector provides regime and vol
            regime = context.get("market_regime", "unknown")
            vol = context.get("volatility", 0.02)
            trend = context.get("trend_strength", 0.0)
            volume = context.get("volume_ratio", 1.0)
            sentiment = context.get("sentiment_score", 0.0)
            
            res = meta.run(regime, vol, trend, volume, sentiment)
            signals["meta_strategy"] = {
                "signal": res.selected_strategy.split("_")[0].upper(), # e.g. "MOMENTUM" -> "BUY"? 
                "confidence": res.confidence_score,
                "reason": res.reasoning
            }
        except Exception as e:
            logger.warning(f"MetaStrategyAI failed: {e}")

        # 11. AI Strategy Discovery
        try:
            # discover_strategies(df)
            # signals["ai_strategy_discovery"] = ...
            pass
        except Exception as e:
            logger.warning(f"AI Strategy Discovery failed: {e}")

        # For the purpose of this pipeline and combining, you can feed a dictionary directly to evaluate()
        return signals

    def score_and_combine(self, module_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Scores each signal and combines them using weighted scoring.
        """
        total_score = 0.0
        total_weight = 0.0
        
        weighted_confidence_sum = 0.0
        
        # Tracking variables
        best_strategy_source = "aggregate"
        highest_conviction = -1.0
        risk_levels = []
        
        for module_name, output in module_outputs.items():
            if module_name not in self.weights:
                continue
                
            weight = self.weights[module_name]
            signal = output.get("signal", "HOLD")
            confidence = float(output.get("confidence", 0.5))
            
            # 1. Score the signal
            score = self._signal_to_score(signal)
            
            # Calculate module contribution and total score
            module_contribution = score * confidence * weight
            
            total_score += module_contribution
            total_weight += weight
            weighted_confidence_sum += confidence * weight
            
            # Track risk levels from modules
            if "risk_level" in output:
                risk_levels.append(float(output["risk_level"]))
                
            # Determine best strategy source (highest conviction)
            conviction = abs(score) * confidence
            if conviction > highest_conviction:
                highest_conviction = conviction
                best_strategy_source = module_name
                
        # Handle edge case where no weights were accumulated
        if total_weight == 0.0:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "risk_level": 0.5,
                "strategy_source": "none"
            }
            
        # Combine signals using weighted scoring
        normalized_score = total_score / total_weight
        
        # Produce final trading decision
        final_signal = self._score_to_signal(normalized_score)
        
        # Calculate final combined metrics
        final_confidence = weighted_confidence_sum / total_weight
        final_risk = sum(risk_levels) / len(risk_levels) if risk_levels else 0.5
        
        return {
            "signal": final_signal,
            "confidence": round(final_confidence, 3),
            "risk_level": round(final_risk, 3),
            "strategy_source": best_strategy_source
        }

    def evaluate(self, module_outputs: Dict[str, Dict[str, Any]], symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """
        Main pipeline entry point. 
        Expects a dictionary containing outputs from all integrated AI modules.
        
        Example Input:
        {
            "indicators_engine": {"signal": "BUY", "confidence": 0.8},
            "market_regime_detector": {"signal": "HOLD", "confidence": 0.9},
            "risk_manager": {"signal": "SELL", "confidence": 0.7, "risk_level": 0.8},
            ...
        }
        """
        # Distinguish between raw feed from live streamer vs module_outputs
        # If it looks like a symbol feed (has AAPL keys), this would need module routing
        # Assuming module_outputs format here for explanation generation:
        decision = self.score_and_combine(module_outputs)
        
        # Extract metadata
        regime = "RANGING"
        sentiment = "NEUTRAL"
        indicators = {}
        
        if "market_regime_detector" in module_outputs:
            regime_val = module_outputs["market_regime_detector"].get("regime", "RANGING")
            if isinstance(regime_val, str): regime = regime_val.upper().replace(" ", "_")
            
        if "sentiment_engine" in module_outputs:
            sent_val = module_outputs["sentiment_engine"].get("sentiment", "NEUTRAL")
            if isinstance(sent_val, str): sentiment = sent_val.upper().replace(" ", "_")
            
        if "indicators_engine" in module_outputs:
            indicators = module_outputs["indicators_engine"].get("indicators", {})
            
        risk_float = decision.get("risk_level", 0.5)
        if risk_float > 0.8: risk_str = "VERY_HIGH"
        elif risk_float > 0.6: risk_str = "HIGH"
        elif risk_float > 0.4: risk_str = "MODERATE"
        elif risk_float > 0.2: risk_str = "LOW"
        else: risk_str = "VERY_LOW"
        
        raw_signal = decision.get("signal", "HOLD")
        confidence = decision.get("confidence", 0.0)
        
        explainer_signal = raw_signal
        if raw_signal == "BUY" and confidence > 0.75: explainer_signal = "STRONG_BUY"
        elif raw_signal == "SELL" and confidence > 0.75: explainer_signal = "STRONG_SELL"
        
        # Only inject explanation if we actually have module combinations
        if module_outputs:
            try:
                ctx = TradeContext(
                    symbol=symbol,
                    signal=explainer_signal,
                    confidence=confidence,
                    market_regime=regime,
                    sentiment=sentiment,
                    risk_level=risk_str,
                    indicators=indicators,
                    signal_breakdown={
                        k: self._signal_to_score(v.get("signal", "HOLD")) * float(v.get("confidence", 0.5))
                        for k, v in module_outputs.items() if isinstance(v, dict)
                    }
                )
                explainer = TradeExplainer()
                report = explainer.explain(ctx)
                # Short readable summary and full explanation
                decision["reason"] = report.plain_text
                decision["reasoning_summary"] = report.summary
            except Exception as e:
                decision["reason"] = f"Explainer error: {e}"
                
        return decision