"""
ML Package — AIGOFIN Core AI/ML Engines
========================================
Import all public classes so callers can do:

    from ML import AIBrain, IndicatorsEngine, LiveDataStream, ...
"""

# Migrated modules from LLM
from .meta_strategy import MetaStrategyAI
from .ai_strategy_discovery import discover_strategies
from .strategy_genome import StrategyGenome
from .model_registry import ModelRegistry
from .position_manager import PositionManager
from .feature_engineering import FeatureEngineer

# Core Engines
from .indicators_engine import IndicatorsEngine
from .anomaly_detector import AnomalyDetector
from .sentiment_engine import SentimentEngine
from .strategy_engine import StrategyEngine
from .strategy_evolver import Strategy
from .rl_trader import TradingEnv
from .portfolio_optimizer import PortfolioOptimizer
from .risk_manager import RiskManager
from .simulator import Simulator
from .trade_explainer import TradeExplainer
from .live_data_stream import LiveDataStream
from .data_pipeline import run_data_pipeline

# Brain (High-level aggregator)
from .ai_brain import AIBrain

import logging
logger = logging.getLogger(__name__)
logger.info("AIGOFIN ML package initialized with migrated LLM modules")

__all__ = [
    "AIBrain",
    "IndicatorsEngine",
    "AnomalyDetector",
    "SentimentEngine",
    "StrategyEngine",
    "Strategy",  # From strategy_evolver
    "TradingEnv",
    "PortfolioOptimizer",
    "RiskManager",
    "Simulator",
    "TradeExplainer",
    "LiveDataStream",
    "MetaStrategyAI",
    "discover_strategies",
    "StrategyGenome",
    "ModelRegistry",
    "PositionManager",
    "FeatureEngineer",
]
