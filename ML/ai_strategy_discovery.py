"""
ML/ai_strategy_discovery.py
Automatically discover profitable trading strategies using evolutionary search.
Upgraded version with:
- Multi asset support
- Parallel backtesting
- Fitness scoring
- Strategy deduplication
- Strategy database persistence
"""

import numpy as np
import pandas as pd
import random
import logging
import json
from typing import Dict, List, Any
from dataclasses import dataclass
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

from .backtest_engine import BacktestEngine
# from .strategy_evolver import Strategy  # removed unused and broken StrategyEvolver import
# from .ai_brain import AIBrain # Moved inside class to avoid circular dependency

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# MULTI ASSET POOL
# ---------------------------------------------------

SYMBOL_POOL = [
    "AAPL",
    "MSFT",
    "TSLA",
    "NVDA",
    "AMZN",
    "META",
    "SPY",
    "QQQ"
]

# ---------------------------------------------------
# STRATEGY CLASS
# ---------------------------------------------------

@dataclass
class TradingStrategy:
    name: str
    entry_conditions: List[Dict[str, Any]]
    exit_conditions: List[Dict[str, Any]]
    indicators_used: List[str]
    params: Dict[str, float]
    regime: str

    def to_dict(self):

        return {
            "strategy_name": self.name,
            "entry_rules": self.entry_conditions,
            "exit_rules": self.exit_conditions,
            "indicators_used": self.indicators_used,
            "params": self.params,
            "regime": self.regime
        }


# ---------------------------------------------------
# INDICATOR LIBRARY
# ---------------------------------------------------

class IndicatorLibrary:

    AVAILABLE_INDICATORS = {

        "RSI": {"period": [14, 21], "thresholds": [30, 70]},

        "MACD": {"fast": [12], "slow": [26], "signal": [9]},

        "VWAP": {"period": [20]},

        "ATR": {"period": [14]},

        "ADX": {"period": [14], "thresholds": [25]},

        "ZScore": {"period": [20]},

        "EMA50": {"period": [50]},

        "EMA200": {"period": [200]},

        "VolumeSpike": {"threshold": [1.5, 2.0, 3.0]}
    }

    @staticmethod
    def random_indicator():

        name = random.choice(list(IndicatorLibrary.AVAILABLE_INDICATORS.keys()))

        config = IndicatorLibrary.AVAILABLE_INDICATORS[name].copy()

        for k, v in config.items():

            if isinstance(v, list):

                config[k] = random.choice(v)

        return {"name": name, "params": config}


# ---------------------------------------------------
# STRATEGY GENERATOR
# ---------------------------------------------------

class StrategyGenerator:

    def generate_random_strategy(self, name):

        entry_conditions = []
        exit_conditions = []

        indicators = []
        params = {}

        regime = random.choice(["bull", "bear", "sideways"])

        num_entry = random.randint(1, 3)

        for _ in range(num_entry):

            ind = IndicatorLibrary.random_indicator()

            indicators.append(ind["name"])

            condition = {

                "indicator": ind["name"],

                "type": random.choice(["lt", "gt", "cross_above", "cross_below"]),

                "threshold": random.uniform(20, 80),

                "params": ind["params"]
            }

            entry_conditions.append(condition)

            params.update(ind["params"])

        num_exit = random.randint(1, 2)

        for _ in range(num_exit):

            ind = IndicatorLibrary.random_indicator()

            indicators.append(ind["name"])

            condition = {

                "indicator": ind["name"],

                "type": random.choice(["lt", "gt", "cross_above", "cross_below"]),

                "threshold": random.uniform(20, 80),

                "params": ind["params"]
            }

            exit_conditions.append(condition)

            params.update(ind["params"])

        return TradingStrategy(

            name=name,

            entry_conditions=entry_conditions,

            exit_conditions=exit_conditions,

            indicators_used=list(set(indicators)),

            params=params,

            regime=regime
        )

    def mutate(self, strategy):

        child = deepcopy(strategy)

        for cond in child.entry_conditions:

            if random.random() < 0.3:

                cond["threshold"] += random.uniform(-10, 10)

        child.name = strategy.name + "_mut_" + str(random.randint(1000, 9999))

        return child


# ---------------------------------------------------
# STRATEGY HASH (FOR DEDUPLICATION)
# ---------------------------------------------------

def strategy_hash(strategy):

    return hash(str(strategy.entry_conditions) + str(strategy.exit_conditions))


# ---------------------------------------------------
# STRATEGY EVALUATOR
# ---------------------------------------------------

class StrategyEvaluator:

    def __init__(self):
        from .backtest_engine import BacktestEngine
        self.backtester = BacktestEngine()

    def evaluate(self, strategy):

        symbol = random.choice(SYMBOL_POOL)

        config = {

            "strategy": strategy.to_dict(),

            "symbol": symbol,

            "start_date": "2020-01-01",

            "end_date": "2025-12-31"
        }

        try:

            result = self.backtester.run_backtest(config)

            profit = result.get("total_return", 0)

            sharpe = result.get("sharpe_ratio", 0)

            drawdown = result.get("max_drawdown", 0)

            winrate = result.get("win_rate", 0)

            fitness = (sharpe * 0.5) + (profit * 0.3) - (drawdown * 0.2)

            return {

                "strategy": strategy,

                "profit": profit,

                "sharpe": sharpe,

                "drawdown": drawdown,

                "winrate": winrate,

                "fitness": fitness
            }

        except Exception as e:

            logger.error(f"Backtest failed {strategy.name}: {e}")

            return {

                "strategy": strategy,

                "profit": -999,

                "sharpe": -999,

                "drawdown": 999,

                "winrate": 0,

                "fitness": -999
            }


# ---------------------------------------------------
# EVOLUTION ENGINE
# ---------------------------------------------------

class EvolutionaryStrategyDiscoverer:

    def __init__(self, population=100, generations=30):

        self.population_size = population

        self.generations = generations

        self.generator = StrategyGenerator()

        self.evaluator = StrategyEvaluator()

        from .ai_brain import AIBrain
        self.ai_brain = AIBrain()

    def run(self):

        logger.info("Starting Strategy Evolution")

        population = [

            self.generator.generate_random_strategy(f"gen0_{i}")

            for i in range(self.population_size)
        ]

        best_strategies = []

        for gen in range(self.generations):

            logger.info(f"Generation {gen}")

            with ThreadPoolExecutor(max_workers=8) as executor:

                results = list(executor.map(self.evaluator.evaluate, population))

            results.sort(key=lambda x: x["fitness"], reverse=True)

            best = results[0]

            logger.info(

                f"Best fitness {best['fitness']:.2f} | sharpe {best['sharpe']:.2f} | profit {best['profit']:.2%}"
            )

            best_strategies.append({

                **best["strategy"].to_dict(),

                "profit": best["profit"],

                "sharpe_ratio": best["sharpe"],

                "drawdown": best["drawdown"],

                "win_rate": best["winrate"]
            })

            elite_size = max(5, self.population_size // 5)

            elite = results[:elite_size]

            new_population = []

            seen = set()

            while len(new_population) < self.population_size:

                parent = random.choice(elite)["strategy"]

                child = self.generator.mutate(parent)

                h = strategy_hash(child)

                if h not in seen:

                    seen.add(h)

                    new_population.append(child)

            population = new_population

        self.save_strategies(best_strategies)

        self.ai_brain.store_strategies(best_strategies)

        logger.info("Evolution complete")

        return best_strategies[:10]

    def save_strategies(self, strategies):

        with open("ML/discovered_strategies.json", "w") as f:

            json.dump(strategies, f, indent=4)


# ---------------------------------------------------
# ENTRY FUNCTION
# ---------------------------------------------------

def discover_strategies(population=50, generations=20):

    engine = EvolutionaryStrategyDiscoverer(population, generations)

    return engine.run()


# ---------------------------------------------------
# CLI TEST
# ---------------------------------------------------

if __name__ == "__main__":

    top = discover_strategies()

    print("\nTOP STRATEGIES\n")

    for i, s in enumerate(top[:3]):

        print("\nStrategy", i + 1)

        print("Name:", s["strategy_name"])

        print("Sharpe:", s["sharpe_ratio"])

        print("Profit:", s["profit"])

        print("Regime:", s["regime"])