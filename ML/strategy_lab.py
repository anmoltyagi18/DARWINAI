"""
strategy_lab.py — AIGOFIN ML Package
======================================
Training loop connecting strategy_evolver and backtest_engine.
Automates generation, backtesting, scoring, mutation, and preservation of trading strategies.
"""

import json
import os
import copy
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

from .strategy_evolver import (
    generate_population,
    select_parents,
    crossover,
    mutate,
    Strategy,
    Gene,
    _compute_fitness,
    _strategy_signals
)
from .backtest_engine import run_backtest, load_historical_data

DB_FILE = "ML/strategies_db.json"

class EvolverBacktestAdapter:
    """
    Adapter bridging vectorized strategy gene extraction into step-by-step
    engine-compliant ticks, acting identically to StrategyEngine.
    """
    def __init__(self, strategy: Strategy, data_source: pd.DataFrame):
        self.strategy = strategy
        self.df = data_source
        
        # Pre-compute vectorized signals for the entire dataset to feed backtrader step-by-step
        self.signals = _strategy_signals(self.strategy, self.df['close'].values)
        self.step = 0

    def generate_signal(self, features: dict) -> dict:
        """Yields the signal for the current step mapped to Backtrader iteration."""
        if self.step < len(self.signals):
            sig = self.signals[self.step]
            self.step += 1
            return {"signal": int(sig)}
        return {"signal": 0}


def strategy_to_dict(strat: Strategy) -> dict:
    """Serializes a genetic strategy."""
    return {
        "generation": strat.generation,
        "fitness": strat.fitness,
        "total_return": strat.total_return,
        "sharpe_ratio": strat.sharpe_ratio,
        "max_drawdown": strat.max_drawdown,
        "win_rate": strat.win_rate,
        "num_trades": strat.num_trades,
        "genes": [
            {
                "indicator": g.indicator,
                "period": g.period,
                "period2": g.period2,
                "threshold": g.threshold,
                "condition": g.condition,
                "action": g.action
            } for g in strat.genes
        ]
    }


def save_to_db(population: List[Strategy], db_path: str = DB_FILE):
    """
    Persists strategies in a local JSON database database to maintain
    gene continuity across multiple run cycles.
    """
    # Overwrites with current population rankings to easily load highest fitness.
    data_to_save = [strategy_to_dict(s) for s in population]
    
    # Create dir if not exist
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    
    with open(db_path, "w") as f:
        json.dump(data_to_save, f, indent=4)


def generate_strategies(size: int = 20) -> List[Strategy]:
    """Generates an initial random population base."""
    return generate_population(size=size, min_genes=1, max_genes=5)


def test_strategies(population: List[Strategy], data_source: Any) -> List[Strategy]:
    """
    Executes backtest evaluation for a population natively relying on
    the `backtest_engine.py` wrapper metrics. Fits to composite score logic.
    """
    df = load_historical_data(data_source)

    for i, strat in enumerate(population):
        adapter = EvolverBacktestAdapter(strat, df)
        
        # Dispatch Strategy to BacktestEngine
        results = run_backtest(df, adapter, initial_cash=100000.0)
        
        if "error" in results:
            strat.fitness = float("-inf")
            continue
            
        strat.total_return = results.get("profit_pct", 0.0) / 100.0
        strat.sharpe_ratio = results.get("sharpe_ratio", 0.0)
        strat.max_drawdown = results.get("max_drawdown", 0.0)
        strat.win_rate = results.get("win_rate", 0.0) / 100.0
        strat.num_trades = results.get("total_trades", 0)

        metrics_for_fitness = {
            "total_return": strat.total_return,
            "sharpe_ratio": strat.sharpe_ratio,
            "max_drawdown": strat.max_drawdown,
            "win_rate": strat.win_rate,
            "num_trades": strat.num_trades,
        }
        
        # Assign composite genetic score
        strat.fitness = _compute_fitness(metrics_for_fitness)

    return population


def evolve_strategies(population: List[Strategy], generation: int, **kwargs) -> List[Strategy]:
    """
    Transforms the current generation through Selection, Crossover, and Mutation.
    """
    import random
    
    pop_size = len(population)
    n_parents = kwargs.get("n_parents", max(2, pop_size // 4))
    elite_size = kwargs.get("elite_size", max(1, pop_size // 10))
    mutation_rate = kwargs.get("mutation_rate", 0.2)
    
    # 1. Elitism extraction
    population.sort(key=lambda s: s.fitness, reverse=True)
    elites = [copy.deepcopy(s) for s in population[:elite_size]]
    
    # 2. Parental Tournament Selection
    parents = select_parents(population, n_parents=n_parents)
    
    # 3. Structural Crossover
    offspring = []
    random.shuffle(parents)
    for i in range(0, len(parents) - 1, 2):
        c1, c2 = crossover(parents[i], parents[i+1])
        offspring.extend([c1, c2])
        
    # 4. Stochastic Mutation
    for child in offspring:
        mutate(child, mutation_rate=mutation_rate)
        child.generation = generation
        
    # 5. Populate deficiencies with fresh genestrings
    n_random = pop_size - len(elites) - len(offspring)
    newcomers = generate_strategies(max(0, n_random))
    for s in newcomers:
        s.generation = generation
            
    return elites + offspring + newcomers


def run_training_loop(data_source: Any, generations: int = 5, pop_size: int = 20) -> Tuple[Optional[Strategy], List[Strategy]]:
    """
    Executes the macro training payload bridging strategy creation, evaluation,
    genetic advancement, and database persistence sequentially.
    """
    print(f"[Strategy Lab] Initializing Training Loop (Gen: {generations}, Size: {pop_size})")
    
    population = generate_strategies(size=pop_size)
    best_ever = None
    
    for gen in range(1, generations + 1):
        population = test_strategies(population, data_source)
        population.sort(key=lambda s: s.fitness, reverse=True)
        
        gen_best = population[0]
        if not best_ever or gen_best.fitness > best_ever.fitness:
            best_ever = copy.deepcopy(gen_best)
            
        print(f"Gen {gen:2d} | Best Fitness: {gen_best.fitness:7.4f} | "
              f"Return: {gen_best.total_return:+.2%} | Sharpe: {gen_best.sharpe_ratio:.2f} | "
              f"Trades: {gen_best.num_trades}")
              
        save_to_db(population)
        
        if gen < generations:
            population = evolve_strategies(population, generation=gen+1)
            
    print("\n[Strategy Lab] Sequence Finalized.")
    if best_ever:
        print(f"Maximum Assessed Fitness: {best_ever.fitness:7.4f}")
    
    return best_ever, population

if __name__ == "__main__":
    import numpy as np
    
    # Generate generic mock environment for debugging
    dates = pd.date_range('2023-01-01', periods=300)
    df_dummy = pd.DataFrame({
        'open': np.random.uniform(90, 110, 300),
        'high': np.random.uniform(105, 115, 300),
        'low': np.random.uniform(85, 95, 300),
        'close': np.random.uniform(90, 110, 300),
        'volume': np.random.uniform(1_000, 5_000, 300)
    }, index=dates)
    
    best, pop = run_training_loop(df_dummy, generations=3, pop_size=10)
