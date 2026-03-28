# =============================================================================
# MODULE: ML/strategy_genome.py
# PROJECT: AIGOFIN - AI Quant Trading Platform
#
# PURPOSE:
#   Evolutionary algorithm that discovers novel trading strategies by
#   evolving a population of strategy "genomes" over many generations.
#
#   Each genome encodes a complete rule-based strategy:
#     indicator, threshold, entry_rule, exit_rule
#
#   Fitness is measured by running each genome as a strategy against
#   historical data and scoring: profit, Sharpe ratio, max drawdown.
#
# INTEGRATES WITH:
#   strategy_evolver.py  — calls evolution_loop() to obtain new strategies
#   backtest_engine.py   — fitness evaluation via backtesting
#
# GENETIC ALGORITHM STEPS:
#   1. generate_population   — random initial genomes
#   2. evaluate_fitness      — backtest each genome → fitness score
#   3. selection             — keep best performers (elitism + tournament)
#   4. crossover             — combine pairs to produce offspring
#   5. mutation              — randomly alter genes to explore search space
#   6. evolution_loop        — run N generations, return best genome
#
# AUTHOR: AIGOFIN System
# =============================================================================

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gene pools — the search space for each genome attribute
# ---------------------------------------------------------------------------

INDICATOR_POOL: List[str] = [
    "RSI", "MACD", "EMA_20", "EMA_50", "VWAP",
    "bollinger_position", "volume_spike", "momentum",
    "macd_histogram", "trend_strength",
]

ENTRY_RULE_POOL: List[str] = [
    "crossover_above",        # Indicator crosses above threshold
    "crossover_below",        # Indicator crosses below threshold
    "value_above",            # Current value > threshold
    "value_below",            # Current value < threshold
    "spike_confirmed",        # Indicator spikes > threshold AND volume spike
    "reversal_up",            # Indicator was below, now above threshold
    "reversal_down",          # Indicator was above, now below threshold
]

EXIT_RULE_POOL: List[str] = [
    "stop_loss_hit",          # Fixed % stop loss
    "take_profit_hit",        # Fixed % take profit
    "indicator_reversal",     # Entry indicator reverses signal
    "time_exit",              # Exit after N bars
    "trailing_stop",          # Trailing stop loss
    "macd_divergence",        # MACD divergence as exit trigger
]

# Threshold search range per indicator (min, max)
THRESHOLD_RANGES: Dict[str, Tuple[float, float]] = {
    "RSI": (20.0, 80.0),
    "MACD": (-2.0, 2.0),
    "EMA_20": (-0.05, 0.05),       # As price ratio offsets
    "EMA_50": (-0.05, 0.05),
    "VWAP": (-0.03, 0.03),
    "bollinger_position": (0.1, 0.9),
    "volume_spike": (1.2, 3.0),
    "momentum": (-0.05, 0.05),
    "macd_histogram": (-1.0, 1.0),
    "trend_strength": (-0.5, 0.5),
}


# ---------------------------------------------------------------------------
# Genome data structure
# ---------------------------------------------------------------------------

@dataclass
class StrategyGenome:
    """
    Represents a single tradeable strategy encoded as a set of genes.

    Genes
    -----
    indicator   : Which technical indicator drives the entry signal.
    threshold   : The critical value the indicator must cross/exceed.
    entry_rule  : Logic applied to indicator vs threshold for entry.
    exit_rule   : Mechanism to close the trade.
    stop_pct    : Stop-loss distance as fraction of entry price.
    tp_pct      : Take-profit distance as fraction of entry price.

    Fitness metrics (populated after evaluation)
    -----------------------------------------------
    fitness     : Composite fitness score (higher = better).
    total_return: Backtested total return.
    sharpe      : Backtested Sharpe ratio.
    max_drawdown: Maximum portfolio drawdown (negative value).
    n_trades    : Number of trades executed in backtest window.
    """

    indicator: str = "RSI"
    threshold: float = 50.0
    entry_rule: str = "value_above"
    exit_rule: str = "stop_loss_hit"
    stop_pct: float = 0.02          # 2% default stop
    tp_pct: float = 0.04            # 4% default target (1:2 RR)

    # Fitness fields — set by evaluate_fitness()
    fitness: float = 0.0
    total_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    n_trades: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "indicator": self.indicator,
            "threshold": round(self.threshold, 4),
            "entry_rule": self.entry_rule,
            "exit_rule": self.exit_rule,
            "stop_pct": round(self.stop_pct, 4),
            "tp_pct": round(self.tp_pct, 4),
            "fitness": round(self.fitness, 4),
            "total_return": round(self.total_return, 4),
            "sharpe": round(self.sharpe, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "n_trades": self.n_trades,
        }


# ---------------------------------------------------------------------------
# Genetic algorithm engine
# ---------------------------------------------------------------------------

class StrategyGenome_Engine:
    """
    Genetic algorithm that evolves a population of StrategyGenome objects
    to discover high-performing trading strategies.

    The fitness function can be provided externally (e.g. a backtest_engine
    call) or uses an internal fast simulator for development/testing.

    Parameters
    ----------
    population_size : int
        Number of genomes per generation. Default 50.
    n_generations : int
        Number of evolution cycles. Default 30.
    elite_pct : float
        Fraction of top performers carried unchanged to next gen. Default 0.1.
    mutation_rate : float
        Probability of mutating each gene. Default 0.15.
    tournament_size : int
        Number of candidates per tournament selection round. Default 3.
    fitness_fn : callable, optional
        External fitness function: fn(genome, df) → (total_return, sharpe, drawdown, n_trades).
        If None, internal simulator is used.
    """

    def __init__(
        self,
        population_size: int = 50,
        n_generations: int = 30,
        elite_pct: float = 0.10,
        mutation_rate: float = 0.15,
        tournament_size: int = 3,
        fitness_fn: Optional[Callable] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        self.population_size = population_size
        self.n_generations = n_generations
        self.elite_pct = elite_pct
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.fitness_fn = fitness_fn
        self.history: List[Dict[str, Any]] = []  # Per-generation summary

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        logger.info(
            f"StrategyGenome_Engine: pop={population_size}, "
            f"gen={n_generations}, mut={mutation_rate}, elite={elite_pct}"
        )

    # ------------------------------------------------------------------
    # Step 1: Population generation
    # ------------------------------------------------------------------

    def generate_population(self) -> List[StrategyGenome]:
        """
        Create a random initial population of StrategyGenome objects.
        Each genome is drawn uniformly from the gene pools.

        Returns
        -------
        List[StrategyGenome]
        """
        population = []
        for _ in range(self.population_size):
            indicator = random.choice(INDICATOR_POOL)
            low, high = THRESHOLD_RANGES.get(indicator, (-1.0, 1.0))

            genome = StrategyGenome(
                indicator=indicator,
                threshold=round(random.uniform(low, high), 4),
                entry_rule=random.choice(ENTRY_RULE_POOL),
                exit_rule=random.choice(EXIT_RULE_POOL),
                stop_pct=round(random.uniform(0.005, 0.05), 4),
                tp_pct=round(random.uniform(0.01, 0.10), 4),
            )
            population.append(genome)

        logger.debug(f"generate_population: {len(population)} genomes created.")
        return population

    # ------------------------------------------------------------------
    # Step 2: Fitness evaluation
    # ------------------------------------------------------------------

    def evaluate_fitness(
        self,
        genome: StrategyGenome,
        df: pd.DataFrame,
    ) -> StrategyGenome:
        """
        Evaluate a single genome by simulating it on the provided
        OHLCV + features DataFrame.

        If an external fitness_fn was provided at construction, it is
        called instead of the internal simulator.

        The composite fitness score weights:
            60% Sharpe ratio (risk-adjusted return)
            30% Total return
            10% Max drawdown penalty

        Parameters
        ----------
        genome : StrategyGenome
        df : pd.DataFrame
            Feature-engineered DataFrame (output of FeatureEngineer).

        Returns
        -------
        StrategyGenome
            Same genome with fitness fields populated.
        """
        if self.fitness_fn is not None:
            total_return, sharpe, drawdown, n_trades = self.fitness_fn(
                genome, df
            )
        else:
            total_return, sharpe, drawdown, n_trades = (
                self._internal_simulate(genome, df)
            )

        genome.total_return = total_return
        genome.sharpe = sharpe
        genome.max_drawdown = drawdown
        genome.n_trades = n_trades

        # Composite fitness (higher = better)
        sharpe_component = max(sharpe, -2.0) * 0.60
        return_component = total_return * 0.30
        # Drawdown is negative, penalty scales up as drawdown worsens
        drawdown_penalty = max(drawdown, -1.0) * 0.10
        genome.fitness = round(
            sharpe_component + return_component + drawdown_penalty, 4
        )

        return genome

    def _internal_simulate(
        self, genome: StrategyGenome, df: pd.DataFrame
    ) -> Tuple[float, float, float, int]:
        """
        Lightweight internal backtest simulator.

        Applies the genome's entry/exit rules bar-by-bar on the feature
        DataFrame, tracks equity, and returns performance metrics.

        Note: This is a fast approximation. For production evaluation,
        inject backtest_engine.py via the fitness_fn parameter.

        Returns
        -------
        Tuple[total_return, sharpe, max_drawdown, n_trades]
        """
        if genome.indicator not in df.columns or "close" not in df.columns:
            logger.warning(
                f"Indicator '{genome.indicator}' missing from DataFrame. "
                f"Returning zero fitness."
            )
            return 0.0, 0.0, 0.0, 0

        close = df["close"].values
        indicator_vals = df[genome.indicator].values
        n = len(close)

        if n < 10:
            return 0.0, 0.0, 0.0, 0

        equity = 1.0
        equity_curve: List[float] = [equity]
        in_trade = False
        entry_price = 0.0
        n_trades = 0
        bars_in_trade = 0

        for i in range(1, n):
            val_prev = indicator_vals[i - 1]
            val_curr = indicator_vals[i]
            price = close[i]

            if not in_trade:
                # --- Entry evaluation ---
                signal = self._evaluate_entry(
                    genome.entry_rule, genome.threshold,
                    val_prev, val_curr
                )
                if signal:
                    in_trade = True
                    entry_price = price
                    n_trades += 1
                    bars_in_trade = 0
            else:
                bars_in_trade += 1
                # --- Exit evaluation ---
                pnl_pct = (price - entry_price) / entry_price
                exit_signal = self._evaluate_exit(
                    genome.exit_rule, genome.threshold,
                    val_curr, pnl_pct,
                    genome.stop_pct, genome.tp_pct,
                    bars_in_trade,
                )
                if exit_signal:
                    trade_return = pnl_pct * 0.95  # Approximate slippage/commission
                    equity *= (1 + trade_return)
                    in_trade = False

            equity_curve.append(equity)

        # --- Performance metrics ---
        equity_arr = np.array(equity_curve)
        total_return = float(equity_arr[-1] - 1.0)

        bar_returns = np.diff(equity_arr) / equity_arr[:-1]
        if bar_returns.std() > 1e-8:
            sharpe = float(
                bar_returns.mean() / bar_returns.std() * np.sqrt(252)
            )
        else:
            sharpe = 0.0

        # Drawdown calculation
        running_max = np.maximum.accumulate(equity_arr)
        drawdowns = (equity_arr - running_max) / running_max
        max_drawdown = float(drawdowns.min())

        return (
            round(total_return, 4),
            round(sharpe, 4),
            round(max_drawdown, 4),
            n_trades,
        )

    @staticmethod
    def _evaluate_entry(
        rule: str,
        threshold: float,
        val_prev: float,
        val_curr: float,
    ) -> bool:
        """Apply entry rule logic. Returns True if entry conditions are met."""
        if rule == "value_above":
            return val_curr > threshold
        elif rule == "value_below":
            return val_curr < threshold
        elif rule == "crossover_above":
            return val_prev <= threshold < val_curr
        elif rule == "crossover_below":
            return val_prev >= threshold > val_curr
        elif rule == "reversal_up":
            return val_prev < threshold and val_curr >= threshold
        elif rule == "reversal_down":
            return val_prev > threshold and val_curr <= threshold
        elif rule == "spike_confirmed":
            return val_curr > threshold * 1.2
        return False

    @staticmethod
    def _evaluate_exit(
        rule: str,
        threshold: float,
        val_curr: float,
        pnl_pct: float,
        stop_pct: float,
        tp_pct: float,
        bars_in_trade: int,
    ) -> bool:
        """Apply exit rule logic. Returns True if exit conditions are met."""
        if rule == "stop_loss_hit":
            return pnl_pct <= -stop_pct
        elif rule == "take_profit_hit":
            return pnl_pct >= tp_pct
        elif rule == "trailing_stop":
            return pnl_pct <= -stop_pct
        elif rule == "indicator_reversal":
            return val_curr < threshold
        elif rule == "time_exit":
            return bars_in_trade >= 20
        elif rule == "macd_divergence":
            return abs(val_curr) < threshold * 0.3
        return False

    # ------------------------------------------------------------------
    # Step 3: Selection
    # ------------------------------------------------------------------

    def selection(
        self, population: List[StrategyGenome]
    ) -> List[StrategyGenome]:
        """
        Select survivors using elitism + tournament selection.

        Elites (top elite_pct%) are always preserved unchanged.
        Remaining slots are filled via tournament selection.

        Parameters
        ----------
        population : List[StrategyGenome]
            Evaluated population (fitness scores populated).

        Returns
        -------
        List[StrategyGenome]
            Survivors for next generation.
        """
        sorted_pop = sorted(population, key=lambda g: g.fitness, reverse=True)
        n_elite = max(1, int(self.population_size * self.elite_pct))

        # Carry over elites unchanged
        survivors = [copy.deepcopy(g) for g in sorted_pop[:n_elite]]

        # Fill remainder via tournament selection
        while len(survivors) < self.population_size:
            candidates = random.sample(population, min(self.tournament_size, len(population)))
            winner = max(candidates, key=lambda g: g.fitness)
            survivors.append(copy.deepcopy(winner))

        logger.debug(
            f"selection: {n_elite} elites preserved, "
            f"{len(survivors) - n_elite} tournament selected."
        )
        return survivors

    # ------------------------------------------------------------------
    # Step 4: Crossover
    # ------------------------------------------------------------------

    def crossover(
        self,
        parent_a: StrategyGenome,
        parent_b: StrategyGenome,
    ) -> Tuple[StrategyGenome, StrategyGenome]:
        """
        Uniform crossover: for each gene, randomly inherit from either parent.

        Parameters
        ----------
        parent_a, parent_b : StrategyGenome

        Returns
        -------
        Tuple[StrategyGenome, StrategyGenome]
            Two offspring genomes.
        """
        def pick(gene_a, gene_b):
            return gene_a if random.random() < 0.5 else gene_b

        child_a = StrategyGenome(
            indicator=pick(parent_a.indicator, parent_b.indicator),
            threshold=pick(parent_a.threshold, parent_b.threshold),
            entry_rule=pick(parent_a.entry_rule, parent_b.entry_rule),
            exit_rule=pick(parent_a.exit_rule, parent_b.exit_rule),
            stop_pct=pick(parent_a.stop_pct, parent_b.stop_pct),
            tp_pct=pick(parent_a.tp_pct, parent_b.tp_pct),
        )
        child_b = StrategyGenome(
            indicator=pick(parent_b.indicator, parent_a.indicator),
            threshold=pick(parent_b.threshold, parent_a.threshold),
            entry_rule=pick(parent_b.entry_rule, parent_a.entry_rule),
            exit_rule=pick(parent_b.exit_rule, parent_a.exit_rule),
            stop_pct=pick(parent_b.stop_pct, parent_a.stop_pct),
            tp_pct=pick(parent_b.tp_pct, parent_a.tp_pct),
        )
        return child_a, child_b

    # ------------------------------------------------------------------
    # Step 5: Mutation
    # ------------------------------------------------------------------

    def mutation(self, genome: StrategyGenome) -> StrategyGenome:
        """
        Apply random mutations to a genome's genes.

        Each gene mutates independently with probability mutation_rate.
        Numeric genes receive Gaussian perturbation; categorical genes
        are replaced with a random value from the pool.

        Parameters
        ----------
        genome : StrategyGenome

        Returns
        -------
        StrategyGenome
            Mutated genome (mutates in place and returns).
        """
        if random.random() < self.mutation_rate:
            genome.indicator = random.choice(INDICATOR_POOL)
            # Reset threshold to be valid for new indicator
            low, high = THRESHOLD_RANGES.get(genome.indicator, (-1.0, 1.0))
            genome.threshold = round(random.uniform(low, high), 4)

        elif random.random() < self.mutation_rate:
            # Perturb threshold without changing indicator
            low, high = THRESHOLD_RANGES.get(genome.indicator, (-1.0, 1.0))
            noise = (high - low) * 0.1 * np.random.randn()
            genome.threshold = round(
                float(np.clip(genome.threshold + noise, low, high)), 4
            )

        if random.random() < self.mutation_rate:
            genome.entry_rule = random.choice(ENTRY_RULE_POOL)

        if random.random() < self.mutation_rate:
            genome.exit_rule = random.choice(EXIT_RULE_POOL)

        if random.random() < self.mutation_rate:
            genome.stop_pct = round(
                float(np.clip(genome.stop_pct * random.uniform(0.5, 1.5), 0.005, 0.10)), 4
            )

        if random.random() < self.mutation_rate:
            genome.tp_pct = round(
                float(np.clip(genome.tp_pct * random.uniform(0.5, 1.5), 0.01, 0.20)), 4
            )

        return genome

    # ------------------------------------------------------------------
    # Step 6: Evolution loop
    # ------------------------------------------------------------------

    def evolution_loop(
        self,
        df: pd.DataFrame,
        n_generations: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[StrategyGenome, List[Dict[str, Any]]]:
        """
        Run the full genetic algorithm evolution cycle.

        Steps per generation:
          1. Evaluate fitness of every genome in the population.
          2. Log generation summary.
          3. Selection → survivors.
          4. Crossover surviving pairs → offspring.
          5. Mutate offspring.

        Parameters
        ----------
        df : pd.DataFrame
            Historical feature DataFrame for backtesting.
        n_generations : int, optional
            Override constructor setting.
        verbose : bool
            If True, print generation progress.

        Returns
        -------
        Tuple[StrategyGenome, List[Dict]]
            (best genome found, generation history list)
        """
        generations = n_generations or self.n_generations
        population = self.generate_population()
        best_ever: Optional[StrategyGenome] = None

        for gen in range(1, generations + 1):
            # --- Evaluate ---
            population = [self.evaluate_fitness(g, df) for g in population]

            # --- Track best ---
            gen_best = max(population, key=lambda g: g.fitness)
            if best_ever is None or gen_best.fitness > best_ever.fitness:
                best_ever = copy.deepcopy(gen_best)

            # --- Generation summary ---
            avg_fitness = np.mean([g.fitness for g in population])
            summary = {
                "generation": gen,
                "best_fitness": round(gen_best.fitness, 4),
                "avg_fitness": round(float(avg_fitness), 4),
                "best_sharpe": round(gen_best.sharpe, 4),
                "best_return": round(gen_best.total_return, 4),
                "best_drawdown": round(gen_best.max_drawdown, 4),
                "best_genome": gen_best.to_dict(),
            }
            self.history.append(summary)

            if verbose:
                logger.info(
                    f"Gen {gen:03d}/{generations} | "
                    f"best_fitness={gen_best.fitness:.4f} | "
                    f"avg={avg_fitness:.4f} | "
                    f"sharpe={gen_best.sharpe:.3f} | "
                    f"return={gen_best.total_return*100:.2f}%"
                )

            if gen == generations:
                break  # Skip unnecessary reproduction on last gen

            # --- Selection ---
            survivors = self.selection(population)

            # --- Crossover ---
            next_generation: List[StrategyGenome] = []
            random.shuffle(survivors)
            for i in range(0, len(survivors) - 1, 2):
                child_a, child_b = self.crossover(
                    survivors[i], survivors[i + 1]
                )
                next_generation.extend([child_a, child_b])

            # Preserve any leftover odd genome
            if len(next_generation) < self.population_size:
                next_generation.append(copy.deepcopy(survivors[-1]))

            # --- Mutation ---
            next_generation = [
                self.mutation(g) for g in next_generation[:self.population_size]
            ]

            population = next_generation

        logger.info(
            f"evolution_loop complete. Best fitness: {best_ever.fitness:.4f} | "
            f"Sharpe: {best_ever.sharpe:.3f} | "
            f"Return: {best_ever.total_return*100:.2f}%"
        )
        return best_ever, self.history
