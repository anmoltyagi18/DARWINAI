"""
strategy_evolver.py
===================
Genetic Algorithm–based Trading Strategy Evolver

Pipeline
--------
1. generate_population  – random strategies
2. evaluate_population  – back-test each strategy on price data
3. select_parents       – tournament selection of top performers
4. crossover            – blend two parent strategies
5. mutate               – random perturbation of genes
6. evolve               – full GA loop returning the best strategy

Quick start
-----------
    import numpy as np
    from strategy_evolver import evolve, generate_price_data

    prices = generate_price_data(seed=42)          # synthetic or real OHLCV
    best, history = evolve(prices, generations=50)
    print(best)
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

INDICATOR_CHOICES = ["sma", "ema", "rsi", "macd", "bollinger", "momentum", "atr"]
CONDITION_CHOICES = ["crossover", "crossunder", "above", "below", "in_band"]
ACTION_CHOICES    = ["buy", "sell", "hold"]


@dataclass
class Gene:
    """A single trading rule: indicator + condition + threshold → action."""
    indicator:  str   = "sma"
    period:     int   = 14          # look-back window
    period2:    int   = 28          # secondary period (e.g., slow SMA)
    threshold:  float = 0.0        # generic threshold (RSI level, band width …)
    condition:  str   = "crossover"
    action:     str   = "buy"

    def __repr__(self) -> str:
        return (f"Gene({self.indicator}[{self.period}/{self.period2}] "
                f"{self.condition} {self.threshold:.2f} → {self.action})")


@dataclass
class Strategy:
    """A strategy is an ordered list of genes evaluated top-to-bottom."""
    genes:       List[Gene] = field(default_factory=list)
    fitness:     float      = float("-inf")
    generation:  int        = 0

    # ---- performance metrics set by evaluate() ----
    total_return:  float = 0.0
    sharpe_ratio:  float = 0.0
    max_drawdown:  float = 0.0
    win_rate:      float = 0.0
    num_trades:    int   = 0

    def __repr__(self) -> str:
        return (f"Strategy(genes={len(self.genes)}, gen={self.generation}, "
                f"fitness={self.fitness:.4f}, return={self.total_return:.2%}, "
                f"sharpe={self.sharpe_ratio:.2f}, trades={self.num_trades})")


# ---------------------------------------------------------------------------
# Helper: synthetic price data
# ---------------------------------------------------------------------------

def generate_price_data(
    n: int = 500,
    start_price: float = 100.0,
    volatility: float = 0.015,
    drift: float = 0.0002,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a synthetic close-price series via geometric Brownian motion.

    Returns
    -------
    np.ndarray of shape (n,)
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, volatility, n)
    prices  = start_price * np.cumprod(1 + returns)
    return prices.astype(np.float64)


# ---------------------------------------------------------------------------
# Technical Indicators
# ---------------------------------------------------------------------------

def _sma(prices: np.ndarray, period: int) -> np.ndarray:
    out = np.full_like(prices, np.nan)
    for i in range(period - 1, len(prices)):
        out[i] = prices[i - period + 1 : i + 1].mean()
    return out


def _ema(prices: np.ndarray, period: int) -> np.ndarray:
    out   = np.full_like(prices, np.nan)
    alpha = 2.0 / (period + 1)
    start = period - 1
    out[start] = prices[:period].mean()
    for i in range(start + 1, len(prices)):
        out[i] = alpha * prices[i] + (1 - alpha) * out[i - 1]
    return out


def _rsi(prices: np.ndarray, period: int) -> np.ndarray:
    delta  = np.diff(prices, prepend=prices[0])
    gains  = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_g  = _sma(gains,  period)
    avg_l  = _sma(losses, period)
    rs     = np.where(avg_l == 0, 100.0, avg_g / avg_l)
    return 100 - (100 / (1 + rs))


def _macd_line(prices: np.ndarray, fast: int, slow: int) -> np.ndarray:
    return _ema(prices, fast) - _ema(prices, slow)


def _bollinger(prices: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    mid = _sma(prices, period)
    std = np.array([
        prices[max(0, i - period + 1): i + 1].std()
        for i in range(len(prices))
    ])
    return mid + 2 * std, mid - 2 * std   # upper, lower


def _momentum(prices: np.ndarray, period: int) -> np.ndarray:
    out = np.full_like(prices, np.nan)
    out[period:] = prices[period:] - prices[:-period]
    return out


def _atr(prices: np.ndarray, period: int) -> np.ndarray:
    """Simplified ATR (no high/low; uses price range proxy)."""
    tr  = np.abs(np.diff(prices, prepend=prices[0]))
    return _sma(tr, period)


# ---------------------------------------------------------------------------
# Signal generation from a single Gene
# ---------------------------------------------------------------------------

def _gene_signal(gene: Gene, prices: np.ndarray) -> np.ndarray:
    """
    Produce a signal array for every bar:
      +1 = buy, -1 = sell, 0 = hold
    """
    n      = len(prices)
    signal = np.zeros(n, dtype=int)
    p1, p2 = max(2, gene.period), max(2, gene.period2)

    ind = gene.indicator
    cond = gene.condition
    act  = 1 if gene.action == "buy" else (-1 if gene.action == "sell" else 0)

    if ind == "sma":
        fast = _sma(prices, p1)
        slow = _sma(prices, p2)
        line1, line2 = fast, slow

    elif ind == "ema":
        fast = _ema(prices, p1)
        slow = _ema(prices, p2)
        line1, line2 = fast, slow

    elif ind == "rsi":
        rsi   = _rsi(prices, p1)
        line1 = rsi
        line2 = np.full(n, gene.threshold if gene.threshold != 0 else 50.0)

    elif ind == "macd":
        macd  = _macd_line(prices, p1, p2)
        sig   = _ema(macd, 9)
        line1, line2 = macd, sig

    elif ind == "bollinger":
        upper, lower = _bollinger(prices, p1)
        # above upper band → signal; below lower → opposite
        if cond == "above":
            line1 = prices;  line2 = upper
        else:
            line1 = prices;  line2 = lower

    elif ind == "momentum":
        mom   = _momentum(prices, p1)
        line1 = mom
        line2 = np.zeros(n)

    else:  # atr
        atr   = _atr(prices, p1)
        line1 = atr
        line2 = np.full(n, gene.threshold)

    # ---- apply condition ----
    for i in range(1, n):
        v1, v2     = line1[i], line2[i]
        v1p, v2p   = line1[i - 1], line2[i - 1]
        if np.isnan(v1) or np.isnan(v2) or np.isnan(v1p) or np.isnan(v2p):
            continue

        triggered = False
        if cond == "crossover":
            triggered = v1p < v2p and v1 >= v2
        elif cond == "crossunder":
            triggered = v1p > v2p and v1 <= v2
        elif cond == "above":
            triggered = v1 > v2
        elif cond == "below":
            triggered = v1 < v2
        elif cond == "in_band":
            triggered = v2p <= v1 <= v2  # within threshold band

        if triggered:
            signal[i] = act

    return signal


def _strategy_signals(strategy: Strategy, prices: np.ndarray) -> np.ndarray:
    """
    Combine gene signals by majority vote; ties → hold.
    """
    if not strategy.genes:
        return np.zeros(len(prices), dtype=int)

    matrix = np.stack([_gene_signal(g, prices) for g in strategy.genes], axis=0)
    combined = matrix.sum(axis=0)
    signals  = np.sign(combined).astype(int)   # +1, -1, or 0
    return signals


# ---------------------------------------------------------------------------
# Back-tester
# ---------------------------------------------------------------------------

def _backtest(
    strategy: Strategy,
    prices:   np.ndarray,
    commission: float = 0.001,
) -> dict:
    """
    Simple long/short back-test.

    Returns dict with performance metrics.
    """
    signals  = _strategy_signals(strategy, prices)
    n        = len(prices)
    position = 0          # +1 long, -1 short, 0 flat
    cash     = 1.0
    equity   = [1.0]
    trades   = 0
    wins     = 0
    entry_p  = 0.0

    for i in range(1, n):
        sig = signals[i]

        # close position on reversal or exit signal
        if position != 0 and (sig == -position or sig == 0):
            ret    = (prices[i] / entry_p - 1) * position - commission
            cash  *= 1 + ret
            if ret > 0:
                wins += 1
            trades   += 1
            position  = 0

        # open new position
        if position == 0 and sig != 0:
            position = sig
            entry_p  = prices[i] * (1 + commission * sig)

        # mark-to-market
        if position != 0:
            mtm = (prices[i] / entry_p - 1) * position
            equity.append(cash * (1 + mtm))
        else:
            equity.append(cash)

    equity = np.array(equity)

    total_return = equity[-1] - 1.0

    # Sharpe (annualised, assume 252 bars/year)
    rets   = np.diff(equity) / equity[:-1]
    std_r  = rets.std()
    sharpe = (rets.mean() / std_r * np.sqrt(252)) if std_r > 1e-9 else 0.0

    # Max drawdown
    peak    = np.maximum.accumulate(equity)
    dd      = (equity - peak) / peak
    max_dd  = dd.min()

    win_rate = (wins / trades) if trades > 0 else 0.0

    return dict(
        total_return = total_return,
        sharpe_ratio = sharpe,
        max_drawdown = max_dd,
        win_rate     = win_rate,
        num_trades   = trades,
    )


# ---------------------------------------------------------------------------
# Fitness function
# ---------------------------------------------------------------------------

def _compute_fitness(metrics: dict) -> float:
    """
    Weighted composite of Sharpe ratio, return, drawdown, and win-rate.
    Penalise strategies with < 5 trades (over-fitted / never trade).
    """
    if metrics["num_trades"] < 5:
        return float("-inf")

    score = (
        0.50 * metrics["sharpe_ratio"]
      + 0.25 * metrics["total_return"] * 10   # scale ~
      + 0.15 * metrics["win_rate"]
      - 0.10 * abs(metrics["max_drawdown"]) * 5
    )
    return score


# ---------------------------------------------------------------------------
# Step 1 – Random strategy generation
# ---------------------------------------------------------------------------

def _random_gene(rng: random.Random) -> Gene:
    period  = rng.randint(2, 50)
    period2 = rng.randint(period + 1, period + 50)
    return Gene(
        indicator  = rng.choice(INDICATOR_CHOICES),
        period     = period,
        period2    = period2,
        threshold  = rng.uniform(-2, 100),
        condition  = rng.choice(CONDITION_CHOICES),
        action     = rng.choice(ACTION_CHOICES),
    )


def generate_population(
    size:      int = 50,
    min_genes: int = 1,
    max_genes: int = 5,
    seed:      Optional[int] = None,
) -> List[Strategy]:
    """
    Step 1 – Generate a population of random strategies.

    Parameters
    ----------
    size      : number of strategies
    min_genes : minimum rules per strategy
    max_genes : maximum rules per strategy
    seed      : reproducibility seed

    Returns
    -------
    List[Strategy]
    """
    rng = random.Random(seed)
    population = []
    for _ in range(size):
        n_genes = rng.randint(min_genes, max_genes)
        genes   = [_random_gene(rng) for _ in range(n_genes)]
        population.append(Strategy(genes=genes))
    return population


# ---------------------------------------------------------------------------
# Step 2 – Evaluate population
# ---------------------------------------------------------------------------

def evaluate_population(
    population: List[Strategy],
    prices:     np.ndarray,
    commission: float = 0.001,
) -> List[Strategy]:
    """
    Step 2 – Back-test every strategy and assign fitness scores.

    Parameters
    ----------
    population : list of Strategy objects
    prices     : 1-D price array
    commission : per-trade friction

    Returns
    -------
    Same list, mutated in-place with fitness set.
    """
    for strat in population:
        metrics        = _backtest(strat, prices, commission)
        strat.total_return  = metrics["total_return"]
        strat.sharpe_ratio  = metrics["sharpe_ratio"]
        strat.max_drawdown  = metrics["max_drawdown"]
        strat.win_rate      = metrics["win_rate"]
        strat.num_trades    = metrics["num_trades"]
        strat.fitness       = _compute_fitness(metrics)
    return population


# ---------------------------------------------------------------------------
# Step 3 – Selection (tournament)
# ---------------------------------------------------------------------------

def select_parents(
    population:       List[Strategy],
    n_parents:        int = 20,
    tournament_size:  int = 5,
    seed:             Optional[int] = None,
) -> List[Strategy]:
    """
    Step 3 – Tournament selection: pick n_parents strategies.

    Parameters
    ----------
    population      : evaluated population
    n_parents       : how many parents to select
    tournament_size : contestants per tournament

    Returns
    -------
    List of selected parent Strategy objects (deep-copied).
    """
    rng     = random.Random(seed)
    parents = []
    viable  = [s for s in population if s.fitness > float("-inf")]
    if not viable:
        viable = population   # fall back if all are infeasible

    for _ in range(n_parents):
        contestants = rng.sample(viable, min(tournament_size, len(viable)))
        winner      = max(contestants, key=lambda s: s.fitness)
        parents.append(copy.deepcopy(winner))
    return parents


# ---------------------------------------------------------------------------
# Step 4 – Crossover
# ---------------------------------------------------------------------------

def crossover(
    parent_a: Strategy,
    parent_b: Strategy,
    seed:     Optional[int] = None,
) -> Tuple[Strategy, Strategy]:
    """
    Step 4 – Single-point crossover on the gene lists.

    Returns two children strategies.
    """
    rng    = random.Random(seed)
    a_genes = copy.deepcopy(parent_a.genes)
    b_genes = copy.deepcopy(parent_b.genes)

    if len(a_genes) > 1 and len(b_genes) > 1:
        pt_a = rng.randint(1, len(a_genes) - 1)
        pt_b = rng.randint(1, len(b_genes) - 1)
        child_a_genes = a_genes[:pt_a] + b_genes[pt_b:]
        child_b_genes = b_genes[:pt_b] + a_genes[pt_a:]
    else:
        # swap single genes
        child_a_genes = b_genes
        child_b_genes = a_genes

    return Strategy(genes=child_a_genes), Strategy(genes=child_b_genes)


# ---------------------------------------------------------------------------
# Step 5 – Mutation
# ---------------------------------------------------------------------------

def mutate(
    strategy:        Strategy,
    mutation_rate:   float = 0.2,
    add_gene_prob:   float = 0.1,
    drop_gene_prob:  float = 0.1,
    seed:            Optional[int] = None,
) -> Strategy:
    """
    Step 5 – Randomly perturb a strategy's genes.

    Possible mutations per gene
    ---------------------------
    * Change indicator
    * ±20 % period tweak
    * Flip condition
    * Flip action
    * Threshold drift

    Additionally, with small probability:
    * Add a new random gene
    * Drop an existing gene (if > 1 remain)

    Parameters
    ----------
    mutation_rate  : probability of mutating each gene attribute
    add_gene_prob  : probability of appending a new gene
    drop_gene_prob : probability of removing a random gene

    Returns
    -------
    Mutated Strategy (in-place, same object).
    """
    rng = random.Random(seed)

    for gene in strategy.genes:
        if rng.random() < mutation_rate:
            gene.indicator = rng.choice(INDICATOR_CHOICES)
        if rng.random() < mutation_rate:
            delta      = rng.uniform(0.8, 1.2)
            gene.period = max(2, int(gene.period * delta))
        if rng.random() < mutation_rate:
            delta       = rng.uniform(0.8, 1.2)
            gene.period2 = max(gene.period + 1, int(gene.period2 * delta))
        if rng.random() < mutation_rate:
            gene.condition = rng.choice(CONDITION_CHOICES)
        if rng.random() < mutation_rate:
            gene.action = rng.choice(ACTION_CHOICES)
        if rng.random() < mutation_rate:
            gene.threshold += rng.gauss(0, 5)

    if rng.random() < add_gene_prob:
        strategy.genes.append(_random_gene(rng))

    if rng.random() < drop_gene_prob and len(strategy.genes) > 1:
        idx = rng.randrange(len(strategy.genes))
        strategy.genes.pop(idx)

    # reset fitness – must be re-evaluated
    strategy.fitness = float("-inf")
    return strategy


# ---------------------------------------------------------------------------
# Step 6 – Full GA loop
# ---------------------------------------------------------------------------

def evolve(
    prices:          np.ndarray,
    generations:     int   = 30,
    population_size: int   = 60,
    n_parents:       int   = 20,
    elite_size:      int   = 5,
    mutation_rate:   float = 0.25,
    commission:      float = 0.001,
    tournament_size: int   = 5,
    min_genes:       int   = 1,
    max_genes:       int   = 5,
    verbose:         bool  = True,
    seed:            Optional[int] = None,
) -> Tuple[Strategy, List[dict]]:
    """
    Step 6 – Run the full genetic algorithm.

    Pipeline per generation
    -----------------------
    1. Evaluate fitness
    2. Preserve elites
    3. Tournament-select parents
    4. Crossover pairs → offspring
    5. Mutate offspring
    6. Fill remainder with fresh random strategies
    7. Repeat

    Parameters
    ----------
    prices          : 1-D close price array
    generations     : number of GA iterations
    population_size : total strategies per generation
    n_parents       : parents selected each generation
    elite_size      : top-n strategies carried forward unchanged
    mutation_rate   : gene mutation probability
    commission      : back-test commission per trade
    tournament_size : contestants per selection tournament
    min_genes       : min rules in a random strategy
    max_genes       : max rules in a random strategy
    verbose         : print progress each generation
    seed            : master RNG seed for reproducibility

    Returns
    -------
    (best_strategy, history)
        best_strategy : Strategy with highest fitness found
        history       : list of dicts with per-generation stats
    """
    rng   = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # ---- initial population ----
    population = generate_population(population_size, min_genes, max_genes, seed=rng.randint(0, 2**31))
    evaluate_population(population, prices, commission)
    population.sort(key=lambda s: s.fitness, reverse=True)

    best_ever = copy.deepcopy(population[0])
    history   = []

    for gen in range(1, generations + 1):

        # ---- elitism: carry forward best unchanged ----
        elites = [copy.deepcopy(s) for s in population[:elite_size]]

        # ---- selection ----
        parents = select_parents(
            population,
            n_parents       = n_parents,
            tournament_size = tournament_size,
            seed            = rng.randint(0, 2**31),
        )

        # ---- crossover → offspring ----
        offspring: List[Strategy] = []
        rng.shuffle(parents)
        for i in range(0, len(parents) - 1, 2):
            c1, c2 = crossover(parents[i], parents[i + 1], seed=rng.randint(0, 2**31))
            offspring.extend([c1, c2])

        # ---- mutation ----
        for child in offspring:
            mutate(child, mutation_rate=mutation_rate, seed=rng.randint(0, 2**31))
            child.generation = gen

        # ---- fill remainder with fresh random strategies ----
        n_random   = population_size - elite_size - len(offspring)
        n_random   = max(0, n_random)
        newcomers  = generate_population(n_random, min_genes, max_genes, seed=rng.randint(0, 2**31))
        for s in newcomers:
            s.generation = gen

        # ---- new population ----
        population = elites + offspring + newcomers
        evaluate_population(population, prices, commission)
        population.sort(key=lambda s: s.fitness, reverse=True)

        gen_best = population[0]
        if gen_best.fitness > best_ever.fitness:
            best_ever = copy.deepcopy(gen_best)

        viable = [s for s in population if s.fitness > float("-inf")]
        avg_fitness = np.mean([s.fitness for s in viable]) if viable else float("nan")

        stats = dict(
            generation    = gen,
            best_fitness  = gen_best.fitness,
            avg_fitness   = avg_fitness,
            best_return   = gen_best.total_return,
            best_sharpe   = gen_best.sharpe_ratio,
            best_drawdown = gen_best.max_drawdown,
            best_trades   = gen_best.num_trades,
        )
        history.append(stats)

        if verbose:
            print(
                f"Gen {gen:3d}/{generations} | "
                f"best_fitness={gen_best.fitness:7.4f} | "
                f"avg_fitness={avg_fitness:7.4f} | "
                f"return={gen_best.total_return:+.2%} | "
                f"sharpe={gen_best.sharpe_ratio:.2f} | "
                f"drawdown={gen_best.max_drawdown:.2%} | "
                f"trades={gen_best.num_trades}"
            )

    return best_ever, history


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def describe_strategy(strategy: Strategy) -> str:
    """Return a human-readable description of a strategy."""
    lines = [
        "=" * 60,
        f"  Strategy  (generation {strategy.generation})",
        "=" * 60,
        f"  Fitness      : {strategy.fitness:.4f}",
        f"  Total Return : {strategy.total_return:+.2%}",
        f"  Sharpe Ratio : {strategy.sharpe_ratio:.2f}",
        f"  Max Drawdown : {strategy.max_drawdown:.2%}",
        f"  Win Rate     : {strategy.win_rate:.2%}",
        f"  # Trades     : {strategy.num_trades}",
        f"  # Genes      : {len(strategy.genes)}",
        "-" * 60,
    ]
    for idx, g in enumerate(strategy.genes, 1):
        lines.append(
            f"  Rule {idx}: IF {g.indicator.upper()}({g.period},{g.period2}) "
            f"{g.condition.upper()} {g.threshold:.1f} → {g.action.upper()}"
        )
    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating synthetic price data …")
    prices = generate_price_data(n=600, seed=0)

    print("Starting genetic algorithm …\n")
    best, history = evolve(
        prices,
        generations     = 30,
        population_size = 80,
        n_parents       = 30,
        elite_size      = 5,
        mutation_rate   = 0.25,
        verbose         = True,
        seed            = 42,
    )

    print()
    print(describe_strategy(best))
