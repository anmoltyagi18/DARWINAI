"""
rl_trader.py
------------
Reinforcement learning trading agent using Stable-Baselines3.

Environment
-----------
State  : [price_norm, rsi, macd_norm, macd_signal_norm, volume_norm,
            position, unrealised_pnl_pct, cash_ratio, steps_since_trade]
Actions: 0 = HOLD  |  1 = BUY  |  2 = SELL
Reward : profit percentage on closed trade + small survival bonus;
        penalty for excessive holding and invalid actions.

Functions
---------
  train_agent(df, ...)   -> trained SB3 model + training metadata dict
  predict_action(model, state_dict) -> {"action": str, "action_id": int,
                                        "confidence": float, "state": ...}

Dependencies
------------
  pip install numpy pandas gymnasium stable-baselines3 torch
  (Optional for faster training: pip install sb3-contrib)
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

# gymnasium (SB3 >=2.x expects gymnasium, not gym)
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_HOLD = 0
ACTION_BUY  = 1
ACTION_SELL = 2
ACTION_NAMES = {ACTION_HOLD: "HOLD", ACTION_BUY: "BUY", ACTION_SELL: "SELL"}

# observation vector length (must match _get_obs())
OBS_DIM = 9


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RSI, MACD, and MACD-signal from OHLCV data.
    Returns a new DataFrame with added columns; NaN rows dropped.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    close = df["close"]

    # --- RSI (14) ---
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["rsi"] = (100 - 100 / (1 + rs)) / 100.0   # normalised 0–1

    # --- MACD (12, 26, 9) ---
    ema12          = close.ewm(span=12, adjust=False).mean()
    ema26          = close.ewm(span=26, adjust=False).mean()
    df["macd"]     = ema12 - ema26
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()

    # rolling normalise MACD by price to make it scale-invariant
    macd_std           = df["macd"].rolling(50).std().replace(0, 1e-9)
    df["macd_norm"]    = df["macd"]     / macd_std
    df["macd_sig_norm"]= df["macd_sig"] / macd_std

    # --- volume (rolling z-score, clipped) ---
    vol_mean       = df["volume"].rolling(20).mean()
    vol_std        = df["volume"].rolling(20).std().replace(0, 1e-9)
    df["vol_norm"] = ((df["volume"] - vol_mean) / vol_std).clip(-3, 3) / 3.0

    # --- price change (1-bar log return, clipped) ---
    df["log_ret"] = np.log(close / close.shift(1)).clip(-0.1, 0.1) / 0.1

    return df.dropna().reset_index(drop=False)   # keep timestamp as column "index"


# ---------------------------------------------------------------------------
# Custom RL environment
# ---------------------------------------------------------------------------

class TradingEnv(gym.Env):
    """
    Single-asset discrete-action trading environment.

    Observation space (Box, shape=(OBS_DIM,)):
        [log_ret, rsi, macd_norm, macd_sig_norm, vol_norm,
         position_flag, unrealised_pnl_pct, cash_ratio, steps_since_trade_norm]

    Action space (Discrete 3):
        0 = HOLD  |  1 = BUY  |  2 = SELL

    Reward:
        On a closed trade   : profit_pct * 100  (capped ±20)
        Every step          : +0.01 survival bonus
        Invalid action      : -0.5 penalty (e.g. BUY when already long)
        Excessive hold      : -0.05 per step after 20 consecutive HOLDs
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital:  float = 100_000.0,
        commission_pct:   float = 0.001,
        max_hold_penalty: int   = 20,
        window_size:      int   = 1,          # kept for API compat; env is step-based
    ):
        super().__init__()
        self.df              = df.reset_index(drop=True)
        self.n_steps         = len(df)
        self.initial_capital = initial_capital
        self.commission_pct  = commission_pct
        self.max_hold_penalty= max_hold_penalty

        self.action_space      = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        # state variables (reset in reset())
        self._step       = 0
        self._position   = 0          # 0=flat, 1=long
        self._entry_price= 0.0
        self._cash       = initial_capital
        self._shares     = 0.0
        self._hold_count = 0
        self._trades: list[dict] = []

    # ------------------------------------------------------------------
    # gym interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed:    Optional[int]  = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # start after warm-up (enough bars for all indicators)
        self._step        = 0
        self._position    = 0
        self._entry_price = 0.0
        self._cash        = self.initial_capital
        self._shares      = 0.0
        self._hold_count  = 0
        self._trades      = []
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        row       = self.df.iloc[self._step]
        price     = float(row["close"])
        reward    = 0.01           # survival bonus

        # --- execute action ---
        invalid = False
        if action == ACTION_BUY:
            if self._position == 0:
                self._shares      = (self._cash * (1 - self.commission_pct)) / price
                self._cash        = 0.0
                self._entry_price = price
                self._position    = 1
                self._hold_count  = 0
            else:
                invalid = True     # already long

        elif action == ACTION_SELL:
            if self._position == 1:
                proceeds          = self._shares * price * (1 - self.commission_pct)
                profit_pct        = (price - self._entry_price) / self._entry_price
                reward           += float(np.clip(profit_pct * 100, -20, 20))
                self._cash        = proceeds
                self._shares      = 0.0
                self._position    = 0
                self._hold_count  = 0
                self._trades.append({
                    "step":       self._step,
                    "entry":      self._entry_price,
                    "exit":       price,
                    "profit_pct": profit_pct,
                })
                self._entry_price = 0.0
            else:
                invalid = True     # nothing to sell

        else:   # HOLD
            self._hold_count += 1

        if invalid:
            reward -= 0.5

        if self._hold_count > self.max_hold_penalty:
            reward -= 0.05

        self._step += 1
        terminated = self._step >= self.n_steps - 1
        truncated  = False

        # force-close on last bar
        if terminated and self._position == 1:
            price_last = float(self.df.iloc[-1]["close"])
            proceeds   = self._shares * price_last * (1 - self.commission_pct)
            profit_pct = (price_last - self._entry_price) / self._entry_price
            reward    += float(np.clip(profit_pct * 100, -20, 20))
            self._cash = proceeds
            self._shares = 0.0
            self._position = 0

        info = {
            "step":      self._step,
            "cash":      self._cash,
            "position":  self._position,
            "n_trades":  len(self._trades),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        row   = self.df.iloc[min(self._step, self.n_steps - 1)]
        price = float(row["close"])
        equity= self._cash + self._shares * price
        pos   = ACTION_NAMES.get(self._position + 1 if self._position else 0, "FLAT")
        print(
            f"step={self._step:5d} | price={price:8.2f} | "
            f"equity={equity:10.2f} | pos={self._position} | trades={len(self._trades)}"
        )

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        idx = min(self._step, self.n_steps - 1)
        row = self.df.iloc[idx]

        price      = float(row["close"])
        unrealised = ((price - self._entry_price) / self._entry_price
                      if self._position == 1 and self._entry_price > 0 else 0.0)
        total_equity = self._cash + self._shares * price
        cash_ratio   = self._cash / total_equity if total_equity > 0 else 1.0
        hold_norm    = min(self._hold_count / self.max_hold_penalty, 1.0)

        obs = np.array([
            float(row.get("log_ret",      0.0)),
            float(row.get("rsi",          0.5)),
            float(row.get("macd_norm",    0.0)),
            float(row.get("macd_sig_norm",0.0)),
            float(row.get("vol_norm",     0.0)),
            float(self._position),                # 0 or 1
            float(np.clip(unrealised, -1, 1)),
            float(np.clip(cash_ratio, 0, 1)),
            float(hold_norm),
        ], dtype=np.float32)

        return obs

    # ------------------------------------------------------------------
    # Helper properties
    # ------------------------------------------------------------------

    def get_trade_log(self) -> list[dict]:
        return list(self._trades)

    def current_equity(self) -> float:
        price = float(self.df.iloc[min(self._step, self.n_steps - 1)]["close"])
        return self._cash + self._shares * price


# ---------------------------------------------------------------------------
# Training callback — prints progress
# ---------------------------------------------------------------------------

class ProgressCallback(BaseCallback):
    def __init__(self, log_interval: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._last_log    = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log >= self.log_interval:
            self._last_log = self.num_timesteps
            if self.verbose:
                print(f"  [RL] timesteps={self.num_timesteps:,}", flush=True)
        return True


# ---------------------------------------------------------------------------
# Public API — train_agent()
# ---------------------------------------------------------------------------

def train_agent(
    df: pd.DataFrame,
    algorithm:       str   = "PPO",         # "PPO", "A2C"
    total_timesteps: int   = 100_000,
    initial_capital: float = 100_000.0,
    commission_pct:  float = 0.001,
    learning_rate:   float = 3e-4,
    n_envs:          int   = 1,
    eval_freq:       int   = 10_000,
    save_path:       Optional[str] = None,
    verbose:         int   = 1,
    seed:            int   = 42,
) -> tuple[Any, dict]:
    """
    Train a Stable-Baselines3 RL agent on historical OHLCV data.

    Parameters
    ----------
    df               : OHLCV DataFrame with DatetimeIndex or integer index.
                       Required columns: open, high, low, close, volume.
    algorithm        : SB3 algorithm — "PPO" (default) or "A2C".
    total_timesteps  : total env steps for training.
    initial_capital  : starting cash for the trading env.
    commission_pct   : round-trip commission fraction.
    learning_rate    : optimizer learning rate.
    n_envs           : number of parallel envs (PPO supports >1).
    eval_freq        : evaluate every N steps and save best model.
    save_path        : if set, saves final model to this path (.zip).
    verbose          : 0=silent, 1=progress, 2=SB3 debug.
    seed             : random seed for reproducibility.

    Returns
    -------
    model            : trained SB3 model (PPO or A2C instance).
    metadata         : dict with training info, feature stats, eval results.
    """
    if verbose:
        print(f"\n[RL Trader] Preparing features...")

    features_df = _compute_indicators(df)

    if verbose:
        print(f"[RL Trader] Dataset: {len(features_df)} bars after indicator warm-up")
        print(f"[RL Trader] Algorithm: {algorithm.upper()} | timesteps: {total_timesteps:,}")

    # split: 80% train, 20% eval
    split      = int(len(features_df) * 0.8)
    train_df   = features_df.iloc[:split].reset_index(drop=True)
    eval_df    = features_df.iloc[split:].reset_index(drop=True)

    if len(train_df) < 50:
        raise ValueError(
            f"Training set too small ({len(train_df)} bars). "
            "Provide at least 200 bars of OHLCV data."
        )

    # build environments
    def make_train_env():
        env = TradingEnv(
            train_df,
            initial_capital=initial_capital,
            commission_pct=commission_pct,
        )
        return Monitor(env)

    def make_eval_env():
        env = TradingEnv(
            eval_df,
            initial_capital=initial_capital,
            commission_pct=commission_pct,
        )
        return Monitor(env)

    train_env = make_vec_env(make_train_env, n_envs=n_envs)
    eval_env  = DummyVecEnv([make_eval_env])

    # algorithm-specific hyper-parameters
    algo_upper = algorithm.upper()
    common_kwargs = dict(
        learning_rate = learning_rate,
        verbose       = max(0, verbose - 1),
        seed          = seed,
    )

    if algo_upper == "PPO":
        model = PPO(
            "MlpPolicy",
            train_env,
            n_steps      = 2048,
            batch_size   = 64,
            n_epochs     = 10,
            gamma        = 0.99,
            gae_lambda   = 0.95,
            clip_range   = 0.2,
            ent_coef     = 0.01,
            policy_kwargs= dict(net_arch=[128, 128]),
            **common_kwargs,
        )
    elif algo_upper == "A2C":
        model = A2C(
            "MlpPolicy",
            train_env,
            n_steps      = 5,
            gamma        = 0.99,
            gae_lambda   = 1.0,
            ent_coef     = 0.01,
            policy_kwargs= dict(net_arch=[128, 128]),
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose 'PPO' or 'A2C'.")

    # callbacks
    callbacks = [ProgressCallback(log_interval=max(1000, total_timesteps // 20), verbose=verbose)]
    if eval_freq > 0 and len(eval_df) > 10:
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path = save_path or "./rl_trader_best",
            log_path             = None,
            eval_freq            = max(eval_freq // n_envs, 1),
            n_eval_episodes      = 3,
            deterministic        = True,
            verbose              = 0,
        )
        callbacks.append(eval_cb)

    if verbose:
        print(f"[RL Trader] Training started...\n")

    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    if verbose:
        print(f"\n[RL Trader] Training complete.")

    # optional: save final model
    if save_path:
        model.save(save_path)
        if verbose:
            print(f"[RL Trader] Model saved to: {save_path}.zip")

    # --- quick eval on held-out set ---
    eval_metrics = _evaluate_model(model, eval_df, initial_capital, commission_pct)

    metadata = {
        "algorithm":        algo_upper,
        "total_timesteps":  total_timesteps,
        "train_bars":       len(train_df),
        "eval_bars":        len(eval_df),
        "obs_dim":          OBS_DIM,
        "actions":          ACTION_NAMES,
        "feature_columns":  ["log_ret", "rsi", "macd_norm", "macd_sig_norm",
                              "vol_norm", "position", "unrealised_pnl_pct",
                              "cash_ratio", "steps_since_trade_norm"],
        "eval_metrics":     eval_metrics,
        "save_path":        save_path,
    }

    if verbose:
        print("\n[RL Trader] Eval metrics on held-out set:")
        for k, v in eval_metrics.items():
            print(f"  {k:<25}: {v}")

    return model, metadata


# ---------------------------------------------------------------------------
# Public API — predict_action()
# ---------------------------------------------------------------------------

def predict_action(
    model:          Any,
    state:          dict,
    deterministic:  bool = True,
) -> dict:
    """
    Predict the next trading action for a given market state.

    Parameters
    ----------
    model         : trained SB3 model returned by train_agent().
    state         : dict with keys matching the observation features:
                    {
                      "log_ret":           float,   # 1-bar log return (-1 to 1)
                      "rsi":               float,   # RSI normalised 0-1
                      "macd_norm":         float,   # MACD / rolling std
                      "macd_sig_norm":     float,   # MACD signal / rolling std
                      "vol_norm":          float,   # volume z-score / 3
                      "position":          int,     # 0=flat, 1=long
                      "unrealised_pnl_pct":float,   # open trade P&L fraction
                      "cash_ratio":        float,   # cash / total equity
                      "steps_since_trade": float,   # normalised 0-1
                    }
                    Missing keys default to 0.
    deterministic : use greedy (True) or stochastic (False) policy.

    Returns
    -------
    dict:
        {
          "action":       str,    # "HOLD", "BUY", or "SELL"
          "action_id":    int,    # 0, 1, or 2
          "confidence":   float,  # max action probability (0-1)
          "probabilities":dict,   # {"HOLD": p, "BUY": p, "SELL": p}
          "state_used":   list,   # raw observation vector
        }
    """
    keys = [
        "log_ret", "rsi", "macd_norm", "macd_sig_norm", "vol_norm",
        "position", "unrealised_pnl_pct", "cash_ratio", "steps_since_trade",
    ]
    obs_vec = np.array(
        [float(state.get(k, 0.0)) for k in keys],
        dtype=np.float32,
    ).reshape(1, -1)

    action_id, _states = model.predict(obs_vec, deterministic=deterministic)
    action_id = int(action_id)

    # extract action probabilities from policy distribution
    import torch
    with torch.no_grad():
        obs_tensor = model.policy.obs_to_tensor(obs_vec)[0]
        dist       = model.policy.get_distribution(obs_tensor)
        probs      = dist.distribution.probs.cpu().numpy().flatten()

    probabilities = {ACTION_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)}
    confidence    = float(probs[action_id])

    return {
        "action":        ACTION_NAMES[action_id],
        "action_id":     action_id,
        "confidence":    round(confidence, 4),
        "probabilities": probabilities,
        "state_used":    obs_vec.flatten().tolist(),
    }


# ---------------------------------------------------------------------------
# State builder helper — convert a raw OHLCV row + portfolio state to a dict
# ---------------------------------------------------------------------------

def build_state(
    df:                pd.DataFrame,
    bar_index:         int,
    position:          int   = 0,
    entry_price:       float = 0.0,
    cash:              float = 100_000.0,
    total_equity:      float = 100_000.0,
    steps_since_trade: int   = 0,
    max_hold:          int   = 20,
) -> dict:
    """
    Convenience function: build a state dict from a features DataFrame row.

    df must have been processed with _compute_indicators() already, or you
    can pass a raw OHLCV df and this function will compute features on-the-fly
    (slower — prefer pre-computing for live loops).
    """
    if "rsi" not in df.columns:
        df = _compute_indicators(df)
    row   = df.iloc[bar_index]
    price = float(row["close"])
    unr   = ((price - entry_price) / entry_price
             if position == 1 and entry_price > 0 else 0.0)
    return {
        "log_ret":            float(row.get("log_ret",       0.0)),
        "rsi":                float(row.get("rsi",           0.5)),
        "macd_norm":          float(row.get("macd_norm",     0.0)),
        "macd_sig_norm":      float(row.get("macd_sig_norm", 0.0)),
        "vol_norm":           float(row.get("vol_norm",      0.0)),
        "position":           float(position),
        "unrealised_pnl_pct": float(np.clip(unr, -1, 1)),
        "cash_ratio":         float(np.clip(cash / max(total_equity, 1e-9), 0, 1)),
        "steps_since_trade":  float(min(steps_since_trade / max(max_hold, 1), 1.0)),
    }


# ---------------------------------------------------------------------------
# Internal evaluation helper
# ---------------------------------------------------------------------------

def _evaluate_model(
    model:           Any,
    eval_df:         pd.DataFrame,
    initial_capital: float,
    commission_pct:  float,
) -> dict:
    """Run one deterministic episode on eval_df and return performance metrics."""
    env   = TradingEnv(eval_df, initial_capital=initial_capital, commission_pct=commission_pct)
    obs, _= env.reset()
    done  = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += reward
        done = terminated or truncated

    trades    = env.get_trade_log()
    final_eq  = env.current_equity()
    n         = len(trades)
    wins      = [t for t in trades if t["profit_pct"] > 0]
    total_ret = (final_eq - initial_capital) / initial_capital

    action_counts = {ACTION_NAMES[a]: 0 for a in range(3)}
    env2, _  = TradingEnv(eval_df).__class__(eval_df).reset() if False else (None, None)
    # re-run to count actions
    env3  = TradingEnv(eval_df, initial_capital=initial_capital, commission_pct=commission_pct)
    obs, _= env3.reset()
    done  = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action_counts[ACTION_NAMES[int(action)]] = action_counts.get(ACTION_NAMES[int(action)], 0) + 1
        obs, _, terminated, truncated, _ = env3.step(int(action))
        done = terminated or truncated

    return {
        "total_return_pct": round(total_ret * 100, 2),
        "final_equity":     round(final_eq, 2),
        "total_trades":     n,
        "win_rate_%":       round(len(wins) / n * 100, 1) if n else 0.0,
        "avg_profit_%":     round(
            float(np.mean([t["profit_pct"] * 100 for t in trades])), 2
        ) if trades else 0.0,
        "total_reward":     round(total_reward, 2),
        "action_counts":    action_counts,
    }


# ---------------------------------------------------------------------------
# Model persistence helpers
# ---------------------------------------------------------------------------

def save_model(model: Any, path: str) -> str:
    """Save model to <path>.zip. Returns the saved path."""
    model.save(path)
    return f"{path}.zip"


def load_model(path: str, algorithm: str = "PPO") -> Any:
    """Load a previously saved model. algorithm must match what was trained."""
    algo_map = {"PPO": PPO, "A2C": A2C}
    cls = algo_map.get(algorithm.upper())
    if cls is None:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose 'PPO' or 'A2C'.")
    return cls.load(path)


# ---------------------------------------------------------------------------
# Demo data generator
# ---------------------------------------------------------------------------

def _generate_demo_data(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    """Synthetic OHLCV series with regime changes."""
    rng    = np.random.default_rng(seed)
    dates  = pd.date_range("2019-01-01", periods=n, freq="1D")
    close  = 100.0
    data   = []
    vol    = 0.012
    drift  = 0.0003

    for i in range(n):
        if i % 200 == 0:
            vol   = rng.choice([0.008, 0.015, 0.025])
            drift = rng.choice([-0.0002, 0.0002, 0.0005])
        ret   = rng.normal(drift, vol)
        o     = close
        close = max(0.5, close * (1 + ret))
        h     = max(o, close) * (1 + abs(rng.normal(0, vol * 0.4)))
        l     = min(o, close) * (1 - abs(rng.normal(0, vol * 0.4)))
        v     = abs(rng.normal(1_500_000, 300_000))
        data.append({"open": o, "high": h, "low": l, "close": close, "volume": v})

    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  rl_trader.py — RL Trading Agent Demo")
    print("=" * 60)

    df = _generate_demo_data(n=1500)
    print(f"\nGenerated {len(df)} bars of synthetic OHLCV data.\n")

    # --- train ---
    model, meta = train_agent(
        df,
        algorithm        = "PPO",
        total_timesteps  = 50_000,   # increase for better results
        initial_capital  = 100_000,
        commission_pct   = 0.001,
        verbose          = 1,
        seed             = 42,
    )

    print("\nTraining metadata:")
    for k, v in meta.items():
        if k != "eval_metrics":
            print(f"  {k}: {v}")

    # --- predict on latest bar ---
    features_df = _compute_indicators(df)
    state = build_state(
        features_df,
        bar_index    = len(features_df) - 1,
        position     = 0,
        cash         = 100_000,
        total_equity = 100_000,
    )

    result = predict_action(model, state, deterministic=True)
    print(f"\nPrediction for latest bar:")
    print(f"  Action      : {result['action']}")
    print(f"  Confidence  : {result['confidence']:.2%}")
    print(f"  Probabilities: {result['probabilities']}")
