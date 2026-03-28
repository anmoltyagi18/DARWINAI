# =============================================================================
# MODULE: ML/feature_engineering.py
# PROJECT: AIGOFIN - AI Quant Trading Platform
#
# PURPOSE:
#   Generates machine learning-ready features from OHLCV price data and
#   technical indicators. Transforms raw market data into a structured
#   feature matrix compatible with scikit-learn, FinRL, and Backtrader.
#
# INPUT COLUMNS EXPECTED:
#   open, high, low, close, volume,
#   RSI, MACD, MACD_signal, EMA_20, EMA_50, VWAP, Volume_MA_20
#
# OUTPUT:
#   A clean, normalized pandas DataFrame ready for ML model training/inference.
#
# AUTHOR: AIGOFIN System
# =============================================================================

import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Generates a comprehensive set of machine learning features from
    OHLCV market data and pre-computed technical indicators.

    Designed to be called once per symbol per timeframe and returns
    a fully enriched, ML-ready DataFrame.
    """

    # Number of periods for rolling volatility calculation
    VOLATILITY_WINDOW: int = 20

    # Number of periods for momentum calculation
    MOMENTUM_WINDOW: int = 10

    # Number of periods for Bollinger Band calculation
    BOLLINGER_WINDOW: int = 20
    BOLLINGER_STD: float = 2.0

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Raw input DataFrame containing OHLCV + indicator columns.
        """
        self._validate_input(df)
        # Work on a copy to never mutate caller data
        self.df: pd.DataFrame = df.copy()
        logger.info(f"FeatureEngineer initialized with {len(self.df)} rows.")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Ensure all required source columns are present."""
        required = {
            "open", "high", "low", "close", "volume",
            "RSI", "MACD", "MACD_signal",
            "EMA_20", "EMA_50", "VWAP", "Volume_MA_20",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Input DataFrame is missing columns: {missing}")

    # ------------------------------------------------------------------
    # Feature builders
    # ------------------------------------------------------------------

    def add_returns(self) -> "FeatureEngineer":
        """
        Compute simple percentage returns and log returns.

        returns      = (close_t / close_{t-1}) - 1
        log_returns  = ln(close_t / close_{t-1})
        """
        self.df["returns"] = self.df["close"].pct_change()
        self.df["log_returns"] = np.log(
            self.df["close"] / self.df["close"].shift(1)
        )
        logger.debug("add_returns: done.")
        return self  # Enable method chaining

    def add_volatility(self) -> "FeatureEngineer":
        """
        Rolling standard deviation of log returns over VOLATILITY_WINDOW periods.
        Represents realised short-term volatility.
        """
        if "log_returns" not in self.df.columns:
            self.add_returns()

        self.df["volatility"] = (
            self.df["log_returns"]
            .rolling(window=self.VOLATILITY_WINDOW)
            .std()
        )
        logger.debug("add_volatility: done.")
        return self

    def add_momentum(self) -> "FeatureEngineer":
        """
        Price momentum: percentage change in close price over MOMENTUM_WINDOW.

        momentum = (close_t / close_{t-N}) - 1
        """
        self.df["momentum"] = self.df["close"].pct_change(
            periods=self.MOMENTUM_WINDOW
        )
        logger.debug("add_momentum: done.")
        return self

    def add_trend_strength(self) -> "FeatureEngineer":
        """
        Trend strength based on separation between fast and slow EMAs.

        trend_strength = (EMA_20 - EMA_50) / EMA_50

        Positive → uptrend; Negative → downtrend; Near zero → sideways.
        """
        self.df["trend_strength"] = (
            (self.df["EMA_20"] - self.df["EMA_50"]) / self.df["EMA_50"]
        )
        logger.debug("add_trend_strength: done.")
        return self

    def add_volume_features(self) -> "FeatureEngineer":
        """
        Detect abnormal volume spikes relative to the 20-period volume MA.

        volume_spike = volume / Volume_MA_20

        Values > 1 indicate above-average activity; > 2 is a strong spike.
        """
        self.df["volume_spike"] = (
            self.df["volume"] / self.df["Volume_MA_20"]
        )
        logger.debug("add_volume_features: done.")
        return self

    def add_vwap_distance(self) -> "FeatureEngineer":
        """
        Price distance from VWAP, normalised by VWAP.

        price_distance_from_vwap = (close - VWAP) / VWAP

        Positive → price above VWAP (bullish intraday bias).
        Negative → price below VWAP (bearish intraday bias).
        """
        self.df["price_distance_from_vwap"] = (
            (self.df["close"] - self.df["VWAP"]) / self.df["VWAP"]
        )
        logger.debug("add_vwap_distance: done.")
        return self

    def add_rsi_scaled(self) -> "FeatureEngineer":
        """
        Scale RSI from [0, 100] to [-1, +1] for neural-network compatibility.

        rsi_scaled = (RSI - 50) / 50
        """
        self.df["rsi_scaled"] = (self.df["RSI"] - 50.0) / 50.0
        logger.debug("add_rsi_scaled: done.")
        return self

    def add_macd_features(self) -> "FeatureEngineer":
        """
        Derive the MACD histogram.

        macd_histogram = MACD - MACD_signal

        Positive histogram → bullish momentum building.
        Negative histogram → bearish momentum building.
        """
        self.df["macd_histogram"] = (
            self.df["MACD"] - self.df["MACD_signal"]
        )
        logger.debug("add_macd_features: done.")
        return self

    def _add_bollinger_position(self) -> None:
        """
        Internal: compute Bollinger Band position.

        bollinger_position = (close - lower_band) / (upper_band - lower_band)

        0 → at lower band (oversold zone)
        1 → at upper band (overbought zone)
        0.5 → at middle band
        """
        rolling_mean = (
            self.df["close"]
            .rolling(window=self.BOLLINGER_WINDOW)
            .mean()
        )
        rolling_std = (
            self.df["close"]
            .rolling(window=self.BOLLINGER_WINDOW)
            .std()
        )
        upper_band = rolling_mean + self.BOLLINGER_STD * rolling_std
        lower_band = rolling_mean - self.BOLLINGER_STD * rolling_std
        band_width = upper_band - lower_band

        # Clip to [0, 1] to handle occasional out-of-band extremes
        self.df["bollinger_position"] = (
            (self.df["close"] - lower_band) / band_width
        ).clip(0.0, 1.0)

    # ------------------------------------------------------------------
    # Master builder
    # ------------------------------------------------------------------

    def build_feature_matrix(
        self,
        drop_na: bool = True,
        feature_cols: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Run all feature-creation methods and return a clean feature matrix.

        Parameters
        ----------
        drop_na : bool
            If True, drop rows that contain NaN values introduced by
            rolling windows. Default = True.
        feature_cols : list, optional
            Subset of feature columns to return. If None, all engineered
            features are returned alongside the source columns.

        Returns
        -------
        pd.DataFrame
            ML-ready feature matrix.
        """
        # Build every feature group
        (
            self
            .add_returns()
            .add_volatility()
            .add_momentum()
            .add_trend_strength()
            .add_volume_features()
            .add_vwap_distance()
            .add_rsi_scaled()
            .add_macd_features()
        )
        self._add_bollinger_position()

        if drop_na:
            before = len(self.df)
            self.df.dropna(inplace=True)
            dropped = before - len(self.df)
            logger.info(f"build_feature_matrix: dropped {dropped} NaN rows.")

        # Define the canonical feature column order
        default_features = [
            # Source columns preserved for context
            "open", "high", "low", "close", "volume",
            "RSI", "MACD", "MACD_signal", "EMA_20", "EMA_50",
            "VWAP", "Volume_MA_20",
            # Engineered features
            "returns",
            "log_returns",
            "volatility",
            "momentum",
            "trend_strength",
            "price_distance_from_vwap",
            "volume_spike",
            "bollinger_position",
            "rsi_scaled",
            "macd_histogram",
        ]

        cols = feature_cols if feature_cols else default_features
        # Only return columns that actually exist (defensive)
        cols = [c for c in cols if c in self.df.columns]

        logger.info(
            f"build_feature_matrix: returning {len(self.df)} rows × "
            f"{len(cols)} columns."
        )
        return self.df[cols].copy()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def get_feature_names() -> list:
        """
        Return the canonical list of engineered feature names.
        Useful for building ML pipelines that reference column names.
        """
        return [
            "returns",
            "log_returns",
            "volatility",
            "momentum",
            "trend_strength",
            "price_distance_from_vwap",
            "volume_spike",
            "bollinger_position",
            "rsi_scaled",
            "macd_histogram",
        ]
