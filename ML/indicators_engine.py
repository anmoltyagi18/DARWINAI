# indicators_engine.py

import pandas as pd
import numpy as np


class IndicatorsEngine:

    def __init__(self, df: pd.DataFrame):
        """
        df must contain columns:
        open, high, low, close, volume
        """
        self.df = df.copy()

    # -----------------------------
    # EMA
    # -----------------------------
    def ema(self, period: int = 20, column: str = "close"):
        self.df[f"EMA_{period}"] = (
            self.df[column]
            .ewm(span=period, adjust=False)
            .mean()
        )

    # -----------------------------
    # RSI
    # -----------------------------
    def rsi(self, period: int = 14):

        delta = self.df["close"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()

        rs = avg_gain / avg_loss

        self.df["RSI"] = 100 - (100 / (1 + rs))

    # -----------------------------
    # MACD
    # -----------------------------
    def macd(self):

        ema12 = self.df["close"].ewm(span=12, adjust=False).mean()
        ema26 = self.df["close"].ewm(span=26, adjust=False).mean()

        self.df["MACD"] = ema12 - ema26
        self.df["MACD_signal"] = self.df["MACD"].ewm(span=9, adjust=False).mean()
        self.df["MACD_hist"] = self.df["MACD"] - self.df["MACD_signal"]

    # -----------------------------
    # Bollinger Bands
    # -----------------------------
    def bollinger_bands(self, period: int = 20):

        ma = self.df["close"].rolling(period).mean()
        std = self.df["close"].rolling(period).std()

        self.df["BB_middle"] = ma
        self.df["BB_upper"] = ma + (2 * std)
        self.df["BB_lower"] = ma - (2 * std)

    # -----------------------------
    # VWAP
    # -----------------------------
    def vwap(self):

        typical_price = (
            self.df["high"]
            + self.df["low"]
            + self.df["close"]
        ) / 3

        cumulative_vp = (typical_price * self.df["volume"]).cumsum()
        cumulative_volume = self.df["volume"].cumsum()

        self.df["VWAP"] = cumulative_vp / cumulative_volume

    # -----------------------------
    # Volume Moving Average
    # -----------------------------
    def volume_ma(self, period: int = 20):

        self.df[f"Volume_MA_{period}"] = (
            self.df["volume"]
            .rolling(period)
            .mean()
        )

    # -----------------------------
    # Run all indicators
    # -----------------------------
    def calculate_all(self):

        self.ema(20)
        self.ema(50)
        self.rsi()
        self.macd()
        self.bollinger_bands()
        self.vwap()
        self.volume_ma()

        return self.df