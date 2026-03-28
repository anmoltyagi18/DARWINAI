"""
anomaly_detector.py
-------------------
ML-based market anomaly detection module.

Detects:
  - Sudden volume spikes
  - Flash crashes
  - Price manipulation patterns (spoofing, layering, pump-and-dump signals)

Algorithms used:
  - Isolation Forest     : general-purpose anomaly scoring
  - Z-score / rolling    : fast univariate spike detection
  - DBSCAN clustering    : manipulation pattern clustering
  - Local Outlier Factor : density-based anomaly detection

Dependencies:
    pip install numpy pandas scikit-learn scipy
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class AnomalyType(str, Enum):
    VOLUME_SPIKE        = "volume_spike"
    FLASH_CRASH         = "flash_crash"
    PUMP_AND_DUMP       = "pump_and_dump"
    SPOOFING            = "spoofing"
    LAYERING            = "layering"
    GENERAL             = "general"


@dataclass
class Anomaly:
    timestamp: pd.Timestamp
    anomaly_type: AnomalyType
    severity: float          # 0.0 - 1.0
    confidence: float        # 0.0 - 1.0
    description: str
    affected_features: list[str] = field(default_factory=list)
    raw_score: float = 0.0

    def __repr__(self) -> str:
        return (
            f"Anomaly({self.anomaly_type.value} | "
            f"severity={self.severity:.2f} | "
            f"confidence={self.confidence:.2f} | "
            f"{self.timestamp})"
        )


@dataclass
class DetectionResult:
    ticker: str
    anomalies: list[Anomaly]
    anomaly_rate: float          # fraction of bars flagged
    risk_score: float            # aggregate 0-100
    summary: str

    def __repr__(self) -> str:
        return (
            f"DetectionResult(ticker={self.ticker} | "
            f"anomalies={len(self.anomalies)} | "
            f"risk_score={self.risk_score:.1f})"
        )


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive ML-ready features from OHLCV data.

    Expected columns: open, high, low, close, volume
    All column names are lowercased internally.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --- price features ---
    df["ret"]            = df["close"].pct_change()
    df["log_ret"]        = np.log(df["close"] / df["close"].shift(1))
    df["intrabar_range"] = (df["high"] - df["low"]) / df["open"].replace(0, np.nan)
    df["body_ratio"]     = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9)
    df["upper_wick"]     = (df["high"] - df[["open","close"]].max(axis=1)) / (df["high"] - df["low"] + 1e-9)
    df["lower_wick"]     = (df[["open","close"]].min(axis=1) - df["low"])  / (df["high"] - df["low"] + 1e-9)

    # --- rolling statistics (20-bar window) ---
    w = 20
    df["ret_zscore"]     = (df["ret"] - df["ret"].rolling(w).mean()) / (df["ret"].rolling(w).std() + 1e-9)
    df["vol_zscore"]     = (df["volume"] - df["volume"].rolling(w).mean()) / (df["volume"].rolling(w).std() + 1e-9)
    df["range_zscore"]   = (df["intrabar_range"] - df["intrabar_range"].rolling(w).mean()) / (df["intrabar_range"].rolling(w).std() + 1e-9)

    # --- momentum & reversal ---
    df["ret_5"]          = df["close"].pct_change(5)
    df["ret_10"]         = df["close"].pct_change(10)
    df["reversal"]       = df["ret"].rolling(3).sum()
    df["vol_price_corr"] = df["volume"].rolling(10).corr(df["close"])

    # --- volatility regime ---
    df["realized_vol"]   = df["log_ret"].rolling(w).std() * np.sqrt(252)

    return df.dropna()


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

class VolumeSpikeDetector:
    """
    Flags bars where volume deviates anomalously from its rolling baseline.
    Uses Z-score threshold + Isolation Forest confirmation.
    """

    def __init__(self, zscore_threshold: float = 3.5, contamination: float = 0.02):
        self.zscore_threshold = zscore_threshold
        self.contamination    = contamination
        self._iso             = IsolationForest(
            contamination=contamination,
            n_estimators=200,
            random_state=42,
        )

    def detect(self, df: pd.DataFrame) -> list[Anomaly]:
        feats = df[["vol_zscore", "volume", "ret"]].values
        scaler = StandardScaler()
        X = scaler.fit_transform(feats)
        iso_labels = self._iso.fit_predict(X)
        iso_scores = self._iso.score_samples(X)

        anomalies = []
        for i, row in enumerate(df.itertuples()):
            z = abs(row.vol_zscore)
            if z >= self.zscore_threshold and iso_labels[i] == -1:
                severity   = min(1.0, (z - self.zscore_threshold) / 5.0 + 0.4)
                confidence = min(1.0, 0.5 + (z / self.zscore_threshold) * 0.25)
                anomalies.append(Anomaly(
                    timestamp         = row.Index,
                    anomaly_type      = AnomalyType.VOLUME_SPIKE,
                    severity          = round(severity, 3),
                    confidence        = round(confidence, 3),
                    description       = (
                        f"Volume {row.volume:,.0f} is {z:.1f}sigma above the 20-bar mean. "
                        f"Price move: {row.ret*100:+.2f}%."
                    ),
                    affected_features = ["volume", "vol_zscore"],
                    raw_score         = float(iso_scores[i]),
                ))
        return anomalies


class FlashCrashDetector:
    """
    Identifies flash crashes: rapid, large price drops followed by
    partial or full recovery within a short window.
    """

    def __init__(
        self,
        drop_threshold: float  = -0.03,
        recovery_bars: int     = 5,
        recovery_frac: float   = 0.5,
        contamination: float   = 0.01,
    ):
        self.drop_threshold = drop_threshold
        self.recovery_bars  = recovery_bars
        self.recovery_frac  = recovery_frac
        self._lof           = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
        )

    def detect(self, df: pd.DataFrame) -> list[Anomaly]:
        closes = df["close"].values
        lows   = df["low"].values

        feats = StandardScaler().fit_transform(
            df[["ret_zscore", "range_zscore", "lower_wick"]].values
        )
        lof_labels = self._lof.fit_predict(feats)
        lof_scores = self._lof.negative_outlier_factor_

        anomalies = []
        for i, row in enumerate(df.itertuples()):
            drop         = row.ret
            intrabar_drop = (row.low - row.open) / (row.open + 1e-9)
            crash_signal  = drop <= self.drop_threshold or intrabar_drop <= self.drop_threshold

            if not crash_signal:
                continue

            end        = min(i + self.recovery_bars, len(closes) - 1)
            bottom     = min(closes[i], lows[i])
            peak_after = max(closes[i:end+1]) if end > i else closes[i]
            drop_size  = abs(bottom - closes[max(i-1, 0)])
            recovery   = (peak_after - bottom) / (drop_size + 1e-9)

            recovered   = recovery >= self.recovery_frac
            lof_outlier = lof_labels[i] == -1

            if lof_outlier or (crash_signal and recovered):
                severity   = min(1.0, abs(drop) / 0.10 + 0.3)
                confidence = 0.6 if (lof_outlier and recovered) else (0.45 if lof_outlier else 0.35)
                anomalies.append(Anomaly(
                    timestamp         = row.Index,
                    anomaly_type      = AnomalyType.FLASH_CRASH,
                    severity          = round(severity, 3),
                    confidence        = round(confidence, 3),
                    description       = (
                        f"Rapid price drop of {drop*100:.2f}% "
                        f"(intrabar: {intrabar_drop*100:.2f}%). "
                        f"Recovery {recovery*100:.0f}% within {self.recovery_bars} bars."
                    ),
                    affected_features = ["ret", "intrabar_range", "lower_wick"],
                    raw_score         = float(lof_scores[i]),
                ))
        return anomalies


class PriceManipulationDetector:
    """
    Detects three manipulation patterns:

    1. Pump-and-dump  -- sharp coordinated rise + volume, then reversal
    2. Spoofing       -- large volume bar with tiny net price move
    3. Layering       -- alternating up/down swings with elevated volume, low net displacement
    """

    def __init__(self, contamination: float = 0.02):
        self.contamination = contamination
        self._iso = IsolationForest(
            contamination=contamination,
            n_estimators=300,
            max_features=0.8,
            random_state=42,
        )

    def _pump_dump_score(self, df: pd.DataFrame) -> pd.Series:
        gain  = df["ret_10"]
        vol_z = df["vol_zscore"]
        rev   = -df["reversal"].shift(-3)
        return (gain * vol_z.clip(0) + rev.clip(0)).fillna(0)

    def _spoofing_score(self, df: pd.DataFrame) -> pd.Series:
        net_move = abs(df["ret"])
        vol_z    = df["vol_zscore"].clip(0)
        return (vol_z / (net_move * 100 + 1)).fillna(0)

    def _layering_score(self, df: pd.DataFrame) -> pd.Series:
        sign_changes  = (np.sign(df["ret"]) != np.sign(df["ret"].shift(1))).astype(int)
        rolling_flips = sign_changes.rolling(5).sum()
        net_disp      = abs(df["ret_5"])
        return (rolling_flips * df["vol_zscore"].clip(0) / (net_disp * 100 + 1)).fillna(0)

    def detect(self, df: pd.DataFrame) -> list[Anomaly]:
        df = df.copy()
        df["_pnd"]   = self._pump_dump_score(df)
        df["_spoof"] = self._spoofing_score(df)
        df["_layer"] = self._layering_score(df)

        feat_cols  = ["_pnd", "_spoof", "_layer", "vol_zscore", "ret_zscore", "body_ratio"]
        X          = StandardScaler().fit_transform(df[feat_cols].fillna(0).values)
        iso_labels = self._iso.fit_predict(X)
        iso_scores = self._iso.score_samples(X)

        dbscan         = DBSCAN(eps=1.5, min_samples=3)
        cluster_labels = dbscan.fit_predict(X)

        type_map = {
            "pump_and_dump": AnomalyType.PUMP_AND_DUMP,
            "spoofing":      AnomalyType.SPOOFING,
            "layering":      AnomalyType.LAYERING,
        }

        anomalies = []
        for i, row in enumerate(df.itertuples()):
            if iso_labels[i] != -1:
                continue

            pnd   = getattr(row, "_pnd",   0)
            spoof = getattr(row, "_spoof", 0)
            layer = getattr(row, "_layer", 0)

            kind_str  = max(
                [("pump_and_dump", pnd), ("spoofing", spoof), ("layering", layer)],
                key=lambda x: x[1],
            )[0]
            kind_enum = type_map[kind_str]
            in_cluster = cluster_labels[i] != -1
            severity   = min(1.0, abs(iso_scores[i]) * 0.6 + 0.3)
            confidence = min(1.0, 0.55 + (0.20 if in_cluster else 0.0))

            descriptions = {
                "pump_and_dump": (
                    f"Pump-and-dump signal: 10-bar gain {row.ret_10*100:+.1f}%, "
                    f"vol {row.vol_zscore:.1f}sigma above mean, reversal follows."
                ),
                "spoofing": (
                    f"Spoofing signal: vol {row.vol_zscore:.1f}sigma spike "
                    f"with near-zero net price move ({row.ret*100:+.3f}%)."
                ),
                "layering": (
                    f"Layering signal: rapid price oscillation with elevated volume "
                    f"and {row.ret_5*100:.2f}% net displacement over 5 bars."
                ),
            }

            anomalies.append(Anomaly(
                timestamp         = row.Index,
                anomaly_type      = kind_enum,
                severity          = round(severity, 3),
                confidence        = round(confidence, 3),
                description       = descriptions[kind_str],
                affected_features = ["volume", "ret", "ret_10", "body_ratio"],
                raw_score         = float(iso_scores[i]),
            ))
        return anomalies


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Unified market anomaly detector.

    Usage
    -----
    >>> detector = AnomalyDetector()
    >>> result = detector.fit_detect(df, ticker="AAPL")
    >>> print(detector.report(result))
    """

    def __init__(
        self,
        volume_zscore_threshold: float = 3.5,
        flash_crash_threshold: float   = -0.03,
        contamination: float           = 0.02,
        recovery_bars: int             = 5,
    ):
        self.volume_detector = VolumeSpikeDetector(
            zscore_threshold=volume_zscore_threshold,
            contamination=contamination,
        )
        self.crash_detector = FlashCrashDetector(
            drop_threshold=flash_crash_threshold,
            recovery_bars=recovery_bars,
            contamination=contamination / 2,
        )
        self.manipulation_detector = PriceManipulationDetector(
            contamination=contamination,
        )

    def fit_detect(self, df: pd.DataFrame, ticker: str = "UNKNOWN") -> DetectionResult:
        """
        Run all detectors on the supplied OHLCV DataFrame.

        Parameters
        ----------
        df     : pd.DataFrame with DatetimeIndex and columns
                 [open, high, low, close, volume]
        ticker : symbol label for the result object

        Returns
        -------
        DetectionResult
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)

        features = _build_features(df)

        vol_anomalies   = self.volume_detector.detect(features)
        crash_anomalies = self.crash_detector.detect(features)
        manip_anomalies = self.manipulation_detector.detect(features)

        all_anomalies = sorted(
            vol_anomalies + crash_anomalies + manip_anomalies,
            key=lambda a: a.timestamp,
        )

        # deduplicate same-timestamp same-type (keep highest severity)
        seen: dict[tuple, Anomaly] = {}
        for a in all_anomalies:
            key = (a.timestamp, a.anomaly_type)
            if key not in seen or a.severity > seen[key].severity:
                seen[key] = a
        all_anomalies = sorted(seen.values(), key=lambda a: a.timestamp)

        n_bars       = len(features)
        anomaly_rate = len(all_anomalies) / n_bars if n_bars else 0.0
        risk_score   = self._compute_risk_score(all_anomalies, anomaly_rate)
        summary      = self._build_summary(ticker, all_anomalies, risk_score, n_bars)

        return DetectionResult(
            ticker       = ticker,
            anomalies    = all_anomalies,
            anomaly_rate = round(anomaly_rate, 4),
            risk_score   = round(risk_score, 2),
            summary      = summary,
        )

    @staticmethod
    def _compute_risk_score(anomalies: list[Anomaly], anomaly_rate: float) -> float:
        if not anomalies:
            return 0.0
        type_weights = {
            AnomalyType.PUMP_AND_DUMP: 1.4,
            AnomalyType.SPOOFING:      1.3,
            AnomalyType.LAYERING:      1.2,
            AnomalyType.FLASH_CRASH:   1.1,
            AnomalyType.VOLUME_SPIKE:  0.9,
            AnomalyType.GENERAL:       1.0,
        }
        weighted   = sum(a.severity * a.confidence * type_weights.get(a.anomaly_type, 1.0) for a in anomalies)
        base       = (weighted / max(len(anomalies), 1)) * 60
        rate_bonus = min(40, anomaly_rate * 2000)
        return min(100.0, base + rate_bonus)

    @staticmethod
    def _build_summary(
        ticker: str,
        anomalies: list[Anomaly],
        risk_score: float,
        n_bars: int,
    ) -> str:
        if not anomalies:
            return f"{ticker}: No anomalies detected across {n_bars} bars. Risk score: 0."
        counts: dict[AnomalyType, int] = {}
        for a in anomalies:
            counts[a.anomaly_type] = counts.get(a.anomaly_type, 0) + 1
        count_str = ", ".join(
            f"{v} {k.value.replace('_',' ')}"
            for k, v in sorted(counts.items(), key=lambda x: -x[1])
        )
        level = (
            "CRITICAL" if risk_score >= 75 else
            "HIGH"     if risk_score >= 50 else
            "MEDIUM"   if risk_score >= 25 else
            "LOW"
        )
        return (
            f"{ticker}: {len(anomalies)} anomalies detected over {n_bars} bars "
            f"[{count_str}]. Risk score: {risk_score:.1f}/100 ({level})."
        )

    def report(self, result: DetectionResult, top_n: int = 10) -> str:
        """Return a formatted text report for a DetectionResult."""
        lines = [
            "=" * 70,
            f"  ANOMALY DETECTION REPORT -- {result.ticker}",
            "=" * 70,
            f"  {result.summary}",
            f"  Anomaly rate : {result.anomaly_rate*100:.2f}%",
            "-" * 70,
        ]
        if not result.anomalies:
            lines.append("  No anomalies found.")
        else:
            top = sorted(result.anomalies, key=lambda a: -(a.severity * a.confidence))[:top_n]
            for i, a in enumerate(top, 1):
                lines += [
                    f"  [{i:02d}] {a.anomaly_type.value.upper()}",
                    f"       Time       : {a.timestamp}",
                    f"       Severity   : {a.severity:.2f}  |  Confidence: {a.confidence:.2f}",
                    f"       Details    : {a.description}",
                    "",
                ]
        lines.append("=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def detect_anomalies(
    df: pd.DataFrame,
    ticker: str = "UNKNOWN",
    **kwargs,
) -> DetectionResult:
    """
    One-call shortcut.

    Parameters
    ----------
    df      : OHLCV DataFrame with DatetimeIndex
    ticker  : symbol string
    **kwargs: forwarded to AnomalyDetector.__init__

    Returns
    -------
    DetectionResult
    """
    return AnomalyDetector(**kwargs).fit_detect(df, ticker=ticker)


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

def _generate_demo_data(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Generate synthetic OHLCV data with injected anomalies."""
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")

    close = 100.0
    opens, highs, lows, closes, volumes = [], [], [], [], []

    for i in range(n):
        ret      = rng.normal(0.0002, 0.008)
        vol_mult = 1.0

        if i == 100:                      # volume spike
            ret      = rng.normal(0.015, 0.005)
            vol_mult = 8.0
        elif i == 200:                    # flash crash
            ret = -0.055
        elif 300 <= i <= 310:             # pump leg
            ret      = rng.normal(0.008, 0.003)
            vol_mult = 4.0
        elif 311 <= i <= 315:             # dump leg
            ret      = rng.normal(-0.012, 0.004)
            vol_mult = 3.0
        elif i == 400:                    # spoofing
            ret      = rng.normal(0.0001, 0.001)
            vol_mult = 6.0

        o     = close
        close = max(0.01, close * (1 + ret))
        h     = max(o, close) * (1 + abs(rng.normal(0, 0.002)))
        l     = min(o, close) * (1 - abs(rng.normal(0, 0.002)))
        v     = abs(rng.normal(1_000_000, 200_000)) * vol_mult

        opens.append(o);  closes.append(close)
        highs.append(h);  lows.append(l)
        volumes.append(v)

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=dates,
    )


if __name__ == "__main__":
    print("Running anomaly_detector demo...\n")
    demo_df  = _generate_demo_data(n=500)
    detector = AnomalyDetector(contamination=0.03)
    result   = detector.fit_detect(demo_df, ticker="DEMO")
    print(detector.report(result))
