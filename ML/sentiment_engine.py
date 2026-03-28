"""
sentiment_engine.py
===================
Financial News Sentiment Analysis Engine.

Fetches news headlines for a stock ticker, runs multi-layer sentiment
analysis, classifies each headline as Positive / Negative / Neutral,
and returns an aggregate sentiment score in [-1, +1].

Layers (applied in priority order)
-----------------------------------
1. Transformer model  — FinancialBERT or fallback to distilbert (best accuracy)
2. VADER              — rule-based, fast, no GPU needed           (fallback)
3. TextBlob           — pattern-based polarity                    (fallback)
4. Finance lexicon    — custom financial keyword dictionary       (always available)

Data Sources (tried in order)
------------------------------
1. yfinance  .news  — free, no key required
2. NewsAPI          — free tier; set env var  NEWS_API_KEY
3. Synthetic        — deterministic demo headlines when nothing else works

Usage
-----
    from sentiment_engine import SentimentEngine, analyze_sentiment

    # Quick one-liner
    result = analyze_sentiment("AAPL")
    print(result)

    # Full control
    engine = SentimentEngine(ticker="TSLA", max_headlines=20, use_transformer=True)
    result = engine.analyze()
    print(result.score)            # float in [-1, +1]
    print(result.label)            # "positive" | "negative" | "neutral"
    print(result.headlines_df)     # per-headline breakdown
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ── optional heavy dependencies ──────────────────────────────────────────────
try:
    import yfinance as yf
    _YF = True
except ImportError:
    _YF = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = True
except ImportError:
    _VADER = False

try:
    from textblob import TextBlob
    _TEXTBLOB = True
except ImportError:
    _TEXTBLOB = False

try:
    from transformers import pipeline as hf_pipeline, logging as hf_logging
    hf_logging.set_verbosity_error()
    _HF = True
except ImportError:
    _HF = False

try:
    import requests as _requests
    _REQUESTS = True
except ImportError:
    _REQUESTS = False


# ─────────────────────────────────────────────────────────────────────────────
# Financial Sentiment Lexicon
# ─────────────────────────────────────────────────────────────────────────────

_POSITIVE_FINANCE = {
    # earnings & growth
    "beat": 0.6, "beats": 0.6, "surpassed": 0.65, "exceeded": 0.6,
    "record": 0.55, "record-high": 0.7, "all-time high": 0.8,
    "growth": 0.5, "grew": 0.5, "soared": 0.75, "surge": 0.7,
    "surged": 0.7, "rally": 0.65, "rallied": 0.65, "jumped": 0.6,
    "climb": 0.5, "climbed": 0.5, "rose": 0.45, "rises": 0.45,
    "gain": 0.5, "gains": 0.5, "profit": 0.55, "profits": 0.55,
    "revenue": 0.35, "strong": 0.45, "outperform": 0.6,
    "upgrade": 0.65, "upgraded": 0.65, "buy": 0.5, "overweight": 0.5,
    "optimistic": 0.55, "bullish": 0.7, "positive": 0.4,
    "dividend": 0.4, "buyback": 0.5, "acquisition": 0.35,
    "partnership": 0.4, "expansion": 0.45, "innovative": 0.45,
    "breakthrough": 0.7, "approved": 0.55, "approval": 0.55,
    "guidance raised": 0.75, "raised guidance": 0.75,
}

_NEGATIVE_FINANCE = {
    "miss": -0.6, "misses": -0.6, "missed": -0.6, "disappoints": -0.65,
    "disappointed": -0.65, "below expectations": -0.7,
    "loss": -0.55, "losses": -0.55, "decline": -0.5, "declined": -0.5,
    "drop": -0.55, "dropped": -0.55, "fell": -0.5, "fall": -0.5,
    "plunge": -0.75, "plunged": -0.75, "crash": -0.8, "crashed": -0.8,
    "tumble": -0.65, "tumbled": -0.65, "slump": -0.6, "slumped": -0.6,
    "weak": -0.45, "weakness": -0.5, "downgrade": -0.65,
    "downgraded": -0.65, "sell": -0.5, "underperform": -0.6,
    "underweight": -0.5, "bearish": -0.7, "negative": -0.4,
    "investigation": -0.65, "lawsuit": -0.6, "fraud": -0.85,
    "recall": -0.6, "layoff": -0.55, "layoffs": -0.6,
    "bankruptcy": -0.9, "default": -0.75, "debt": -0.35,
    "warning": -0.5, "cautious": -0.4, "concern": -0.4,
    "risk": -0.3, "cut": -0.4, "cuts": -0.4, "guidance cut": -0.75,
    "lowered guidance": -0.75, "tariff": -0.4, "inflation": -0.35,
    "recession": -0.65, "slowdown": -0.5,
}


def _lexicon_score(text: str) -> float:
    """Score a text string using the finance lexicon. Returns [-1, +1]."""
    text_lower = text.lower()
    scores: List[float] = []

    # Multi-word phrases first
    for phrase, score in {**_POSITIVE_FINANCE, **_NEGATIVE_FINANCE}.items():
        if " " in phrase and phrase in text_lower:
            scores.append(score)

    # Single words
    words = re.findall(r"\b\w+\b", text_lower)
    for w in words:
        if w in _POSITIVE_FINANCE:
            scores.append(_POSITIVE_FINANCE[w])
        elif w in _NEGATIVE_FINANCE:
            scores.append(_NEGATIVE_FINANCE[w])

    if not scores:
        return 0.0
    return float(np.clip(np.mean(scores), -1.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HeadlineSentiment:
    """Sentiment result for a single headline."""
    headline: str
    score: float                    # [-1, +1]
    label: str                      # positive | negative | neutral
    confidence: float               # [0, 1]
    method: str                     # which analyser fired
    source: str = ""
    published: str = ""


@dataclass
class SentimentResult:
    """Aggregate sentiment result for a ticker."""
    ticker: str
    score: float                              # aggregate score [-1, +1]
    label: str                                # positive | negative | neutral
    confidence: float                         # average confidence
    n_headlines: int
    positive_pct: float
    negative_pct: float
    neutral_pct: float
    headlines: List[HeadlineSentiment] = field(default_factory=list)
    headlines_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    analysed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    method_used: str = ""

    # ── score → label helper ─────────────────────────────────────────────────
    @staticmethod
    def score_to_label(score: float, pos_thresh: float = 0.05,
                       neg_thresh: float = -0.05) -> str:
        if score >= pos_thresh:
            return "positive"
        if score <= neg_thresh:
            return "negative"
        return "neutral"

    def __str__(self) -> str:
        bar_len = 30
        filled = int((self.score + 1) / 2 * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        sign = "+" if self.score >= 0 else ""
        return (
            f"\n{'═'*58}\n"
            f"  Sentiment Analysis  ─  {self.ticker}\n"
            f"{'═'*58}\n"
            f"  Score    : {sign}{self.score:.4f}   [{bar}]\n"
            f"  Label    : {self.label.upper()}\n"
            f"  Confidence: {self.confidence*100:.1f}%\n"
            f"  Headlines: {self.n_headlines}\n"
            f"  Breakdown: ✅ {self.positive_pct:.1f}%  "
            f"⬜ {self.neutral_pct:.1f}%  "
            f"❌ {self.negative_pct:.1f}%\n"
            f"  Method   : {self.method_used}\n"
            f"  At       : {self.analysed_at}\n"
            f"{'─'*58}\n"
            + (
                "\n".join(
                    f"  {'▲' if h.label=='positive' else ('▼' if h.label=='negative' else '●')}"
                    f" [{h.score:+.2f}] {h.headline[:70]}"
                    for h in self.headlines[:8]
                )
                + ("\n  ..." if len(self.headlines) > 8 else "")
            )
            + f"\n{'═'*58}\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment Engine
# ─────────────────────────────────────────────────────────────────────────────

class SentimentEngine:
    """
    Financial news sentiment engine for a single stock ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. "AAPL").
    max_headlines : int
        Maximum number of headlines to analyse (default 30).
    use_transformer : bool
        Attempt to load a HuggingFace transformer model (default True).
        Set False to skip and use VADER/TextBlob/lexicon only.
    transformer_model : str
        HuggingFace model ID for sentiment.  Defaults to a FinancialBERT
        variant with automatic fallback to distilbert-base-uncased-finetuned-sst-2-english.
    pos_threshold : float
        Score boundary above which a headline is "positive" (default 0.05).
    neg_threshold : float
        Score boundary below which a headline is "negative" (default -0.05).
    news_api_key : str | None
        NewsAPI.org key.  Falls back to env var NEWS_API_KEY.
    """

    _DEFAULT_MODEL = "ProsusAI/finbert"
    _FALLBACK_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(
        self,
        ticker: str,
        max_headlines: int = 30,
        use_transformer: bool = True,
        transformer_model: str = _DEFAULT_MODEL,
        pos_threshold: float = 0.05,
        neg_threshold: float = -0.05,
        news_api_key: Optional[str] = None,
    ) -> None:
        self.ticker = ticker.upper()
        self.max_headlines = max_headlines
        self.use_transformer = use_transformer
        self.transformer_model = transformer_model
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.news_api_key = news_api_key or os.getenv("NEWS_API_KEY", "")

        self._transformer = None      # lazy-loaded
        self._vader = None            # lazy-loaded

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self) -> SentimentResult:
        """Fetch headlines and run full sentiment analysis."""
        raw_headlines = self._fetch_headlines()
        if not raw_headlines:
            warnings.warn(f"No headlines found for {self.ticker}; using synthetic data.")
            raw_headlines = _synthetic_headlines(self.ticker)

        headlines = raw_headlines[: self.max_headlines]
        results = [self._score_headline(h) for h in headlines]
        return self._aggregate(results)

    def score_text(self, text: str) -> HeadlineSentiment:
        """Score a single arbitrary text string."""
        return self._score_headline({"title": text, "source": "custom", "published": ""})

    # ── Headline Fetching ─────────────────────────────────────────────────────

    def _fetch_headlines(self) -> List[dict]:
        """Try yfinance → NewsAPI → synthetic, return list of dicts."""
        headlines: List[dict] = []

        # 1. yfinance
        if _YF:
            try:
                tk = yf.Ticker(self.ticker)
                news = tk.news or []
                for item in news:
                    title = item.get("title", "")
                    if title:
                        headlines.append({
                            "title": title,
                            "source": item.get("publisher", "yfinance"),
                            "published": _ts_to_str(item.get("providerPublishTime", 0)),
                        })
            except Exception:
                pass

        # 2. NewsAPI
        if len(headlines) < 5 and self.news_api_key and _REQUESTS:
            try:
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": f"{self.ticker} stock",
                    "apiKey": self.news_api_key,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": self.max_headlines,
                    "from": (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d"),
                }
                resp = _requests.get(url, params=params, timeout=8)
                if resp.status_code == 200:
                    for art in resp.json().get("articles", []):
                        title = art.get("title", "")
                        if title and "[Removed]" not in title:
                            headlines.append({
                                "title": title,
                                "source": art.get("source", {}).get("name", "NewsAPI"),
                                "published": art.get("publishedAt", ""),
                            })
            except Exception:
                pass

        return headlines

    # ── Scoring Pipeline ──────────────────────────────────────────────────────

    def _score_headline(self, item: dict) -> HeadlineSentiment:
        text = item.get("title", "")
        score, confidence, method = self._run_analyser(text)
        label = SentimentResult.score_to_label(score, self.pos_threshold, self.neg_threshold)
        return HeadlineSentiment(
            headline=text,
            score=round(score, 4),
            label=label,
            confidence=round(confidence, 4),
            method=method,
            source=item.get("source", ""),
            published=item.get("published", ""),
        )

    def _run_analyser(self, text: str) -> Tuple[float, float, str]:
        """Return (score, confidence, method_name) for a text."""
        if not text.strip():
            return 0.0, 0.0, "none"

        # ── Layer 1: Transformer ──────────────────────────────────────────────
        if self.use_transformer and _HF:
            try:
                result = self._get_transformer()(text[:512])[0]
                raw_label: str = result["label"].lower()
                conf: float = float(result["score"])
                # Map labels from different model families
                if raw_label in ("positive", "label_2", "pos"):
                    score = conf
                elif raw_label in ("negative", "label_0", "neg"):
                    score = -conf
                else:                            # neutral / label_1
                    score = 0.0
                    conf = max(conf, 0.5)
                # blend with lexicon for finance-aware calibration
                lex = _lexicon_score(text)
                score = float(np.clip(0.75 * score + 0.25 * lex, -1.0, 1.0))
                return score, conf, self.transformer_model.split("/")[-1]
            except Exception:
                pass  # fall through to next layer

        # ── Layer 2: VADER ────────────────────────────────────────────────────
        if _VADER:
            try:
                va = self._get_vader()
                compound = va.polarity_scores(text)["compound"]  # already in [-1,+1]
                lex = _lexicon_score(text)
                score = float(np.clip(0.65 * compound + 0.35 * lex, -1.0, 1.0))
                conf = abs(score) * 0.85 + 0.15
                return score, conf, "vader+lexicon"
            except Exception:
                pass

        # ── Layer 3: TextBlob ─────────────────────────────────────────────────
        if _TEXTBLOB:
            try:
                tb_score = TextBlob(text).sentiment.polarity       # [-1, +1]
                lex = _lexicon_score(text)
                score = float(np.clip(0.6 * tb_score + 0.4 * lex, -1.0, 1.0))
                conf = abs(score) * 0.75 + 0.15
                return score, conf, "textblob+lexicon"
            except Exception:
                pass

        # ── Layer 4: Pure Lexicon (always available) ──────────────────────────
        score = _lexicon_score(text)
        conf = min(abs(score) * 0.9 + 0.1, 1.0)
        return score, conf, "lexicon"

    # ── Aggregation ───────────────────────────────────────────────────────────

    def _aggregate(self, results: List[HeadlineSentiment]) -> SentimentResult:
        if not results:
            return SentimentResult(
                ticker=self.ticker, score=0.0, label="neutral",
                confidence=0.0, n_headlines=0,
                positive_pct=0.0, negative_pct=0.0, neutral_pct=0.0,
            )

        scores = np.array([h.score for h in results])
        confidences = np.array([h.confidence for h in results])

        # Confidence-weighted mean score
        if confidences.sum() > 0:
            agg_score = float(np.average(scores, weights=confidences))
        else:
            agg_score = float(scores.mean())
        agg_score = float(np.clip(agg_score, -1.0, 1.0))

        labels = [h.label for h in results]
        n = len(labels)
        pos_pct = labels.count("positive") / n * 100
        neg_pct = labels.count("negative") / n * 100
        neu_pct = labels.count("neutral") / n * 100

        # Determine dominant method
        methods = [h.method for h in results]
        method_used = max(set(methods), key=methods.count)

        df = pd.DataFrame([{
            "headline": h.headline,
            "score": h.score,
            "label": h.label,
            "confidence": h.confidence,
            "method": h.method,
            "source": h.source,
            "published": h.published,
        } for h in results])

        return SentimentResult(
            ticker=self.ticker,
            score=round(agg_score, 4),
            label=SentimentResult.score_to_label(agg_score, self.pos_threshold, self.neg_threshold),
            confidence=round(float(confidences.mean()), 4),
            n_headlines=n,
            positive_pct=round(pos_pct, 1),
            negative_pct=round(neg_pct, 1),
            neutral_pct=round(neu_pct, 1),
            headlines=results,
            headlines_df=df,
            method_used=method_used,
        )

    # ── Lazy Loaders ──────────────────────────────────────────────────────────

    def _get_transformer(self):
        if self._transformer is None:
            for model_id in (self.transformer_model, self._FALLBACK_MODEL):
                try:
                    self._transformer = hf_pipeline(
                        "sentiment-analysis",
                        model=model_id,
                        truncation=True,
                        max_length=512,
                    )
                    self.transformer_model = model_id
                    break
                except Exception:
                    continue
            if self._transformer is None:
                raise RuntimeError("Could not load any transformer model.")
        return self._transformer

    def _get_vader(self):
        if self._vader is None:
            self._vader = SentimentIntensityAnalyzer()
        return self._vader


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts_to_str(ts: int) -> str:
    try:
        return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ""


def _synthetic_headlines(ticker: str) -> List[dict]:
    """Deterministic demo headlines for offline / test use."""
    headlines = [
        f"{ticker} beats Q3 earnings estimates, stock surges after hours",
        f"Analysts upgrade {ticker} to Buy with raised price target",
        f"{ticker} announces record revenue and strong forward guidance",
        f"CEO of {ticker} expresses bullish outlook for next quarter",
        f"{ticker} expands into new markets with strategic partnership",
        f"{ticker} faces regulatory investigation over pricing practices",
        f"Supply chain disruptions weigh on {ticker} outlook",
        f"{ticker} misses revenue estimates amid slowing demand",
        f"Investors cautious as {ticker} cuts full-year guidance",
        f"{ticker} layoffs signal cost concerns, shares decline",
        f"{ticker} launches innovative product line to strong reviews",
        f"Market uncertainty creates headwinds for {ticker} growth plans",
    ]
    return [{"title": h, "source": "synthetic", "published": ""} for h in headlines]


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Function
# ─────────────────────────────────────────────────────────────────────────────

def analyze_sentiment(
    ticker: str,
    max_headlines: int = 30,
    use_transformer: bool = True,
    news_api_key: Optional[str] = None,
) -> SentimentResult:
    """
    One-call sentiment analysis for a stock ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker, e.g. "AAPL".
    max_headlines : int
        Max headlines to analyse (default 30).
    use_transformer : bool
        Attempt to use a HuggingFace FinBERT model (default True).
    news_api_key : str | None
        NewsAPI key (or set env var NEWS_API_KEY).

    Returns
    -------
    SentimentResult
        .score  : float in [-1, +1]
        .label  : "positive" | "negative" | "neutral"
        .headlines_df : per-headline DataFrame

    Examples
    --------
    >>> result = analyze_sentiment("MSFT")
    >>> print(result)
    >>> print(result.score)      # e.g. 0.3241
    >>> print(result.label)      # "positive"
    """
    engine = SentimentEngine(
        ticker=ticker,
        max_headlines=max_headlines,
        use_transformer=use_transformer,
        news_api_key=news_api_key,
    )
    return engine.analyze()


def compare_sentiments(
    tickers: List[str],
    max_headlines: int = 20,
    use_transformer: bool = True,
) -> pd.DataFrame:
    """
    Run sentiment analysis on multiple tickers and return a comparison DataFrame.

    Parameters
    ----------
    tickers : list[str]
    max_headlines : int
    use_transformer : bool

    Returns
    -------
    pd.DataFrame  sorted by score descending, columns:
        ticker, score, label, confidence, n_headlines,
        positive_pct, negative_pct, neutral_pct, method_used
    """
    rows = []
    for t in tickers:
        r = analyze_sentiment(t, max_headlines=max_headlines, use_transformer=use_transformer)
        rows.append({
            "ticker": r.ticker,
            "score": r.score,
            "label": r.label,
            "confidence": r.confidence,
            "n_headlines": r.n_headlines,
            "positive_pct": r.positive_pct,
            "negative_pct": r.negative_pct,
            "neutral_pct": r.neutral_pct,
            "method_used": r.method_used,
        })
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DEMO_TICKERS = ["AAPL", "TSLA", "MSFT"]

    print("\n🔷  Sentiment Engine — Demo Run")
    print(f"   Tickers : {DEMO_TICKERS}")
    print(f"   Note    : Uses synthetic headlines if yfinance unavailable\n")

    for ticker in DEMO_TICKERS:
        result = analyze_sentiment(ticker, use_transformer=False)  # fast demo
        print(result)

    print("\n📊  Side-by-side Comparison:")
    comparison = compare_sentiments(DEMO_TICKERS, use_transformer=False)
    print(comparison.to_string(index=False))
