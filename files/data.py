"""
data.py — Fetch OHLCV data from Yahoo Finance via yfinance.
"""

import pandas as pd
import yfinance as yf
from typing import Optional


def fetch_ohlcv(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.

    Args:
        ticker:   Yahoo Finance ticker symbol (e.g. "AAPL", "BTC-USD").
        period:   Data period when start/end are not provided.
                  Valid values: 1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max
        interval: Bar interval.
                  Valid values: 1m 2m 5m 15m 30m 60m 90m 1h 1d 5d 1wk 1mo 3mo
        start:    Start date string "YYYY-MM-DD" (overrides period).
        end:      End date string "YYYY-MM-DD" (overrides period).

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex (UTC-aware for intraday, date for daily+)

    Raises:
        ValueError: If no data is returned for the given parameters.
    """
    kwargs = dict(ticker=ticker, interval=interval, auto_adjust=True, progress=False)
    if start or end:
        kwargs["start"] = start
        kwargs["end"] = end
    else:
        kwargs["period"] = period

    raw = yf.download(**kwargs, multi_level_column=False)

    if raw.empty:
        raise ValueError(
            f"No data returned for ticker='{ticker}' "
            f"period='{period}' interval='{interval}'. "
            "Check the symbol and date range."
        )

    # Normalise column names
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df = df.dropna()
    return df
