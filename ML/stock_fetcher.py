"""
stock_fetcher.py — AIGOFIN ML Package
======================================
Unified data fetching layer for historical OHLCV data.
Handles symbol normalization and yfinance requests.
"""

from __future__ import annotations
import yfinance as yf
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def normalize_symbol(symbol: str) -> str:
    """
    Standardize symbol format for Yahoo Finance.
    - Converts .NSE to .NS
    - Ensures uppercase
    """
    s = symbol.upper().strip()
    if s.endswith(".NSE"):
        s = s.replace(".NSE", ".NS")
    return s

def fetch_ohlcv(
    symbol:   str,
    period:   str = "1mo",
    interval: str = "1d",
    timeout:  int = 10
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Yahoo Finance.
    
    Parameters
    ----------
    symbol   : Ticker symbol (e.g. "AAPL", "RELIANCE.NS")
    period   : 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max
    interval : 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    
    Returns
    -------
    pd.DataFrame with OHLCV columns.
    """
    clean_symbol = normalize_symbol(symbol)
    logger.info(f"Fetching data for {clean_symbol} ({period}/{interval})")
    
    try:
        ticker = yf.Ticker(clean_symbol)
        df = ticker.history(period=period, interval=interval, timeout=timeout)
        
        if df.empty:
            logger.warning(f"No data returned for {clean_symbol}. Trying fallback search...")
            # Fallback — sometimes yfinance needs a direct download
            df = yf.download(clean_symbol, period=period, interval=interval, progress=False, timeout=timeout)
            
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {clean_symbol}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Test fetch
    test_df = fetch_ohlcv("RELIANCE.NS", period="5d", interval="1h")
    print(f"Fetched {len(test_df)} bars for RELIANCE.NS")
    print(test_df.tail())
