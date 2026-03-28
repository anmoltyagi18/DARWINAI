"""
stock_fetcher.py
Fetch OHLCV stock data from Yahoo Finance using yfinance.
"""

import yfinance as yf
import pandas as pd


def fetch_ohlcv(
    symbol: str,
    period: str = "1mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a given stock symbol.

    Args:
        symbol:   Ticker symbol, e.g. "AAPL", "TSLA", "^NSEI"
        period:   Data period. Valid values:
                  1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        interval: Data interval. Valid values:
                  1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo

    Returns:
        pandas DataFrame with columns:
            Open, High, Low, Close, Volume
        indexed by Date (DatetimeIndex).

    Raises:
        ValueError: If the symbol is invalid or no data is returned.
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(
            f"No data returned for symbol '{symbol}'. "
            "Check the ticker and try again."
        )

    # Keep only the OHLCV columns
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[ohlcv_cols]

    # Ensure the index is timezone-naive for easier downstream use
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df.index.name = "Date"
    return df


def print_summary(df: pd.DataFrame, symbol: str) -> None:
    """Print a quick summary of the fetched data."""
    print(f"\n{'='*50}")
    print(f"  OHLCV Data — {symbol.upper()}")
    print(f"{'='*50}")
    print(f"  Rows      : {len(df)}")
    print(f"  From      : {df.index.min().date()}")
    print(f"  To        : {df.index.max().date()}")
    print(f"  Avg Close : ${df['Close'].mean():.2f}")
    print(f"  Avg Volume: {int(df['Volume'].mean()):,}")
    print(f"{'='*50}\n")
    print(df.tail(10).to_string())
    print()


# ── CLI entry-point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch OHLCV stock data from Yahoo Finance."
    )
    parser.add_argument(
        "symbol",
        type=str,
        help="Stock ticker symbol, e.g. AAPL",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="1mo",
        help="Data period (default: 1mo). E.g. 1d, 5d, 1mo, 3mo, 1y, max",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Data interval (default: 1d). E.g. 1m, 5m, 1h, 1d, 1wk",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        metavar="FILE",
        help="Optional path to save the DataFrame as a CSV file.",
    )

    args = parser.parse_args()

    df = fetch_ohlcv(
        symbol=args.symbol,
        period=args.period,
        interval=args.interval,
    )

    print_summary(df, args.symbol)

    if args.csv:
        df.to_csv(args.csv)
        print(f"Saved to {args.csv}")
