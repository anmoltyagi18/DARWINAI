"""
example.py — Demonstrate the IndicatorEngine with a few real tickers.

Run with:
    pip install yfinance pandas numpy
    python example.py
"""

from indicator_engine import IndicatorEngine


def demo_latest(ticker: str, **kwargs) -> None:
    """Print the latest signal for a single ticker."""
    print(f"\n{'='*60}")
    print(f"  {ticker}  — Latest Signal")
    print("="*60)
    engine = IndicatorEngine(ticker, **kwargs)
    print(engine.run_latest_json())


def demo_summary(ticker: str, period: str = "3mo") -> None:
    """Print a buy/sell/hold summary table for the last N bars."""
    engine = IndicatorEngine(ticker, period=period)
    df = engine.run_dataframe()

    counts = df["action"].value_counts()
    last5  = df[["close", "action", "score", "confidence"]].tail(5)

    print(f"\n{'='*60}")
    print(f"  {ticker}  — Summary ({period})")
    print("="*60)
    print(f"  BUY={counts.get('BUY',0)}  SELL={counts.get('SELL',0)}  HOLD={counts.get('HOLD',0)}")
    print(f"\n  Last 5 bars:\n{last5.to_string()}\n")


def demo_custom_params() -> None:
    """Show how to override indicator parameters."""
    print(f"\n{'='*60}")
    print("  Custom params — TSLA weekly, aggressive thresholds")
    print("="*60)
    engine = IndicatorEngine(
        "TSLA",
        period="1y",
        interval="1wk",
        rsi_period=10,
        macd_fast=8,
        macd_slow=21,
        macd_signal=5,
        ma_short=10,
        ma_long=30,
        buy_threshold=0.20,      # more sensitive
        sell_threshold=-0.20,
    )
    print(engine.run_latest_json())


if __name__ == "__main__":
    # 1. Latest signal for Apple (daily)
    demo_latest("AAPL", period="6mo")

    # 2. Latest signal for Bitcoin
    demo_latest("BTC-USD", period="1y")

    # 3. Summary table — count BUY/SELL/HOLD over last 3 months
    demo_summary("MSFT", period="3mo")

    # 4. Custom indicator parameters
    demo_custom_params()
