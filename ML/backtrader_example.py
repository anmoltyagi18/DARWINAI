"""
backtrader_example.py — AIGOFIN ML Package
=============================================
A self-contained, minimal working example of using Backtrader 
integrated into the AIGOFIN system. This script demonstrates:
1. Setting up a Backtrader Strategy
2. Supplying synthetic pandas DataFrame data to Cerebro
3. Running with indicators and Analyzers
4. Printing the performance metrics
"""

import datetime
import pandas as pd
import numpy as np
import backtrader as bt

class SmaCrossStrategy(bt.Strategy):
    """
    Example Strategy:
    - Buys when 12-EMA crosses above 26-EMA and RSI is not overbought.
    - Sells when 12-EMA crosses below 26-EMA.
    """
    params = (
        ('fast', 12),
        ('slow', 26),
        ('rsi_period', 14),
        ('rsi_overbought', 70),
    )

    def __init__(self):
        # Initialize indicators
        self.fast_ema = bt.indicators.EMA(self.data.close, period=self.params.fast)
        self.slow_ema = bt.indicators.EMA(self.data.close, period=self.params.slow)
        self.rsi = bt.indicators.RSI_SMA(self.data.close, period=self.params.rsi_period)
        
        # Crossover signal
        self.crossover = bt.indicators.CrossOver(self.fast_ema, self.slow_ema)

    def next(self):
        # We are not in the market
        if not self.position:
            # Bullish crossover + RSI filter
            if self.crossover > 0 and self.rsi < self.params.rsi_overbought:
                self.buy()
        
        # We are in the market
        else:
            # Bearish crossover
            if self.crossover < 0:
                self.close()

def generate_synthetic_data(days=365):
    """Generate synthetic stock data as a Pandas DataFrame."""
    dates = pd.date_range(end=datetime.date.today(), periods=days)
    df = pd.DataFrame(index=dates)
    
    np.random.seed(42)
    # Random walk with slight upward drift
    returns = np.random.normal(0.0005, 0.02, days)
    prices = 100 * np.exp(np.cumsum(returns))
    
    df['open'] = prices * np.random.uniform(0.99, 1.01, days)
    df['high'] = df['open'] * np.random.uniform(1.0, 1.03, days)
    df['low'] = df['open'] * np.random.uniform(0.97, 1.0, days)
    df['close'] = prices
    df['volume'] = np.random.uniform(1000, 50000, days).astype(int)
    
    return df

def run_example():
    print("=" * 60)
    print("  AIGOFIN Backtrader Minimal Example")
    print("=" * 60)
    
    cerebro = bt.Cerebro()
    
    # 1. Add Data Feed
    df = generate_synthetic_data(400)
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    
    # 2. Add Strategy
    cerebro.addstrategy(SmaCrossStrategy)
    
    # 3. Configure Broker
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.001)
    
    # 4. Add Analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.0, timeframe=bt.TimeFrame.Days)
    
    print(f"Starting Portfolio Value: ${cerebro.broker.getvalue():,.2f}")
    
    results = cerebro.run()
    strat = results[0]
    
    print(f"Final Portfolio Value:  ${cerebro.broker.getvalue():,.2f}")
    print("-" * 60)
    
    # Extract Metrics
    trades = strat.analyzers.trades.get_analysis()
    dd = strat.analyzers.drawdown.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()
    
    total = trades.total.closed if 'total' in trades and 'closed' in trades.total else 0
    won = trades.won.total if 'won' in trades and 'total' in trades.won else 0
    win_rate = (won / total * 100) if total > 0 else 0
    
    print("PERFORMANCE SUMMARY:")
    print(f"  Total Trades: {total} (Won: {won}, Win Rate: {win_rate:.1f}%)")
    print(f"  Max Drawdown: {dd.max.drawdown if 'max' in dd else 0:.2f}%")
    print(f"  Sharpe Ratio: {sharpe.get('sharperatio', 0.0):.3f}")
    print("=" * 60)

if __name__ == "__main__":
    run_example()
