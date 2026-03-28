"""Smoke test for trade_opportunity_analyzer and strategy_classifier."""
import sys, os
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

np.random.seed(42)
n = 200
dates = pd.date_range('2023-06-01', periods=n, freq='B')
close = np.cumprod(1 + np.random.randn(n) * 0.015) * 170
df = pd.DataFrame({
    'open':   close * (1 + np.random.randn(n) * 0.003),
    'high':   close * (1 + abs(np.random.randn(n)) * 0.008),
    'low':    close * (1 - abs(np.random.randn(n)) * 0.008),
    'close':  close,
    'volume': np.random.uniform(1e6, 5e6, n),
}, index=dates)

PASS = 0
FAIL = 0

def chk(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        print(f"  [PASS] {label}")
        PASS += 1
    else:
        print(f"  [FAIL] {label}  {detail}")
        FAIL += 1

# ── Test 1: trade_opportunity_analyzer ───────────────────────────────────────
print("\n=== trade_opportunity_analyzer ===")
try:
    from ML.trade_opportunity_analyzer import (
        find_best_buy_point, find_best_sell_point, analyze_opportunities
    )
    chk("import OK", True)

    buy = find_best_buy_point(df)
    chk("find_best_buy_point returns dict", isinstance(buy, dict))
    chk("buy has date",  "date" in buy)
    chk("buy has price", "price" in buy and buy["price"] > 0)
    chk("buy has signal_type", "signal_type" in buy)
    print(f"     date={buy['date']}  price={buy['price']}  type={buy['signal_type']}")

    sell = find_best_sell_point(df, after_buy_idx=0)
    chk("find_best_sell_point returns dict", isinstance(sell, dict))
    chk("sell has date",  "date" in sell)
    chk("sell has price", "price" in sell and sell["price"] > 0)
    print(f"     date={sell['date']}  price={sell['price']}  type={sell['signal_type']}")

    opp = analyze_opportunities(df, symbol="AAPL")
    chk("analyze_opportunities returns dict", isinstance(opp, dict))
    chk("profit_pct present",   "profit_pct" in opp)
    chk("holding_days present", "holding_days" in opp and opp["holding_days"] >= 0)
    chk("buy signals list",     isinstance(opp["all_buy_signals"], list))
    chk("sell signals list",    isinstance(opp["all_sell_signals"], list))

    print(f"     profit_pct={opp['profit_pct']}%  holding={opp['holding_days']} days")
    print(f"     buy signals detected: {len(opp['all_buy_signals'])}")
    print(f"     sell signals detected: {len(opp['all_sell_signals'])}")
    print(f"     best_buy  {opp['best_buy_date']} @ ${opp['best_buy_price']}")
    print(f"     best_sell {opp['best_sell_date']} @ ${opp['best_sell_price']}")

except Exception as e:
    chk("trade_opportunity_analyzer", False, str(e))
    import traceback; traceback.print_exc()

# ── Test 2: strategy_classifier ──────────────────────────────────────────────
print("\n=== strategy_classifier ===")
try:
    from ML.strategy_classifier import classify_strategy
    chk("import OK", True)

    result = classify_strategy(
        indicators={
            'rsi': 42.0, 'adx': 31.0, 'macd_histogram': 1.2,
            'sma_20': 172.0, 'sma_50': 168.0, 'sma_200': 155.0,
            'bb_pct': 0.55, 'volume_ratio': 1.3, 'price': 175.0,
        },
        signals={'signal_type': 'Momentum Breakout'},
        holding_days=5,
    )
    chk("returns dict", isinstance(result, dict))
    chk("strategy present",       "strategy" in result and result["strategy"])
    chk("trade_type present",     "trade_type" in result and result["trade_type"])
    chk("holding_period present", "holding_period" in result)
    chk("confidence 0-1",         0.0 <= result.get("confidence", -1) <= 1.0)
    chk("all_scores dict",        isinstance(result.get("all_scores"), dict))
    chk("5 strategy scores",      len(result.get("all_scores", {})) == 5)

    print(f"     strategy      = {result['strategy']}")
    print(f"     trade_type    = {result['trade_type']}")
    print(f"     holding_period= {result['holding_period']}")
    print(f"     confidence    = {result['confidence']}")
    print(f"     rationale     = {result['rationale']}")
    print("     scores:")
    for k, v in result['all_scores'].items():
        bar = '#' * int(v * 20)
        print(f"       {k:20s} {v:.3f}  {bar}")

    # Test with RSI oversold → expect Mean Reversion
    result2 = classify_strategy(
        indicators={'rsi': 22.0, 'adx': 15.0, 'bb_pct': 0.03, 'price': 160.0},
        signals={'signal_type': 'RSI Reversal'},
        holding_days=3,
    )
    chk("mean_reversion detected for oversold RSI", result2["strategy"] == "Mean Reversion",
        f"got {result2['strategy']}")

    # Test with strong ADX + breakout → expect Momentum Breakout
    result3 = classify_strategy(
        indicators={'rsi': 58.0, 'adx': 42.0, 'sma_20': 175.0, 'sma_50': 168.0, 'price': 178.0},
        signals={'signal_type': 'Momentum Breakout'},
        holding_days=4,
    )
    chk("momentum detected for high ADX breakout", result3["strategy"] == "Momentum Breakout",
        f"got {result3['strategy']}")

except Exception as e:
    chk("strategy_classifier", False, str(e))
    import traceback; traceback.print_exc()

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 50)
total = PASS + FAIL
print(f"  Tests: {total}   Passed: {PASS}   Failed: {FAIL}")
print("  STATUS:", "ALL PASSED [OK]" if FAIL == 0 else f"{FAIL} FAILURES [FAIL]")
print("=" * 50)
sys.exit(0 if FAIL == 0 else 1)
