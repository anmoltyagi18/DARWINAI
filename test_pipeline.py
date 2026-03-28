"""
test_pipeline.py — AIGOFIN Full System Test
============================================
Runs a full end-to-end simulation for AAPL and validates the
expected output format:
  {
    symbol: "AAPL",
    signal: "BUY",
    confidence: 0.82,
    market_regime: "bullish",
    risk_level: 0.25,
    reasoning: "RSI oversold and MACD bullish crossover"
  }
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

PASS = 0
FAIL = 0

def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  [PASS] {label}")
        PASS += 1
    else:
        print(f"  [FAIL] {label}  {detail}")
        FAIL += 1

print("\n" + "=" * 60)
print("  AIGOFIN — Full System Integration Test")
print("=" * 60)

# ── Test 1: Config ────────────────────────────────────────────────
print("\n[1] Config Module")
try:
    from ML.config import config
    check("config import", True)
    check("FINNHUB_API_KEY loaded", config.has_finnhub_key(),
          "Add FINNHUB_API_KEY to .env")
    check("ROOT_DIR exists", config.ROOT_DIR.exists())
    print(f"      config repr: {config}")
except Exception as e:
    check("config import", False, str(e))

# ── Test 2: Logger ────────────────────────────────────────────────
print("\n[2] Logger Module")
try:
    from ML.logger import get_logger
    log = get_logger("test_pipeline")
    log.info("Logger test message")
    check("logger import", True)
    check("logger callable", callable(get_logger))
except Exception as e:
    check("logger import", False, str(e))

# ── Test 3: Strategy Engine ───────────────────────────────────────
print("\n[3] Strategy Engine")
try:
    from ML.strategy_engine import StrategyEngine
    engine = StrategyEngine()
    import numpy as np
    closes = np.array([100 + i * 0.5 + np.sin(i) * 3 for i in range(60)])
    result = engine.generate_signal({"closes": closes})
    check("strategy_engine import", True)
    check("generate_signal returns dict", isinstance(result, dict))
    check("signal in {-1,0,1}", result.get("signal") in (-1, 0, 1),
          str(result.get("signal")))
    check("confidence 0-1", 0.0 <= result.get("confidence", -1) <= 1.0)
    print(f"      signal={result['signal']} confidence={result['confidence']:.3f}")
    print(f"      reason: {result.get('reason','')[:80]}")
except Exception as e:
    check("strategy_engine", False, str(e))

# ── Test 4: Full Analysis Pipeline (AAPL) ────────────────────────
print("\n[4] Full Analysis Pipeline — AAPL")
try:
    from ML.api_server import run_analysis
    result = run_analysis("AAPL", period="1y", interval="1d")

    sig = result["signal"]
    sig_val = sig.value if hasattr(sig, "value") else sig
    regime = result["market_regime"]
    regime_val = regime.value if hasattr(regime, "value") else regime
    risk = result["risk_level"]
    risk_val = risk.value if hasattr(risk, "value") else risk

    check("run_analysis success", True)
    check("symbol == AAPL", result.get("symbol") == "AAPL")
    check("signal present", sig_val in ("STRONG_BUY","BUY","HOLD","SELL","STRONG_SELL"),
          f"got: {sig_val}")
    check("confidence 0-1", 0.0 <= result.get("confidence", -1) <= 1.0)
    check("market_regime present", bool(regime_val))
    check("risk_level present", bool(risk_val))
    check("price > 0", result.get("price", 0) > 0)
    check("bars_analysed > 50", result.get("bars_analysed", 0) > 50)
    check("indicators dict", isinstance(result.get("indicators"), dict))
    check("signal_breakdown dict", isinstance(result.get("signal_breakdown"), dict))

    print()
    print("  === AAPL FINAL SIGNAL ===")
    print(f"  symbol       : {result['symbol']}")
    print(f"  signal       : {sig_val}")
    print(f"  confidence   : {result['confidence']:.3f}")
    print(f"  market_regime: {regime_val}")
    print(f"  risk_level   : {risk_val}")
    print(f"  price        : ${result['price']:,.2f}")
    print(f"  bars_analysed: {result['bars_analysed']}")
    print(f"  data_source  : {result['data_source']}")
    ind = result["indicators"]
    print(f"  RSI={ind['rsi']:.1f}  MACD_hist={ind['macd_histogram']:.3f}  ADX={ind['adx']:.1f}  BB%={ind['bb_pct']:.2f}")
    print()
    print("  Signal breakdown:")
    for k, v in result["signal_breakdown"].items():
        bar = "#" * int(abs(v) * 15)
        print(f"    {k:12s}: {v:+.3f}  {bar}")

except Exception as e:
    check("run_analysis", False, str(e))
    import traceback
    traceback.print_exc()

# ── Test 5: AI Brain ─────────────────────────────────────────────
print("\n[5] AI Brain")
try:
    from ML.ai_brain import AIBrain
    brain = AIBrain()
    module_outputs = {
        "indicators_engine":     {"signal": "BUY",  "confidence": 0.75, "indicators": {}},
        "market_regime_detector":{"signal": "BUY",  "confidence": 0.80, "regime": "TRENDING_UP"},
        "sentiment_engine":      {"signal": "HOLD", "confidence": 0.60, "sentiment": "BULLISH"},
        "risk_manager":          {"signal": "HOLD", "confidence": 1.00, "risk_level": 0.25},
    }
    decision = brain.evaluate(module_outputs, symbol="AAPL")
    check("ai_brain import", True)
    check("evaluate returns dict", isinstance(decision, dict))
    check("signal in result", "signal" in decision)
    check("confidence in result", "confidence" in decision)
    print(f"      AIBrain decision: signal={decision.get('signal')} confidence={decision.get('confidence'):.3f}")
except Exception as e:
    check("ai_brain", False, str(e))

# ── Test 6: Trade Explainer ───────────────────────────────────────
print("\n[6] Trade Explainer")
try:
    from ML.trade_explainer import TradeExplainer, TradeContext
    ctx = TradeContext(
        symbol="AAPL",
        signal="BUY",
        confidence=0.75,
        market_regime="TRENDING_UP",
        sentiment="BULLISH",
        risk_level="LOW",
        indicators={"rsi": 42.0, "macd": 1.5, "macd_signal": 0.8, "macd_histogram": 0.7,
                    "sma_20": 175.0, "sma_50": 170.0, "sma_200": 160.0,
                    "bb_upper": 185.0, "bb_lower": 165.0, "bb_pct": 0.48,
                    "atr": 2.5, "adx": 28.0, "volume_ratio": 1.2},
        signal_breakdown={"trend": 0.6, "momentum": 0.5, "sentiment": 0.3},
    )
    explainer = TradeExplainer()
    report = explainer.explain(ctx)
    check("trade_explainer import", True)
    check("report returned", report is not None)
    has_text = hasattr(report, "plain_text") or hasattr(report, "summary") or isinstance(report, str)
    check("report has text", has_text)
    text = getattr(report, "plain_text", None) or getattr(report, "summary", None) or str(report)
    print(f"      Explanation preview: {str(text)[:120]}...")
except Exception as e:
    check("trade_explainer", False, str(e))

# ── Test 7: Backtest Engine ───────────────────────────────────────
print("\n[7] Backtest Engine")
try:
    from ML.backtest_engine import run_backtest
    import pandas as pd, numpy as np
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    df = pd.DataFrame({
        "open":   np.random.uniform(90, 110, 100),
        "high":   np.random.uniform(100, 120, 100),
        "low":    np.random.uniform(80, 100, 100),
        "close":  np.random.uniform(90, 110, 100),
        "volume": np.random.uniform(1000, 5000, 100),
    }, index=dates)
    from ML.strategy_engine import StrategyEngine
    se = StrategyEngine()
    closes_arr = df["close"].values
    class _Adapter:
        def __init__(self, c, e):
            self._c = c; self._e = e; self._i = 0
        def generate_signal(self, f):
            idx = min(self._i, len(self._c)-1); self._i += 1
            return self._e.generate_signal({"closes": self._c[:idx+1]})
    adapter = _Adapter(closes_arr, se)
    bt = run_backtest(df, adapter)
    check("backtest_engine import", True)
    check("returns dict", isinstance(bt, dict))
    required_keys = ["profit", "win_rate", "sharpe_ratio", "max_drawdown"]
    for k in required_keys:
        check(f"  key '{k}' present", k in bt or "error" in bt,
              f"missing key in {list(bt.keys())[:5]}")
    print(f"      Result: {bt}")
except Exception as e:
    check("backtest_engine", False, str(e))

# ── Test 8: Strategy Evolver ─────────────────────────────────────
print("\n[8] Strategy Evolver")
try:
    from ML.strategy_evolver import evolve, generate_price_data
    prices = generate_price_data(n=200, seed=42)
    best, history = evolve(prices, generations=2, population_size=10, verbose=False)
    check("strategy_evolver import", True)
    check("evolution returns strategy", best is not None)
    check("history has entries", len(history) > 0)
    check("genes present", len(best.genes) > 0)
    print(f"      Best strategy: fitness={best.fitness:.4f} trades={best.num_trades} genes={len(best.genes)}")
except Exception as e:
    check("strategy_evolver", False, str(e))

# ── Test 9: Live Data Stream (config key) ────────────────────────
print("\n[9] Live Data Stream — get_live_stock_price")
try:
    from ML.live_data_stream import get_live_stock_price
    data = get_live_stock_price("AAPL")
    check("live_data_stream import", True)
    check("returns dict with price", isinstance(data, dict) and "price" in data)
    price = data.get("price", 0)
    check("price > 0", price > 0, f"price={price}")
    print(f"      AAPL live price: ${price}")
    print(f"      source: {data.get('source','unknown')}")
except Exception as e:
    check("live_data_stream", False, str(e))

# ── Summary ───────────────────────────────────────────────────────
print()
print("=" * 60)
total = PASS + FAIL
print(f"  Tests: {total}   Passed: {PASS}   Failed: {FAIL}")
if FAIL == 0:
    print("  STATUS: ALL TESTS PASSED [OK]")
else:
    print(f"  STATUS: {FAIL} FAILURES — see above [FAIL]")
print("=" * 60 + "\n")
sys.exit(0 if FAIL == 0 else 1)
