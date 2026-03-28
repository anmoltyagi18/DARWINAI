"""
python_engine.py  –  Minimal Flask stub for the Python AI engine.
Run:  pip install flask && python python_engine.py
"""

from flask import Flask, request, jsonify
import random, time

app = Flask(__name__)

# ── /stock-data ──────────────────────────────────────────────────────────────
@app.route("/stock-data")
def stock_data():
    symbol   = request.args.get("symbol",   "AAPL")
    interval = request.args.get("interval", "1d")
    limit    = int(request.args.get("limit", 100))

    base  = 150.0
    data  = []
    for i in range(limit):
        base += random.uniform(-2, 2)
        data.append({
            "timestamp": int(time.time()) - (limit - i) * 86400,
            "open":  round(base - random.uniform(0, 1), 2),
            "high":  round(base + random.uniform(0, 2), 2),
            "low":   round(base - random.uniform(0, 2), 2),
            "close": round(base, 2),
            "volume": random.randint(10_000_000, 50_000_000),
        })

    return jsonify({"symbol": symbol, "interval": interval, "data": data})

# ── /analyze ─────────────────────────────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
def analyze():
    body   = request.get_json(force=True) or {}
    symbol = body.get("symbol", "AAPL")
    model  = body.get("model",  "lstm")

    return jsonify({
        "symbol":     symbol,
        "model":      model,
        "prediction": {
            "direction":   random.choice(["bullish", "bearish", "neutral"]),
            "confidence":  round(random.uniform(0.55, 0.95), 4),
            "price_target": round(random.uniform(140, 200), 2),
            "horizon_days": body.get("horizon_days", 5),
        },
        "features_used": ["RSI", "MACD", "Bollinger", "Volume_MA"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

# ── /strategy ────────────────────────────────────────────────────────────────
@app.route("/strategy")
def strategy():
    symbol  = request.args.get("symbol",  "AAPL")
    risk    = request.args.get("risk",    "medium")
    horizon = request.args.get("horizon", "short")

    actions = {
        "low":    ("BUY",  0.05),
        "medium": ("BUY",  0.10),
        "high":   ("BUY",  0.20),
    }
    action, size = actions.get(risk, ("HOLD", 0.0))

    return jsonify({
        "symbol":       symbol,
        "risk_profile": risk,
        "horizon":      horizon,
        "strategy": {
            "action":       action,
            "position_size": size,
            "entry_price":  round(random.uniform(148, 155), 2),
            "stop_loss":    round(random.uniform(140, 147), 2),
            "take_profit":  round(random.uniform(160, 175), 2),
        },
        "rationale": f"{action} signal based on {risk}-risk {horizon}-term profile.",
    })

# ── /signals ─────────────────────────────────────────────────────────────────
@app.route("/signals")
def signals():
    symbol      = request.args.get("symbol",    "AAPL")
    signal_type = request.args.get("type",      "all")
    timeframe   = request.args.get("timeframe", "1d")

    signals_list = [
        {"indicator": "RSI",        "value": round(random.uniform(30, 70), 2),  "signal": random.choice(["BUY", "SELL", "HOLD"])},
        {"indicator": "MACD",       "value": round(random.uniform(-2, 2), 4),   "signal": random.choice(["BUY", "SELL", "HOLD"])},
        {"indicator": "Bollinger",  "value": round(random.uniform(0.1, 0.9), 4),"signal": random.choice(["BUY", "SELL", "HOLD"])},
        {"indicator": "Volume_MA",  "value": round(random.uniform(0.8, 1.5), 4),"signal": random.choice(["BUY", "SELL", "HOLD"])},
        {"indicator": "Stochastic", "value": round(random.uniform(20, 80), 2),  "signal": random.choice(["BUY", "SELL", "HOLD"])},
    ]

    if signal_type != "all":
        signals_list = [s for s in signals_list if s["signal"].lower() == signal_type.lower()]

    buy_count  = sum(1 for s in signals_list if s["signal"] == "BUY")
    sell_count = sum(1 for s in signals_list if s["signal"] == "SELL")
    consensus  = "BUY" if buy_count > sell_count else ("SELL" if sell_count > buy_count else "HOLD")

    return jsonify({
        "symbol":    symbol,
        "timeframe": timeframe,
        "signals":   signals_list,
        "consensus": consensus,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
