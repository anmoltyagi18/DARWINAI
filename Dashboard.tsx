"use client";

import React, { useState, useCallback, useEffect, useRef } from "react";
import { api, AnalyzeResponse, OpportunityResponse, BacktestResponse, LiveResponse } from "../lib/api";

const SIGNAL_COLOR: Record<string, string> = {
  STRONG_BUY: "#4fffb0",
  BUY:        "#6b46ff",
  HOLD:       "#f5a623",
  SELL:       "#ff4d6d",
  STRONG_SELL:"#ff2244",
};

// Mock helpers removed for production integration

// ─── Candlestick Chart with Markers ─────────────────────────────────────────

function CandlestickChart({ symbol, hasOpp }: { symbol: string; hasOpp: boolean }) {
  interface Candle { open: number; high: number; low: number; close: number; x: number }
  const candles = useRef<Candle[]>([]);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const gen: Candle[] = [];
    let p = 150 + Math.random() * 50;
    for (let i = 0; i < 42; i++) {
      const o = p, c = p + (Math.random() - 0.47) * 6;
      const h = Math.max(o, c) + Math.random() * 4;
      const l = Math.min(o, c) - Math.random() * 4;
      p = c;
      gen.push({ open: o, high: h, low: l, close: c, x: i * 14 + 10 });
    }
    candles.current = gen; setMounted(true);
  }, [symbol]);

  if (!mounted) return (
    <div style={{ height: 280, display: "flex", alignItems: "center", justifyContent: "center", color: "#555" }}>
      Loading chart…
    </div>
  );

  const all = candles.current.flatMap(c => [c.high, c.low]);
  const mn = Math.min(...all), mx = Math.max(...all), rng = mx - mn || 1;
  const toY = (p: number) => 230 - ((p - mn) / rng) * 200;

  // Approximate marker positions (3rd and 26th candle)
  const buyX  = hasOpp ? candles.current[10]?.x ?? -1 : -1;
  const sellX = hasOpp ? candles.current[28]?.x ?? -1 : -1;
  const buyY  = buyX  >= 0 ? toY(candles.current[10].close)  : 0;
  const sellY = sellX >= 0 ? toY(candles.current[28].close) : 0;

  return (
    <div style={{ position: "relative" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
        <span style={{ fontSize: 12, color: "#666", letterSpacing: "0.06em", textTransform: "uppercase" }}>
          {symbol} · Price Action
        </span>
        <div style={{ display: "flex", gap: 14 }}>
          {hasOpp && <>
            <span style={{ fontSize: 11, color: "#6b46ff", display: "flex", alignItems: "center", gap: 5 }}>
              <span style={{ width: 7, height: 7, background: "#6b46ff", borderRadius: "50%", display: "inline-block" }} /> Buy
            </span>
            <span style={{ fontSize: 11, color: "#ff4d6d", display: "flex", alignItems: "center", gap: 5 }}>
              <span style={{ width: 7, height: 7, background: "#ff4d6d", borderRadius: "50%", display: "inline-block" }} /> Sell
            </span>
          </>}
        </div>
      </div>
      <svg width="100%" height="250" viewBox="0 0 600 250" preserveAspectRatio="xMidYMid meet" style={{ display: "block" }}>
        {/* Grid */}
        {[50, 125, 200].map(y => (
          <line key={y} x1="0" y1={y} x2="600" y2={y} stroke="#1f1f1f" strokeWidth="1" />
        ))}

        {/* Candles */}
        {candles.current.map((c, i) => {
          const bull = c.close >= c.open;
          const col = bull ? "#4fffb0" : "#ff4d6d";
          const bodyY = toY(Math.max(c.open, c.close));
          const bH = Math.max(1.5, Math.abs(toY(c.open) - toY(c.close)));
          return (
            <g key={i}>
              <line x1={c.x} y1={toY(c.high)} x2={c.x} y2={toY(c.low)} stroke={col} strokeWidth="1.1" />
              <rect x={c.x - 4} y={bodyY} width="8" height={bH} fill={col} rx="1" />
            </g>
          );
        })}

        {/* Buy marker */}
        {buyX >= 0 && <>
          <line x1={buyX} y1={0} x2={buyX} y2={250} stroke="#6b46ff" strokeWidth="1.2" strokeDasharray="4 3" opacity="0.7" />
          <polygon points={`${buyX},${buyY - 18} ${buyX - 7},${buyY - 4} ${buyX + 7},${buyY - 4}`} fill="#6b46ff" opacity="0.9" />
          <text x={buyX + 5} y={16} fill="#6b46ff" fontSize="9" fontFamily="monospace" opacity="0.85">BUY</text>
        </>}

        {/* Sell marker */}
        {sellX >= 0 && <>
          <line x1={sellX} y1={0} x2={sellX} y2={250} stroke="#ff4d6d" strokeWidth="1.2" strokeDasharray="4 3" opacity="0.7" />
          <polygon points={`${sellX},${sellY + 18} ${sellX - 7},${sellY + 4} ${sellX + 7},${sellY + 4}`} fill="#ff4d6d" opacity="0.9" />
          <text x={sellX + 5} y={16} fill="#ff4d6d" fontSize="9" fontFamily="monospace" opacity="0.85">SELL</text>
        </>}
      </svg>
    </div>
  );
}

// ─── Score Bar ───────────────────────────────────────────────────────────────

function ScoreBar({ label, score }: { label: string; score: number }) {
  const pct = Math.round(score * 100);
  const col = pct >= 60 ? "#6b46ff" : pct >= 35 ? "#f5a623" : "#444";
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: 12, color: "#888" }}>{label}</span>
        <span style={{ fontSize: 12, fontFamily: "monospace", color: col }}>{pct}%</span>
      </div>
      <div style={{ height: 3, background: "#1a1a1a", borderRadius: 2, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${pct}%`, background: col, transition: "width 0.9s cubic-bezier(.25,.8,.25,1)", borderRadius: 2 }} />
      </div>
    </div>
  );
}

// ─── Main Component ──────────────────────────────────────────────────────────

export default function Dashboard() {
  const [inputVal, setInputVal] = useState("AAPL");
  const [symbol, setSymbol] = useState("");
  const [data, setData] = useState<AnalyzeResponse | null>(null);
  const [opp, setOpp] = useState<OpportunityResponse | null>(null);
  const [backtest, setBacktest] = useState<BacktestResponse | null>(null);
  const [live, setLive] = useState<LiveResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analyzed, setAnalyzed] = useState(false);

  const performAnalysis = async (ticker: string) => {
    const t = ticker.toUpperCase().trim();
    if (!t) return;
    setLoading(true); setError(null); setSymbol(t); setAnalyzed(false);
    try {
      const [a, o] = await Promise.all([
        api.analyzeStock(t),
        api.getOpportunity(t)
      ]);
      setData(a); setOpp(o); setAnalyzed(true);
    } catch (err: any) {
      setError(err.message || "Analysis failed.");
    } finally {
      setLoading(false);
    }
  };

  const performBacktest = async () => {
    if (!symbol) return;
    setLoading(true); setError(null);
    try {
      const res = await api.backtestStock(symbol);
      setBacktest(res);
    } catch (err: any) {
      setError(err.message || "Backtest failed.");
    } finally {
      setLoading(false);
    }
  };

  const getLiveSignal = async () => {
    if (!symbol) return;
    setLoading(true); setError(null);
    try {
      const res = await api.liveTrade(symbol);
      setLive(res);
    } catch (err: any) {
      setError(err.message || "Live signal failed.");
    } finally {
      setLoading(false);
    }
  };

  const sigKey = (data?.signal || "HOLD").toUpperCase();
  const sigCol = SIGNAL_COLOR[sigKey] || "#888";

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:ital,wght@0,700;1,700&display=swap');

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        body {
          background: #000;
          color: #e8e8e8;
          font-family: 'Inter', system-ui, sans-serif;
          min-height: 100vh;
        }

        /* ── Navbar ── */
        .nav {
          position: fixed; top: 0; left: 0; right: 0; z-index: 100;
          display: flex; align-items: center; justify-content: center;
          padding: 20px 40px;
        }
        .nav-inner {
          display: flex; align-items: center; gap: 8px;
          background: rgba(255,255,255,0.04);
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 100px;
          padding: 8px 8px 8px 20px;
          backdrop-filter: blur(12px);
        }
        .nav-logo {
          font-size: 16px; font-weight: 600; color: #fff;
          letter-spacing: -0.3px; margin-right: 8px;
        }
        .nav-logo span { color: #6b46ff; }
        .nav-links { display: flex; align-items: center; gap: 4px; }
        .nav-link {
          padding: 6px 14px; font-size: 13px; color: #888; font-weight: 400;
          border-radius: 100px; cursor: pointer; transition: color 0.2s, background 0.2s;
          border: none; background: none;
        }
        .nav-link:hover { color: #fff; background: rgba(255,255,255,0.06); }
        .nav-cta {
          padding: 8px 20px; font-size: 13px; font-weight: 600; color: #fff;
          background: #6b46ff; border: none; border-radius: 100px;
          cursor: pointer; transition: filter 0.2s, transform 0.2s;
          margin-left: 4px;
        }
        .nav-cta:hover:not(:disabled) { filter: brightness(1.15); transform: translateY(-1px); }
        .nav-cta:disabled { opacity: 0.6; cursor: not-allowed; }

        /* ── Hero ── */
        .hero {
          min-height: 100vh;
          display: flex; flex-direction: column; align-items: center; justify-content: center;
          text-align: center;
          padding: 120px 40px 60px;
        }

        .hero-badge {
          display: inline-flex; align-items: center; gap: 8px;
          padding: 6px 14px; border-radius: 100px;
          background: rgba(255,255,255,0.05);
          border: 1px solid rgba(255,255,255,0.1);
          font-size: 12px; color: #aaa; margin-bottom: 36px;
        }
        .hero-badge-dot { width: 7px; height: 7px; border-radius: 50%; background: #4fffb0;
          box-shadow: 0 0 8px #4fffb0; animation: pulse-dot 2s infinite; }

        .hero-title {
          font-size: clamp(44px, 7vw, 88px);
          font-weight: 700; line-height: 1.05; letter-spacing: -2px;
          margin-bottom: 24px; max-width: 800px;
        }
        .hero-title em {
          font-style: italic;
          font-family: 'Playfair Display', serif;
          color: #6b46ff;
        }

        .hero-sub {
          font-size: 16px; color: #666; max-width: 480px;
          line-height: 1.6; margin-bottom: 48px;
        }

        .hero-search {
          display: flex; align-items: center; gap: 10px;
          background: rgba(255,255,255,0.05);
          border: 1px solid rgba(255,255,255,0.1);
          border-radius: 100px; padding: 8px 8px 8px 24px;
          width: 100%; max-width: 480px;
          transition: border-color 0.2s;
        }
        .hero-search:focus-within { border-color: rgba(107,70,255,0.5); }

        .hero-input {
          flex: 1; background: none; border: none; outline: none;
          font-size: 16px; color: #e8e8e8; font-family: 'Inter', monospace;
          letter-spacing: 0.04em;
        }
        .hero-input::placeholder { color: #444; }

        .hero-btn {
          padding: 10px 24px; background: #6b46ff; color: #fff;
          font-weight: 600; font-size: 14px; border: none; border-radius: 100px;
          cursor: pointer; transition: filter 0.2s, transform 0.2s; white-space: nowrap;
        }
        .hero-btn:hover:not(:disabled) { filter: brightness(1.15); transform: translateY(-1px); }
        .hero-btn:disabled { opacity: 0.6; cursor: not-allowed; }

        /* ── Dashboard Content ── */
        .dash-wrap {
          max-width: 1160px; margin: 0 auto;
          padding: 0 40px 80px;
        }

        .section-divider {
          display: flex; align-items: center; gap: 16px;
          margin-bottom: 40px;
        }
        .section-divider-line { flex: 1; height: 1px; background: #1a1a1a; }
        .section-divider-label {
          font-size: 11px; color: #444; text-transform: uppercase;
          letter-spacing: 0.12em; white-space: nowrap;
        }

        /* Main signal banner */
        .signal-banner {
          display: flex; align-items: center; justify-content: space-between;
          padding: 32px 40px;
          border: 1px solid rgba(255,255,255,0.07);
          border-radius: 20px;
          background: #080808;
          margin-bottom: 24px;
        }

        .signal-left h2 { font-size: 52px; font-weight: 700; letter-spacing: -2px; margin-bottom: 4px; }
        .signal-left .price { font-size: 28px; color: #666; font-weight: 300; font-family: monospace; }

        .signal-badge {
          font-size: 32px; font-weight: 700; letter-spacing: 3px;
          padding: 12px 32px; border-radius: 12px;
          font-family: 'Inter', monospace;
        }

        /* Two-column grid */
        .main-grid {
          display: grid; grid-template-columns: 1fr 360px;
          gap: 20px; align-items: start;
        }
        @media (max-width: 900px) { .main-grid { grid-template-columns: 1fr; } }

        /* Card */
        .card {
          background: #080808;
          border: 1px solid rgba(255,255,255,0.07);
          border-radius: 20px;
          padding: 28px;
          margin-bottom: 20px;
        }

        .card-label {
          font-size: 10px; font-weight: 600; text-transform: uppercase;
          letter-spacing: 0.12em; color: #555; margin-bottom: 20px;
        }

        .metric-row {
          display: flex; justify-content: space-between; align-items: center;
          padding: 12px 0; border-bottom: 1px solid #111;
        }
        .metric-row:last-child { border-bottom: none; }
        .metric-name { font-size: 13px; color: #666; }
        .metric-val  { font-size: 14px; font-weight: 500; font-family: monospace; }

        /* Opportunity 2x2 */
        .opp-grid {
          display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
          margin-bottom: 20px;
        }
        .opp-cell {
          background: #0d0d0d;
          border: 1px solid rgba(255,255,255,0.06);
          border-radius: 14px; padding: 16px 18px;
        }
        .opp-cell-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: #555; margin-bottom: 8px; }
        .opp-cell-value { font-size: 20px; font-weight: 700; font-family: monospace; }
        .opp-cell-sub   { font-size: 11px; color: #555; margin-top: 4px; }

        /* Profit highlight */
        .profit-row {
          display: flex; justify-content: space-between; align-items: center;
          padding: 16px 20px;
          border-radius: 12px; margin-bottom: 12px;
        }
        .profit-label { font-size: 12px; }
        .profit-value { font-size: 24px; font-weight: 700; font-family: monospace; }

        /* Progress bar */
        .pbar { height: 3px; background: #111; border-radius: 2px; overflow: hidden; margin-top: 12px; }
        .pbar-fill { height: 100%; border-radius: 2px; transition: width 1s cubic-bezier(.25,.8,.25,1); }

        /* Confidence ring */
        .conf-display {
          display: flex; align-items: baseline; gap: 12px; margin-bottom: 12px;
        }
        .conf-num { font-size: 56px; font-weight: 700; font-family: monospace; letter-spacing: -2px; }
        .conf-label { font-size: 13px; color: #555; }

        /* Strategy trade type chip */
        .chip {
          display: inline-flex; align-items: center;
          padding: 4px 12px; border-radius: 100px;
          font-size: 11px; font-weight: 600; letter-spacing: 0.04em;
        }

        @keyframes pulse-dot {
          0%, 100% { opacity: 1; transform: scale(1); }
          50%       { opacity: 0.4; transform: scale(0.8); }
        }
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(16px); }
          to   { opacity: 1; transform: translateY(0px); }
        }
        .fade-up { animation: fadeUp 0.7s ease-out forwards; }
      `}</style>

      {/* ── NAVBAR ── */}
      <nav className="nav">
        <div className="nav-inner">
          <span className="nav-logo">AI<span>GO</span>FIN</span>
          <div className="nav-links">
            {["Markets", "Strategy", "Backtest", "Docs"].map(l => (
              <button key={l} className="nav-link">{l}</button>
            ))}
          </div>
          <button
            className="nav-cta"
            onClick={() => performAnalysis(inputVal)}
            disabled={loading}
          >
            {loading ? "Analyzing…" : "Analyze"}
          </button>
        </div>
      </nav>

      {/* ── ERROR MESSAGE ── */}
      {error && (
        <div style={{
          position: "fixed", top: 80, left: "50%", transform: "translateX(-50%)",
          background: "#ff4d6d", color: "#fff", padding: "12px 24px", borderRadius: "8px",
          zIndex: 1000, boxShadow: "0 4px 20px rgba(0,0,0,0.5)", fontSize: "14px"
        }}>
          {error}
          <button onClick={() => setError(null)} style={{ marginLeft: 16, background: "none", border: "none", color: "#fff", cursor: "pointer", fontWeight: "bold" }}>✕</button>
        </div>
      )}

      {/* ── HERO ── */}
      <section className="hero">
        <div className="hero-badge">
          <span className="hero-badge-dot" />
          {analyzed ? `${symbol} analyzed — live signal ready` : "AI-powered trading intelligence · Real-time signals"}
        </div>

        <h1 className="hero-title">
          The truly <em>Limitless</em><br />trading engine.
        </h1>
        <p className="hero-sub">
          Real-time AI signals, optimal entry &amp; exit detection, and full strategy classification — all in one dashboard.
        </p>

        <div className="hero-search">
          <input
            className="hero-input"
            placeholder="Enter ticker symbol (e.g. AAPL, TSLA, MSFT)"
            value={inputVal}
            onChange={e => setInputVal(e.target.value)}
            onKeyDown={e => e.key === "Enter" && performAnalysis(inputVal)}
          />
          <button className="hero-btn" onClick={() => performAnalysis(inputVal)} disabled={loading}>
            {loading ? "Analyze now" : "Analyze now"}
          </button>
        </div>
        <div style={{ marginTop: 20, display: "flex", gap: 10 }}>
           <button className="nav-link" style={{ border: "1px solid #333" }} onClick={performBacktest} disabled={loading || !symbol}>Backtest</button>
           <button className="nav-link" style={{ border: "1px solid #333" }} onClick={getLiveSignal} disabled={loading || !symbol}>Live Signal</button>
        </div>
      </section>

      {/* ── DASHBOARD RESULTS ── */}
      {data && analyzed && (
        <div className="dash-wrap fade-up">
          {/* Divider */}
          <div className="section-divider">
            <div className="section-divider-line" />
            <span className="section-divider-label">AI Signal — {symbol}</span>
            <div className="section-divider-line" />
          </div>

          {/* Signal Banner */}
          <div className="signal-banner">
            <div className="signal-left">
              <h2>{data.symbol}</h2>
              <div className="price">${data.price?.toFixed(2) ?? "---.--"}</div>
              <div style={{ marginTop: 8, fontSize: 13, color: "#555" }}>
                {new Date(data.timestamp || "").toLocaleString("en-IN", { timeZone: "Asia/Kolkata" })}
              </div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: "#555", textTransform: "uppercase", letterSpacing: "0.1em", textAlign: "right", marginBottom: 8 }}>
                AI Signal
              </div>
              <div className="signal-badge" style={{ background: `${sigCol}14`, color: sigCol, border: `1px solid ${sigCol}33` }}>
                {data.signal}
              </div>
            </div>
          </div>

          {/* Main 2-column grid */}
          <div className="main-grid">

            {/* LEFT */}
            <div>
              {/* Candlestick Chart */}
              <div className="card">
                <div className="card-label">Price Action + Entry / Exit</div>
                <CandlestickChart symbol={symbol} hasOpp={!!opp} />
              </div>

              {/* Opportunity Panel */}
              {opp && (
                <div className="card" style={{ borderLeft: "2px solid #6b46ff" }}>
                  <div className="card-label">Trade Opportunity Analyzer</div>
                  
                  <div className="opp-grid">
                    <div className="opp-cell" style={{ borderLeft: "2px solid #4fffb0" }}>
                      <div className="opp-cell-label">Best Buy Time</div>
                      <div className="opp-cell-value" style={{ color: "#4fffb0" }}>${opp.best_buy_price?.toFixed(2)}</div>
                      <div className="opp-cell-sub">{opp.best_buy_date}</div>
                      <div className="opp-cell-sub" style={{ fontStyle: "italic" }}>{opp.buy_signal_type}</div>
                    </div>

                    <div className="opp-cell" style={{ borderLeft: "2px solid #ff4d6d" }}>
                      <div className="opp-cell-label">Best Sell Time</div>
                      <div className="opp-cell-value" style={{ color: "#ff4d6d" }}>${opp.best_sell_price?.toFixed(2)}</div>
                      <div className="opp-cell-sub">{opp.best_sell_date}</div>
                      <div className="opp-cell-sub" style={{ fontStyle: "italic" }}>{opp.sell_signal_type}</div>
                    </div>

                    <div className="opp-cell">
                      <div className="opp-cell-label">Profit Simulation</div>
                      <div className="opp-cell-value" style={{ color: (opp.profit_pct||0) >= 0 ? "#4fffb0" : "#ff4d6d", fontSize: 26 }}>
                        {(opp.profit_pct||0) >= 0 ? "+" : ""}{opp.profit_pct?.toFixed(2)}%
                      </div>
                      <div className="opp-cell-sub">{opp.holding_days}-day hold</div>
                    </div>

                    <div className="opp-cell" style={{ borderLeft: "2px solid #6b46ff" }}>
                      <div className="opp-cell-label">Strategy Used</div>
                      <div className="opp-cell-value" style={{ color: "#9d7fff", fontSize: 15, lineHeight: 1.3 }}>{opp.strategy}</div>
                      <div style={{ marginTop: 8 }}>
                        <span className="chip" style={{ background: "#f5a62314", color: "#f5a623", border: "1px solid #f5a62333" }}>
                          {opp.trade_type}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Rationale */}
                  <div style={{ fontSize: 13, color: "#666", lineHeight: 1.7, borderTop: "1px solid #111", paddingTop: 16 }}>
                    <span style={{ color: "#9d7fff" }}>Strategy rationale — </span>
                    {opp.strategy_rationale} Expected hold: <span style={{ color: "#f5a623" }}>{opp.holding_period}</span>.
                  </div>

                  {/* Scores */}
                  {opp.all_scores && (
                    <div style={{ marginTop: 24 }}>
                      <div className="card-label">Strategy Score Breakdown</div>
                      {Object.entries(opp.all_scores).map(([k, v]) => (
                        <ScoreBar key={k} label={k} score={v} />
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* AI Reasoning */}
              <div className="card" style={{ borderLeft: "2px solid #6b46ff" }}>
                <div className="card-label">AI Reasoning Engine</div>
                <div style={{ fontSize: 13, color: "#888", lineHeight: 1.8 }}>
                  {data.analysis_notes && data.analysis_notes.length > 0 
                    ? data.analysis_notes.map((note, idx) => <p key={idx} style={{ marginBottom: 8 }}>• {note}</p>)
                    : "No reasoning provided — ensure Python engine is online."}
                </div>
              </div>
            </div>

            {/* RIGHT */}
            <div>
              {/* Confidence */}
              <div className="card">
                <div className="card-label">Signal Confidence</div>
                <div className="conf-display">
                  <span className="conf-num" style={{ color: sigCol }}>
                    {(data.confidence * 100).toFixed(0) ?? "--"}<span style={{ fontSize: 28, color: "#444" }}>%</span>
                  </span>
                  <span className="conf-label">conviction</span>
                </div>
                <div className="pbar">
                  <div className="pbar-fill" style={{ width: `${(data.confidence * 100) ?? 0}%`, background: sigCol }} />
                </div>
              </div>

              {/* Market Intelligence */}
              <div className="card">
                <div className="card-label">Market Intelligence</div>
                {[
                  { name: "Market Regime", val: opp?.market_regime || data.market_regime || "Unknown", col: "#f5a623" },
                  { name: "Risk Level",    val: data.risk_level || "Medium", col: (data.risk_level || "").toLowerCase().includes("low") ? "#4fffb0" : "#ff4d6d" },
                  { name: "Social Sentiment", val: data.sentiment || "--", col: (data.sentiment || "").toLowerCase().includes("bullish") ? "#4fffb0" : "#ff4d6d" },
                ].map(({ name, val, col }) => (
                  <div className="metric-row" key={name}>
                    <span className="metric-name">{name}</span>
                    <span className="metric-val" style={{ color: col }}>{val}</span>
                  </div>
                ))}
              </div>

              {/* Trade Classification */}
              {opp && (
                <div className="card">
                  <div className="card-label">Trade Classification</div>
                  {[
                    { name: "Strategy",   val: opp.strategy,           col: "#9d7fff" },
                    { name: "Trade Type", val: opp.trade_type,          col: "#f5a623" },
                    { name: "Hold Period",val: opp.holding_period,     col: "#888" },
                    { name: "Confidence", val: `${Math.round((opp.strategy_confidence||0)*100)}%`, col: "#4fffb0" },
                  ].map(({ name, val, col }) => (
                    <div className="metric-row" key={name}>
                      <span className="metric-name">{name}</span>
                      <span className="metric-val" style={{ color: col, fontSize: 13 }}>{val}</span>
                    </div>
                  ))}
                  <div className="pbar" style={{ marginTop: 16 }}>
                    <div className="pbar-fill" style={{
                      width: `${Math.round((opp.strategy_confidence||0)*100)}%`,
                      background: "linear-gradient(90deg, #6b46ff, #4fffb0)",
                    }} />
                  </div>
                </div>
              )}

              {/* Quick Entry / Exit */}
              {opp && (
                <div className="card">
                  <div className="card-label">Quick Entry / Exit</div>
                  <div className="profit-row" style={{ background: "#0a1a0f", border: "1px solid #4fffb014" }}>
                    <div>
                      <div style={{ fontSize: 10, color: "#4fffb0", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 }}>Entry</div>
                      <div className="profit-value" style={{ color: "#4fffb0" }}>${opp.best_buy_price?.toFixed(2)}</div>
                      <div style={{ fontSize: 11, color: "#555", marginTop: 2 }}>{opp.buy_signal_type}</div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <div style={{ fontSize: 11, color: "#555" }}>{opp.best_buy_date}</div>
                    </div>
                  </div>
                  <div className="profit-row" style={{ background: "#1a0509", border: "1px solid #ff4d6d14" }}>
                    <div>
                      <div style={{ fontSize: 10, color: "#ff4d6d", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 }}>Exit</div>
                      <div className="profit-value" style={{ color: "#ff4d6d" }}>${opp.best_sell_price?.toFixed(2)}</div>
                      <div style={{ fontSize: 11, color: "#555", marginTop: 2 }}>{opp.sell_signal_type}</div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <div style={{ fontSize: 11, color: "#555" }}>{opp.best_sell_date}</div>
                    </div>
                  </div>
                  <div style={{
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                    padding: "16px 20px", borderRadius: 12, marginTop: 2,
                    background: (opp.profit_pct||0) >= 0 ? "#0a1a0f" : "#1a0509",
                    border: `1px solid ${(opp.profit_pct||0) >= 0 ? "#4fffb0" : "#ff4d6d"}14`,
                  }}>
                    <span style={{ fontSize: 12, color: "#555" }}>Simulated Profit</span>
                    <span style={{
                      fontFamily: "monospace", fontSize: 22, fontWeight: 700,
                      color: (opp.profit_pct||0) >= 0 ? "#4fffb0" : "#ff4d6d"
                    }}>
                      {(opp.profit_pct||0) >= 0 ? "+" : ""}{opp.profit_pct?.toFixed(2)}%
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
