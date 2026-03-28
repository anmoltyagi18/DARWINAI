"use client";

import React, { useState, useEffect, useRef, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { api, LiveResponse } from "../../lib/api";
import { Sidebar } from "@/components/Sidebar";
import { 
  Activity, 
  TrendingUp, 
  TrendingDown, 
  Wallet, 
  PieChart, 
  History, 
  Settings, 
  Bell, 
  User,
  ArrowUpRight,
  ArrowDownRight,
  Shield,
  Zap,
  LayoutDashboard,
  CircleDollarSign,
  Search
} from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";

// ─── DASHBOARD COMPONENTS ──────────────────────────────────────────────────


function MetricCard({ title, value, change, isPositive, sub }: any) {
  return (
    <div className="glass p-6 rounded-2xl border-white/5 premium-card">
      <div className="text-[10px] uppercase tracking-widest text-white/30 font-bold mb-4">{title}</div>
      <div className="flex items-baseline gap-3 mb-2">
        <div className="text-3xl font-bold tracking-tighter">{value}</div>
        <div className={`text-xs font-bold flex items-center gap-1 ${isPositive ? "text-secondary" : "text-red-500"}`}>
          {isPositive ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
          {change}
        </div>
      </div>
      {sub && <div className="text-[10px] text-white/20 font-medium uppercase tracking-wider">{sub}</div>}
    </div>
  );
}

// ─── LIVE CHART ─────────────────────────────────────────────────────────────

function LiveRealtimeChart({ dataPoints }: { dataPoints: number[] }) {
  if (dataPoints.length === 0) return <div className="h-[300px] flex items-center justify-center text-white/20">Awaiting Signal...</div>;

  const min = Math.min(...dataPoints);
  const max = Math.max(...dataPoints);
  const range = max - min || 1;
  const padding = range * 0.1;
  
  const width = 800;
  const height = 300;
  
  const toX = (i: number) => (i / (Math.max(dataPoints.length - 1, 1))) * width;
  const toY = (v: number) => height - ((v - (min - padding)) / (range + padding * 2)) * height;

  const points = dataPoints.map((v, i) => `${toX(i)},${toY(v)}`).join(" ");

  return (
    <div className="relative w-full h-[300px]">
      <svg width="100%" height="300" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" className="overflow-visible">
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map(p => (
          <line 
            key={p} 
            x1="0" 
            y1={height * p} 
            x2={width} 
            y2={height * p} 
            stroke="white" 
            strokeOpacity="0.05" 
            strokeWidth="1" 
          />
        ))}
        
        {/* Glow Path */}
        <polyline
          fill="none"
          stroke="var(--color-primary)"
          strokeWidth="4"
          points={points}
          className="glow-primary opacity-50"
        />
        
        {/* Main Path */}
        <polyline
          fill="none"
          stroke="white"
          strokeWidth="2"
          points={points}
          className="transition-all duration-500"
        />

        {/* Gradient fill */}
        <path
          d={`M ${points} V ${height} H 0 Z`}
          fill="url(#chartGradient)"
          className="opacity-20"
        />

        <defs>
          <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--color-primary)" />
            <stop offset="100%" stopColor="transparent" />
          </linearGradient>
        </defs>
      </svg>
    </div>
  );
}

// ─── MAIN CONTENT ───────────────────────────────────────────────────────────

function LiveTradingContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const symbol = (searchParams.get("symbol") || api.getActiveSymbol()).toUpperCase();
  
  const [liveData, setLiveData] = useState<LiveResponse | null>(null);
  const [priceHistory, setPriceHistory] = useState<number[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchInput, setSearchInput] = useState(symbol);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchInput.trim()) {
      router.push(`/live?symbol=${searchInput.trim().toUpperCase()}`);
    }
  };

  useEffect(() => {
    let interval: any;
    
    // 1. Pre-fetch history to avoid "constant" line
    const initHistory = async () => {
      try {
        const hist = await api.getChartData(symbol, "1mo", "1d");
        if (Array.isArray(hist) && hist.length > 0) {
          setPriceHistory(hist.map((h: any) => h.close).slice(-50));
        }
      } catch (e) {
        console.error("Failed to load history:", e);
      }
    };
    initHistory();

    const fetchData = async () => {
      try {
        const res = await api.liveTrade(symbol);
        setLiveData(res);
        setPriceHistory(prev => {
          const next = [...prev, res.price];
          return next.slice(-50); 
        });
        setLoading(false);
      } catch (err) {
        console.error("Live fetch failed", err);
      }
    };

    fetchData();
    interval = setInterval(fetchData, 2000); // Polling every 2s

    return () => clearInterval(interval);
  }, [symbol]);

  return (
    <div className="flex h-screen bg-black overflow-hidden">
      {/* ── SIDEBAR ── */}
      <Sidebar />

      {/* ── MAIN ── */}
      <main className="flex-1 overflow-y-auto p-12">
        <header className="flex justify-between items-center mb-12">
          <div className="flex items-center gap-8 flex-1 max-w-2xl">
            <div>
              <h1 className="text-4xl font-bold tracking-tight mb-2 uppercase">{symbol} Live Stream</h1>
              <p className="text-white/30 text-xs">Real-time neural monitoring and trade execution path.</p>
            </div>
            
            <form onSubmit={handleSearch} className="relative flex-1 ml-8">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-white/20 w-4 h-4" />
              <input 
                type="text"
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                placeholder="Search Ticker (e.g. RELIANCE.NS)"
                className="w-full bg-white/5 border border-white/10 rounded-xl py-2 pl-10 pr-4 text-xs text-white focus:outline-none focus:ring-1 focus:ring-primary transition-all font-mono uppercase"
              />
            </form>
          </div>
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2 px-4 py-2 glass rounded-full text-xs font-bold text-secondary">
              <div className="w-2 h-2 rounded-full bg-secondary shadow-[0_0_8px_rgba(34,197,94,0.4)]" />
              LIVE DATA
            </div>
            <Bell className="w-5 h-5 text-white/30 cursor-pointer hover:text-white transition-colors" />
            <User className="w-5 h-5 text-white/30 cursor-pointer hover:text-white transition-colors" />
          </div>
        </header>

        {/* Top Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          <MetricCard 
            title="Portfolio Value" 
            value={new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR' }).format(priceHistory[priceHistory.length-1] * 12450 || 3162673)} 
            change="+₹24,431" 
            isPositive={true} 
            sub="Neural Estimate"
          />
          <MetricCard 
            title="Signal Confidence" 
            value={`${((liveData?.confidence ?? 0.532) * 100).toFixed(1)}%`} 
            change="+1.2%" 
            isPositive={true} 
            sub="Neural AI Engine"
          />
          <MetricCard 
            title="Current LTP" 
            value={new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR' }).format(liveData?.price || 0)} 
            change="0.45" 
            isPositive={liveData?.signal.includes("BUY")} 
            sub={`${liveData?.market_regime || "Ranging"} Market`}
          />
          <MetricCard 
            title="Neural Signal" 
            value={liveData?.signal || "HOLD"} 
            change={`${((liveData?.confidence ?? 0) * 100).toFixed(0)}%`} 
            isPositive={true} 
            sub="Conviction Level"
          />
        </div>

        {/* Chart Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 glass p-10 rounded-[40px] border-white/5">
            <div className="flex justify-between items-center mb-12">
              <div className="card-label">Performance Topology</div>
              <div className="flex gap-4">
                <button className="text-[10px] font-bold text-white/30 hover:text-white transition-colors">1H</button>
                <button className="text-[10px] font-bold text-white hover:text-white transition-colors underline underline-offset-4 decoration-primary decoration-2">LIVE</button>
                <button className="text-[10px] font-bold text-white/30 hover:text-white transition-colors">1D</button>
              </div>
            </div>
            <LiveRealtimeChart dataPoints={priceHistory} />
            <div className="mt-8 pt-8 border-t border-white/5 flex justify-between items-center">
              <div className="flex gap-12">
                <div>
                   <div className="text-[10px] text-white/20 uppercase font-bold mb-1">Max Price</div>
                   <div className="text-xl font-bold">₹{Math.max(...priceHistory, 0).toFixed(2)}</div>
                </div>
                <div>
                   <div className="text-[10px] text-white/20 uppercase font-bold mb-1">Min Price</div>
                   <div className="text-xl font-bold">₹{Math.min(...priceHistory, 99999).toFixed(2)}</div>
                </div>
              </div>
              <div className="flex items-center gap-4">
                 <Shield className="w-5 h-5 text-secondary glow-secondary" />
                 <span className="text-xs font-bold text-white/40">Secured by TruthGuard X Protocols</span>
              </div>
            </div>
          </div>

          <div className="flex flex-col gap-8">
            <div className="glass p-10 rounded-[40px] border-white/5 flex-1">
               <div className="text-[10px] uppercase font-bold tracking-[0.2em] text-white/30 mb-8">Asset Allocations</div>
               <div className="relative h-64 flex items-center justify-center">
                  {/* Dynamic Ring Visual */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <svg className="w-full h-full transform -rotate-90">
                      <circle cx="50%" cy="50%" r="90" fill="transparent" stroke="rgba(255,255,255,0.05)" strokeWidth="25" />
                      <circle 
                        cx="50%" cy="50%" r="90" 
                        fill="transparent" 
                        stroke="var(--color-primary)" 
                        strokeWidth="25" 
                        strokeDasharray="565" 
                        strokeDashoffset={565 * (1 - (liveData?.confidence ?? 0.5))} 
                        strokeLinecap="round"
                        className="transition-all duration-1000"
                      />
                    </svg>
                  </div>
                  <div className="text-center z-10 transition-all duration-500">
                    <div className="text-sm font-bold text-white/40 uppercase tracking-widest">{symbol}</div>
                    <div className="text-4xl font-black tracking-tighter">{((liveData?.confidence ?? 0.5)*100).toFixed(0)}%</div>
                  </div>
               </div>
               <div className="space-y-4 mt-8">
                  <div className="flex justify-between text-xs">
                    <span className="text-white/40">Portfolio Weight</span>
                    <span className="font-bold">{(liveData?.risk_level ?? 0.5 * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-white/40">Neural Conviction</span>
                    <span className="font-bold">HIGH</span>
                  </div>
               </div>
            </div>
            
            <button className="w-full py-6 rounded-[30px] bg-primary text-white font-bold glow-primary hover:brightness-110 active:scale-95 transition-all">
              EXECUTE LIVE REBALANCING
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

export default function LiveTradingPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-black flex items-center justify-center text-white">Initializing Neural Stream...</div>}>
      <LiveTradingContent />
    </Suspense>
  );
}
