'use client';

import React, { useState, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { Sidebar } from '@/components/Sidebar';
import { api, BacktestResponse } from '@/lib/api';
import { 
  Play, 
  Search, 
  TrendingUp, 
  TrendingDown, 
  BarChart3, 
  Calendar, 
  DollarSign, 
  Percent, 
  ShieldCheck,
  Zap,
  History,
  Info
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// ─── COMPONENTS ─────────────────────────────────────────────────────────────

function StatCard({ title, value, sub, icon: Icon, color = "cyan" }: any) {
  const colorMap: any = {
    cyan: "text-cyan-400 bg-cyan-500/10 border-cyan-500/20 shadow-cyan-500/5",
    green: "text-green-400 bg-green-500/10 border-green-500/20 shadow-green-500/5",
    red: "text-red-400 bg-red-500/10 border-red-500/20 shadow-red-500/5",
    purple: "text-purple-400 bg-purple-500/10 border-purple-500/20 shadow-purple-500/5"
  };

  return (
    <div className={`p-6 rounded-2xl border ${colorMap[color]} shadow-lg backdrop-blur-md`}>
      <div className="flex justify-between items-start mb-4">
        <div className="p-2 rounded-lg bg-black/20">
          <Icon size={18} />
        </div>
        <div className="text-[10px] font-bold uppercase tracking-widest opacity-50">{title}</div>
      </div>
      <div className="text-3xl font-black tracking-tighter mb-1 uppercase">{value}</div>
      <div className="text-[10px] opacity-40 font-mono tracking-tight">{sub}</div>
    </div>
  );
}

function BacktestContent() {
  const searchParams = useSearchParams();
  const initialSymbol = (searchParams.get('symbol') || api.getActiveSymbol()).toUpperCase();
  
  const [symbol, setSymbol] = useState(initialSymbol);
  const [searchInput, setSearchInput] = useState(initialSymbol);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<BacktestResponse | null>(null);
  const [config, setConfig] = useState({
    cash: 1000000,
    commission: 0.1
  });

  const runBacktest = async (sym: string) => {
    setLoading(true);
    setResults(null);
    try {
      const data = await api.backtestStock(sym);
      setResults(data);
    } catch (err) {
      console.error("Backtest failed:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchInput) {
      setSymbol(searchInput.toUpperCase());
      runBacktest(searchInput.toUpperCase());
    }
  };

  return (
    <div className="flex h-screen bg-[#06080c] text-white font-sans selection:bg-cyan-500/30">
      <Sidebar />
      
      <main className="flex-1 overflow-y-auto p-12 custom-scrollbar">
        {/* Header */}
        <header className="flex justify-between items-end mb-12">
          <div>
            <div className="flex items-center gap-3 mb-2">
               <div className="w-8 h-8 rounded-lg bg-cyan-500/20 flex items-center justify-center border border-cyan-500/30">
                  <History className="text-cyan-400" size={18} />
               </div>
               <span className="text-[10px] font-bold text-cyan-400/60 uppercase tracking-[0.3em]">Module // Quantitative Backtest</span>
            </div>
            <h1 className="text-5xl font-black tracking-tighter uppercase italic">
              Strategy <span className="text-cyan-500">Validator</span>
            </h1>
          </div>

          <form onSubmit={handleSubmit} className="flex gap-4">
            <div className="relative group">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-white/20 group-focus-within:text-cyan-400 transition-colors" size={16} />
              <input 
                type="text" 
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                placeholder="SYMBOL (e.g. TCS.NS)"
                className="bg-black/40 border border-white/5 rounded-xl py-3 pl-12 pr-6 text-sm font-mono w-64 focus:outline-none focus:border-cyan-500/50 transition-all placeholder:text-white/10 uppercase"
              />
            </div>
            <button 
              type="submit"
              disabled={loading}
              className="bg-cyan-500 hover:bg-cyan-400 disabled:opacity-50 text-black px-8 py-3 rounded-xl font-bold text-xs uppercase tracking-widest transition-all flex items-center gap-2 shadow-[0_0_20px_rgba(6,182,212,0.3)] active:scale-95"
            >
              <Play size={14} fill="currentColor" />
              {loading ? 'Simulating...' : 'Run Simulation'}
            </button>
          </form>
        </header>

        {/* Control Panel */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-12">
           <div className="lg:col-span-2 p-8 rounded-3xl bg-black/40 border border-white/5 backdrop-blur-sm relative overflow-hidden group">
              <div className="absolute top-0 right-0 p-8 opacity-5 group-hover:opacity-10 transition-opacity">
                 <Zap size={120} />
              </div>
              <h3 className="text-lg font-bold mb-6 flex items-center gap-2">
                 <Info size={16} className="text-cyan-400" />
                 Simulation Parameters
              </h3>
              <div className="grid grid-cols-2 gap-8">
                 <div>
                    <label className="text-[10px] font-bold text-white/20 uppercase tracking-widest mb-3 block">Initial Capital (INR)</label>
                    <div className="relative">
                       <input 
                          type="number" 
                          value={config.cash}
                          onChange={(e) => setConfig({...config, cash: Number(e.target.value)})}
                          className="w-full bg-white/5 border border-white/10 rounded-xl py-4 px-6 text-2xl font-black font-mono focus:outline-none focus:border-cyan-500/50"
                       />
                       <span className="absolute right-6 top-1/2 -translate-y-1/2 text-cyan-400/40 font-mono">₹</span>
                    </div>
                 </div>
                 <div>
                    <label className="text-[10px] font-bold text-white/20 uppercase tracking-widest mb-3 block">Trade Commission (%)</label>
                    <div className="relative">
                       <input 
                          type="number" 
                          value={config.commission}
                          onChange={(e) => setConfig({...config, commission: Number(e.target.value)})}
                          className="w-full bg-white/5 border border-white/10 rounded-xl py-4 px-6 text-2xl font-black font-mono focus:outline-none focus:border-cyan-500/50"
                       />
                       <span className="absolute right-6 top-1/2 -translate-y-1/2 text-cyan-400/40 font-mono">%</span>
                    </div>
                 </div>
              </div>
           </div>

           <div className="p-8 rounded-3xl bg-gradient-to-br from-cyan-500/10 to-transparent border border-cyan-500/20 flex flex-col justify-center">
              <div className="text-[10px] font-bold text-cyan-400 uppercase tracking-[0.2em] mb-4">Neural Engine Status</div>
              <div className="flex items-center gap-4 mb-2">
                 <div className="w-3 h-3 rounded-full bg-cyan-500 shadow-[0_0_10px_rgba(6,182,212,1)]" />
                 <span className="text-2xl font-black tracking-tighter uppercase italic">Ready to Sim</span>
              </div>
              <p className="text-white/40 text-[10px] font-medium leading-relaxed uppercase">
                 AI strategy weights: Multi-modal fusion with ATR volatility normalization active.
              </p>
           </div>
        </div>

        {/* Results */}
        <AnimatePresence mode="wait">
          {results ? (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-8"
            >
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <StatCard 
                  title="Net Profit" 
                  value={`₹${results.profit.toLocaleString()}`} 
                  sub={`${results.profit_pct >= 0 ? '+' : ''}${results.profit_pct}% Total Return`}
                  icon={DollarSign}
                  color={results.profit >= 0 ? "green" : "red"}
                />
                <StatCard 
                  title="Win Rate" 
                  value={`${results.win_rate}%`} 
                  sub={`Out of ${results.total_trades} trade signals`}
                  icon={Percent}
                  color="cyan"
                />
                <StatCard 
                  title="Max Drawdown" 
                  value={`${results.max_drawdown}%`} 
                  sub="Maximum equity decline"
                  icon={TrendingDown}
                  color="red"
                />
                <StatCard 
                  title="Sharpe Ratio" 
                  value={results.sharpe_ratio.toFixed(2)} 
                  sub="Risk-adjusted efficiency"
                  icon={ShieldCheck}
                  color="purple"
                />
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                 <div className="lg:col-span-2 p-8 rounded-3xl bg-black/40 border border-white/5">
                    <div className="flex justify-between items-center mb-12">
                       <h3 className="text-lg font-black tracking-tighter uppercase italic">Equity Growth Chart</h3>
                       <div className="flex gap-4">
                          <div className="flex items-center gap-2">
                             <div className="w-2 h-2 rounded-full bg-cyan-500" />
                             <span className="text-[10px] font-bold text-white/40 uppercase">Benchmark</span>
                          </div>
                          <div className="flex items-center gap-2">
                             <div className="w-2 h-2 rounded-full bg-green-400" />
                             <span className="text-[10px] font-bold text-white/40 uppercase">AI Strategy</span>
                          </div>
                       </div>
                    </div>
                    {/* Placeholder for Equity Chart */}
                    <div className="h-[300px] flex items-center justify-center bg-white/2 rounded-2xl border border-dashed border-white/10 relative overflow-hidden group">
                       <div className="absolute inset-0 opacity-10 flex items-end pointer-events-none">
                          <div className="w-full h-[60%] bg-gradient-to-t from-cyan-500 to-transparent" style={{ clipPath: 'polygon(0 80%, 10% 70%, 20% 75%, 30% 50%, 40% 60%, 50% 30%, 60% 45%, 70% 20%, 80% 35%, 90% 10%, 100% 15%, 100% 100%, 0 100%)' }} />
                       </div>
                       <div className="relative z-10 text-[10px] font-bold text-white/20 uppercase tracking-[0.4em] group-hover:text-cyan-400 transition-colors">
                          Visualizing performance history...
                       </div>
                    </div>
                 </div>

                 <div className="p-8 rounded-3xl bg-black/40 border border-white/5">
                    <h3 className="text-lg font-black tracking-tighter uppercase italic mb-8">Trade Metrics</h3>
                    <div className="space-y-6">
                       <MetricRow label="Avg Trade Profit" value={`₹${(results.profit / (results.total_trades || 1)).toLocaleString()}`} />
                       <MetricRow label="Total Samples" value={`${results.bars} bars`} />
                       <MetricRow label="Final Valuation" value={`₹${results.final_value.toLocaleString()}`} />
                       <MetricRow label="Data Source" value={results.data_source} />
                    </div>
                    <div className="mt-12 p-4 rounded-xl bg-cyan-500/5 border border-cyan-500/10">
                       <div className="text-[10px] font-bold text-cyan-400 uppercase tracking-widest mb-2">Simulation Note</div>
                       <p className="text-[10px] text-white/40 leading-relaxed italic">
                         Strategy employs dynamic volatility exit with ATR thresholding. Past performance does not guarantee future institutional yields.
                       </p>
                    </div>
                 </div>
              </div>
            </motion.div>
          ) : (
            <div className="h-[60vh] flex flex-col items-center justify-center text-white/10">
               <BarChart3 size={64} className="mb-6 opacity-40" />
               <p className="text-sm font-bold uppercase tracking-[0.5em] italic">Awaiting Quantitative Parameters</p>
               {loading && <div className="mt-8 flex gap-2">
                  {[0,1,2].map(i => (
                    <motion.div 
                      key={i}
                      animate={{ height: [12, 32, 12] }}
                      transition={{ duration: 0.6, repeat: Infinity, delay: i * 0.1 }}
                      className="w-1 bg-cyan-500"
                    />
                  ))}
               </div>}
            </div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}

function MetricRow({ label, value }: any) {
  return (
    <div className="flex justify-between items-end border-b border-white/5 pb-3">
       <div className="text-[10px] font-bold text-white/20 uppercase">{label}</div>
       <div className="text-sm font-black italic">{value}</div>
    </div>
  );
}

export default function BacktestPage() {
  return (
    <Suspense fallback={<div>Loading validator kernel...</div>}>
      <BacktestContent />
    </Suspense>
  );
}
