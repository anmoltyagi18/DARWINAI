'use client';

import React, { useState, Suspense, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import { Sidebar } from '@/components/Sidebar';
import { api, StrategyResponse } from '@/lib/api';
import { 
  Zap, 
  Dna, 
  Cpu, 
  Binary, 
  Search, 
  ChevronRight, 
  ShieldCheck, 
  TrendingUp, 
  Target,
  Sparkles,
  Info
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// ─── COMPONENTS ─────────────────────────────────────────────────────────────

function GeneCard({ gene, index }: { gene: any, index: number }) {
  const isBuy = gene.action === "BUY";
  
  return (
    <motion.div 
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.1 }}
      className="p-4 rounded-xl bg-white/2 border border-white/5 flex items-center justify-between group hover:bg-white/5 transition-all"
    >
      <div className="flex items-center gap-4">
        <div className={`w-8 h-8 rounded-lg flex items-center justify-center font-bold text-xs ${isBuy ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'}`}>
          {index + 1}
        </div>
        <div>
           <div className="text-[10px] font-bold text-white/20 uppercase tracking-widest">{gene.indicator} // {gene.condition}</div>
           <div className="text-sm font-black italic tracking-tight">
              {gene.indicator}({gene.period}) <span className="text-cyan-400">{gene.condition}</span> {gene.threshold}
           </div>
        </div>
      </div>
      <div className={`px-4 py-1.5 rounded-lg text-[10px] font-black tracking-widest uppercase border ${isBuy ? 'bg-green-500/10 text-green-400 border-green-500/20' : 'bg-red-500/10 text-red-400 border-red-500/20'}`}>
        {gene.action}
      </div>
    </motion.div>
  );
}

function LabContent() {
  const searchParams = useSearchParams();
  const initialSymbol = (searchParams.get('symbol') || api.getActiveSymbol()).toUpperCase();
  
  const [symbol, setSymbol] = useState(initialSymbol);
  const [searchInput, setSearchInput] = useState(initialSymbol);
  const [loading, setLoading] = useState(false);
  const [evolution, setEvolution] = useState<StrategyResponse | null>(null);

  const startEvolution = async (sym: string) => {
    setLoading(true);
    setEvolution(null);
    try {
      const data = await api.evolveStrategy(sym);
      setEvolution(data);
    } catch (err) {
      console.error("Evolution failed:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchInput) {
      setSymbol(searchInput.toUpperCase());
      startEvolution(searchInput.toUpperCase());
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
               <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center border border-purple-500/30">
                  <Dna className="text-purple-400" size={18} />
               </div>
               <span className="text-[10px] font-bold text-purple-400/60 uppercase tracking-[0.3em]">Module // Genetic Strategy Lab</span>
            </div>
            <h1 className="text-5xl font-black tracking-tighter uppercase italic">
              Neural <span className="text-purple-500">Evolution</span>
            </h1>
          </div>

          <form onSubmit={handleSearch} className="flex gap-4">
            <div className="relative">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-white/20" size={16} />
              <input 
                type="text" 
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                placeholder="SYMBOL (e.g. INFY.NS)"
                className="bg-black/40 border border-white/5 rounded-xl py-3 pl-12 pr-6 text-sm font-mono w-64 focus:outline-none focus:border-purple-500/50 transition-all uppercase"
              />
            </div>
            <button 
              type="submit"
              disabled={loading}
              className="bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white px-8 py-3 rounded-xl font-bold text-xs uppercase tracking-widest transition-all flex items-center gap-2 shadow-[0_0_20px_rgba(147,51,234,0.3)]"
            >
              <Cpu size={14} />
              {loading ? 'Evolving...' : 'Start Evolution'}
            </button>
          </form>
        </header>

        {/* Hero Section */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8 mb-12">
           <div className="lg:col-span-3 p-10 rounded-3xl bg-gradient-to-br from-purple-600/10 to-transparent border border-white/5 relative overflow-hidden group">
              <div className="absolute top-0 right-0 p-12 opacity-5 scale-150 rotate-12 group-hover:scale-125 transition-transform duration-1000">
                 <Binary size={200} />
              </div>
              <div className="relative z-10 max-w-2xl">
                 <h2 className="text-3xl font-black tracking-tight uppercase mb-4 italic">Next-Gen Strategy <span className="text-purple-400">Breeding</span></h2>
                 <p className="text-white/40 leading-relaxed text-sm mb-8 uppercase font-medium tracking-tight">
                    Simulating 5,000+ generations of technical indicator combinations using a tournament-selection genetic algorithm. The goal is to find non-linear alpha in historical volatility clusters.
                 </p>
                 <div className="flex gap-8">
                    <div className="flex items-center gap-3">
                       <span className="text-xs font-bold text-purple-400">01.</span>
                       <span className="text-[10px] font-black uppercase text-white/20">Mutation Rate: 5%</span>
                    </div>
                    <div className="flex items-center gap-3">
                       <span className="text-xs font-bold text-purple-400">02.</span>
                       <span className="text-[10px] font-black uppercase text-white/20">Crossover: Two-Point</span>
                    </div>
                    <div className="flex items-center gap-3">
                       <span className="text-xs font-bold text-purple-400">03.</span>
                       <span className="text-[10px] font-black uppercase text-white/20">Elite Preservation: On</span>
                    </div>
                 </div>
              </div>
           </div>

           <div className="p-8 rounded-3xl bg-black/40 border border-white/5 flex flex-col justify-center text-center">
              <div className="text-[10px] font-bold text-white/20 uppercase tracking-widest mb-6">Population Strength</div>
              <div className="text-6xl font-black text-cyan-400 mb-2 italic tracking-tighter">84<span className="text-xl">%</span></div>
              <div className="text-[10px] font-black text-white/40 uppercase tracking-widest">Average Fitness</div>
              <div className="mt-8 h-1 bg-white/5 rounded-full overflow-hidden">
                 <motion.div animate={{ width: "84%" }} className="h-full bg-cyan-500" />
              </div>
           </div>
        </div>

        {/* Results Area */}
        <AnimatePresence mode="wait">
           {evolution ? (
             <motion.div 
               initial={{ opacity: 0, y: 30 }}
               animate={{ opacity: 1, y: 0 }}
               className="grid grid-cols-1 lg:grid-cols-2 gap-12"
             >
                {/* Genes List */}
                <div className="space-y-6">
                   <div className="flex items-center justify-between mb-8">
                      <h3 className="text-xl font-bold tracking-tighter uppercase italic flex items-center gap-2">
                         <Binary size={18} className="text-purple-400" />
                         Evolved Genes
                      </h3>
                      <span className="text-[10px] font-bold text-white/20 uppercase">Dominancy Threshold {evolution.best_strategy.fitness.toFixed(2)}</span>
                   </div>
                   <div className="space-y-3">
                      {evolution.best_strategy.genes.map((gene, i) => (
                        <GeneCard key={i} gene={gene} index={i} />
                      ))}
                   </div>
                   <div className="pt-8 flex items-center gap-4 text-white/20">
                      <Sparkles size={16} />
                      <p className="text-[10px] font-medium leading-relaxed uppercase">
                        AI has successfully converged on a strategy that prioritizes volatility compression as a breakout signal.
                      </p>
                   </div>
                </div>

                {/* Performance Summary */}
                <div className="p-8 rounded-3xl bg-black/40 border border-white/5 relative overflow-hidden h-fit">
                   <div className="flex items-center gap-3 mb-10">
                      <TrendingUp size={20} className="text-cyan-400" />
                      <h3 className="text-xl font-bold tracking-tighter uppercase italic">Institutional Yield</h3>
                   </div>

                   <div className="grid grid-cols-2 gap-12 mb-12">
                      <div className="space-y-1">
                         <div className="text-[10px] font-bold text-white/20 uppercase">Total Return</div>
                         <div className="text-4xl font-black text-green-400">{(evolution.best_strategy.total_return * 100).toFixed(2)}%</div>
                      </div>
                      <div className="space-y-1">
                         <div className="text-[10px] font-bold text-white/20 uppercase">Win Rate</div>
                         <div className="text-4xl font-black text-cyan-400">{(evolution.best_strategy.win_rate * 100).toFixed(1)}%</div>
                      </div>
                      <div className="space-y-1">
                         <div className="text-[10px] font-bold text-white/20 uppercase">Sharpe Ratio</div>
                         <div className="text-4xl font-black text-purple-400">{evolution.best_strategy.sharpe_ratio.toFixed(2)}</div>
                      </div>
                      <div className="space-y-1">
                         <div className="text-[10px] font-bold text-white/20 uppercase">Num Trades</div>
                         <div className="text-4xl font-black text-white">{evolution.best_strategy.num_trades}</div>
                      </div>
                   </div>

                   <div className="p-6 rounded-2xl bg-cyan-500/5 border border-cyan-500/10 mb-8">
                      <div className="flex items-center gap-3 mb-3">
                         <Target size={16} className="text-cyan-400" />
                         <span className="text-[10px] font-bold text-cyan-400 uppercase tracking-widest">Convergence Report</span>
                      </div>
                      <p className="text-[10px] text-white/40 leading-relaxed uppercase font-medium">
                        The evolved strategy is robust against market noise and shows high persistence in Trending and Breakout regimes. 
                        Recommended for high-conviction institutional positions.
                      </p>
                   </div>

                   <button className="w-full py-4 rounded-xl border border-white/10 hover:border-cyan-500/50 hover:text-cyan-400 transition-all font-black text-[10px] uppercase tracking-[0.3em] flex items-center justify-center gap-3 active:scale-[0.98]">
                      Export Strategy Weights
                      <ChevronRight size={14} />
                   </button>
                </div>
             </motion.div>
           ) : (
             <div className="h-[50vh] flex flex-col items-center justify-center text-white/10">
                <motion.div 
                  animate={loading ? { rotate: 360 } : {}} 
                  transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                >
                  <Cpu size={64} className="mb-6 opacity-30" />
                </motion.div>
                <p className="text-sm font-bold uppercase tracking-[0.5em] italic">Initializing Genetic Simulation Hub</p>
             </div>
           )}
        </AnimatePresence>
      </main>
    </div>
  );
}

export default function LabPage() {
  return (
    <Suspense fallback={<div>Connecting neural evolution kernel...</div>}>
      <LabContent />
    </Suspense>
  );
}
