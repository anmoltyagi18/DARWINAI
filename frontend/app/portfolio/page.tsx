'use client';

import React, { useState, useEffect } from 'react';
import { Sidebar } from '@/components/Sidebar';
import { 
  Wallet, 
  TrendingUp, 
  PieChart, 
  ArrowUpRight, 
  ArrowDownRight, 
  ShieldCheck, 
  Briefcase, 
  Clock,
  ChevronRight,
  Plus
} from 'lucide-react';
import { motion } from 'framer-motion';

// ─── MOCK DATA ─────────────────────────────────────────────────────────────

const HOLDINGS = [
  { symbol: 'RELIANCE.NS', name: 'Reliance Industries', qty: 150, avgPrice: 2854.20, currentPrice: 2942.15, value: 441322.50, pnl: 13192.50, pnlPct: 3.08 },
  { symbol: 'TCS.NS', name: 'Tata Consultancy', qty: 85, avgPrice: 3912.45, currentPrice: 4102.30, value: 348695.50, pnl: 16137.25, pnlPct: 4.85 },
  { symbol: 'HDFCBANK.NS', name: 'HDFC Bank', qty: 320, avgPrice: 1442.10, currentPrice: 1421.50, value: 454880.00, pnl: -6592.00, pnlPct: -1.43 },
  { symbol: 'INFY.NS', name: 'Infosys', qty: 210, avgPrice: 1610.80, currentPrice: 1645.20, value: 345492.00, pnl: 7224.00, pnlPct: 2.13 },
  { symbol: 'BTC-INR', name: 'Bitcoin / INR', qty: 0.045, avgPrice: 5240000, currentPrice: 5821430, value: 261964.35, pnl: 26164.35, pnlPct: 11.08 }
];

// ─── COMPONENTS ─────────────────────────────────────────────────────────────

function HoldingRow({ holding }: any) {
  const isPositive = holding.pnl >= 0;
  
  return (
    <div className="group grid grid-cols-6 items-center p-6 border-b border-white/5 hover:bg-white/2 transition-all">
      <div className="col-span-2">
         <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center font-bold text-xs group-hover:border-cyan-500/30 transition-colors uppercase">
               {holding.symbol.split('.')[0].substring(0,2)}
            </div>
            <div>
               <div className="text-sm font-black italic tracking-tight">{holding.symbol}</div>
               <div className="text-[10px] font-bold text-white/20 uppercase tracking-widest">{holding.name}</div>
            </div>
         </div>
      </div>
      <div className="text-right">
         <div className="text-sm font-mono font-bold">{holding.qty}</div>
         <div className="text-[10px] text-white/20 uppercase font-bold">Qty</div>
      </div>
      <div className="text-right">
         <div className="text-sm font-mono font-bold">₹{holding.avgPrice.toLocaleString()}</div>
         <div className="text-[10px] text-white/20 uppercase font-bold">Avg Price</div>
      </div>
      <div className="text-right">
         <div className="text-sm font-mono font-black text-white">₹{holding.value.toLocaleString()}</div>
         <div className="text-[10px] text-white/20 uppercase font-bold tracking-widest">Market Value</div>
      </div>
      <div className="text-right">
         <div className={`text-sm font-black italic flex items-center justify-end gap-1 ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
            {isPositive ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
            {holding.pnlPct}%
         </div>
         <div className={`text-[10px] font-bold uppercase ${isPositive ? 'text-green-400/40' : 'text-red-400/40'}`}>
            {isPositive ? '+' : ''}₹{holding.pnl.toLocaleString()}
         </div>
      </div>
    </div>
  );
}

export default function PortfolioPage() {
  const totalValue = HOLDINGS.reduce((acc, h) => acc + h.value, 0) + 124500.25; // Base cash
  const totalPnL = HOLDINGS.reduce((acc, h) => acc + h.pnl, 0);
  const pnlPct = (totalPnL / totalValue) * 100;

  return (
    <div className="flex h-screen bg-[#06080c] text-white font-sans selection:bg-cyan-500/30">
      <Sidebar />
      
      <main className="flex-1 overflow-y-auto p-12 custom-scrollbar">
        {/* Header */}
        <header className="flex justify-between items-start mb-12">
          <div>
            <div className="flex items-center gap-3 mb-2">
               <div className="w-8 h-8 rounded-lg bg-green-500/20 flex items-center justify-center border border-green-500/30">
                  <Wallet className="text-green-400" size={18} />
               </div>
               <span className="text-[10px] font-bold text-green-400/60 uppercase tracking-[0.3em]">Module // Institutional Custody</span>
            </div>
            <h1 className="text-5xl font-black tracking-tighter uppercase italic">
              Portfolio <span className="text-green-500">Inventory</span>
            </h1>
          </div>

          <div className="flex gap-4">
             <button className="p-3 rounded-xl border border-white/5 hover:border-white/20 transition-all text-white/40 hover:text-white">
                <Clock size={20} />
             </button>
             <button className="bg-green-600 hover:bg-green-500 text-white px-8 py-3 rounded-xl font-bold text-xs uppercase tracking-widest transition-all flex items-center gap-2 shadow-[0_0_20px_rgba(34,197,94,0.3)]">
                <Plus size={14} />
                Deploy Capital
             </button>
          </div>
        </header>

        {/* Portfolio Overview */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-12">
           <div className="lg:col-span-2 p-10 rounded-3xl bg-black/40 border border-white/5 relative overflow-hidden flex justify-between items-center group">
              <div className="relative z-10">
                 <div className="text-[10px] font-bold text-white/30 uppercase tracking-[0.3em] mb-4">Net Liquidating Value</div>
                 <div className="text-6xl font-black tracking-tighter mb-4 italic">₹{totalValue.toLocaleString(undefined, { minimumFractionDigits: 2 })}</div>
                 <div className="flex items-center gap-6">
                    <div className="flex items-center gap-2 text-green-400">
                       <ArrowUpRight size={18} />
                       <span className="text-lg font-black tracking-tight">{pnlPct.toFixed(2)}%</span>
                       <span className="text-[10px] font-bold uppercase tracking-widest opacity-60">Session Gain</span>
                    </div>
                    <div className="h-4 w-px bg-white/10" />
                    <div className="text-[10px] font-black uppercase text-white/20 tracking-widest">
                       Cash Balance: <span className="text-white/60">₹1,24,500.25</span>
                    </div>
                 </div>
              </div>
              <div className="absolute top-0 right-0 p-12 opacity-5 scale-150 rotate-12 group-hover:scale-110 transition-transform duration-1000">
                 <ShieldCheck size={200} />
              </div>
           </div>

           <div className="p-10 rounded-3xl bg-gradient-to-br from-green-500/10 to-transparent border border-white/5 flex flex-col justify-center">
              <div className="flex items-center gap-3 mb-8">
                 <PieChart size={20} className="text-green-400" />
                 <h3 className="text-lg font-black tracking-tighter uppercase italic">Asset Allocation</h3>
              </div>
              <div className="space-y-4">
                 <AllocationBar label="Equities" pct={75} color="bg-green-500" />
                 <AllocationBar label="Crypto" pct={12} color="bg-cyan-500" />
                 <AllocationBar label="Cash" pct={13} color="bg-white/20" />
              </div>
           </div>
        </div>

        {/* Holdings Table */}
        <div className="rounded-3xl bg-black/40 border border-white/5 overflow-hidden">
           <div className="p-8 border-b border-white/5 flex justify-between items-center bg-white/2">
              <h3 className="text-lg font-black tracking-tighter uppercase italic flex items-center gap-3">
                 <Briefcase size={18} className="text-green-400" />
                 Core Holdings
              </h3>
              <div className="text-[10px] font-bold text-white/30 uppercase tracking-widest">Total Assets: {HOLDINGS.length}</div>
           </div>
           
           <div>
              {HOLDINGS.map((h, i) => (
                <HoldingRow key={i} holding={h} />
              ))}
           </div>

           <div className="p-8 flex justify-center border-t border-white/5">
              <button className="text-[10px] font-black uppercase tracking-[0.4em] text-white/20 hover:text-green-400 transition-colors flex items-center gap-3">
                 Generate Performance Audit
                 <ChevronRight size={14} />
              </button>
           </div>
        </div>
      </main>
    </div>
  );
}

function AllocationBar({ label, pct, color }: any) {
  return (
    <div className="space-y-2">
       <div className="flex justify-between text-[10px] font-bold uppercase tracking-widest">
          <span className="text-white/40">{label}</span>
          <span>{pct}%</span>
       </div>
       <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
          <motion.div initial={{ width: 0 }} animate={{ width: `${pct}%` }} className={`h-full ${color}`} />
       </div>
    </div>
  );
}
