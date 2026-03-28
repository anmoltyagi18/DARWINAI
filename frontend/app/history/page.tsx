'use client';

import React from 'react';
import { Sidebar } from '@/components/Sidebar';
import { 
  History, 
  ArrowUpRight, 
  ArrowDownRight, 
  Search, 
  Filter, 
  Download, 
  ShieldCheck, 
  Clock,
  ExternalLink
} from 'lucide-react';
import { motion } from 'framer-motion';

// ─── MOCK DATA ─────────────────────────────────────────────────────────────

const TRANSACTIONS = [
  { id: 'TX-9842', symbol: 'RELIANCE.NS', type: 'BUY', price: 2942.15, qty: 50, date: '2026-03-27 10:42', status: 'COMPLETED', total: 147107.50 },
  { id: 'TX-9841', symbol: 'TCS.NS', type: 'BUY', price: 4102.30, qty: 25, date: '2026-03-27 09:15', status: 'COMPLETED', total: 102557.50 },
  { id: 'TX-9840', symbol: 'HDFCBANK.NS', type: 'SELL', price: 1421.50, qty: 100, date: '2026-03-26 15:30', status: 'COMPLETED', total: 142150.00 },
  { id: 'TX-9839', symbol: 'INFY.NS', type: 'BUY', price: 1645.20, qty: 75, date: '2026-03-26 11:10', status: 'COMPLETED', total: 123390.00 },
  { id: 'TX-9838', symbol: 'SBIN.NS', type: 'SELL', price: 742.10, qty: 500, date: '2026-03-25 14:05', status: 'COMPLETED', total: 371050.00 },
  { id: 'TX-9837', symbol: 'TATAMOTORS.NS', type: 'BUY', price: 982.45, qty: 200, date: '2026-03-25 09:20', status: 'COMPLETED', total: 196490.00 },
  { id: 'TX-9836', symbol: 'ICICIBANK.NS', type: 'SELL', price: 1085.20, qty: 150, date: '2026-03-24 16:15', status: 'COMPLETED', total: 162780.00 },
  { id: 'TX-9835', symbol: 'AXISBANK.NS', type: 'BUY', price: 1054.10, qty: 100, date: '2026-03-24 10:45', status: 'COMPLETED', total: 105410.00 }
];

// ─── COMPONENTS ─────────────────────────────────────────────────────────────

function TransactionRow({ tx, index }: any) {
  const isBuy = tx.type === 'BUY';
  
  return (
    <motion.div 
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.05 }}
      className="group grid grid-cols-7 items-center p-5 border-b border-white/5 hover:bg-white/2 transition-all cursor-pointer"
    >
      <div className="col-span-1 text-[10px] font-mono font-bold text-white/20 uppercase">{tx.id}</div>
      <div className="col-span-2">
         <div className="flex items-center gap-4">
            <div className={`w-8 h-8 rounded-lg flex items-center justify-center font-bold text-xs ${isBuy ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'}`}>
               {isBuy ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
            </div>
            <div>
               <div className="text-sm font-black italic tracking-tight">{tx.symbol}</div>
               <div className="text-[10px] font-bold text-white/20 uppercase tracking-widest">{tx.type} Sequence</div>
            </div>
         </div>
      </div>
      <div className="text-right">
         <div className="text-sm font-mono font-bold">₹{tx.price.toLocaleString()}</div>
         <div className="text-[10px] text-white/20 uppercase font-bold tracking-widest">Entry Price</div>
      </div>
      <div className="text-right">
         <div className="text-sm font-mono font-bold">{tx.qty} units</div>
         <div className="text-[10px] text-white/20 uppercase font-bold tracking-widest">Volume</div>
      </div>
      <div className="text-right">
         <div className="text-sm font-mono font-black text-white">₹{tx.total.toLocaleString()}</div>
         <div className="text-[10px] text-white/20 uppercase font-bold tracking-widest">Gross Total</div>
      </div>
      <div className="text-right flex justify-end items-center gap-3">
         <div className="text-[10px] font-bold text-white/30 uppercase tracking-tighter text-right leading-none">
            {tx.date.split(' ')[0]}<br/>{tx.date.split(' ')[1]}
         </div>
         <ExternalLink size={12} className="text-white/10 group-hover:text-cyan-400 transition-colors" />
      </div>
    </motion.div>
  );
}

export default function HistoryPage() {
  return (
    <div className="flex h-screen bg-[#06080c] text-white font-sans selection:bg-cyan-500/30">
      <Sidebar />
      
      <main className="flex-1 overflow-y-auto p-12 custom-scrollbar">
        {/* Header */}
        <header className="flex justify-between items-start mb-12">
          <div>
            <div className="flex items-center gap-3 mb-2">
               <div className="w-8 h-8 rounded-lg bg-cyan-500/20 flex items-center justify-center border border-cyan-500/30">
                  <History className="text-cyan-400" size={18} />
               </div>
               <span className="text-[10px] font-bold text-white/20 uppercase tracking-[0.3em]">Module // Immutable Audit log</span>
            </div>
            <h1 className="text-5xl font-black tracking-tighter uppercase italic">
              Transaction <span className="text-cyan-500">History</span>
            </h1>
          </div>

          <div className="flex gap-4">
             <div className="relative group">
                <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-white/20 group-focus-within:text-cyan-400 transition-colors" size={16} />
                <input 
                   type="text" 
                   placeholder="Filter transactions..."
                   className="bg-black/40 border border-white/5 rounded-xl py-3 pl-12 pr-6 text-sm font-mono w-64 focus:outline-none focus:border-cyan-500/50 transition-all placeholder:text-white/10"
                />
             </div>
             <button className="p-3 rounded-xl border border-white/5 hover:border-white/20 transition-all text-white/40 hover:text-white flex items-center gap-2 text-[10px] font-bold uppercase tracking-widest">
                <Download size={16} />
                Export CSV
             </button>
          </div>
        </header>

        {/* Audit Banner */}
        <div className="p-8 rounded-3xl bg-gradient-to-br from-cyan-500/10 to-transparent border border-white/5 mb-12 flex justify-between items-center group overflow-hidden relative">
           <div className="relative z-10 flex items-center gap-8">
              <div className="w-16 h-16 rounded-2xl bg-black/40 border border-white/10 flex items-center justify-center text-cyan-400 shadow-inner">
                 <ShieldCheck size={32} />
              </div>
              <div>
                 <h3 className="text-xl font-black tracking-tighter uppercase italic mb-1">Institutional Integrity</h3>
                 <p className="text-[10px] text-white/40 font-medium uppercase tracking-widest max-w-md">
                    All simulated executions are timestamped and signed by the Quant Kernel audit module. Direct modifications are restricted at the kernel level.
                 </p>
              </div>
           </div>
           <div className="relative z-10 text-right">
              <div className="text-[10px] font-bold text-white/20 uppercase tracking-[0.3em] mb-2">Uptime Validation</div>
              <div className="flex items-center gap-3 justify-end text-green-400">
                 <Clock size={16} />
                 <span className="text-lg font-black tracking-tighter italic">99.98% SLAV</span>
              </div>
           </div>
           <div className="absolute inset-0 bg-cyan-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-1000" />
        </div>

        {/* Transactions List */}
        <div className="rounded-3xl bg-black/40 border border-white/5 overflow-hidden">
           <div className="p-6 border-b border-white/5 grid grid-cols-7 text-[10px] font-black uppercase text-white/20 tracking-widest bg-white/2">
              <div className="col-span-1">ID</div>
              <div className="col-span-2">Instrument</div>
              <div className="text-right">Price</div>
              <div className="text-right">Volume</div>
              <div className="text-right">Total</div>
              <div className="text-right">Timestamp</div>
           </div>
           
           <div className="divide-y divide-white/2">
              {TRANSACTIONS.map((tx, i) => (
                <TransactionRow key={i} tx={tx} index={i} />
              ))}
           </div>

           <div className="p-8 flex justify-center bg-white/1">
              <button className="text-[10px] font-black uppercase tracking-[0.4em] text-white/10 hover:text-white transition-colors">
                 Load older records
              </button>
           </div>
        </div>
      </main>
    </div>
  );
}
