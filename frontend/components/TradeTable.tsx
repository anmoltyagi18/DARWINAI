'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Clock } from 'lucide-react';

interface HistoryRow {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export const TradeTable: React.FC<{ symbol?: string }> = ({ symbol }) => {
  const [history, setHistory] = useState<HistoryRow[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!symbol) return;
    const fetchHistory = async () => {
      setLoading(true);
      try {
        const res = await fetch(`http://localhost:8000/history/${symbol}?days=10`);
        const result = await res.json();
        if (result && result.data) {
          setHistory(result.data);
        }
      } catch (err) {
        console.error('Failed to fetch history:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, [symbol]);

  const formatINR = (val: number) => 
    new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR' }).format(val);

  const formatVol = (vol: number) => {
    if (vol >= 1e6) return (vol / 1e6).toFixed(1) + 'M';
    if (vol >= 1e3) return (vol / 1e3).toFixed(1) + 'k';
    return vol.toString();
  };

  return (
    <div className="bg-[#111827] rounded-xl border border-white/5 overflow-hidden flex flex-col h-full shadow-2xl">
      {/* Table Header */}
      <div className="px-6 py-4 border-b border-white/5 bg-black/20 flex items-center justify-between">
        <h3 className="text-xs font-bold uppercase tracking-widest text-white/60 flex items-center">
          <Activity size={14} className="mr-2 text-cyan-400" />
          Price History ({symbol || '...'})
        </h3>
        <span className="text-[10px] text-cyan-400 font-mono uppercase tracking-widest opacity-60 inline-block w-32 text-right">
          {loading ? 'Fetching tapes...' : 'Live Stream'}
        </span>
      </div>

      <div className="flex-1 overflow-x-auto custom-scrollbar">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="bg-white/[0.02] border-b border-white/5">
              <th className="px-6 py-3 text-[10px] font-bold text-white/30 uppercase tracking-widest whitespace-nowrap">Date</th>
              <th className="px-6 py-3 text-[10px] font-bold text-white/30 uppercase tracking-widest">Open</th>
              <th className="px-6 py-3 text-[10px] font-bold text-white/30 uppercase tracking-widest">High</th>
              <th className="px-6 py-3 text-[10px] font-bold text-white/30 uppercase tracking-widest">Low</th>
              <th className="px-6 py-3 text-[10px] font-bold text-white/30 uppercase tracking-widest">Close</th>
              <th className="px-6 py-3 text-[10px] font-bold text-white/30 uppercase tracking-widest text-right">Volume</th>
            </tr>
          </thead>
          <tbody>
            <AnimatePresence>
              {history.map((row, i) => (
                <motion.tr 
                  key={row.date}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="group hover:bg-white/[0.03] border-b border-white/[0.03] transition-colors"
                >
                  <td className="px-6 py-4">
                    <div className="flex items-center text-xs font-mono text-cyan-400/80 tracking-tighter">
                      <Clock size={12} className="mr-1.5 opacity-50" />
                      {row.date}
                    </div>
                  </td>
                  <td className="px-6 py-4 text-xs font-mono text-white/80">
                    {formatINR(row.open)}
                  </td>
                  <td className="px-6 py-4 text-xs font-mono text-green-400/90">
                    {formatINR(row.high)}
                  </td>
                  <td className="px-6 py-4 text-xs font-mono text-red-400/90">
                    {formatINR(row.low)}
                  </td>
                  <td className="px-6 py-4 text-xs font-bold font-mono text-white">
                    {formatINR(row.close)}
                  </td>
                  <td className="px-6 py-4 text-right">
                    <span className="text-[11px] font-mono text-white/40 tracking-widest bg-white/5 px-2 py-1 rounded">
                      {formatVol(row.volume)}
                    </span>
                  </td>
                </motion.tr>
              ))}
            </AnimatePresence>
            {!loading && history.length === 0 && (
               <tr>
                 <td colSpan={6} className="text-center py-8 text-white/20 text-xs font-mono">No historical tape available</td>
               </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Table Footer */}
      <div className="p-3 bg-black/20 border-t border-white/5 flex justify-center">
         <button className="text-[10px] uppercase tracking-[0.2em] text-white/20 hover:text-cyan-400 transition-colors font-bold">
            Full History Mode
         </button>
      </div>
    </div>
  );
};
