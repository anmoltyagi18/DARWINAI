'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Target, Shield, Zap, AlertTriangle, PieChart, Activity, Globe } from 'lucide-react';
import { AnalyzeResponse } from '@/lib/api';

interface AIInsightsPanelProps {
  data: AnalyzeResponse | null;
  loading?: boolean;
  currentMode: 'intraday' | 'swing' | 'institutional';
  onModeChange: (mode: 'intraday' | 'swing' | 'institutional') => void;
}

export const AIInsightsPanel: React.FC<AIInsightsPanelProps> = ({ 
  data, 
  loading, 
  currentMode, 
  onModeChange 
}) => {
  if (loading || !data) {
    return (
      <div className="bg-[#111827] rounded-xl border border-white/5 p-6 space-y-6 h-full shadow-2xl animate-pulse">
        <div className="h-4 w-1/2 bg-white/5 rounded"></div>
        <div className="h-24 w-full bg-white/5 rounded-xl"></div>
        <div className="space-y-4">
          {[1, 2, 3, 4].map(i => (
            <div key={i} className="h-10 w-full bg-white/5 rounded-lg"></div>
          ))}
        </div>
      </div>
    );
  }

  const isBuy = data.signal.includes('BUY');
  const signalColor = isBuy ? 'text-green-400' : data.signal.includes('SELL') ? 'text-red-400' : 'text-gray-400';
  const glowColor = isBuy ? 'shadow-[0_0_30px_rgba(34,197,94,0.3)]' : data.signal.includes('SELL') ? 'shadow-[0_0_30px_rgba(239,68,68,0.3)]' : '';

  return (
    <div className="bg-[#111827] rounded-xl border border-white/5 overflow-hidden flex flex-col h-full shadow-2xl">
      {/* Header */}
      <div className="p-4 border-b border-white/5 bg-black/20 flex items-center justify-between">
        <h3 className="text-sm font-bold uppercase tracking-widest text-white/60 flex items-center">
          <Zap size={14} className="mr-2 text-cyan-400" />
          Neural Insights
        </h3>
        <span className="text-[10px] text-white/30 font-mono">ID: {Math.random().toString(36).substr(2, 9).toUpperCase()}</span>
      </div>

      <div className="p-6 space-y-8 flex-1 overflow-y-auto custom-scrollbar">
        {/* Strategy Selector */}
        <div className="flex bg-black/40 rounded-xl p-1 border border-white/5">
           {[
             { id: 'intraday', label: 'Intraday', icon: <Zap size={10} /> },
             { id: 'swing', label: 'Swing', icon: <Activity size={10} /> },
             { id: 'institutional', label: 'Institutional', icon: <Globe size={10} /> }
           ].map((mode) => (
             <button
               key={mode.id}
               onClick={() => onModeChange(mode.id as any)}
               className={`flex-1 flex items-center justify-center space-x-1.5 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-wider transition-all ${
                 currentMode === mode.id 
                   ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30' 
                   : 'text-white/30 hover:text-white/60'
               }`}
             >
               {mode.icon}
               <span>{mode.label}</span>
             </button>
           ))}
        </div>

        {/* Signal Hero */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className={`relative p-6 rounded-2xl bg-black/40 border border-white/5 flex flex-col items-center justify-center text-center ${glowColor}`}
        >
          <div className="absolute top-2 right-4 text-[10px] font-mono text-white/20 uppercase tracking-tighter">Conviction Score</div>
          <div className="text-4xl font-black italic tracking-tighter mb-1 font-mono">
            <span className={signalColor}>{data.signal}</span>
          </div>
          <div className="text-[10px] text-white/30 uppercase tracking-[0.2em] font-mono mt-1 mb-4 flex items-center">
             <span className="w-1.5 h-1.5 rounded-full bg-cyan-500 mr-2 opacity-70"></span>
             Current: {new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR' }).format(data.price)}
          </div>
          <div className="text-5xl font-bold tracking-tighter text-white">
            {(data.confidence * 100).toFixed(1)}%
          </div>
          <p className="text-[10px] text-white/40 uppercase mt-2 tracking-widest font-medium italic">Execute with Precision</p>
        </motion.div>

        {/* Confidence Meter (Circular Progress) */}
        <div className="grid grid-cols-2 gap-4">
          <StatCard 
            label="Market Regime" 
            value={data.market_regime} 
            icon={<Target size={14} />} 
            color="text-cyan-400" 
          />
          <StatCard 
            label="Sentiment" 
            value={data.sentiment} 
            icon={<Activity size={14} />} 
            color="text-purple-400" 
          />
          <StatCard 
            label="Risk Score" 
            value={data.risk_level} 
            icon={<Shield size={14} />} 
            color="text-orange-400" 
          />
          <StatCard 
            label="Profit Target" 
            value={`+${data.trade_params.take_profit_pct.toFixed(2)}%`} 
            icon={<TrendingUp size={14} />} 
            color="text-green-400" 
          />
        </div>

        {/* Detailed Metrics */}
        <div className="space-y-4">
           <h4 className="text-xs font-bold text-white/40 uppercase tracking-widest flex items-center">
              <PieChart size={12} className="mr-2" />
              Risk Metrics
           </h4>
           <div className="space-y-3">
              <MetricRow label="Volatility (Annualized)" value={`${(data.risk_metrics.daily_volatility * 100).toFixed(2)}%`} />
              <MetricRow label="Sharpe Proxy" value={data.risk_metrics.sharpe_proxy.toFixed(2)} />
              <MetricRow label="Max Drawdown" value={`${(data.risk_metrics.max_drawdown * 100).toFixed(2)}%`} />
           </div>
        </div>

        {/* AI Rational Notes */}
        <div className="pt-4 border-t border-white/5">
           <h4 className="text-xs font-bold text-white/40 uppercase tracking-widest flex items-center mb-4">
              <Zap size={12} className="mr-2 text-yellow-400" />
              Discovery Reasonings
           </h4>
           <div className="space-y-3">
              {data.analysis_notes.slice(0, 3).map((note, i) => (
                <div key={i} className="text-[11px] text-white/70 bg-white/5 p-3 rounded-lg border-l-2 border-cyan-500/50 leading-relaxed italic">
                  "{note}"
                </div>
              ))}
           </div>
        </div>
      </div>

      {/* Footer Actions */}
      <div className="p-4 bg-black/40 border-t border-white/5 grid grid-cols-2 gap-2">
         <button className="py-2 bg-green-500/10 hover:bg-green-500/20 text-green-400 text-xs font-bold rounded-lg border border-green-500/20 transition-all uppercase tracking-widest shadow-lg shadow-green-500/5">
            Execute Buy
         </button>
         <button className="py-2 bg-red-500/10 hover:bg-red-500/20 text-red-400 text-xs font-bold rounded-lg border border-red-500/20 transition-all uppercase tracking-widest shadow-lg shadow-red-500/5">
            Execute Sell
         </button>
      </div>
    </div>
  );
};

const StatCard = ({ label, value, icon, color }: any) => (
  <div className="p-3 bg-black/40 rounded-xl border border-white/5">
    <div className="flex items-center text-[10px] text-white/40 uppercase tracking-tighter mb-1">
      <span className={`mr-1.5 ${color}`}>{icon}</span>
      {label}
    </div>
    <div className="text-xs font-bold text-white truncate">{value}</div>
  </div>
);

const MetricRow = ({ label, value }: any) => (
  <div className="flex items-center justify-between py-1 border-b border-white/[0.03]">
    <span className="text-[11px] text-white/50">{label}</span>
    <span className="text-[11px] font-mono font-bold text-white">{value}</span>
  </div>
);
