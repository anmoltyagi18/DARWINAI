'use client';

import React, { useState, useEffect } from 'react';
import { Sidebar } from '@/components/Sidebar';
import { ChartPanel } from '@/components/ChartPanel';
import { AIInsightsPanel } from '@/components/AIInsightsPanel';
import { TradeTable } from '@/components/TradeTable';
import { api, AnalyzeResponse } from '@/lib/api';
import { Search, Bell, Globe, Command, ChevronDown, User, AlertTriangle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const COMMON_STOCKS = [
  { symbol: 'RELIANCE.NS', name: 'Reliance Industries' },
  { symbol: 'HDFCBANK.NS', name: 'HDFC Bank' },
  { symbol: 'TCS.NS', name: 'Tata Consultancy' },
  { symbol: 'INFY.NS', name: 'Infosys' },
  { symbol: 'ICICIBANK.NS', name: 'ICICI Bank' },
  { symbol: 'SBIN.NS', name: 'State Bank of India' },
  { symbol: 'AAPL', name: 'Apple Inc.' },
  { symbol: 'MSFT', name: 'Microsoft' },
  { symbol: 'TSLA', name: 'Tesla Inc.' },
  { symbol: 'NVDA', name: 'NVIDIA Corp.' },
  { symbol: 'BTC-USD', name: 'Bitcoin (INR/Equiv)' },
];

export default function DashboardPage() {
  const [symbol, setSymbol] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [analyzeData, setAnalyzeData] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [strategyMode, setStrategyMode] = useState<'intraday' | 'swing' | 'institutional'>('swing');

  const modeConfigs = {
    intraday: { period: '5d', interval: '15m' },
    swing: { period: '1mo', interval: '1h' },
    institutional: { period: '1y', interval: '1d' }
  };

  const performAnalysis = async (sym: string) => {
    setError(null);
    setLoading(true);
    try {
      const config = modeConfigs[strategyMode];
      const data = await api.analyzeStock(sym, config.period, config.interval);
      setAnalyzeData(data);
      setSymbol(sym);
      api.setActiveSymbol(sym);
    } catch (err: any) {
      console.error('Analysis failed:', err);
      setError(`Failed to analyze "${sym}". Please ensure the ticker is correct (e.g., add .NS for NSE).`);
      setAnalyzeData(null);
      setSymbol('');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const lastSymbol = api.getActiveSymbol();
    if (lastSymbol) {
      setSymbol(lastSymbol);
      setSearchInput(lastSymbol);
      performAnalysis(lastSymbol);
    }
  }, [strategyMode]);

  const filteredStocks = COMMON_STOCKS.filter(stock => 
    stock.symbol.toLowerCase().includes(searchInput.toLowerCase()) || 
    stock.name.toLowerCase().includes(searchInput.toLowerCase())
  );

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    const input = searchInput.trim();
    if (input) {
      // Smart lookup: if user typed a known name, use the symbol
      const foundMatch = COMMON_STOCKS.find(s => s.name.toLowerCase() === input.toLowerCase());
      const ticker = foundMatch ? foundMatch.symbol : input.toUpperCase().replace(/\s+/g, '');
      performAnalysis(ticker);
    }
  };

  return (
    <div className="flex h-screen bg-[#0b0f14] text-gray-300 font-sans selection:bg-cyan-500/30">
      {/* 1. LEFT SIDEBAR */}
      <Sidebar />

      {/* MAIN CONTENT AREA */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
        
        {/* TOP HEADER (GLOBAL CONTROLS) */}
        <header className="h-16 border-b border-white/5 bg-black/20 backdrop-blur-md flex items-center justify-between px-8 z-20">
           <div className="flex items-center flex-1 max-w-xl">
              <form onSubmit={handleSearch} className="relative w-full group">
                 <button type="submit" className="absolute left-3 top-1/2 -translate-y-1/2 text-white/20 group-focus-within:text-cyan-400 hover:text-cyan-300 transition-colors z-10">
                    <Search size={18} />
                 </button>
                 <input 
                    type="text"
                    value={searchInput}
                    onChange={(e) => { setSearchInput(e.target.value); if (error) setError(null); }}
                    onFocus={() => setShowSuggestions(true)}
                    onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
                    placeholder="Search (e.g. AAPL, TSLA, RELIANCE.NS, TCS.BO...)"
                    className="w-full bg-white/5 border border-white/10 rounded-xl py-2 pl-10 pr-4 text-sm text-white focus:outline-none focus:ring-2 focus:ring-cyan-500/40 focus:bg-white/10 transition-all tracking-tight uppercase font-mono"
                 />
                 <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center space-x-1 pointer-events-none">
                    <kbd className="px-1.5 py-0.5 rounded bg-white/5 border border-white/10 text-[10px] text-white/40 font-mono">⌘</kbd>
                    <kbd className="px-1.5 py-0.5 rounded bg-white/5 border border-white/10 text-[10px] text-white/40 font-mono">K</kbd>
                 </div>

                 {/* Autocomplete Dropdown */}
                 <AnimatePresence>
                   {showSuggestions && searchInput.trim().length > 0 && filteredStocks.length > 0 && (
                     <motion.div 
                       initial={{ opacity: 0, y: 10 }}
                       animate={{ opacity: 1, y: 0 }}
                       exit={{ opacity: 0, y: 5 }}
                       className="absolute top-[110%] left-0 w-full bg-[#121820] border border-white/10 rounded-xl shadow-2xl overflow-hidden z-50 flex flex-col"
                     >
                       {filteredStocks.map((stock) => (
                         <div 
                           key={stock.symbol}
                           onMouseDown={(e) => {
                             e.preventDefault();
                             setSearchInput(stock.symbol);
                             setShowSuggestions(false);
                             performAnalysis(stock.symbol);
                           }}
                           className="flex items-center justify-between px-4 py-3 hover:bg-white/5 cursor-pointer border-b border-white/5 last:border-0 transition-colors"
                         >
                           <span className="text-white font-mono font-bold tracking-tight">{stock.symbol}</span>
                           <span className="text-white/40 text-xs truncate ml-4">{stock.name}</span>
                         </div>
                       ))}
                     </motion.div>
                   )}
                 </AnimatePresence>
              </form>
           </div>

           <div className="flex items-center space-x-6 ml-8">
              <div className="flex items-center space-x-1 px-3 py-1.5 bg-green-500/10 rounded-full border border-green-500/20 text-[10px] font-bold text-green-400 uppercase tracking-widest shadow-sm shadow-green-500/5">
                <Globe size={12} />
                <span>Live Feed</span>
              </div>
              <div className="h-8 w-px bg-white/5"></div>
              <div className="flex items-center space-x-4">
                 <button className="relative text-white/40 hover:text-white transition-colors">
                    <Bell size={20} />
                    <span className="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full border-2 border-[#0b0f14]"></span>
                 </button>
                 <button className="flex items-center space-x-2 p-1 rounded-lg hover:bg-white/5 transition-colors border border-transparent hover:border-white/5">
                    <div className="w-8 h-8 rounded bg-cyan-500/10 flex items-center justify-center text-cyan-400 font-bold border border-cyan-500/20 shadow-lg shadow-cyan-500/5">
                       DS
                    </div>
                    <ChevronDown size={14} className="text-white/20" />
                 </button>
              </div>
           </div>
        </header>

        {/* BODY AREA */}
        {symbol && analyzeData ? (
          <div className="flex-1 flex overflow-hidden p-6 gap-6">
            {/* 2. CENTER PANEL (CHART + TABLE) */}
            <div className="flex-[3] flex flex-col min-w-0 gap-6 overflow-hidden">
               {/* Chart Section */}
               <div className="flex-[2] min-h-0">
                  <ChartPanel symbol={symbol} />
               </div>
               
               {/* Logs Section */}
               <div className="flex-1 min-h-0">
                  <TradeTable symbol={symbol} />
               </div>
            </div>

            {/* 3. RIGHT PANEL (AI INSIGHTS) */}
            <div className="flex-1 min-w-[320px] max-w-[420px] h-full overflow-hidden">
               <AIInsightsPanel 
                 data={analyzeData} 
                 loading={loading} 
                 currentMode={strategyMode}
                 onModeChange={setStrategyMode}
               />
            </div>
          </div>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-center p-8 animate-in fade-in duration-1000">
             <div className="w-24 h-24 bg-cyan-500/5 rounded-full flex items-center justify-center mb-6 border border-cyan-500/10 shadow-[0_0_100px_-20px_rgba(6,182,212,0.15)]">
               <Search size={40} className="text-cyan-500/40" />
             </div>
             <h2 className="text-2xl font-medium text-white/80 mb-3 tracking-tight">AI Terminal Active</h2>
             <p className="text-white/40 max-w-md font-mono text-sm leading-relaxed">
               System online. Awaiting ticker symbol to initialize multi-model intelligence scan, backtest engine, and live pattern detection.
             </p>
             
             {error && (
               <motion.div 
                 initial={{ opacity: 0, y: 10 }}
                 animate={{ opacity: 1, y: 0 }}
                 className="mt-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-xs font-mono max-w-md"
               >
                 <AlertTriangle size={14} className="inline mr-2 mb-0.5" />
                 {error}
               </motion.div>
             )}

             <div className="mt-8 flex items-center space-x-3 text-xs font-mono text-white/30">
               <kbd className="px-2 py-1 rounded bg-white/5 border border-white/10">⌘K</kbd>
               <span>to focus search</span>
             </div>
          </div>
        )}
      </main>

      {/* Global Background Accents */}
      <div className="fixed top-0 right-0 w-[500px] h-[500px] bg-cyan-500/5 rounded-full blur-[120px] -z-10 pointer-events-none"></div>
      <div className="fixed bottom-0 left-64 w-[400px] h-[400px] bg-purple-600/5 rounded-full blur-[120px] -z-10 pointer-events-none"></div>
    </div>
  );
}
