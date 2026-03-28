'use client';

import React, { useEffect, useRef, useState } from 'react';
import { 
  createChart, 
  ColorType, 
  ISeriesApi, 
  Time, 
  SeriesType
} from 'lightweight-charts';
import { api } from '@/lib/api';
import { 
  Maximize2, 
  RefreshCw, 
  ZoomIn, 
  ZoomOut, 
  TrendingUp, 
  Activity, 
  BarChart3, 
  AlertCircle 
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface ChartPanelProps {
  symbol: string;
  signals?: { time: number; type: 'buy' | 'sell'; price: number }[];
}

type ChartType = 'candle' | 'line' | 'area';

export const ChartPanel: React.FC<ChartPanelProps> = ({ symbol, signals = [] }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);
  const seriesRef = useRef<ISeriesApi<SeriesType> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeframe, setTimeframe] = useState('1h');
  const [chartType, setChartType] = useState<ChartType>('candle');

  // 1. Unified Chart lifecycle
  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create the chart instance
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#0b0f14' },
        textColor: '#d1d5db',
      },
      grid: {
        vertLines: { color: '#1f2937' },
        horzLines: { color: '#1f2937' },
      },
      crosshair: { mode: 1 },
      rightPriceScale: { borderColor: '#374151' },
      timeScale: {
        borderColor: '#374151',
        timeVisible: true,
      },
      localization: {
        priceFormatter: (p: number) => 
          new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR' }).format(p),
      },
      width: chartContainerRef.current.clientWidth,
      height: 500,
    });

    chartRef.current = chart;

    const handleResize = () => {
      chart.applyOptions({ width: chartContainerRef.current?.clientWidth });
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []);

  // 2. Synchronize series and type
  useEffect(() => {
    if (!chartRef.current) return;

    // Clean up old series
    if (seriesRef.current) {
       try {
          chartRef.current.removeSeries(seriesRef.current);
       } catch (e) { /* ignore cleanup errors */ }
       seriesRef.current = null;
    }

    // Add new series
    let series: ISeriesApi<SeriesType>;
    if (chartType === 'candle') {
      series = chartRef.current.addCandlestickSeries({
        upColor: '#22c55e',
        downColor: '#ef4444',
        borderVisible: false,
        wickUpColor: '#22c55e',
        wickDownColor: '#ef4444',
      });
    } else if (chartType === 'line') {
      series = chartRef.current.addLineSeries({
        color: '#06b6d4',
        lineWidth: 2,
      });
    } else {
      series = chartRef.current.addAreaSeries({
        topColor: 'rgba(6, 182, 212, 0.4)',
        bottomColor: 'rgba(6, 182, 212, 0.0)',
        lineColor: '#06b6d4',
        lineWidth: 2,
      });
    }

    seriesRef.current = series;
    refreshData();
  }, [chartType]);

  // 3. Data Synchronization
  const refreshData = async () => {
    if (!seriesRef.current || !chartRef.current) return;
    setLoading(true);
    setError(null);

    try {
      const data = await api.getChartData(symbol, '1mo', timeframe);
      
      if (!data || data.length === 0) {
        throw new Error(`Market connectivity established, but no historical tape was returned for ${symbol}. Please ensure the ticker exists.`);
      }

      let formatted: any[];
      if (chartType === 'candle') {
        formatted = data.map(d => ({
          time: d.time as Time,
          open: d.open,
          high: d.high,
          low: d.low,
          close: d.close,
        }));
      } else {
        formatted = data.map(d => ({
          time: d.time as Time,
          value: d.close,
        }));
      }
      
      seriesRef.current.setData(formatted);

      if (signals.length > 0) {
        const markers = signals.map(s => ({
          time: s.time as Time,
          position: s.type === 'buy' ? 'belowBar' : 'aboveBar',
          color: s.type === 'buy' ? '#22c55e' : '#ef4444',
          shape: s.type === 'buy' ? 'arrowUp' : 'arrowDown',
          text: s.type.toUpperCase(),
        }));
        seriesRef.current.setMarkers(markers as any);
      }

      chartRef.current.timeScale().fitContent();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshData();
  }, [symbol, timeframe, signals]);

  return (
    <div className="bg-[#111827] rounded-[32px] border border-white/5 overflow-hidden flex flex-col h-full shadow-2xl relative group">
      <div className="p-6 flex items-center justify-between border-b border-white/5 bg-black/20 backdrop-blur-md">
        <div className="flex items-center space-x-6">
          <div className="flex flex-col">
             <div className="flex items-center gap-2 mb-1">
                <div className={`w-2 h-2 rounded-full ${loading ? 'bg-cyan-500 animate-pulse' : 'bg-green-500'} shadow-[0_0_10px_rgba(34,197,94,0.4)]`}></div>
                <span className="text-[10px] font-bold text-white/20 uppercase tracking-[0.2em]">{loading ? 'Synchronizing' : 'Live Data Stream'}</span>
             </div>
             <h2 className="text-2xl font-black italic tracking-tighter text-white">
                {symbol} <span className="text-cyan-500">/ INR</span>
             </h2>
          </div>

          <div className="h-10 w-px bg-white/5 mx-2"></div>

          {/* Timeframe selector */}
          <div className="flex bg-black/40 rounded-xl p-1 border border-white/5">
            {['1m', '5m', '15m', '1h', '1d'].map(tf => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                className={`px-4 py-1.5 text-[10px] font-black uppercase rounded-lg transition-all ${
                  timeframe === tf 
                    ? 'bg-cyan-500 text-black shadow-[0_0_20px_rgba(6,182,212,0.4)]' 
                    : 'text-gray-500 hover:text-white'
                }`}
              >
                {tf}
              </button>
            ))}
          </div>

          {/* Chart type selector */}
          <div className="flex bg-black/40 rounded-xl p-1 border border-white/5">
             <button 
                onClick={() => setChartType('candle')}
                className={`p-2 rounded-lg transition-all ${chartType === 'candle' ? 'bg-white/10 text-cyan-400' : 'text-gray-600 hover:text-white'}`}
                title="Candlestick"
             >
                <BarChart3 size={16} />
             </button>
             <button 
                onClick={() => setChartType('line')}
                className={`p-2 rounded-lg transition-all ${chartType === 'line' ? 'bg-white/10 text-cyan-400' : 'text-gray-600 hover:text-white'}`}
                title="Line Chart"
             >
                <TrendingUp size={16} />
             </button>
             <button 
                onClick={() => setChartType('area')}
                className={`p-2 rounded-lg transition-all ${chartType === 'area' ? 'bg-white/10 text-cyan-400' : 'text-gray-600 hover:text-white'}`}
                title="Area Chart"
             >
                <Activity size={16} />
             </button>
          </div>
        </div>

        <div className="flex items-center space-x-3">
           <button onClick={refreshData} className="p-3 text-gray-500 hover:text-white bg-white/5 rounded-xl border border-white/5 transition-all">
              <RefreshCw size={18} className={loading ? 'animate-spin' : ''} />
           </button>
           <button className="p-3 text-gray-500 hover:text-white bg-white/5 rounded-xl border border-white/5 transition-all">
              <Maximize2 size={18} />
           </button>
        </div>
      </div>

      <div className="flex-1 relative min-h-[500px] overflow-hidden">
        <div ref={chartContainerRef} className="w-full h-full" />
        
        <AnimatePresence mode="wait">
          {error && (
            <motion.div 
              initial={{ opacity: 0 }} 
              animate={{ opacity: 1 }} 
              exit={{ opacity: 0 }}
              className="absolute inset-0 z-50 bg-[#0b0f14]/80 backdrop-blur-sm flex items-center justify-center p-8 text-center"
            >
              <div className="max-w-md">
                <AlertCircle className="w-12 h-12 text-red-500/50 mx-auto mb-6" />
                <h3 className="text-xl font-black uppercase italic text-white mb-2">Service Offline</h3>
                <p className="text-[10px] font-bold text-white/40 mb-8 leading-relaxed uppercase tracking-widest">{error}</p>
                <button 
                  onClick={refreshData}
                  className="px-10 py-4 bg-white text-black font-black text-[10px] uppercase tracking-[0.2em] rounded-2xl hover:bg-cyan-400 transition-all active:scale-95"
                >
                  Reconnect to Terminal
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};
