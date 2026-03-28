'use client';

import { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceDot,
} from 'recharts';

export interface CandleData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface TradeSignal {
  index: number;
  type: 'buy' | 'sell';
  price: number;
  time: string;
}

interface CandlestickChartProps {
  data: CandleData[];
  signals?: TradeSignal[];
  height?: number;
}

export function CandlestickChart({ data, signals = [], height = 400 }: CandlestickChartProps) {
  const chartData = useMemo(
    () =>
      data.map((candle, index) => ({
        index,
        time: candle.time,
        close: candle.close,
        open: candle.open,
        high: candle.high,
        low: candle.low,
      })),
    [data]
  );

  const buySignals = useMemo(
    () => signals.filter((s) => s.type === 'buy'),
    [signals]
  );

  const sellSignals = useMemo(
    () => signals.filter((s) => s.type === 'sell'),
    [signals]
  );

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2540" />
        <XAxis
          dataKey="time"
          stroke="#9ca3af"
          style={{ fontSize: '12px' }}
          tick={{ fill: '#9ca3af' }}
          interval={Math.floor(data.length / 6)}
        />
        <YAxis
          stroke="#9ca3af"
          style={{ fontSize: '12px' }}
          tick={{ fill: '#9ca3af' }}
          domain={['dataMin - 10', 'dataMax + 10']}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: '#0f1428',
            border: '1px solid #1e2540',
            borderRadius: '8px',
            color: '#e4e7eb',
          }}
          labelStyle={{ color: '#e4e7eb' }}
          formatter={(value: any) => {
            if (typeof value === 'number') {
              return `$${value.toFixed(2)}`;
            }
            return value;
          }}
          cursor={{ fill: 'rgba(16, 185, 129, 0.1)' }}
        />
        <Legend wrapperStyle={{ color: '#9ca3af' }} />

        {/* Close Price Line */}
        <Line
          type="monotone"
          dataKey="close"
          stroke="#06b6d4"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
          name="Close Price"
        />

        {/* High Price Line - subtle */}
        <Line
          type="monotone"
          dataKey="high"
          stroke="#10b981"
          strokeWidth={1}
          dot={false}
          isAnimationActive={false}
          opacity={0.3}
          name="High"
          strokeDasharray="5 5"
        />

        {/* Low Price Line - subtle */}
        <Line
          type="monotone"
          dataKey="low"
          stroke="#ef4444"
          strokeWidth={1}
          dot={false}
          isAnimationActive={false}
          opacity={0.3}
          name="Low"
          strokeDasharray="5 5"
        />

        {/* Buy Signals */}
        {buySignals.map((signal) => (
          <ReferenceDot
            key={`buy-${signal.index}`}
            x={chartData[signal.index]?.time}
            y={signal.price}
            r={6}
            fill="#10b981"
            stroke="#0a0e27"
            strokeWidth={2}
          />
        ))}

        {/* Sell Signals */}
        {sellSignals.map((signal) => (
          <ReferenceDot
            key={`sell-${signal.index}`}
            x={chartData[signal.index]?.time}
            y={signal.price}
            r={6}
            fill="#ef4444"
            stroke="#0a0e27"
            strokeWidth={2}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
