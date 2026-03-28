'use client';

import { useState, useEffect } from 'react';
import { Plus, Trash2, TrendingUp, X } from 'lucide-react';

export interface Strategy {
  id: string;
  name: string;
  type: 'ma-crossover' | 'rsi' | 'macd' | 'custom';
  parameters: Record<string, number>;
  signals: number; // Number of signals generated
  winRate: number; // Percentage
  active: boolean;
}

interface StrategyPanelProps {
  data?: Array<{ close: number; high: number; low: number }>;
  onStrategyChange?: (strategies: Strategy[]) => void;
}

const DEFAULT_STRATEGIES: Strategy[] = [
  {
    id: '1',
    name: 'Fast MA Crossover',
    type: 'ma-crossover',
    parameters: { fast: 10, slow: 30 },
    signals: 12,
    winRate: 62,
    active: true,
  },
  {
    id: '2',
    name: 'RSI Overbought',
    type: 'rsi',
    parameters: { period: 14, threshold: 70 },
    signals: 8,
    winRate: 58,
    active: false,
  },
  {
    id: '3',
    name: 'MACD Strategy',
    type: 'macd',
    parameters: { fast: 12, slow: 26, signal: 9 },
    signals: 15,
    winRate: 65,
    active: false,
  },
];

export function StrategyPanel({ data, onStrategyChange }: StrategyPanelProps) {
  const [strategies, setStrategies] = useState<Strategy[]>(DEFAULT_STRATEGIES);
  const [showNew, setShowNew] = useState(false);
  const [newStrategy, setNewStrategy] = useState({
    name: '',
    type: 'custom' as const,
    param1: '20',
    param2: '50',
  });

  useEffect(() => {
    onStrategyChange?.(strategies);
  }, [strategies, onStrategyChange]);

  const handleAddStrategy = () => {
    if (!newStrategy.name.trim()) return;

    const strategy: Strategy = {
      id: Date.now().toString(),
      name: newStrategy.name,
      type: newStrategy.type,
      parameters: {
        param1: parseInt(newStrategy.param1),
        param2: parseInt(newStrategy.param2),
      },
      signals: Math.floor(Math.random() * 20) + 5,
      winRate: Math.floor(Math.random() * 30) + 50,
      active: false,
    };

    setStrategies([...strategies, strategy]);
    setNewStrategy({ name: '', type: 'custom', param1: '20', param2: '50' });
    setShowNew(false);
  };

  const toggleStrategy = (id: string) => {
    setStrategies(
      strategies.map((s) => (s.id === id ? { ...s, active: !s.active } : s))
    );
  };

  const deleteStrategy = (id: string) => {
    setStrategies(strategies.filter((s) => s.id !== id));
  };

  const getTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      'ma-crossover': 'bg-blue-900/30 text-blue-400',
      rsi: 'bg-purple-900/30 text-purple-400',
      macd: 'bg-orange-900/30 text-orange-400',
      custom: 'bg-cyan-900/30 text-cyan-400',
    };
    return colors[type] || colors.custom;
  };

  return (
    <div className="bg-card border border-border rounded-lg p-6 h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-primary" />
          <h2 className="text-lg font-semibold text-foreground">Trading Strategies</h2>
        </div>
        <button
          onClick={() => setShowNew(!showNew)}
          className="p-1.5 hover:bg-secondary rounded-lg transition-colors"
          title="Add new strategy"
        >
          {showNew ? (
            <X className="h-5 w-5 text-muted-foreground" />
          ) : (
            <Plus className="h-5 w-5 text-muted-foreground" />
          )}
        </button>
      </div>

      {/* New Strategy Form */}
      {showNew && (
        <div className="mb-4 p-4 bg-secondary rounded-lg border border-border space-y-3">
          <input
            type="text"
            placeholder="Strategy name"
            value={newStrategy.name}
            onChange={(e) => setNewStrategy({ ...newStrategy, name: e.target.value })}
            className="w-full px-3 py-2 bg-input border border-border rounded text-foreground placeholder-muted-foreground text-sm focus:outline-none focus:ring-2 focus:ring-primary"
          />

          <select
            value={newStrategy.type}
            onChange={(e) =>
              setNewStrategy({
                ...newStrategy,
                type: e.target.value as any,
              })
            }
            className="w-full px-3 py-2 bg-input border border-border rounded text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-primary"
          >
            <option value="custom">Custom Strategy</option>
            <option value="ma-crossover">MA Crossover</option>
            <option value="rsi">RSI</option>
            <option value="macd">MACD</option>
          </select>

          <div className="grid grid-cols-2 gap-2">
            <input
              type="number"
              placeholder="Param 1"
              value={newStrategy.param1}
              onChange={(e) => setNewStrategy({ ...newStrategy, param1: e.target.value })}
              className="px-3 py-2 bg-input border border-border rounded text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-primary"
            />
            <input
              type="number"
              placeholder="Param 2"
              value={newStrategy.param2}
              onChange={(e) => setNewStrategy({ ...newStrategy, param2: e.target.value })}
              className="px-3 py-2 bg-input border border-border rounded text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>

          <button
            onClick={handleAddStrategy}
            disabled={!newStrategy.name.trim()}
            className="w-full bg-primary text-primary-foreground py-2 px-3 rounded font-medium text-sm hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Create Strategy
          </button>
        </div>
      )}

      {/* Strategy List */}
      <div className="flex-1 overflow-y-auto space-y-2">
        {strategies.map((strategy) => (
          <div
            key={strategy.id}
            className={`p-3 rounded-lg border transition-all ${
              strategy.active
                ? 'bg-secondary border-primary'
                : 'bg-input border-border hover:border-primary/50'
            }`}
          >
            <div className="flex items-start justify-between mb-2">
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={strategy.active}
                    onChange={() => toggleStrategy(strategy.id)}
                    className="w-4 h-4 cursor-pointer accent-primary"
                  />
                  <h3 className="font-medium text-foreground">{strategy.name}</h3>
                </div>
                <div className="flex items-center gap-2 mt-1">
                  <span className={`text-xs px-2 py-1 rounded ${getTypeColor(strategy.type)}`}>
                    {strategy.type.replace('-', ' ').toUpperCase()}
                  </span>
                </div>
              </div>
              <button
                onClick={() => deleteStrategy(strategy.id)}
                className="p-1 text-muted-foreground hover:text-accent transition-colors"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </div>

            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-background/50 px-2 py-1.5 rounded">
                <div className="text-muted-foreground">Signals</div>
                <div className="font-semibold text-primary">{strategy.signals}</div>
              </div>
              <div className="bg-background/50 px-2 py-1.5 rounded">
                <div className="text-muted-foreground">Win Rate</div>
                <div
                  className={`font-semibold ${
                    strategy.winRate >= 60 ? 'text-primary' : 'text-amber-400'
                  }`}
                >
                  {strategy.winRate}%
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Performance Summary */}
      <div className="mt-4 pt-4 border-t border-border text-xs space-y-2">
        <div className="flex justify-between">
          <span className="text-muted-foreground">Active Strategies</span>
          <span className="text-foreground font-semibold">
            {strategies.filter((s) => s.active).length}/{strategies.length}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">Avg Win Rate</span>
          <span className="text-foreground font-semibold">
            {Math.round(strategies.reduce((sum, s) => sum + s.winRate, 0) / strategies.length)}%
          </span>
        </div>
      </div>
    </div>
  );
}
