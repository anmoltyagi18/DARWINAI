const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface AnalyzeResponse {
  symbol: string;
  signal: string;
  confidence: number;
  market_regime: string;
  sentiment: string;
  risk_level: string;
  price: number;
  price_change_pct: number;
  indicators: Record<string, number>;
  risk_metrics: {
    daily_volatility: number;
    max_drawdown: number;
    sharpe_proxy: number;
  };
  trade_params: {
    stop_loss_pct: number;
    take_profit_pct: number;
    position_size_pct: number;
    risk_reward_ratio: number;
  };
  analysis_notes: string[];
  timestamp: string;
}

export interface OpportunityResponse {
  symbol: string;
  best_buy_date: string;
  best_buy_price: number;
  best_sell_date: string;
  best_sell_price: number;
  profit_pct: number;
  buy_signal_type: string;
  sell_signal_type: string;
  holding_days: number;
  strategy: string;
  trade_type: string;
  holding_period: string;
  strategy_confidence: number;
  strategy_rationale: string;
  all_scores: Record<string, number>;
  chart_markers: {
    buy_signals: Array<{ date: string; price: number; score: number; signal_type: string }>;
    sell_signals: Array<{ date: string; price: number; score: number; signal_type: string }>;
  };
  current_signal: string;
  current_confidence: number;
  market_regime: string;
}

export interface BacktestResponse {
  symbol: string;
  profit: number;
  profit_pct: number;
  win_rate: number;
  max_drawdown: number;
  sharpe_ratio: number;
  total_trades: number;
  starting_value: number;
  final_value: number;
  bars: number;
  data_source: string;
  timestamp: string;
}

export interface StrategyResponse {
  symbol: string;
  best_strategy: {
    fitness: number;
    total_return: number;
    sharpe_ratio: number;
    win_rate: number;
    num_trades: number;
    genes: Array<{
      indicator: string;
      period: number;
      threshold: number;
      condition: string;
      action: string;
    }>;
  };
  timestamp: string;
}

export interface LiveResponse {
  symbol: string;
  price: number;
  signal: string;
  confidence: number;
  market_regime: string;
  risk_level: number;
}

export interface CandleData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

const wrapFetch = async <T>(url: string, options?: RequestInit): Promise<T> => {
  try {
    const res = await fetch(url, options);
    if (!res.ok) {
      const errorText = await res.text().catch(() => "Response parsing error");
      throw new Error(`API Error [${res.status}]: ${errorText || res.statusText}`);
    }
    return res.json();
  } catch (err: any) {
    if (err.name === 'TypeError' && err.message === 'Failed to fetch') {
      throw new Error("Backend server is unreachable. Please ensure 'run_server.py' is running on port 8000.");
    }
    throw err;
  }
};

export const api = {
  async analyzeStock(symbol: string, period: string = "1mo", interval: string = "1d"): Promise<AnalyzeResponse> {
    return wrapFetch(`${API_BASE}/analyze/${symbol}?period=${period}&interval=${interval}`);
  },

  async getOpportunity(symbol: string): Promise<OpportunityResponse> {
    return wrapFetch(`${API_BASE}/opportunity/${symbol}`);
  },

  async backtestStock(symbol: string): Promise<BacktestResponse> {
    return wrapFetch(`${API_BASE}/backtest/${symbol}`);
  },

  async liveTrade(symbol: string): Promise<LiveResponse> {
    return wrapFetch(`${API_BASE}/live/${symbol}`);
  },

  async getChartData(symbol: string, period: string = "1mo", interval: string = "1h"): Promise<CandleData[]> {
    return wrapFetch(`${API_BASE}/chart/${symbol}?period=${period}&interval=${interval}`);
  },

  async evolveStrategy(symbol: string): Promise<StrategyResponse> {
    return wrapFetch(`${API_BASE}/strategy/${symbol}`);
  },

  // ── Persistence Helpers ──
  getActiveSymbol(): string {
    if (typeof window !== "undefined") {
      return localStorage.getItem("aigofin_active_symbol") || "RELIANCE.NS";
    }
    return "RELIANCE.NS";
  },

  setActiveSymbol(symbol: string) {
    if (typeof window !== "undefined") {
      localStorage.setItem("aigofin_active_symbol", symbol);
    }
  }
};
