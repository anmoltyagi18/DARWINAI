export interface CandleData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface StockData {
  symbol: string;
  name: string;
  currentPrice: number;
  priceChange: number;
  percentChange: number;
  volume: number;
  data: CandleData[];
}

// Mock stock data
const STOCK_DATA: Record<string, StockData> = {
  AAPL: {
    symbol: 'AAPL',
    name: 'Apple Inc.',
    currentPrice: 195.45,
    priceChange: 2.85,
    percentChange: 1.48,
    volume: 52350000,
    data: generateCandles(195.45, 50),
  },
  MSFT: {
    symbol: 'MSFT',
    name: 'Microsoft Corporation',
    currentPrice: 428.65,
    priceChange: 5.32,
    percentChange: 1.26,
    volume: 23450000,
    data: generateCandles(428.65, 50),
  },
  GOOGL: {
    symbol: 'GOOGL',
    name: 'Alphabet Inc.',
    currentPrice: 142.80,
    priceChange: -1.45,
    percentChange: -1.01,
    volume: 31200000,
    data: generateCandles(142.80, 50),
  },
  AMZN: {
    symbol: 'AMZN',
    name: 'Amazon.com Inc.',
    currentPrice: 210.55,
    priceChange: 3.15,
    percentChange: 1.52,
    volume: 45680000,
    data: generateCandles(210.55, 50),
  },
  TSLA: {
    symbol: 'TSLA',
    name: 'Tesla Inc.',
    currentPrice: 248.30,
    priceChange: -8.20,
    percentChange: -3.19,
    volume: 125600000,
    data: generateCandles(248.30, 50),
  },
  META: {
    symbol: 'META',
    name: 'Meta Platforms Inc.',
    currentPrice: 502.18,
    priceChange: 12.45,
    percentChange: 2.54,
    volume: 18900000,
    data: generateCandles(502.18, 50),
  },
  NVDA: {
    symbol: 'NVDA',
    name: 'NVIDIA Corporation',
    currentPrice: 878.95,
    priceChange: 22.15,
    percentChange: 2.58,
    volume: 35400000,
    data: generateCandles(878.95, 50),
  },
  JPM: {
    symbol: 'JPM',
    name: 'JPMorgan Chase & Co.',
    currentPrice: 225.30,
    priceChange: 1.85,
    percentChange: 0.83,
    volume: 8950000,
    data: generateCandles(225.30, 50),
  },
};

function generateCandles(basePrice: number, count: number): CandleData[] {
  const candles: CandleData[] = [];
  let price = basePrice;

  for (let i = 0; i < count; i++) {
    const timeStr = new Date(Date.now() - (count - i) * 24 * 60 * 60 * 1000)
      .toLocaleDateString('en-US', { month: 'short', day: 'numeric' });

    // Random walk for realistic price movement
    const randomChange = (Math.random() - 0.48) * 10; // Slightly bullish bias
    price = Math.max(price + randomChange, basePrice * 0.85);

    const volatility = basePrice * 0.02;
    const open = price + (Math.random() - 0.5) * volatility;
    const close = price + (Math.random() - 0.5) * volatility;
    const high = Math.max(open, close) + Math.random() * volatility;
    const low = Math.min(open, close) - Math.random() * volatility;

    candles.push({
      time: timeStr,
      open: parseFloat(open.toFixed(2)),
      close: parseFloat(close.toFixed(2)),
      high: parseFloat(high.toFixed(2)),
      low: parseFloat(low.toFixed(2)),
      volume: Math.floor(Math.random() * 100000000) + 10000000,
    });
  }

  return candles;
}

export function getStockData(symbol: string): StockData {
  return STOCK_DATA[symbol] || STOCK_DATA['AAPL'];
}

export function getAllStocks(): StockData[] {
  return Object.values(STOCK_DATA);
}

// Generate buy/sell signals based on simple moving average crossover
export function generateSignals(data: CandleData[]) {
  const signals = [];
  const fastMA = 10;
  const slowMA = 30;

  if (data.length < slowMA) return signals;

  for (let i = slowMA; i < data.length; i++) {
    const fastAvg = data
      .slice(i - fastMA + 1, i + 1)
      .reduce((sum, d) => sum + d.close, 0) / fastMA;
    const slowAvg = data
      .slice(i - slowMA + 1, i + 1)
      .reduce((sum, d) => sum + d.close, 0) / slowMA;

    const prevFastAvg = data
      .slice(i - fastMA, i)
      .reduce((sum, d) => sum + d.close, 0) / fastMA;
    const prevSlowAvg = data
      .slice(i - slowMA, i)
      .reduce((sum, d) => sum + d.close, 0) / slowMA;

    // Crossover signal
    if (prevFastAvg <= prevSlowAvg && fastAvg > slowAvg) {
      signals.push({ index: i, type: 'buy' as const, price: data[i].close, time: data[i].time });
    } else if (prevFastAvg >= prevSlowAvg && fastAvg < slowAvg) {
      signals.push({ index: i, type: 'sell' as const, price: data[i].close, time: data[i].time });
    }
  }

  return signals;
}
