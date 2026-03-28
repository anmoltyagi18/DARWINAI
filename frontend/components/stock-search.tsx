'use client';

import { useState, useMemo } from 'react';
import { Search, Star } from 'lucide-react';

const POPULAR_STOCKS = [
  { symbol: 'AAPL', name: 'Apple Inc.' },
  { symbol: 'MSFT', name: 'Microsoft Corporation' },
  { symbol: 'GOOGL', name: 'Alphabet Inc.' },
  { symbol: 'AMZN', name: 'Amazon.com Inc.' },
  { symbol: 'TSLA', name: 'Tesla Inc.' },
  { symbol: 'META', name: 'Meta Platforms Inc.' },
  { symbol: 'NVDA', name: 'NVIDIA Corporation' },
  { symbol: 'JPM', name: 'JPMorgan Chase & Co.' },
];

interface StockSearchProps {
  onSelectStock: (symbol: string) => void;
  selectedStock?: string;
}

export function StockSearch({ onSelectStock, selectedStock }: StockSearchProps) {
  const [query, setQuery] = useState('');
  const [favorites, setFavorites] = useState<string[]>(['AAPL']);
  const [showDropdown, setShowDropdown] = useState(false);

  const filteredStocks = useMemo(() => {
    if (!query.trim()) return POPULAR_STOCKS;
    return POPULAR_STOCKS.filter(
      (stock) =>
        stock.symbol.toUpperCase().includes(query.toUpperCase()) ||
        stock.name.toUpperCase().includes(query.toUpperCase())
    );
  }, [query]);

  const toggleFavorite = (symbol: string) => {
    setFavorites((prev) =>
      prev.includes(symbol) ? prev.filter((s) => s !== symbol) : [...prev, symbol]
    );
  };

  const handleSelectStock = (symbol: string) => {
    onSelectStock(symbol);
    setShowDropdown(false);
    setQuery('');
  };

  return (
    <div className="space-y-4">
      <div className="relative">
        <div className="relative">
          <Search className="absolute left-3 top-3 h-5 w-5 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search stocks..."
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setShowDropdown(true);
            }}
            onFocus={() => setShowDropdown(true)}
            className="w-full pl-10 pr-4 py-2 bg-input border border-border rounded-lg text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary"
          />
        </div>

        {showDropdown && (
          <div className="absolute top-full mt-2 w-full bg-card border border-border rounded-lg shadow-lg z-50 max-h-64 overflow-y-auto">
            {filteredStocks.length > 0 ? (
              filteredStocks.map((stock) => (
                <button
                  key={stock.symbol}
                  onClick={() => handleSelectStock(stock.symbol)}
                  className="w-full px-4 py-3 flex items-center justify-between hover:bg-secondary transition-colors border-b border-border last:border-b-0"
                >
                  <div className="text-left">
                    <div className="font-semibold text-foreground">{stock.symbol}</div>
                    <div className="text-sm text-muted-foreground">{stock.name}</div>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleFavorite(stock.symbol);
                    }}
                    className={`p-1 ${
                      favorites.includes(stock.symbol)
                        ? 'text-primary'
                        : 'text-muted-foreground hover:text-primary'
                    }`}
                  >
                    <Star className="h-4 w-4 fill-current" />
                  </button>
                </button>
              ))
            ) : (
              <div className="px-4 py-8 text-center text-muted-foreground">No stocks found</div>
            )}
          </div>
        )}
      </div>

      {favorites.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
            Favorites
          </div>
          <div className="flex flex-wrap gap-2">
            {favorites.map((symbol) => (
              <button
                key={symbol}
                onClick={() => handleSelectStock(symbol)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  selectedStock === symbol
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-secondary text-foreground hover:bg-input'
                }`}
              >
                {symbol}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
