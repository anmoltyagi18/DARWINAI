'use client';

import { useState, useEffect } from 'react';
import { Zap, Loader } from 'lucide-react';

interface AIExplanationPanelProps {
  symbol?: string;
  currentPrice?: number;
  priceChange?: number;
  volume?: number;
}

export function AIExplanationPanel({
  symbol = 'AAPL',
  currentPrice = 195.45,
  priceChange = 2.5,
  volume = 45000000,
}: AIExplanationPanelProps) {
  const [explanation, setExplanation] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    generateExplanation();
  }, [symbol, currentPrice, priceChange]);

  const generateExplanation = async () => {
    setLoading(true);
    setError('');
    setExplanation('');

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol,
          currentPrice,
          priceChange,
          volume,
        }),
      });

      if (!response.ok) throw new Error('Failed to generate analysis');

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');

      let result = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = new TextDecoder().decode(value);
        result += text;
        setExplanation(result);
      }
    } catch (err) {
      setError('Unable to generate AI analysis. Please try again.');
      console.error('Error generating analysis:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-card border border-border rounded-lg p-6 h-full flex flex-col">
      <div className="flex items-center gap-2 mb-4">
        <Zap className="h-5 w-5 text-primary" />
        <h2 className="text-lg font-semibold text-foreground">AI Analysis</h2>
      </div>

      <div className="flex-1 overflow-y-auto space-y-3">
        {loading && !explanation && (
          <div className="flex items-center gap-2 text-muted-foreground">
            <Loader className="h-4 w-4 animate-spin" />
            <span>Analyzing market data...</span>
          </div>
        )}

        {error && (
          <div className="text-sm text-red-400 bg-red-900/20 border border-red-900/50 rounded p-3">
            {error}
          </div>
        )}

        {explanation && (
          <div className="text-sm text-muted-foreground leading-relaxed space-y-2">
            {explanation.split('\n').map((paragraph, idx) => (
              <p key={idx}>{paragraph}</p>
            ))}
          </div>
        )}

        {!loading && !explanation && !error && (
          <div className="text-sm text-muted-foreground">
            Click "Analyze" to get AI-powered market insights for {symbol}
          </div>
        )}
      </div>

      <button
        onClick={generateExplanation}
        disabled={loading}
        className="mt-4 w-full bg-primary text-primary-foreground py-2 px-4 rounded-lg font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {loading ? 'Analyzing...' : 'Analyze'}
      </button>
    </div>
  );
}
