export async function POST(req: Request) {
  try {
    const { symbol, currentPrice, priceChange, volume } = await req.json();

    // Mock AI response with streaming
    const mockAnalysis = `${symbol} is currently trading at $${currentPrice.toFixed(2)}, ${
      priceChange >= 0 ? 'up' : 'down'
    } ${Math.abs(priceChange).toFixed(2)}% today.

The trading volume is at ${(volume / 1000000).toFixed(1)}M shares, indicating ${
      volume > 50000000 ? 'strong' : 'moderate'
    } market activity.

Technical Analysis:
- The price is ${priceChange >= 0 ? 'in an uptrend' : 'in a downtrend'} based on recent price action
- Support levels are forming around key moving averages
- RSI indicators suggest ${priceChange >= 0 ? 'cautious' : 'potential'} entry points

Market Sentiment:
${
  priceChange >= 0
    ? 'Positive momentum suggests continued bullish pressure. Look for pullbacks to buy.'
    : 'Bearish sentiment is present. Traders may wait for stabilization before entering.'
}

Recommendation: Monitor key support/resistance levels before making trading decisions. Consider using stop-loss orders to manage risk.`;

    // Simulate streaming by breaking the response into chunks
    const chunks = mockAnalysis.split('\n');

    const stream = new ReadableStream({
      async start(controller) {
        for (const chunk of chunks) {
          controller.enqueue(`${chunk}\n`);
          await new Promise((resolve) => setTimeout(resolve, 50)); // Simulate streaming delay
        }
        controller.close();
      },
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
      },
    });
  } catch (error) {
    console.error('Analysis error:', error);
    return new Response('Error generating analysis', { status: 500 });
  }
}
