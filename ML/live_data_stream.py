"""
live_data_stream.py
===================
Real-Time Stock Data Stream via Finnhub WebSocket API

Features
--------
- WebSocket connection to Finnhub's live trade feed
- Subscribe / unsubscribe to stock symbols dynamically
- In-memory price store with configurable history depth
- Candlestick aggregation (1-min OHLCV bars) from raw ticks
- Heartbeat / auto-reconnect on connection drops
- Thread-safe access for downstream AI engine consumption
- Event callbacks: on_tick, on_bar, on_error, on_connect, on_disconnect
- REST fallback: fetch latest quote when WebSocket is unavailable

Quick start
-----------
    from live_data_stream import LiveDataStream

    stream = LiveDataStream(api_key="your_finnhub_key")
    stream.subscribe(["AAPL", "MSFT", "TSLA"])
    stream.start()

    # Blocking helper — print ticks for 30 seconds
    import time
    time.sleep(30)

    snapshot = stream.snapshot("AAPL")
    bars     = stream.bars("AAPL", n=10)
    feed     = stream.ai_feed(["AAPL", "MSFT"])

    stream.stop()

Environment variable alternative
---------------------------------
    export FINNHUB_API_KEY=your_key
    python live_data_stream.py
"""

from __future__ import annotations

import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Set

# ── optional deps (graceful degradation if not installed) ───────────────────
try:
    import websocket          # pip install websocket-client
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False

try:
    import requests           # pip install requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("live_data_stream")


# ════════════════════════════════════════════════════════════════════════════
# Data Structures
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Tick:
    """Single trade tick received from Finnhub."""
    symbol:    str
    price:     float
    volume:    float
    timestamp: float          # Unix ms → seconds internally
    source:    str = "ws"     # "ws" | "rest" | "mock"

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "datetime": self.dt.isoformat(),
        }


@dataclass
class OHLCBar:
    """1-minute OHLCV candlestick bar."""
    symbol:    str
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float
    timestamp: float          # bar open time (Unix seconds)
    tick_count: int = 0

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)

    @property
    def change_pct(self) -> float:
        return (self.close - self.open) / self.open if self.open else 0.0

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "datetime":   self.dt.isoformat(),
            "change_pct": round(self.change_pct, 6),
        }


@dataclass
class SymbolSnapshot:
    """Latest rolled-up state for one symbol — consumed by the AI engine."""
    symbol:          str
    last_price:      float
    bid:             float        = 0.0
    ask:             float        = 0.0
    volume_today:    float        = 0.0
    tick_count:      int          = 0
    last_tick_ts:    float        = 0.0
    price_change_1m: float        = 0.0   # % change over last minute
    price_change_5m: float        = 0.0
    vwap_1m:         float        = 0.0   # volume-weighted avg price (1 min)
    high_1m:         float        = 0.0
    low_1m:          float        = 0.0
    status:          str          = "active"  # "active" | "stale" | "halted"

    @property
    def spread(self) -> float:
        return self.ask - self.bid if self.ask and self.bid else 0.0

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.last_tick_ts, tz=timezone.utc)

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "spread":   round(self.spread, 4),
            "datetime": self.dt.isoformat(),
        }


# ════════════════════════════════════════════════════════════════════════════
# In-Memory Price Store
# ════════════════════════════════════════════════════════════════════════════

class PriceStore:
    """
    Thread-safe in-memory store for ticks, OHLC bars, and snapshots.

    Parameters
    ----------
    max_ticks_per_symbol : rolling window of raw ticks kept in memory
    max_bars_per_symbol  : rolling window of 1-min OHLC bars
    bar_seconds          : bar aggregation period in seconds (default 60)
    stale_threshold      : seconds without a tick before status = "stale"
    """

    def __init__(
        self,
        max_ticks_per_symbol: int = 500,
        max_bars_per_symbol:  int = 200,
        bar_seconds:          int = 60,
        stale_threshold:      int = 30,
    ):
        self._lock              = threading.RLock()
        self.max_ticks          = max_ticks_per_symbol
        self.max_bars           = max_bars_per_symbol
        self.bar_seconds        = bar_seconds
        self.stale_threshold    = stale_threshold

        # symbol → deque of Tick
        self._ticks: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_ticks_per_symbol)
        )
        # symbol → deque of OHLCBar
        self._bars:  Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_bars_per_symbol)
        )
        # symbol → current open bar being filled
        self._open_bar: Dict[str, Optional[OHLCBar]] = {}
        # symbol → SymbolSnapshot
        self._snapshots: Dict[str, SymbolSnapshot] = {}

    # ── write path ───────────────────────────────────────────────────────────

    def ingest(self, tick: Tick) -> Optional[OHLCBar]:
        """
        Ingest a tick, update snapshot, aggregate into OHLC bar.
        Returns the completed OHLCBar if a bar just closed, else None.
        """
        with self._lock:
            sym = tick.symbol
            self._ticks[sym].append(tick)
            self._update_snapshot(sym, tick)
            return self._aggregate_bar(sym, tick)

    def _update_snapshot(self, sym: str, tick: Tick) -> None:
        snap = self._snapshots.get(sym)
        now  = tick.timestamp

        # compute 1m and 5m price changes
        change_1m = change_5m = 0.0
        ticks_list = list(self._ticks[sym])
        if len(ticks_list) > 1:
            cutoff_1m = now - 60
            cutoff_5m = now - 300
            older_1m  = [t for t in ticks_list if t.timestamp <= cutoff_1m]
            older_5m  = [t for t in ticks_list if t.timestamp <= cutoff_5m]
            if older_1m:
                change_1m = (tick.price - older_1m[-1].price) / older_1m[-1].price
            if older_5m:
                change_5m = (tick.price - older_5m[-1].price) / older_5m[-1].price

        # compute 1m VWAP, high, low
        recent = [t for t in ticks_list if t.timestamp >= now - 60]
        if recent:
            total_vol  = sum(t.volume for t in recent)
            vwap = (
                sum(t.price * t.volume for t in recent) / total_vol
                if total_vol > 0 else tick.price
            )
            high_1m = max(t.price for t in recent)
            low_1m  = min(t.price for t in recent)
        else:
            vwap = tick.price
            high_1m = low_1m = tick.price

        vol_today = (snap.volume_today if snap else 0) + tick.volume

        self._snapshots[sym] = SymbolSnapshot(
            symbol          = sym,
            last_price      = tick.price,
            volume_today    = vol_today,
            tick_count      = (snap.tick_count if snap else 0) + 1,
            last_tick_ts    = now,
            price_change_1m = round(change_1m, 6),
            price_change_5m = round(change_5m, 6),
            vwap_1m         = round(vwap, 4),
            high_1m         = high_1m,
            low_1m          = low_1m,
            status          = "active",
        )

    def _aggregate_bar(self, sym: str, tick: Tick) -> Optional[OHLCBar]:
        """Aggregate tick into the current open bar; return bar if closed."""
        bar_start = (tick.timestamp // self.bar_seconds) * self.bar_seconds
        current   = self._open_bar.get(sym)
        closed    = None

        if current is None:
            # start first bar
            self._open_bar[sym] = OHLCBar(
                symbol    = sym,
                open      = tick.price,
                high      = tick.price,
                low       = tick.price,
                close     = tick.price,
                volume    = tick.volume,
                timestamp = bar_start,
                tick_count= 1,
            )
        elif bar_start > current.timestamp:
            # close old bar, start new one
            self._bars[sym].append(current)
            closed = current
            self._open_bar[sym] = OHLCBar(
                symbol    = sym,
                open      = tick.price,
                high      = tick.price,
                low       = tick.price,
                close     = tick.price,
                volume    = tick.volume,
                timestamp = bar_start,
                tick_count= 1,
            )
        else:
            # update existing bar
            current.high       = max(current.high, tick.price)
            current.low        = min(current.low,  tick.price)
            current.close      = tick.price
            current.volume    += tick.volume
            current.tick_count += 1

        return closed

    def mark_stale(self) -> List[str]:
        """Mark symbols with no recent ticks as stale. Returns list of stale symbols."""
        stale = []
        now   = time.time()
        with self._lock:
            for sym, snap in self._snapshots.items():
                if now - snap.last_tick_ts > self.stale_threshold:
                    snap.status = "stale"
                    stale.append(sym)
        return stale

    # ── read path ────────────────────────────────────────────────────────────

    def snapshot(self, symbol: str) -> Optional[SymbolSnapshot]:
        with self._lock:
            return self._snapshots.get(symbol.upper())

    def snapshots(self, symbols: Optional[List[str]] = None) -> Dict[str, SymbolSnapshot]:
        with self._lock:
            if symbols:
                return {s.upper(): self._snapshots[s.upper()]
                        for s in symbols if s.upper() in self._snapshots}
            return dict(self._snapshots)

    def ticks(self, symbol: str, n: int = 50) -> List[Tick]:
        with self._lock:
            t = list(self._ticks.get(symbol.upper(), []))
            return t[-n:]

    def bars(self, symbol: str, n: int = 60) -> List[OHLCBar]:
        with self._lock:
            b = list(self._bars.get(symbol.upper(), []))
            open_b = self._open_bar.get(symbol.upper())
            if open_b:
                b = b + [open_b]
            return b[-n:]

    def prices(self, symbol: str, n: int = 100) -> List[float]:
        """Return the last n close prices (from bars) for technical analysis."""
        bars = self.bars(symbol, n)
        return [b.close for b in bars]

    def all_symbols(self) -> List[str]:
        with self._lock:
            return list(self._snapshots.keys())

    def stats(self) -> dict:
        with self._lock:
            return {
                sym: {
                    "ticks": len(self._ticks[sym]),
                    "bars":  len(self._bars[sym]),
                    "status": self._snapshots[sym].status
                        if sym in self._snapshots else "unknown",
                }
                for sym in set(list(self._ticks.keys()) + list(self._snapshots.keys()))
            }


# ════════════════════════════════════════════════════════════════════════════
# Finnhub REST Quote (fallback)
# ════════════════════════════════════════════════════════════════════════════

class FinnhubREST:
    """Thin wrapper around Finnhub's quote endpoint for REST fallback."""

    BASE = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def quote(self, symbol: str) -> Optional[Tick]:
        if not REQUESTS_AVAILABLE:
            log.warning("requests not installed; REST fallback unavailable.")
            return None
        try:
            url = f"{self.BASE}/quote"
            r   = requests.get(url, params={"symbol": symbol, "token": self.api_key}, timeout=5)
            r.raise_for_status()
            data = r.json()
            if data.get("c", 0) == 0:
                return None
            return Tick(
                symbol    = symbol.upper(),
                price     = float(data["c"]),
                volume    = 0.0,
                timestamp = float(data.get("t", time.time())),
                source    = "rest",
            )
        except Exception as exc:
            log.error(f"REST quote failed for {symbol}: {exc}")
            return None

    def bulk_quote(self, symbols: List[str]) -> Dict[str, Tick]:
        results = {}
        for sym in symbols:
            tick = self.quote(sym)
            if tick:
                results[sym] = tick
        return results


# ════════════════════════════════════════════════════════════════════════════
# Mock Streamer (demo / testing without a real API key)
# ════════════════════════════════════════════════════════════════════════════

class MockStreamer(threading.Thread):
    """
    Generates synthetic price ticks using GBM for testing without
    a real Finnhub API key.
    """

    BASE_PRICES = {
        "AAPL": 189.50, "MSFT": 415.20, "GOOGL": 175.80, "AMZN": 185.40,
        "TSLA": 178.20, "NVDA": 875.40, "META": 505.60, "SPY":  452.80,
        "QQQ":  390.10, "NFLX": 650.30,
    }

    def __init__(
        self,
        symbols:       List[str],
        store:         PriceStore,
        tick_interval: float = 0.5,
        on_tick:       Optional[Callable] = None,
        on_bar:        Optional[Callable] = None,
    ):
        super().__init__(daemon=True, name="MockStreamer")
        self.symbols       = [s.upper() for s in symbols]
        self.store         = store
        self.tick_interval = tick_interval
        self.on_tick       = on_tick
        self.on_bar        = on_bar
        self._stop_event   = threading.Event()
        # initialise GBM state
        self._prices = {
            s: self.BASE_PRICES.get(s, 100.0 + hash(s) % 200)
            for s in self.symbols
        }
        self._vols = {s: 0.015 + (hash(s) % 10) / 1000 for s in self.symbols}

    def run(self) -> None:
        import random
        log.info(f"MockStreamer started for {self.symbols}")
        while not self._stop_event.is_set():
            for sym in self.symbols:
                # GBM step
                drift   = 0.0001
                vol     = self._vols[sym]
                ret     = random.gauss(drift, vol)
                self._prices[sym] *= (1 + ret)
                price  = round(self._prices[sym], 4)
                volume = abs(random.gauss(500, 300))

                tick = Tick(
                    symbol    = sym,
                    price     = price,
                    volume    = volume,
                    timestamp = time.time(),
                    source    = "mock",
                )
                closed_bar = self.store.ingest(tick)

                if self.on_tick:
                    try:
                        self.on_tick(tick)
                    except Exception:
                        pass

                if closed_bar and self.on_bar:
                    try:
                        self.on_bar(closed_bar)
                    except Exception:
                        pass

            self._stop_event.wait(self.tick_interval)
        log.info("MockStreamer stopped.")

    def stop(self) -> None:
        self._stop_event.set()


# ════════════════════════════════════════════════════════════════════════════
# Finnhub WebSocket Streamer
# ════════════════════════════════════════════════════════════════════════════

class FinnhubStreamer:
    """
    Manages the Finnhub WebSocket connection and routes trade messages
    into PriceStore.

    Finnhub WebSocket docs: https://finnhub.io/docs/api/websocket-trades

    Message format received
    -----------------------
    {
      "type": "trade",
      "data": [
        {"s": "AAPL", "p": 150.0, "v": 100, "t": 1234567890000, "c": [...]}
      ]
    }
    """

    WS_URL = "wss://ws.finnhub.io"

    def __init__(
        self,
        api_key:         str,
        symbols:         List[str],
        store:           PriceStore,
        on_tick:         Optional[Callable[[Tick], None]] = None,
        on_bar:          Optional[Callable[[OHLCBar], None]] = None,
        on_error:        Optional[Callable[[Exception], None]] = None,
        on_connect:      Optional[Callable[[], None]] = None,
        on_disconnect:   Optional[Callable[[], None]] = None,
        reconnect_delay: int   = 5,
        max_reconnects:  int   = 20,
        ping_interval:   int   = 20,
    ):
        if not WS_AVAILABLE:
            raise RuntimeError(
                "websocket-client is not installed.\n"
                "Run:  pip install websocket-client"
            )
        self.api_key         = api_key
        self.symbols         = [s.upper() for s in symbols]
        self.store           = store
        self.on_tick         = on_tick
        self.on_bar          = on_bar
        self.on_error        = on_error
        self.on_connect      = on_connect
        self.on_disconnect   = on_disconnect
        self.reconnect_delay = reconnect_delay
        self.max_reconnects  = max_reconnects
        self.ping_interval   = ping_interval

        self._ws:            Optional[websocket.WebSocketApp] = None
        self._ws_thread:     Optional[threading.Thread] = None
        self._stop_event     = threading.Event()
        self._connected      = threading.Event()
        self._reconnect_count = 0
        self._subscribed:    Set[str] = set()

    # ── public API ───────────────────────────────────────────────────────────

    def start(self) -> None:
        self._stop_event.clear()
        self._reconnect_count = 0
        self._connect()

    def stop(self) -> None:
        self._stop_event.set()
        if self._ws:
            self._ws.close()
        log.info("FinnhubStreamer stopped.")

    def subscribe(self, symbol: str) -> None:
        sym = symbol.upper()
        if sym not in self._subscribed and self._connected.is_set() and self._ws:
            self._ws.send(json.dumps({"type": "subscribe", "symbol": sym}))
            self._subscribed.add(sym)
            log.info(f"  ↳ Subscribed: {sym}")

    def unsubscribe(self, symbol: str) -> None:
        sym = symbol.upper()
        if sym in self._subscribed and self._ws:
            self._ws.send(json.dumps({"type": "unsubscribe", "symbol": sym}))
            self._subscribed.discard(sym)
            log.info(f"  ↳ Unsubscribed: {sym}")

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    # ── internal WebSocket handlers ───────────────────────────────────────────

    def _connect(self) -> None:
        url = f"{self.WS_URL}?token={self.api_key}"
        self._ws = websocket.WebSocketApp(
            url,
            on_open    = self._on_open,
            on_message = self._on_message,
            on_error   = self._on_error,
            on_close   = self._on_close,
        )
        self._ws_thread = threading.Thread(
            target = self._ws.run_forever,
            kwargs = {"ping_interval": self.ping_interval, "ping_timeout": 10},
            daemon = True,
            name   = "FinnhubWS",
        )
        self._ws_thread.start()

    def _on_open(self, ws) -> None:
        self._connected.set()
        self._reconnect_count = 0
        log.info("WebSocket connected to Finnhub.")
        if self.on_connect:
            self.on_connect()
        # subscribe to all requested symbols
        for sym in self.symbols:
            self.subscribe(sym)

    def _on_message(self, ws, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = msg.get("type")

        if msg_type == "trade":
            for trade in msg.get("data", []):
                self._handle_trade(trade)

        elif msg_type == "ping":
            ws.send(json.dumps({"type": "pong"}))

        elif msg_type == "error":
            log.error(f"Finnhub error: {msg.get('msg')}")

    def _handle_trade(self, trade: dict) -> None:
        try:
            sym  = trade["s"].upper()
            ts   = trade["t"] / 1000.0   # ms → seconds
            tick = Tick(
                symbol    = sym,
                price     = float(trade["p"]),
                volume    = float(trade.get("v", 0)),
                timestamp = ts,
                source    = "ws",
            )
            closed_bar = self.store.ingest(tick)

            if self.on_tick:
                self.on_tick(tick)
            if closed_bar and self.on_bar:
                self.on_bar(closed_bar)

        except (KeyError, ValueError, TypeError) as exc:
            log.debug(f"Malformed trade message: {exc}  raw={trade}")

    def _on_error(self, ws, error: Exception) -> None:
        log.warning(f"WebSocket error: {error}")
        self._connected.clear()
        if self.on_error:
            self.on_error(error)

    def _on_close(self, ws, code, reason) -> None:
        self._connected.clear()
        self._subscribed.clear()
        log.warning(f"WebSocket closed (code={code}, reason={reason})")
        if self.on_disconnect:
            self.on_disconnect()
        if not self._stop_event.is_set():
            self._attempt_reconnect()

    def _attempt_reconnect(self) -> None:
        if self._reconnect_count >= self.max_reconnects:
            log.error("Max reconnection attempts reached. Giving up.")
            return
        self._reconnect_count += 1
        delay = min(self.reconnect_delay * self._reconnect_count, 60)
        log.info(f"Reconnecting in {delay}s (attempt {self._reconnect_count}/{self.max_reconnects})…")
        time.sleep(delay)
        if not self._stop_event.is_set():
            self._connect()


# ════════════════════════════════════════════════════════════════════════════
# Stale-Check Background Worker
# ════════════════════════════════════════════════════════════════════════════

class StaleChecker(threading.Thread):
    """Periodically marks symbols with no recent ticks as stale."""

    def __init__(self, store: PriceStore, interval: int = 10):
        super().__init__(daemon=True, name="StaleChecker")
        self.store    = store
        self.interval = interval
        self._stop    = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(self.interval):
            stale = self.store.mark_stale()
            if stale:
                log.debug(f"Stale symbols: {stale}")

    def stop(self) -> None:
        self._stop.set()


# ════════════════════════════════════════════════════════════════════════════
# AI Decision Pusher Background Worker
# ════════════════════════════════════════════════════════════════════════════

class AIPusher(threading.Thread):
    """Periodically pushes market feed to an external AI Decision Engine."""

    def __init__(self, stream: 'LiveDataStream', ai_engine: Any, interval: int = 5):
        super().__init__(daemon=True, name="AIPusher")
        self.stream    = stream
        self.ai_engine = ai_engine
        self.interval  = interval
        self._stop     = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(self.interval):
            feed = self.stream.ai_feed()
            if not feed:
                continue
            try:
                # Assuming ai_engine has an evaluate method (like AIBrain)
                decision = self.ai_engine.evaluate(feed)
                log.info(f"[AI Decision Update] Pushed structured frame. Returns: {decision}")
            except Exception as e:
                log.error(f"Failed to push update to AI engine: {e}")

    def stop(self) -> None:
        self._stop.set()


# ════════════════════════════════════════════════════════════════════════════
# Main LiveDataStream Façade
# ════════════════════════════════════════════════════════════════════════════

class LiveDataStream:
    """
    High-level façade for the real-time stock data pipeline.

    Automatically selects the best available backend:
      1. Finnhub WebSocket  (api_key + websocket-client installed)
      2. Finnhub REST poll  (api_key + requests installed, no WebSocket)
      3. Mock streamer      (no api_key or demo=True)

    Parameters
    ----------
    api_key          : Finnhub API key (or set FINNHUB_API_KEY env var)
    symbols          : initial list of symbols to subscribe to
    demo             : force mock data even if api_key is provided
    max_ticks        : rolling tick window per symbol
    max_bars         : rolling OHLC bar window per symbol
    bar_seconds      : bar aggregation period (default 60 = 1-min bars)
    stale_threshold  : seconds before a symbol is marked stale
    on_tick          : callback(Tick) fired on every incoming tick
    on_bar           : callback(OHLCBar) fired when a bar closes
    on_error         : callback(Exception) fired on connection errors
    on_connect       : callback() fired on successful (re)connection
    on_disconnect    : callback() fired on disconnection
    """

    def __init__(
        self,
        api_key:         Optional[str] = None,
        symbols:         Optional[List[str]] = None,
        demo:            bool = False,
        max_ticks:       int  = 500,
        max_bars:        int  = 200,
        bar_seconds:     int  = 60,
        stale_threshold: int  = 30,
        on_tick:         Optional[Callable[[Tick], None]] = None,
        on_bar:          Optional[Callable[[OHLCBar], None]] = None,
        on_error:        Optional[Callable[[Exception], None]] = None,
        on_connect:      Optional[Callable[[], None]] = None,
        on_disconnect:   Optional[Callable[[], None]] = None,
        ai_engine:       Optional[Any] = None,
        decision_interval: int = 5,
    ):
        self._api_key   = api_key or os.getenv("FINNHUB_API_KEY", "")
        self._symbols   = [s.upper() for s in (symbols or [])]
        self._demo      = demo or not self._api_key
        
        self.ai_engine  = ai_engine
        self.decision_interval = decision_interval

        # callbacks
        self.on_tick       = on_tick
        self.on_bar        = on_bar
        self.on_error      = on_error
        self.on_connect    = on_connect
        self.on_disconnect = on_disconnect

        # core store
        self.store = PriceStore(
            max_ticks_per_symbol = max_ticks,
            max_bars_per_symbol  = max_bars,
            bar_seconds          = bar_seconds,
            stale_threshold      = stale_threshold,
        )

        # workers (initialised in start())
        self._streamer:      Optional[FinnhubStreamer | MockStreamer] = None
        self._rest:          Optional[FinnhubREST] = None
        self._stale_checker: Optional[StaleChecker] = None
        self._ai_pusher:     Optional[AIPusher]     = None
        self._running        = False

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> "LiveDataStream":
        """Start streaming. Non-blocking — returns self for chaining."""
        if self._running:
            log.warning("Stream already running.")
            return self

        if self._demo or not WS_AVAILABLE:
            mode = "MOCK" if self._demo else "MOCK (websocket-client not installed)"
            log.info(f"[LiveDataStream] Starting in {mode} mode for {self._symbols}")
            self._streamer = MockStreamer(
                symbols       = self._symbols,
                store         = self.store,
                tick_interval = 0.4,
                on_tick       = self._dispatch_tick,
                on_bar        = self._dispatch_bar,
            )
            self._streamer.start()
        else:
            log.info(f"[LiveDataStream] Starting Finnhub WebSocket for {self._symbols}")
            self._rest = FinnhubREST(self._api_key)
            self._streamer = FinnhubStreamer(
                api_key       = self._api_key,
                symbols       = self._symbols,
                store         = self.store,
                on_tick       = self._dispatch_tick,
                on_bar        = self._dispatch_bar,
                on_error      = self.on_error,
                on_connect    = self.on_connect,
                on_disconnect = self.on_disconnect,
            )
            self._streamer.start()

        self._stale_checker = StaleChecker(self.store)
        self._stale_checker.start()
        
        if self.ai_engine:
            self._ai_pusher = AIPusher(self, self.ai_engine, interval=self.decision_interval)
            self._ai_pusher.start()
            log.info(f"[LiveDataStream] AIPusher activated every {self.decision_interval}s")
            
        self._running = True
        return self

    def stop(self) -> None:
        """Gracefully shut down all background threads."""
        if not self._running:
            return
        log.info("[LiveDataStream] Stopping …")
        if self._streamer:
            self._streamer.stop()
        if self._stale_checker:
            self._stale_checker.stop()
        if self._ai_pusher:
            self._ai_pusher.stop()
        self._running = False
        log.info("[LiveDataStream] Stopped.")

    def __enter__(self) -> "LiveDataStream":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ── subscription management ───────────────────────────────────────────────

    def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to additional symbols after start()."""
        for sym in symbols:
            sym = sym.upper()
            if sym not in self._symbols:
                self._symbols.append(sym)
            if isinstance(self._streamer, FinnhubStreamer):
                self._streamer.subscribe(sym)
            elif isinstance(self._streamer, MockStreamer):
                if sym not in self._streamer.symbols:
                    self._streamer.symbols.append(sym)
                    self._streamer._prices[sym] = MockStreamer.BASE_PRICES.get(sym, 100.0)
                    self._streamer._vols[sym]   = 0.015

    def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols."""
        for sym in symbols:
            sym = sym.upper()
            if sym in self._symbols:
                self._symbols.remove(sym)
            if isinstance(self._streamer, FinnhubStreamer):
                self._streamer.unsubscribe(sym)

    # ── read API (consumed by AI engine) ─────────────────────────────────────

    def snapshot(self, symbol: str) -> Optional[SymbolSnapshot]:
        """Latest rolled-up state for a single symbol."""
        return self.store.snapshot(symbol)

    def snapshots(self, symbols: Optional[List[str]] = None) -> Dict[str, SymbolSnapshot]:
        """Latest state for multiple (or all) symbols."""
        return self.store.snapshots(symbols)

    def ticks(self, symbol: str, n: int = 50) -> List[Tick]:
        """Last n raw ticks for a symbol."""
        return self.store.ticks(symbol, n)

    def bars(self, symbol: str, n: int = 60) -> List[OHLCBar]:
        """Last n closed OHLC bars (+ current open bar) for a symbol."""
        return self.store.bars(symbol, n)

    def prices(self, symbol: str, n: int = 100) -> List[float]:
        """Last n close prices — ready for the technical indicator engine."""
        return self.store.prices(symbol, n)

    def ai_feed(self, symbols: Optional[List[str]] = None) -> Dict[str, dict]:
        """
        Produce a structured dict consumed directly by the AI trading engine.

        Returns
        -------
        {
          "AAPL": {
            "snapshot": {...},
            "prices":   [150.1, 150.3, ...],   # last 100 closes
            "last_bar": {...} | None,
          },
          ...
        }
        """
        syms   = [s.upper() for s in symbols] if symbols else self._symbols
        result = {}
        for sym in syms:
            snap = self.snapshot(sym)
            bars = self.bars(sym, n=100)
            result[sym] = {
                "snapshot": snap.to_dict() if snap else None,
                "prices":   [b.close for b in bars],
                "last_bar": bars[-1].to_dict() if bars else None,
            }
        return result

    def wait_for_data(
        self,
        symbols: Optional[List[str]] = None,
        timeout: float = 10.0,
    ) -> bool:
        """
        Block until at least one tick has arrived for all requested symbols,
        or until timeout. Returns True if all symbols have data.
        """
        syms    = [s.upper() for s in symbols] if symbols else self._symbols
        deadline = time.time() + timeout
        while time.time() < deadline:
            if all(self.snapshot(s) is not None for s in syms):
                return True
            time.sleep(0.1)
        return False

    @property
    def connected(self) -> bool:
        if isinstance(self._streamer, FinnhubStreamer):
            return self._streamer.connected
        return self._running   # mock is always "connected"

    @property
    def running(self) -> bool:
        return self._running

    # ── REST snapshot (one-shot, no WebSocket) ────────────────────────────────

    def fetch_quotes(self, symbols: Optional[List[str]] = None) -> Dict[str, Tick]:
        """
        One-shot REST quote fetch (useful for bootstrapping or when WebSocket
        is not available).
        """
        if not self._api_key:
            log.warning("No API key — cannot fetch REST quotes.")
            return {}
        if not REQUESTS_AVAILABLE:
            log.warning("requests not installed — cannot fetch REST quotes.")
            return {}
        rest = self._rest or FinnhubREST(self._api_key)
        syms = [s.upper() for s in symbols] if symbols else self._symbols
        ticks = rest.bulk_quote(syms)
        for tick in ticks.values():
            self.store.ingest(tick)
        return ticks

    # ── internal dispatchers ──────────────────────────────────────────────────

    def _dispatch_tick(self, tick: Tick) -> None:
        if self.on_tick:
            try:
                self.on_tick(tick)
            except Exception as exc:
                log.error(f"on_tick callback error: {exc}")

    def _dispatch_bar(self, bar: OHLCBar) -> None:
        if self.on_bar:
            try:
                self.on_bar(bar)
            except Exception as exc:
                log.error(f"on_bar callback error: {exc}")

    # ── diagnostics ───────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "running":    self._running,
            "connected":  self.connected,
            "mode":       "mock" if self._demo else "live",
            "symbols":    self._symbols,
            "store_stats":self.store.stats(),
        }

    def print_status(self) -> None:
        st = self.status()
        bar = "─" * 60
        print(f"\n{'═' * 60}")
        print(f"  LiveDataStream  [{st['mode'].upper()}]  "
              f"{'🟢 connected' if st['connected'] else '🔴 disconnected'}")
        print(f"{'═' * 60}")
        for sym, info in st["store_stats"].items():
            snap = self.snapshot(sym)
            price = f"${snap.last_price:>10,.4f}" if snap else "         —"
            chg   = f"{snap.price_change_1m:+.2%}" if snap else "     —"
            status_icon = "🟢" if (snap and snap.status == "active") else "🟡"
            print(
                f"  {status_icon} {sym:<8}  {price}  ({chg} 1m)  "
                f"ticks={info['ticks']:>4}  bars={info['bars']:>3}"
            )
        print(bar)


# ════════════════════════════════════════════════════════════════════════════
# CLI Demo
# ════════════════════════════════════════════════════════════════════════════

def _run_demo(api_key: str = "", duration: int = 20) -> None:
    """
    Interactive demo:
    - starts stream (mock if no api_key)
    - prints live ticks for `duration` seconds
    - shows final AI feed snapshot
    """

    SYMBOLS = ["AAPL", "MSFT", "TSLA", "NVDA", "SPY"]
    tick_count = [0]

    # ── callbacks ────────────────────────────────────────────────────────────

    def on_tick(tick: Tick):
        tick_count[0] += 1
        if tick_count[0] % (len(SYMBOLS) * 3) == 0:   # throttle console output
            ts = datetime.fromtimestamp(tick.timestamp, tz=timezone.utc).strftime("%H:%M:%S")
            print(
                f"  [{ts}]  {tick.symbol:<6}  ${tick.price:>10,.4f}"
                f"  vol={tick.volume:>7.0f}  [{tick.source}]"
            )

    def on_bar(bar: OHLCBar):
        arrow = "▲" if bar.change_pct >= 0 else "▼"
        print(
            f"\n  ── BAR CLOSED ──  {bar.symbol}  {bar.dt.strftime('%H:%M')}  "
            f"O={bar.open:.2f}  H={bar.high:.2f}  L={bar.low:.2f}  C={bar.close:.2f}  "
            f"V={bar.volume:.0f}  {arrow}{abs(bar.change_pct):.2%}\n"
        )

    def on_connect():
        log.info("🟢  Connected to Finnhub.")

    def on_disconnect():
        log.warning("🔴  Disconnected from Finnhub.")

    # ── stream ───────────────────────────────────────────────────────────────
    print("═" * 60)
    print("  LIVE DATA STREAM — DEMO")
    print("═" * 60)
    print(f"  Symbols  : {SYMBOLS}")
    print(f"  Duration : {duration}s")
    print(f"  API key  : {'provided' if api_key else 'none (mock mode)'}")
    print("═" * 60)
    print()

    stream = LiveDataStream(
        api_key      = api_key,
        symbols      = SYMBOLS,
        demo         = not api_key,
        on_tick      = on_tick,
        on_bar       = on_bar,
        on_connect   = on_connect,
        on_disconnect= on_disconnect,
    )

    # Graceful shutdown on Ctrl-C
    def _sigint(sig, frame):
        print("\n  Interrupted — shutting down …")
        stream.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, _sigint)

    stream.start()
    print("  Streaming … (press Ctrl-C to stop early)\n")

    # wait for initial data
    stream.wait_for_data(timeout=5.0)

    # run for duration seconds
    time.sleep(duration)

    # ── final report ─────────────────────────────────────────────────────────
    print("\n")
    stream.print_status()

    print("\n  AI FEED SNAPSHOT")
    print("─" * 60)
    feed = stream.ai_feed()
    for sym, data in feed.items():
        snap = data["snapshot"]
        bar  = data["last_bar"]
        n_prices = len(data["prices"])
        if snap:
            print(
                f"  {sym:<6}  price=${snap['last_price']:>10,.4f}  "
                f"1m_chg={snap['price_change_1m']:+.2%}  "
                f"5m_chg={snap['price_change_5m']:+.2%}  "
                f"prices={n_prices}"
            )
        else:
            print(f"  {sym:<6}  no data yet")
    print("─" * 60)
    print(f"\n  Total ticks received: {tick_count[0]}")

    stream.stop()


# ── Live Stock Price Fetcher ──────────────────────────────────────────────────

def get_live_stock_price(symbol: str) -> dict:
    """
    Fetch real-time stock price from Finnhub API.
    Falls back to yfinance if Finnhub is unavailable or fails.
    """
    import requests
    import os
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
        
    try:
        import yfinance as yf
    except ImportError:
        yf = None

    api_key = os.getenv("FINNHUB_API_KEY", "")
    
    if api_key:
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol={symbol.upper()}&token={api_key}"
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            
            if data and "c" in data and data["c"] != 0:
                return {
                    "symbol": symbol.upper(),
                    "price": float(data["c"]),
                    "high": float(data["h"]),
                    "low": float(data["l"]),
                    "open": float(data["o"]),
                    "previous close": float(data["pc"]),
                    "previous_close": float(data["pc"])
                }
        except Exception as e:
            log.error(f"[Finnhub] REST API failed for {symbol}: {e}. Falling back to yfinance.")
    
    # Fallback to yfinance
    if yf:
        try:
            ticker = yf.Ticker(symbol)
            todays_data = ticker.history(period="1d")
            if not todays_data.empty:
                return {
                    "symbol": symbol.upper(),
                    "price": float(todays_data['Close'].iloc[-1]),
                    "high": float(todays_data['High'].iloc[-1]),
                    "low": float(todays_data['Low'].iloc[-1]),
                    "open": float(todays_data['Open'].iloc[-1]),
                    "previous close": float(todays_data['Close'].iloc[0]),
                    "previous_close": float(todays_data['Close'].iloc[0]) # Proxy for daily open/prev_close if only 1 bar
                }
        except Exception as e:
            log.error(f"[yfinance] Fallback failed for {symbol}: {e}")
            
    # Default zero-filled response
    return {
        "symbol": symbol.upper(),
        "price": 0.0,
        "high": 0.0,
        "low": 0.0,
        "open": 0.0,
        "previous close": 0.0,
        "previous_close": 0.0
    }


if __name__ == "__main__":
    key = os.getenv("FINNHUB_API_KEY", "")
    _run_demo(api_key=key, duration=20)
