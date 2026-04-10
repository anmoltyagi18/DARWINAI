from __future__ import annotations

"""
run_server.py — AIGOFIN System Launcher
=========================================
Starts the FastAPI AI trading server with:
  - .env loading (FINNHUB_API_KEY and other secrets)
  - Automatic free-port detection
  - Browser auto-open to /docs
  - Clean console output with server URL

Usage
-----
    python run_server.py
    python run_server.py --port 8000
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

# ── Load .env before any ML imports ──────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
_ENV_FILE = _ROOT / ".env"

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_ENV_FILE, override=False)
except ImportError:
    # Manual .env parse if python-dotenv not installed
    if _ENV_FILE.exists():
        with open(_ENV_FILE) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip())

# ── Import uvicorn after env is loaded ────────────────────────────────────────
try:
    import uvicorn
except ImportError:
    print("❌  uvicorn not installed.  Run:  pip install uvicorn")
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def get_free_port(preferred: int = 0) -> int:
    """Return a free TCP port. Uses preferred port if available, else auto."""
    if preferred:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", preferred))
                return preferred
        except OSError:
            pass  # preferred port in use — fall through to auto-detect

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        return s.getsockname()[1]


def open_browser_after_delay(port: int, delay: float = 1.5) -> None:
    """Open /docs in the default browser after a short delay."""
    time.sleep(delay)
    url = f"http://localhost:{port}/docs"
    try:
        webbrowser.open(url)
    except Exception as exc:
        print(f"⚠  Could not open browser: {exc}")


def print_banner(port: int) -> None:
    bar = "═" * 54
    finnhub = os.getenv("FINNHUB_API_KEY", "")
    fh_status = "✅  key loaded" if finnhub else "⚠   NOT SET (using yfinance fallback)"
    print(f"\n{bar}")
    print(f"  🤖  AIGOFIN AI TRADING ENGINE")
    print(bar)
    print(f"  Server URL : http://localhost:{port}")
    print(f"  Swagger UI : http://localhost:{port}/docs")
    print(f"  ReDoc      : http://localhost:{port}/redoc")
    print(f"  Finnhub    : {fh_status}")
    print(f"{bar}\n")


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # Parse optional --port argument
    preferred_port = 0
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--port" and i < len(sys.argv):
            try:
                preferred_port = int(sys.argv[i + 1])
            except (IndexError, ValueError):
                pass
        elif arg.startswith("--port="):
            try:
                preferred_port = int(arg.split("=", 1)[1])
            except ValueError:
                pass

    port = get_free_port(preferred_port or 8000)

    print_banner(port)

    # Run diagnostics
    try:
        from ML.health_check import run_diagnostics
        if not run_diagnostics():
            print("\n⚠ WARNING: Some ML packages are missing. The server will start, but some AI features or backtesting might fail.")
            time.sleep(2)
    except ImportError as e:
        print(f"\n⚠ WARNING: Could not run health check ({e}). Continuing...")
        

    try:
        uvicorn.run(
            "ML.api_server:app",
            host="127.0.0.1",
            port=port,
            reload=False,       # use True during development with --reload flag
            log_level="info",
        )
    except Exception as exc:
        print(f"\n❌  Server crashed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
