"""
health_check.py — AIGOFIN ML Package
======================================
Runs a startup diagnostic to ensure all critical dependencies
and environment configurations are present.
"""

import sys
import os

def run_diagnostics() -> bool:
    """Check required packages and environment. Returns True if core systems pass."""
    print()
    print("=" * 54)
    print("  AIGOFIN STARTUP DIAGNOSTICS")
    print("=" * 54)
    
    passed = True
    
    # ── 1. Python Version ────────────────────────────────────────────────
    py_ver = sys.version_info
    if py_ver.major == 3 and py_ver.minor >= 9:
        print(f"  [x] Python Version: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    else:
        print(f"  [!] Python Version: {py_ver.major}.{py_ver.minor}.{py_ver.micro} (Recommended: 3.9+)")
        
    # ── 2. Core Dependencies ─────────────────────────────────────────────
    packages = {
        "fastapi": "API Framework",
        "uvicorn": "ASGI Server",
        "yfinance": "Market Data",
        "numpy": "Math Operations",
        "scipy": "Stats Engine",
        "backtrader": "Backtest Engine",
        "stable_baselines3": "RL Agent",
        "gymnasium": "RL Environment"
    }
    
    for pkg, desc in packages.items():
        try:
            __import__(pkg)
            print(f"  [x] {pkg:<18} : OK ({desc})")
        except ImportError:
            print(f"  [FAIL] {pkg:<15}- Missing! Run: pip install {pkg}")
            passed = False
            
    # ── 3. Environment Config ────────────────────────────────────────────
    fh_key = os.getenv("FINNHUB_API_KEY", "")
    if fh_key:
        print(f"  [x] FINNHUB_API_KEY  : Loaded")
    else:
        print(f"  [!] FINNHUB_API_KEY  : NOT SET (yfinance fallback will be used)")
        
    print("=" * 54)
    print(f"  DIAGNOSTIC RESULT: {'PASS' if passed else 'FAIL'}")
    print("=" * 54)
    print()
    
    return passed

if __name__ == "__main__":
    success = run_diagnostics()
    sys.exit(0 if success else 1)
