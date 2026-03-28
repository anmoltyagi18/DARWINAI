# AIGOFIN Institutional Trading Terminal

A professional-grade AI trading platform featuring real-time analysis, genetic strategy evolution, and risk management.

## 🚀 Quick Start

### 1. Backend (FastAPI)
```bash
# Install dependencies
pip install -r requirements.txt

# Start the AI engine
python run_server.py --port 8000
```

### 2. Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev
```

## 🌐 Deployment Guide

### Phase 1: Backend (Render/Railway)
1. Create a new **Web Service** from this repository.
2. Set the build command to `pip install -r requirements.txt`.
3. Set the start command to `python run_server.py --port $PORT`.
4. Add Environment Variable: `FINNHUB_API_KEY` (Get from Finnhub.io).

### Phase 2: Frontend (Vercel)
1. Create a new **Project** from the `frontend/` directory.
2. Add Environment Variable: `NEXT_PUBLIC_API_URL` (Point to your Backend URL).
3. Deploy.

---

## 🛠️ Tech Stack
- **Frontend**: Next.js 15, Tailwind CSS, Framer Motion, Lightweight Charts.
- **Backend**: FastAPI, yfinance, Pandas, NumPy, Scikit-learn.
- **AI**: Multi-module consensus engine (Trend, Momentum, Volatility, Sentiment).
