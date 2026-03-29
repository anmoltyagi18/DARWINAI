# AIGOFIN Institutional Trading Terminal

A professional-grade AI trading platform featuring real-time analysis, genetic strategy evolution, and risk management.

## 🚀 One-Click Deployment Guide

### Phase 1: Backend Deployment (Render.com)

1.  **Create New Service**: Select **Web Service** and connect your GitHub repo.
2.  **Configuration**:
    *   **Runtime**: Python 3
    *   **Root Directory**: (Leave Blank)
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `python run_server.py --port $PORT`
3.  **Environment Variables**:
    *   `FINNHUB_API_KEY`: (Get from Finnhub.io)
4.  **Copy the URL**: Once live, copy your Render URL (e.g., `https://aigofin-api.onrender.com`).

### Phase 2: Frontend Deployment (Vercel.com)

1.  **Create New Project**: Select your GitHub repo.
2.  **Configuration**:
    *   **Framework Preset**: Next.js
    *   **Root Directory**: `frontend`
3.  **Environment Variables**:
    *   Add `NEXT_PUBLIC_API_URL` and paste your Render URL from Phase 1.
4.  **Deploy**: Your full site is now live!

---

## 🛠️ Local Development

```bash
# Start Backend
python run_server.py --port 8000

# Start Frontend
cd frontend
npm run dev
```

## 🧠 Core Systems
- **Genetic Strategy Engine**: Evolves trading genomes based on volatility and momentum.
- **RL Decision Agent**: Stabilized PPO agent for high-frequency signal validation.
- **Universal Multi-Horizon Visualization**: Real-time candlestick charts with technical indicators.
