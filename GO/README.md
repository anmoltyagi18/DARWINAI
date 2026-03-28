# Go Trading API (Gin + Python AI Engine)

A Go REST API built with Gin that acts as the gateway layer, forwarding
requests to a Python AI engine for stock analysis, signals, and strategy.

```
┌──────────────┐   REST   ┌────────────────────┐   REST   ┌──────────────────┐
│   Client     │ ───────► │  Go API (Gin :8080) │ ───────► │ Python AI Engine │
│  (curl/app)  │          │                    │          │   (Flask :5000)  │
└──────────────┘          └────────────────────┘          └──────────────────┘
```

## Endpoints

| Method | Path                  | Description                          | Query params                              |
|--------|-----------------------|--------------------------------------|-------------------------------------------|
| GET    | `/health`             | Health check                         | —                                         |
| GET    | `/api/v1/stock-data`  | OHLCV price data                     | `symbol`, `interval`, `limit`             |
| POST   | `/api/v1/analyze`     | AI price prediction                  | JSON body: `symbol`, `model`, `horizon_days` |
| GET    | `/api/v1/strategy`    | Trading strategy recommendation      | `symbol`, `risk`, `horizon`               |
| GET    | `/api/v1/signals`     | Technical indicator signals          | `symbol`, `type`, `timeframe`             |

## Quick Start

### Option A — Docker Compose (recommended)
```bash
docker-compose up --build
```

### Option B — Local

**Python engine**
```bash
pip install flask
python python_engine.py          # listens on :5000
```

**Go API**
```bash
go mod tidy
PYTHON_ENGINE_URL=http://localhost:5000 go run main.go   # listens on :8080
```

## Example Requests

```bash
# Stock data
curl "http://localhost:8080/api/v1/stock-data?symbol=TSLA&limit=10"

# Analyze
curl -X POST http://localhost:8080/api/v1/analyze \
     -H "Content-Type: application/json" \
     -d '{"symbol":"AAPL","model":"lstm","horizon_days":5}'

# Strategy
curl "http://localhost:8080/api/v1/strategy?symbol=AAPL&risk=high&horizon=short"

# Signals
curl "http://localhost:8080/api/v1/signals?symbol=MSFT&type=all&timeframe=1h"
```

## Environment Variables

| Variable           | Default                  | Description                    |
|--------------------|--------------------------|--------------------------------|
| `PORT`             | `8080`                   | Go API port                    |
| `ENV`              | `development`            | `development` or `production`  |
| `PYTHON_ENGINE_URL`| `http://localhost:5000`  | Python AI engine base URL      |

## Project Structure

```
go-trading-api/
├── main.go                  # Entry point, router setup
├── config/
│   └── config.go            # Environment configuration
├── handlers/
│   └── ai_handler.go        # All four endpoint handlers + HTTP proxy helpers
├── middleware/
│   └── middleware.go        # CORS + Request-ID middleware
├── python_engine.py         # Python Flask stub (AI engine)
├── Dockerfile               # Go service image
├── Dockerfile.python        # Python service image
├── docker-compose.yml       # Orchestration
└── go.mod
```
