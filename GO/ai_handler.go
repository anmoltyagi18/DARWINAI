package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

// AIHandler forwards requests to the Python AI engine.
type AIHandler struct {
	engineURL  string
	httpClient *http.Client
}

func NewAIHandler(engineURL string) *AIHandler {
	return &AIHandler{
		engineURL: engineURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// ─────────────────────────────────────────────
// GET /api/v1/stock-data?symbol=AAPL&interval=1d
// ─────────────────────────────────────────────
func (h *AIHandler) GetStockData(c *gin.Context) {
	symbol   := c.DefaultQuery("symbol", "AAPL")
	interval := c.DefaultQuery("interval", "1d")
	limit    := c.DefaultQuery("limit", "100")

	url := fmt.Sprintf("%s/stock-data?symbol=%s&interval=%s&limit=%s",
		h.engineURL, symbol, interval, limit)

	h.proxyGet(c, url)
}

// ─────────────────────────────────────────────
// POST /api/v1/analyze
// Body: { "symbol": "AAPL", "model": "lstm", ... }
// ─────────────────────────────────────────────
func (h *AIHandler) Analyze(c *gin.Context) {
	var body map[string]interface{}
	if err := c.ShouldBindJSON(&body); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "invalid_request",
			"message": "Request body must be valid JSON",
			"details": err.Error(),
		})
		return
	}

	// Attach request metadata
	body["request_id"] = c.GetString("RequestID")
	body["timestamp"]  = time.Now().UTC().Format(time.RFC3339)

	h.proxyPost(c, h.engineURL+"/analyze", body)
}

// ─────────────────────────────────────────────
// GET /api/v1/strategy?symbol=AAPL&risk=low
// ─────────────────────────────────────────────
func (h *AIHandler) GetStrategy(c *gin.Context) {
	symbol   := c.DefaultQuery("symbol", "AAPL")
	risk     := c.DefaultQuery("risk", "medium")
	horizon  := c.DefaultQuery("horizon", "short")

	url := fmt.Sprintf("%s/strategy?symbol=%s&risk=%s&horizon=%s",
		h.engineURL, symbol, risk, horizon)

	h.proxyGet(c, url)
}

// ─────────────────────────────────────────────
// GET /api/v1/signals?symbol=AAPL&type=all
// ─────────────────────────────────────────────
func (h *AIHandler) GetSignals(c *gin.Context) {
	symbol     := c.DefaultQuery("symbol", "AAPL")
	signalType := c.DefaultQuery("type", "all")
	timeframe  := c.DefaultQuery("timeframe", "1d")

	url := fmt.Sprintf("%s/signals?symbol=%s&type=%s&timeframe=%s",
		h.engineURL, symbol, signalType, timeframe)

	h.proxyGet(c, url)
}

// ─────────────────────────────────────────────
// GET /api/v1/live/:symbol
// Proxies to Python /live/{symbol}
// ─────────────────────────────────────────────
func (h *AIHandler) GetLive(c *gin.Context) {
	symbol := c.Param("symbol")
	if symbol == "" {
		symbol = c.DefaultQuery("symbol", "AAPL")
	}
	url := fmt.Sprintf("%s/live/%s", h.engineURL, symbol)
	h.proxyGet(c, url)
}

// ─────────────────────────────────────────────
// GET /api/v1/backtest/:symbol
// Proxies to Python /backtest/{symbol}
// ─────────────────────────────────────────────
func (h *AIHandler) GetBacktest(c *gin.Context) {
	symbol := c.Param("symbol")
	if symbol == "" {
		symbol = c.DefaultQuery("symbol", "AAPL")
	}
	period      := c.DefaultQuery("period", "1y")
	initialCash := c.DefaultQuery("initial_cash", "100000")
	commission  := c.DefaultQuery("commission", "0.001")

	url := fmt.Sprintf("%s/backtest/%s?period=%s&initial_cash=%s&commission=%s",
		h.engineURL, symbol, period, initialCash, commission)

	h.proxyGet(c, url)
}

// ─────────────────────────────────────────────
// GET /api/v1/explain/:symbol
// Proxies to Python /explain/{symbol}
// ─────────────────────────────────────────────
func (h *AIHandler) GetExplain(c *gin.Context) {
	symbol := c.Param("symbol")
	if symbol == "" {
		symbol = c.DefaultQuery("symbol", "AAPL")
	}
	period := c.DefaultQuery("period", "1y")
	url := fmt.Sprintf("%s/explain/%s?period=%s", h.engineURL, symbol, period)
	h.proxyGet(c, url)
}

// ─────────────────────────────────────────────
// GET /api/v1/analyze-full/:symbol
// Combines analyze + explain in one Go call
// ─────────────────────────────────────────────
func (h *AIHandler) GetAnalyzeFull(c *gin.Context) {
	symbol := c.Param("symbol")
	if symbol == "" {
		symbol = c.DefaultQuery("symbol", "AAPL")
	}
	period   := c.DefaultQuery("period", "1y")
	interval := c.DefaultQuery("interval", "1d")

	url := fmt.Sprintf("%s/analyze/%s?period=%s&interval=%s",
		h.engineURL, symbol, period, interval)

	h.proxyGet(c, url)
}

// ─────────────────────────────────────────────
// GET /api/v1/opportunity/:symbol
// Proxies to Python /opportunity/{symbol}
// Returns: best buy/sell dates, profit %, strategy, trade type
// ─────────────────────────────────────────────
func (h *AIHandler) GetOpportunity(c *gin.Context) {
	symbol := c.Param("symbol")
	if symbol == "" {
		symbol = c.DefaultQuery("symbol", "AAPL")
	}
	period   := c.DefaultQuery("period", "1y")
	interval := c.DefaultQuery("interval", "1d")

	url := fmt.Sprintf("%s/opportunity/%s?period=%s&interval=%s",
		h.engineURL, symbol, period, interval)

	h.proxyGet(c, url)
}

// ─────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────

func (h *AIHandler) proxyGet(c *gin.Context, url string) {
	req, err := http.NewRequestWithContext(c.Request.Context(), http.MethodGet, url, nil)
	if err != nil {
		c.JSON(http.StatusInternalServerError, engineError("failed to build request", err))
		return
	}
	h.addHeaders(req, c)
	h.do(c, req)
}

func (h *AIHandler) proxyPost(c *gin.Context, url string, body interface{}) {
	payload, err := json.Marshal(body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, engineError("failed to serialize body", err))
		return
	}

	req, err := http.NewRequestWithContext(c.Request.Context(), http.MethodPost, url,
		bytes.NewBuffer(payload))
	if err != nil {
		c.JSON(http.StatusInternalServerError, engineError("failed to build request", err))
		return
	}
	req.Header.Set("Content-Type", "application/json")
	h.addHeaders(req, c)
	h.do(c, req)
}

func (h *AIHandler) addHeaders(req *http.Request, c *gin.Context) {
	req.Header.Set("Accept", "application/json")
	req.Header.Set("X-Request-ID", c.GetString("RequestID"))
	req.Header.Set("X-Forwarded-For", c.ClientIP())
}

func (h *AIHandler) do(c *gin.Context, req *http.Request) {
	resp, err := h.httpClient.Do(req)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{
			"error":   "engine_unavailable",
			"message": "Could not reach Python AI engine",
			"details": err.Error(),
		})
		return
	}
	defer resp.Body.Close()

	rawBody, err := io.ReadAll(resp.Body)
	if err != nil {
		c.JSON(http.StatusBadGateway, engineError("failed to read engine response", err))
		return
	}

	// Pass the engine's status code and body straight through
	var parsed interface{}
	if err := json.Unmarshal(rawBody, &parsed); err != nil {
		// Engine returned non-JSON — wrap it
		c.JSON(resp.StatusCode, gin.H{
			"error":   "invalid_engine_response",
			"message": "Engine returned non-JSON payload",
			"raw":     string(rawBody),
		})
		return
	}

	c.JSON(resp.StatusCode, parsed)
}

func engineError(msg string, err error) gin.H {
	return gin.H{
		"error":   "engine_error",
		"message": msg,
		"details": err.Error(),
	}
}
