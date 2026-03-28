package main

import (
	"log"
	"os"

	"github.com/gin-gonic/gin"
	"go-trading-api/config"
	"go-trading-api/handlers"
	"go-trading-api/middleware"
)

func main() {
	cfg := config.Load()

	if cfg.Env == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	r := gin.New()

	// Middleware
	r.Use(gin.Logger())
	r.Use(gin.Recovery())
	r.Use(middleware.CORS())
	r.Use(middleware.RequestID())

	// Health check
	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok", "service": "go-trading-api"})
	})

	// API v1 routes
	v1 := r.Group("/api/v1")
	{
		aiHandler := handlers.NewAIHandler(cfg.PythonEngineURL)

		// Existing routes
		v1.GET("/stock-data",     aiHandler.GetStockData)
		v1.POST("/analyze",       aiHandler.Analyze)
		v1.GET("/strategy",       aiHandler.GetStrategy)
		v1.GET("/signals",        aiHandler.GetSignals)

		// New routes — proxied to Python AI server
		v1.GET("/live/:symbol",        aiHandler.GetLive)
		v1.GET("/backtest/:symbol",    aiHandler.GetBacktest)
		v1.GET("/explain/:symbol",     aiHandler.GetExplain)
		v1.GET("/analyze/:symbol",     aiHandler.GetAnalyzeFull)
		v1.GET("/opportunity/:symbol", aiHandler.GetOpportunity)
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = cfg.Port
	}

	log.Printf("🚀 Server starting on port %s", port)
	log.Printf("🤖 Python AI Engine: %s", cfg.PythonEngineURL)

	if err := r.Run(":" + port); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}
