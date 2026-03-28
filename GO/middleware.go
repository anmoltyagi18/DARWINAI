package middleware

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/gin-gonic/gin"
)

// CORS sets permissive CORS headers. Tighten for production.
func CORS() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin",  "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Authorization, X-Request-ID")
		c.Header("Access-Control-Max-Age",       "86400")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		c.Next()
	}
}

// RequestID attaches a unique request ID to every request.
func RequestID() gin.HandlerFunc {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	return func(c *gin.Context) {
		id := c.GetHeader("X-Request-ID")
		if id == "" {
			id = fmt.Sprintf("%d-%d", time.Now().UnixNano(), rng.Intn(100000))
		}
		c.Set("RequestID", id)
		c.Header("X-Request-ID", id)
		c.Next()
	}
}
