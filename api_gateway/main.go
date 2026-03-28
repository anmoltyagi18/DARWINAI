package main

import (
	"bytes"
	"io"
	"log"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"
)

// Default Python backend URL (overridable via ENV)
var PythonAPIBase = "http://127.0.0.1:8000"

func init() {
	if envBase := os.Getenv("PYTHON_API_URL"); envBase != "" {
		PythonAPIBase = envBase
	}
}

// forwardRequest universally pipes the request straight to the AI Engine
func forwardRequest(c *gin.Context) {
	targetPath := c.Request.URL.Path
	if c.Request.URL.RawQuery != "" {
		targetPath += "?" + c.Request.URL.RawQuery
	}

	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read request body"})
		return
	}

	req, err := http.NewRequest(c.Request.Method, PythonAPIBase+targetPath, bytes.NewBuffer(body))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create forward request"})
		return
	}

	// Clone original headers
	for k, v := range c.Request.Header {
		req.Header[k] = v
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("[Gateway Error] Proxy failure to python core: %v", err)
		c.JSON(http.StatusBadGateway, gin.H{"error": "Failed to communicate with Python AI server"})
		return
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read response from Python AI server"})
		return
	}

	// Mirror exact response back out
	for k, v := range resp.Header {
		for _, headerVal := range v {
			c.Header(k, headerVal)
		}
	}
	c.Data(resp.StatusCode, resp.Header.Get("Content-Type"), respBody)
}

func main() {
	// Let Gin do default logger/recovery middleware out of the box
	r := gin.Default()

	log.Printf("Starting AIGOFIN API Gateway. Targeting Python core at %s", PythonAPIBase)

	r.Any("/analyze/*path", forwardRequest)
	r.Any("/opportunity/*path", forwardRequest)
	r.Any("/backtest/*path", forwardRequest)
	r.Any("/live/*path", forwardRequest)
	r.Any("/chart/*path", forwardRequest)
	r.Any("/strategy/*path", forwardRequest)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("API Gateway HTTP proxy listening on port :%s", port)
	if err := r.Run(":" + port); err != nil {
		log.Fatal("Failed to start server: ", err)
	}
}
