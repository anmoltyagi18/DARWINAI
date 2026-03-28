package config

import "os"

type Config struct {
	Port            string
	Env             string
	PythonEngineURL string
}

func Load() *Config {
	return &Config{
		Port:            getEnv("PORT", "8080"),
		Env:             getEnv("ENV", "development"),
		PythonEngineURL: getEnv("PYTHON_ENGINE_URL", "http://localhost:5000"),
	}
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
