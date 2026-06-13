# Anomaly Detection Sidecar (Day 19)

AI-powered sidecar container that detects anomalies in application metrics using statistical methods (z-score, threshold, IQR) with optional LLM root-cause analysis.

## Features

- **Z-score detection** -- identifies statistical outliers from rolling window baseline
- **Static threshold detection** -- configurable per-metric threshold rules
- **IQR detection** -- interquartile range outlier detection
- Rolling window management with configurable size
- Batch metric processing
- Optional AI-powered root-cause analysis via LLM
- JSON and text output formats
- Ollama (local) with OpenAI fallback

## Architecture

```
main.py (CLI) --> detector.py (statistical engine)
                   |
                   v
              llm.py (Ollama/OpenAI)  <-- optional analysis
                   |
                   v
           Anomaly Report (text/JSON)
```

**Module overview:**
- `src/detector.py` -- z-score, threshold, and IQR anomaly detection with rolling windows
- `src/llm.py` -- dual-backend LLM client (Ollama at localhost:11434, OpenAI fallback)
- `src/main.py` -- CLI entry point

## Requirements

- Python 3.11+
- Ollama running locally (or OPENAI_API_KEY for remote fallback)
- No third-party packages required (stdlib only)

## Quick Start

```bash
git clone <repo-url> && cd anomaly-detection-sidecar

# Create metrics JSON file
cat > metrics.json << 'EOF'
[
  {"name": "cpu_usage", "value": 45.2, "timestamp": 1000, "labels": {"host": "web-1"}},
  {"name": "cpu_usage", "value": 47.1, "timestamp": 1001, "labels": {"host": "web-1"}},
  {"name": "cpu_usage", "value": 44.8, "timestamp": 1002, "labels": {"host": "web-1"}},
  {"name": "cpu_usage", "value": 46.5, "timestamp": 1003, "labels": {"host": "web-1"}},
  {"name": "cpu_usage", "value": 43.9, "timestamp": 1004, "labels": {"host": "web-1"}},
  {"name": "cpu_usage", "value": 98.7, "timestamp": 1005, "labels": {"host": "web-1"}},
  {"name": "error_rate", "value": 12.5, "timestamp": 1005, "labels": {"service": "api"}},
  {"name": "latency_p99", "value": 2500, "timestamp": 1005, "labels": {"endpoint": "/api/v1"}}
]
EOF

# Run anomaly detection
python -m src.main -i metrics.json

# With threshold rules
python -m src.main -i metrics.json --threshold "error_rate=5" --threshold "latency_p99=2000"

# With AI analysis
python -m src.main -i metrics.json --threshold "error_rate=5" --analyze --format json
```

## Usage

```bash
# Basic detection
python -m src.main -i metrics.json

# Custom z-score threshold
python -m src.main -i metrics.json --zscore 2.5

# Custom window size
python -m src.main -i metrics.json --window 100

# JSON output with AI analysis
python -m src.main -i metrics.json --analyze --format json -o report.json

# Read from stdin
cat metrics.json | python -m src.main
```

## Running Tests

```bash
python -m venv .venv
source .venv/bin/activate
pip install pytest
python -m pytest tests/ -v
```

## Docker

```bash
docker compose up -d
docker compose run anomaly-detector -i /app/data/metrics.json --analyze
```
