FROM python:3.11-slim

LABEL maintainer="anomaly-detection-sidecar"
LABEL description="AI-powered anomaly detection sidecar for application metrics"

WORKDIR /app

COPY src/ ./src/

ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--help"]
