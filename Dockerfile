# Asian Odds Betting Monitor - Docker Image
# ==========================================
#
# Build: docker build -t betting-monitor:latest .
# Run:   docker run -d --env-file .env betting-monitor:latest
#
# Or use docker-compose (recommended):
# docker-compose up -d

FROM python:3.11-slim

# Metadata
LABEL maintainer="Asian Odds Betting Monitor"
LABEL description="24/7 Automated betting monitor with Telegram notifications"
LABEL version="1.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        git \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 betmonitor && \
    chown -R betmonitor:betmonitor /app

# Switch to non-root user
USER betmonitor

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Environment variables (override with .env or docker-compose)
ENV PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO \
    LIVE_UPDATE_INTERVAL=60

# Expose port (if you add a web dashboard later)
# EXPOSE 8501

# Run the monitoring script
CMD ["python", "start_live_monitoring.py"]
