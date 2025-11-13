# AsianOdds Live Betting Monitor
# Multi-stage build for optimized image size

FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ============================================================
# Final stage
# ============================================================

FROM python:3.10-slim

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY . /app/

# Create non-root user
RUN useradd -m -u 1000 betmonitor && \
    chown -R betmonitor:betmonitor /app

# Create directories for data persistence
RUN mkdir -p /app/ai_system/cache /app/ai_system/logs /app/ai_system/models && \
    chown -R betmonitor:betmonitor /app/ai_system

# Switch to non-root user
USER betmonitor

# Environment variables (override via docker run -e)
ENV TELEGRAM_BOT_TOKEN=""
ENV TELEGRAM_CHAT_ID=""
ENV LOG_LEVEL="INFO"
ENV UPDATE_INTERVAL="60"
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=5m --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Expose port (if adding web dashboard in future)
EXPOSE 8080

# Entry point
ENTRYPOINT ["python", "-u", "start_live_monitoring.py"]

# Default command arguments
CMD ["--interval", "60", "--log-level", "INFO"]
