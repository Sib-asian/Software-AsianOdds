FROM python:3.11-slim

WORKDIR /app

# Copy lean requirements for Fly.io deployment first to leverage Docker layer caching
COPY requirements-fly.txt .
RUN pip install --no-cache-dir -r requirements-fly.txt

# Copy application code
COPY . .

# Streamlit defaults to 8501; Fly.io injects PORT
ENV PORT=8501

CMD ["streamlit", "run", "Frontendcloud.py", "--server.port", "${PORT}", "--server.address", "0.0.0.0"]
