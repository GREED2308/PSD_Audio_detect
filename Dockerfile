# syntax=docker/dockerfile:1
FROM python:3.13-slim

# Install system audio dependencies required by librosa and soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

# Create non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy requirements and install dependencies first to leverage Docker layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files and assign ownership to non-root user
COPY . .
RUN chown -R appuser:appuser /app

# Run as non-root user
USER appuser

EXPOSE 8080

# Healthcheck to verify Streamlit server readiness using python urllib
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/_stcore/health').getcode() == 200 else 1)"

CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true"]

