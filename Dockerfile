# ============================================================
# Stage 1: builder — install deps with pip into /install
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /install

# System deps needed to compile some Python packages (numpy, onnxruntime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for cache-friendly layer ordering
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install/prefix -r requirements.txt

# ============================================================
# Stage 2: runtime — lean final image
# ============================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install/prefix /usr/local

# Create a non-root user for security
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid appuser --shell /bin/bash --create-home appuser

# Copy project source (everything not in .dockerignore)
COPY . .

# Ensure the data directory for SQLite exists and is writable
RUN mkdir -p /app/data && chown -R appuser:appuser /app

USER appuser

# Expose the application port
EXPOSE 8000

# Health check — calls the /health endpoint every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Production entrypoint:
#   --workers 1      : single worker (async; multiple workers need shared session store)
#   --loop asyncio   : force asyncio event loop (avoids uvloop conflicts)
#   --no-access-log  : reduces noise (structured logging is in the app)
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "asyncio", \
     "--no-access-log"]
