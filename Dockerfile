# 🐳 Multi-stage Dockerfile for 1D-Ensemble
# Optimized for production deployment with minimal image size

# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt pyproject.toml ./
COPY ensemble_1d/ ./ensemble_1d/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels .

# ============================================
# Stage 2: Runtime
# ============================================
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN useradd -m -u 1000 mluser && \
    mkdir -p /app && \
    chown -R mluser:mluser /app

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /build/wheels /tmp/wheels

# Install Python packages from wheels
RUN pip install --no-cache-dir /tmp/wheels/* && \
    rm -rf /tmp/wheels

# Copy application code
COPY --chown=mluser:mluser . .

# Switch to non-root user
USER mluser

# Expose ports
EXPOSE 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import ensemble_1d; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
