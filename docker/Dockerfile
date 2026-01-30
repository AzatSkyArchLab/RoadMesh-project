# RoadMesh Production Dockerfile
# Multi-stage build for smaller final image

# =============================================================================
# Stage 1: Build
# =============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels \
    -e .

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy application code
COPY src/ src/
COPY configs/ configs/

# Create directories
RUN mkdir -p data/cache checkpoints logs

# Environment variables
ENV PYTHONPATH=/app/src
ENV ROADMESH_CONFIG_PATH=/app/configs/base.yaml

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API server
CMD ["python", "-m", "uvicorn", "roadmesh.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
