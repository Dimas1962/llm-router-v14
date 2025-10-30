# Unified LLM Router v2.0 - Production Dockerfile
FROM python:3.11-slim

# Metadata
LABEL maintainer="your.email@example.com"
LABEL version="2.0.0"
LABEL description="Unified LLM Router v2.0 with 21 components"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

# Create non-root user for security
RUN useradd -m -u 1000 router && \
    chown -R router:router /app

USER router

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TQDM_DISABLE=1

# Expose port (if running API server)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.unified.unified_router import UnifiedRouter; print('OK')" || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
