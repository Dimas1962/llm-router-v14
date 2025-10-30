# Unified Router v2.0 - Production Deployment Guide

## Quick Start

```bash
# 1. Build distribution packages
python setup.py sdist bdist_wheel

# 2. Build Docker image
docker build -t unified-router:v2.0 .

# 3. Run with Docker Compose
docker-compose up -d
```

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Installation](#local-installation)
3. [Docker Deployment](#docker-deployment)
4. [Production Configuration](#production-configuration)
5. [Monitoring & Observability](#monitoring--observability)
6. [Integration with MindRest](#integration-with-mindrest)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **Python**: 3.10 or higher
- **Memory**: Minimum 4GB RAM (8GB+ recommended)
- **CPU**: 2+ cores recommended
- **Disk**: 5GB+ free space (for models and cache)
- **Docker**: 20.10+ (for containerized deployment)
- **Docker Compose**: 2.0+ (optional, for orchestration)

### Dependencies

All dependencies are listed in `requirements.txt`:
- **Core**: numpy, pydantic
- **ML**: faiss-cpu, sentence-transformers, torch
- **API**: fastapi, uvicorn
- **Monitoring**: prometheus-client

---

## Local Installation

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3. Install Package

```bash
# Development mode (editable)
pip install -e .

# Production mode
pip install .
```

### 4. Verify Installation

```bash
python -c "from src.unified.unified_router import UnifiedRouter; print('✓ Installation successful')"
```

### 5. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run unified router tests only
pytest tests/unified/ -v

# Quick smoke test
pytest tests/unified/test_unified_router.py::test_unified_initialization -v
```

---

## Docker Deployment

### 1. Build Docker Image

```bash
# Standard build
docker build -t unified-router:v2.0 .

# With build arguments
docker build \
  --build-arg PYTHON_VERSION=3.11 \
  -t unified-router:v2.0 .

# Multi-platform build (for ARM64 + AMD64)
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t unified-router:v2.0 \
  --push .
```

### 2. Run Container

```bash
# Simple run
docker run -d \
  --name unified-router \
  -p 8000:8000 \
  unified-router:v2.0

# With environment variables
docker run -d \
  --name unified-router \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  -e ANTHROPIC_API_KEY=your-key \
  -v $(pwd)/logs:/app/logs \
  unified-router:v2.0

# With resource limits
docker run -d \
  --name unified-router \
  -p 8000:8000 \
  --memory=4g \
  --cpus=2 \
  unified-router:v2.0
```

### 3. Docker Compose Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f unified-router

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

### 4. Health Check

```bash
# Check container health
docker ps

# Test the router
docker exec -it unified-router-v2 python -c "
from src.unified.unified_router import UnifiedRouter
router = UnifiedRouter()
print('✓ Router healthy')
"
```

---

## Production Configuration

### 1. Environment Variables

Create `.env` file:

```bash
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Router Configuration
ROUTER_MAX_CACHE_SIZE=1000
ROUTER_ENABLE_BATCHING=true
ROUTER_ENABLE_QUALITY_CHECK=true
ROUTER_ENABLE_SNAPSHOTS=true
ROUTER_ENABLE_MONITORING=true

# Performance
ROUTER_MAX_WORKERS=4
ROUTER_TIMEOUT=30

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Redis (optional)
REDIS_URL=redis://redis:6379/0
```

### 2. Quality Thresholds

Configure in your application:

```python
router = UnifiedRouter(
    enable_batching=True,
    enable_quality_check=True,
    enable_snapshots=True,
    enable_monitoring=True
)

# Configure quality thresholds
router.quality.focus_threshold = 7.0  # 0-10
router.quality.result_threshold = 7.0  # 0-10
router.quality.require_fact_verification = True
```

### 3. Concurrency Settings

```python
# For high-throughput scenarios
import asyncio
from src.unified.unified_router import UnifiedRouter

router = UnifiedRouter(enable_batching=True)

# Process multiple requests concurrently
semaphore = asyncio.Semaphore(10)  # Max 10 concurrent

async def process_with_limit(request):
    async with semaphore:
        return await router.route(request)
```

---

## Monitoring & Observability

### 1. Prometheus Metrics

The router exposes metrics at `/metrics`:

```python
from prometheus_client import start_http_server

# Start metrics server
start_http_server(8001)  # Metrics on port 8001

# Access metrics
curl http://localhost:8001/metrics
```

**Key Metrics:**
- `router_requests_total` - Total requests processed
- `router_requests_failed` - Failed requests
- `router_latency_seconds` - Request latency histogram
- `router_cache_hits` - Cache hit count
- `router_cache_misses` - Cache miss count
- `router_quality_retries` - Quality check retries

### 2. Grafana Dashboards

Access Grafana at `http://localhost:3000` (default credentials: admin/admin)

**Import Dashboard:**
1. Go to Dashboards → Import
2. Upload `grafana-dashboard.json` (create from template)
3. Select Prometheus data source

### 3. Logging

Configure structured logging:

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            'timestamp': record.created,
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module
        })

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.root.addHandler(handler)
logging.root.setLevel(logging.INFO)
```

### 4. Health Check Endpoint

Create a simple health check:

```python
from fastapi import FastAPI
from src.unified.unified_router import UnifiedRouter

app = FastAPI()
router = UnifiedRouter()

@app.get("/health")
async def health_check():
    try:
        stats = router.get_stats()
        return {
            "status": "healthy",
            "version": "2.0.0",
            "components": 21,
            "requests_total": stats["unified"]["total_requests"]
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/metrics")
async def metrics():
    return router.get_stats()
```

---

## Integration with MindRest

### Option 1: Direct Integration

```python
# In your MindRest application
from src.unified.unified_router import UnifiedRouter, UnifiedRequest, RoutingStrategy

class MindRestLLMRouter:
    def __init__(self):
        self.router = UnifiedRouter(
            enable_batching=True,
            enable_quality_check=True,
            enable_snapshots=True,
            enable_monitoring=True
        )

    async def route_request(
        self,
        query: str,
        user_context: dict,
        priority: str = "normal"
    ):
        """Route LLM request through unified router"""

        request = UnifiedRequest(
            query=query,
            context=user_context.get("history", ""),
            user_id=user_context.get("user_id"),
            strategy=RoutingStrategy.BALANCED,
            metadata={"app": "mindrest"}
        )

        response = await self.router.route(request)

        return {
            "model": response.model,
            "result": response.result,
            "quality_score": response.quality_score,
            "confidence": response.confidence,
            "passed_checks": response.passed_quality_check
        }
```

### Option 2: API Integration

Run router as microservice:

```python
# router_api.py
from fastapi import FastAPI, HTTPException
from src.unified.unified_router import UnifiedRouter, UnifiedRequest, RoutingStrategy
import uvicorn

app = FastAPI(title="Unified Router API", version="2.0.0")
router = UnifiedRouter()

@app.post("/route")
async def route_request(
    query: str,
    context: str = None,
    strategy: str = "balanced",
    user_id: str = None
):
    try:
        request = UnifiedRequest(
            query=query,
            context=context,
            user_id=user_id,
            strategy=RoutingStrategy[strategy.upper()]
        )

        response = await router.route(request)

        return {
            "model": response.model,
            "result": response.result,
            "quality_score": response.quality_score,
            "confidence": response.confidence,
            "latency": response.latency,
            "passed_quality_check": response.passed_quality_check
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Call from MindRest:

```python
import httpx

async def route_via_api(query: str, context: str = None):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/route",
            json={
                "query": query,
                "context": context,
                "strategy": "balanced"
            }
        )
        return response.json()
```

### Option 3: Docker Integration

Add to your MindRest `docker-compose.yml`:

```yaml
services:
  mindrest:
    build: .
    depends_on:
      - unified-router
    environment:
      - ROUTER_URL=http://unified-router:8000

  unified-router:
    image: unified-router:v2.0
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
```

---

## Build & Package

### 1. Create Distribution Packages

```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Build source and wheel distributions
python setup.py sdist bdist_wheel

# Verify packages
ls -lh dist/
# unified-llm-router-2.0.0.tar.gz
# unified_llm_router-2.0.0-py3-none-any.whl
```

### 2. Install from Package

```bash
# Install from wheel
pip install dist/unified_llm_router-2.0.0-py3-none-any.whl

# Install from source distribution
pip install dist/unified-llm-router-2.0.0.tar.gz
```

### 3. Publish to PyPI (Optional)

```bash
# Install twine
pip install twine

# Upload to PyPI
twine upload dist/*

# Install from PyPI
pip install unified-llm-router
```

---

## Troubleshooting

### Common Issues

#### 1. FAISS Installation Errors

```bash
# Try CPU version first
pip install faiss-cpu

# For GPU support
pip install faiss-gpu
```

#### 2. Torch Installation Issues

```bash
# Install specific torch version
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
```

#### 3. Docker Build Failures

```bash
# Clear Docker cache
docker builder prune -a

# Build with no cache
docker build --no-cache -t unified-router:v2.0 .
```

#### 4. Memory Issues

```bash
# Reduce cache size
router.context_v2.max_cache_size = 500

# Disable snapshots if memory-constrained
router = UnifiedRouter(enable_snapshots=False)
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
router = UnifiedRouter(
    enable_monitoring=True
)
```

### Performance Tuning

```python
# For high-throughput scenarios
router = UnifiedRouter(
    enable_batching=True,  # Enable batching
    enable_quality_check=False,  # Disable if not needed
    enable_snapshots=False,  # Disable if not needed
    enable_monitoring=True  # Keep for observability
)

# Adjust context cache
router.context_v2.max_cache_size = 2000  # Increase cache

# Reduce quality check overhead
router.quality.require_fact_verification = False
```

---

## Support & Resources

- **Documentation**: See `BENCHMARK_RESULTS.md`
- **Tests**: Run `pytest tests/` to verify installation
- **Benchmarks**: Run `python benchmark_unified.py --help`
- **Issues**: Check logs in `logs/` directory
- **Performance**: Run `python compare_v1_v2.py` for metrics

## Next Steps

1. **Configure monitoring** - Set up Prometheus + Grafana
2. **Tune for your workload** - Adjust cache sizes, thresholds
3. **Integrate with your system** - Choose integration method
4. **Set up alerts** - Monitor quality scores, latency, errors
5. **Optimize costs** - Track model usage, adjust strategies

---

**Version**: 2.0.0
**Last Updated**: 2025
**Status**: Production Ready
