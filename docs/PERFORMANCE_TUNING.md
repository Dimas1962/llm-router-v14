# Unified Router v2.0 - Performance Tuning Guide

## Table of Contents

1. [Performance Targets](#performance-targets)
2. [Profiling & Diagnosis](#profiling--diagnosis)
3. [Optimization Strategies](#optimization-strategies)
4. [Component-Specific Tuning](#component-specific-tuning)
5. [Production Configurations](#production-configurations)
6. [Troubleshooting](#troubleshooting)

---

## Performance Targets

### Baseline Metrics

| Metric | Target | Acceptable | Critical |
|--------|--------|------------|----------|
| **Latency (p50)** | < 150ms | < 300ms | > 500ms |
| **Latency (p95)** | < 300ms | < 600ms | > 1000ms |
| **Throughput** | 10+ QPS | 5-10 QPS | < 5 QPS |
| **Cache Hit Rate** | 80%+ | 60-80% | < 60% |
| **Memory Usage** | < 2GB | 2-4GB | > 4GB |
| **CPU Usage** | < 50% | 50-75% | > 75% |
| **Error Rate** | < 0.1% | 0.1-1% | > 1% |

---

## Profiling & Diagnosis

### 1. Measure Current Performance

```python
import time
import asyncio
from src.unified.unified_router import UnifiedRouter, UnifiedRequest

async def profile_request():
    router = UnifiedRouter()

    start = time.time()
    request = UnifiedRequest(query="Test query")
    response = await router.route(request)
    elapsed = time.time() - start

    print(f"Latency: {elapsed*1000:.2f}ms")
    print(f"Model: {response.model}")

    # Get detailed stats
    stats = router.get_stats()
    print(f"Cache hit rate: {stats['context_v2']['cache_hits']/(stats['context_v2']['cache_hits']+stats['context_v2']['cache_misses']):.2%}")
    print(f"Quality checks: {stats['quality']['total_checks']}")

asyncio.run(profile_request())
```

### 2. Identify Bottlenecks

```python
import cProfile
import pstats

# Profile the route function
profiler = cProfile.Profile()
profiler.enable()

# Your routing code
response = await router.route(request)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### 3. Monitor Component Performance

```bash
# Run benchmark to establish baseline
python benchmark_unified.py --queries 100 --concurrent 10

# Monitor in production
python -m prometheus_client --port 8001  # Expose metrics
```

---

## Optimization Strategies

### 1. Reduce Latency

#### a) Disable Non-Essential Features

```python
# Minimal configuration for lowest latency
router = UnifiedRouter(
    enable_batching=False,      # -20ms
    enable_quality_check=False, # -50ms
    enable_snapshots=False,     # -10ms
    enable_monitoring=True      # Keep for observability
)
```

**Expected improvement**: 50-80ms reduction

#### b) Use Faster Models

```python
# Configure Eagle to prefer fast models
router.eagle.models = [
    "qwen2.5-coder-7b",      # Fast
    "deepseek-coder-6.7b",   # Fast
    "gpt-3.5-turbo"          # Medium
]

# Set strategy
request = UnifiedRequest(
    query=query,
    strategy=RoutingStrategy.CASCADE  # Tries fast models first
)
```

#### c) Optimize Context Assembly

```python
# Reduce context budget
router.context_v2.default_budget = 2000  # Was 8000

# Use minimal sizing
from src.v2.context_sizing import ContextSizingStrategy

budget = router.context_sizer.calculate_budget(
    query=query,
    model_name="gpt-3.5-turbo",
    strategy=ContextSizingStrategy.MINIMAL  # 25% of max
)
```

---

### 2. Increase Throughput

#### a) Enable Batching

```python
router = UnifiedRouter(enable_batching=True)

# Configure batch parameters
router.batching.max_batch_size = 20        # Process 20 at once
router.batching.max_wait_time_ms = 50     # Wait up to 50ms
router.batching.worker_count = 4          # Parallel workers
```

**Expected improvement**: 3-5x throughput increase

#### b) Concurrent Processing

```python
import asyncio

async def process_concurrent(queries):
    router = UnifiedRouter()

    # Use semaphore for controlled concurrency
    semaphore = asyncio.Semaphore(20)  # Max 20 concurrent

    async def process_one(query):
        async with semaphore:
            request = UnifiedRequest(query=query)
            return await router.route(request)

    # Process all concurrently
    tasks = [process_one(q) for q in queries]
    return await asyncio.gather(*tasks)

# Process 100 queries concurrently
queries = [f"Query {i}" for i in range(100)]
results = await process_concurrent(queries)
```

#### c) Optimize Python Runtime

```bash
# Use uvloop for faster event loop
pip install uvloop

# In your code:
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
```

**Expected improvement**: 20-30% faster asyncio operations

---

### 3. Improve Cache Hit Rate

#### a) Increase Cache Size

```python
# Larger cache for more hits
router.context_v2.max_cache_size = 5000  # Was 1000

# Configure associative memory
router.memory.max_entries = 10000  # Was default
```

#### b) Cache Warming

```python
# Pre-populate cache with common queries
common_queries = [
    "What is Python?",
    "Explain machine learning",
    "How to use async/await",
    # ... more common queries
]

for query in common_queries:
    request = UnifiedRequest(query=query)
    await router.route(request)

# Now cache is warm
stats = router.get_stats()
print(f"Cache entries: {stats['context_v2']['cache_size']}")
```

#### c) Adjust Similarity Threshold

```python
# More lenient matching = higher hit rate
router.memory.similarity_threshold = 0.7  # Was 0.8

# Trade-off: May return less precise matches
```

---

### 4. Reduce Memory Usage

#### a) Limit Cache Sizes

```python
# For memory-constrained environments
router.context_v2.max_cache_size = 200    # Smaller cache
router.memory.max_entries = 500           # Fewer memory entries
router.episodic.max_episodes = 1000       # Fewer episodes
```

#### b) Disable Snapshots

```python
# Snapshots consume memory
router = UnifiedRouter(enable_snapshots=False)
```

#### c) Use Context Compression

```python
from src.v2.recursive_compressor import RecursiveCompressor, CompressionLevel

compressor = RecursiveCompressor()

# Compress large contexts
if len(context) > 10000:
    result = compressor.compress(
        text=context,
        target_size=5000,
        level=CompressionLevel.MEDIUM
    )
    context = result.compressed_text
```

---

## Component-Specific Tuning

### ContextManager v2

```python
# High throughput
router.context_v2.max_cache_size = 2000
router.context_v2.default_budget = 4000

# Low latency
router.context_v2.max_cache_size = 500
router.context_v2.default_budget = 1000

# Memory constrained
router.context_v2.max_cache_size = 100
router.context_v2.default_budget = 2000
```

### Eagle ELO

```python
# Faster model selection
router.eagle.k_factor = 16  # Lower = less aggressive updates

# Focus on specific models
router.eagle.exclude_models = ["slow-model-1", "slow-model-2"]

# Faster convergence
router.eagle.initial_rating = 1600  # Higher starting point
```

### CARROT

```python
# More cost-conscious
router.carrot.cost_weight = 0.7  # Prioritize cost (vs quality)

# Faster selection
router.carrot.max_candidates = 3  # Evaluate fewer models
```

### SelfCheckSystem

```python
# Stricter quality (slower but better)
router.quality.focus_threshold = 8.0
router.quality.result_threshold = 8.0

# More lenient (faster)
router.quality.focus_threshold = 5.0
router.quality.result_threshold = 5.0
router.quality.require_fact_verification = False
```

### RuntimeAdapter

```python
# More aggressive adaptation
router.runtime.high_load_threshold = 0.6  # Was 0.7
router.runtime.critical_threshold = 0.8   # Was 0.9

# Less sensitive
router.runtime.measurement_interval = 60  # Check every 60s (was 30s)
```

---

## Production Configurations

### 1. High-Throughput Configuration

**Use Case**: API serving, batch processing

```python
router = UnifiedRouter(
    enable_batching=True,        # Essential for throughput
    enable_quality_check=False,  # Disable for speed
    enable_snapshots=False,      # Not needed
    enable_monitoring=True       # Keep for metrics
)

# Optimize batch settings
router.batching.max_batch_size = 50
router.batching.max_wait_time_ms = 100
router.batching.worker_count = 8

# Large cache
router.context_v2.max_cache_size = 10000

# Minimal context
router.context_v2.default_budget = 2000
```

**Expected**: 20+ QPS, < 300ms latency

---

### 2. High-Quality Configuration

**Use Case**: Critical applications, research

```python
router = UnifiedRouter(
    enable_batching=False,       # Sequential for consistency
    enable_quality_check=True,   # Essential
    enable_snapshots=True,       # For debugging
    enable_monitoring=True
)

# Strict quality
router.quality.focus_threshold = 8.0
router.quality.result_threshold = 8.0
router.quality.require_fact_verification = True

# Quality-focused routing
request = UnifiedRequest(
    query=query,
    strategy=RoutingStrategy.QUALITY_FOCUSED
)

# Generous context
router.context_v2.default_budget = 8000
```

**Expected**: 5-10 QPS, < 500ms latency, high quality

---

### 3. Balanced Configuration

**Use Case**: General production use

```python
router = UnifiedRouter(
    enable_batching=True,        # Moderate batching
    enable_quality_check=True,   # With lenient thresholds
    enable_snapshots=False,      # Not needed
    enable_monitoring=True
)

# Moderate batching
router.batching.max_batch_size = 10
router.batching.max_wait_time_ms = 50

# Balanced quality
router.quality.focus_threshold = 6.0
router.quality.result_threshold = 6.0

# Balanced strategy
request = UnifiedRequest(
    query=query,
    strategy=RoutingStrategy.BALANCED
)

# Moderate context
router.context_v2.default_budget = 4000
```

**Expected**: 10-15 QPS, < 200ms latency, good quality

---

### 4. Low-Resource Configuration

**Use Case**: Edge devices, resource-constrained environments

```python
router = UnifiedRouter(
    enable_batching=False,       # Too much overhead
    enable_quality_check=False,  # Disable
    enable_snapshots=False,      # Disable
    enable_monitoring=False      # Disable to save memory
)

# Minimal cache
router.context_v2.max_cache_size = 50
router.memory.max_entries = 100

# Fast models only
router.cascade.tiers = {
    "fast": ["qwen2.5-coder-7b"]
}

# Minimal context
router.context_v2.default_budget = 1000

# Cost-aware routing
request = UnifiedRequest(
    query=query,
    strategy=RoutingStrategy.COST_AWARE
)
```

**Expected**: 5 QPS, < 150ms latency, < 1GB memory

---

## Troubleshooting

### Issue: High Latency

**Diagnosis:**
```python
stats = router.get_stats()
print(f"Avg latency: {stats['unified']['avg_latency']:.3f}s")

# Check quality retries
if stats['unified']['quality_retries'] > 0:
    print("Quality checks causing retries")
```

**Solutions:**
1. Disable quality checks: `enable_quality_check=False`
2. Lower quality thresholds
3. Use CASCADE strategy (tries fast models first)
4. Reduce context budget

---

### Issue: Low Throughput

**Diagnosis:**
```bash
python benchmark_unified.py --queries 100 --concurrent 10
```

**Solutions:**
1. Enable batching: `enable_batching=True`
2. Increase concurrency limit
3. Use uvloop for faster async
4. Scale horizontally (multiple instances)

---

### Issue: Low Cache Hit Rate

**Diagnosis:**
```python
stats = router.get_stats()
hits = stats['context_v2']['cache_hits']
misses = stats['context_v2']['cache_misses']
rate = hits / (hits + misses) if (hits + misses) > 0 else 0
print(f"Cache hit rate: {rate:.2%}")
```

**Solutions:**
1. Increase cache size: `max_cache_size = 2000`
2. Lower similarity threshold: `similarity_threshold = 0.7`
3. Warm cache with common queries
4. Check query diversity (too diverse = low hits)

---

### Issue: High Memory Usage

**Diagnosis:**
```python
import psutil
process = psutil.Process()
print(f"Memory: {process.memory_info().rss / 1024 / 1024:.0f}MB")
```

**Solutions:**
1. Reduce cache sizes
2. Disable snapshots: `enable_snapshots=False`
3. Use compression for large contexts
4. Clear cache periodically: `router.context_v2.clear_cache()`

---

### Issue: Quality Check Failures

**Diagnosis:**
```python
stats = router.get_stats()
if stats['quality']['total_checks'] > 0:
    fail_rate = stats['quality']['failed'] / stats['quality']['total_checks']
    print(f"Quality fail rate: {fail_rate:.2%}")
```

**Solutions:**
1. Lower thresholds: `focus_threshold = 5.0`
2. Disable fact verification: `require_fact_verification = False`
3. Use better models (QUALITY_FOCUSED strategy)
4. Improve prompt quality

---

## Performance Monitoring

### Prometheus Metrics

```python
from prometheus_client import start_http_server, Counter, Histogram

# Start metrics server
start_http_server(8001)

# Define custom metrics
requests_total = Counter('router_requests', 'Total requests')
latency_histogram = Histogram('router_latency', 'Request latency')

# Instrument code
@latency_histogram.time()
async def instrumented_route(request):
    requests_total.inc()
    return await router.route(request)
```

### Logging

```python
import logging

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Log slow requests
@latency_histogram.time()
async def logged_route(request):
    start = time.time()
    response = await router.route(request)
    elapsed = time.time() - start

    if elapsed > 0.5:  # Slow request
        logging.warning(f"Slow request: {elapsed:.3f}s - {request.query[:50]}")

    return response
```

---

**Version**: 2.0.0
**Last Updated**: 2025
