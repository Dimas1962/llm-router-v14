# Unified Router v2.0 - API Reference

## Core Classes

### UnifiedRouter

Main entry point for LLM routing with 21 integrated components.

#### Constructor

```python
UnifiedRouter(
    enable_batching: bool = True,
    enable_quality_check: bool = True,
    enable_snapshots: bool = True,
    enable_monitoring: bool = True
)
```

**Parameters:**
- `enable_batching`: Enable request batching for throughput
- `enable_quality_check`: Enable quality verification and auto-retry
- `enable_snapshots`: Enable state capture
- `enable_monitoring`: Enable metrics collection

**Example:**
```python
from src.unified.unified_router import UnifiedRouter

router = UnifiedRouter(
    enable_batching=True,
    enable_quality_check=True
)
```

---

#### route()

```python
async def route(
    request: UnifiedRequest
) -> UnifiedResponse
```

Main routing method. Processes request through 7-step pipeline.

**Parameters:**
- `request`: UnifiedRequest object

**Returns:**
- `UnifiedResponse` with model selection, result, quality score

**Example:**
```python
from src.unified.unified_router import UnifiedRequest, RoutingStrategy

request = UnifiedRequest(
    query="Explain machine learning",
    strategy=RoutingStrategy.BALANCED,
    user_id="user123"
)

response = await router.route(request)

print(f"Model: {response.model}")
print(f"Result: {response.result}")
print(f"Quality: {response.quality_score}")
print(f"Confidence: {response.confidence}")
```

---

#### get_stats()

```python
def get_stats() -> Dict[str, Any]
```

Get comprehensive statistics from all 21 components.

**Returns:**
- Dictionary with nested stats for each component

**Example:**
```python
stats = router.get_stats()

print(f"Total requests: {stats['unified']['total_requests']}")
print(f"Cache hits: {stats['context_v2']['cache_hits']}")
print(f"Quality pass rate: {stats['quality']['pass_rate']:.2%}")
```

---

### UnifiedRequest

Request data transfer object.

```python
@dataclass
class UnifiedRequest:
    query: str
    context: Optional[str] = None
    user_id: Optional[str] = None
    priority: Priority = Priority.NORMAL
    strategy: RoutingStrategy = RoutingStrategy.BALANCED
    max_cost: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Example:**
```python
request = UnifiedRequest(
    query="Debug this Python code",
    context="Previously discussed async/await",
    user_id="user456",
    strategy=RoutingStrategy.QUALITY_FOCUSED,
    metadata={"source": "ide"}
)
```

---

### UnifiedResponse

Response data transfer object.

```python
@dataclass
class UnifiedResponse:
    model: str                      # Selected model
    result: str                     # Generated response
    quality_score: float            # 0-1 quality score
    cost: float                     # Estimated cost
    latency: float                  # Response time (seconds)
    confidence: float               # 0-1 routing confidence
    reasoning: str                  # Routing decision explanation
    passed_quality_check: bool      # Quality verification result
    snapshot_id: Optional[str]      # State snapshot ID
    metadata: Dict[str, Any]        # Additional metadata
```

---

### RoutingStrategy (Enum)

```python
class RoutingStrategy(Enum):
    QUALITY_FOCUSED = "quality_focused"  # Eagle ELO
    COST_AWARE = "cost_aware"            # CARROT
    BALANCED = "balanced"                # Hybrid
    CASCADE = "cascade"                  # Multi-tier
```

---

### Priority (Enum)

```python
class Priority(Enum):
    URGENT = "urgent"    # Process immediately
    HIGH = "high"        # Priority processing
    NORMAL = "normal"    # Standard queue
    LOW = "low"          # Background processing
```

---

## Component APIs

### Eagle ELO (Quality-focused routing)

```python
from router.eagle import EagleELO

eagle = EagleELO(memory=memory_instance)

# Get best model
model, score = eagle.get_best_model(
    query="Complex reasoning task",
    k=5,                      # Top-k candidates
    filter_task_type=None,    # Optional filter
    exclude_models=[]         # Exclude specific models
)

# Update ratings
eagle.update_rating(
    winner_model="gpt-4",
    loser_model="gpt-3.5-turbo",
    actual_result=1.0  # 1.0 = winner won, 0.0 = loser won
)
```

---

### CARROT (Cost-aware routing)

```python
from router.carrot import CARROT

carrot = CARROT()

# Select cost-optimal model
model, predictions = carrot.select(
    query="Simple query",
    budget=0.5,          # Max cost threshold
    model_ids=None,      # Specific models only
    task_type="general",
    complexity=0.3,
    context_size=0
)

print(f"Selected: {model}")
print(f"Quality: {predictions['quality']}")
print(f"Cost: {predictions['cost']}")
```

---

### AssociativeMemory (Semantic caching)

```python
from router.memory import AssociativeMemory

memory = AssociativeMemory()

# Search for similar queries
results = memory.search(
    query="What is Python?",
    k=3,            # Top-k results
    threshold=0.8   # Similarity threshold
)

# Store new result
memory.store(
    query="What is Python?",
    result="Python is a programming language...",
    model="gpt-4",
    metadata={"timestamp": "2025-01-01"}
)
```

---

### SelfCheckSystem (Quality verification)

```python
from src.v2.self_check import SelfCheckSystem

quality = SelfCheckSystem(
    focus_threshold=6.0,      # Min focus score
    result_threshold=6.0,     # Min result score
    require_fact_verification=True
)

# Check quality
result = quality.check(
    query="Explain quantum computing",
    result="Quantum computing uses...",
    context="Previous discussion about physics"
)

print(f"Passed: {result.passed}")
print(f"Focus score: {result.focus_score}/10")
print(f"Result score: {result.result_score}/10")
print(f"Issues: {result.issues}")
```

---

### ContextManager v2 (Advanced caching)

```python
from src.v2.context_manager import ContextManager

ctx = ContextManager(max_cache_size=1000, default_budget=8000)

# Get context with caching
context_response = await ctx.get_context(
    query="Explain neural networks",
    max_tokens=4000,
    complexity=0.7
)

print(f"Context: {context_response.context}")
print(f"Fragments: {context_response.fragment_count}")
print(f"Tokens: {context_response.token_count}")

# Get statistics
stats = ctx.get_statistics()
print(f"Cache hits: {stats['cache_hits']}")
print(f"Cache misses: {stats['cache_misses']}")
```

---

### DynamicContextSizer (Token budgeting)

```python
from src.v2.context_sizing import DynamicContextSizer, ContextSizingStrategy

sizer = DynamicContextSizer()

# Calculate budget
budget = sizer.calculate_budget(
    query="Long complex query...",
    model_name="gpt-4",
    complexity=0.8,
    history=[],
    strategy=ContextSizingStrategy.ADAPTIVE
)

print(f"Total tokens: {budget.total_tokens}")
print(f"Query tokens: {budget.query_tokens}")
print(f"Context tokens: {budget.context_tokens}")
print(f"Response reserve: {budget.response_reserve}")
```

---

### StateSnapshotSystem (State management)

```python
from src.v2.state_snapshot import StateSnapshotSystem

snapshot_system = StateSnapshotSystem()

# Capture snapshot
components = router.get_all_components()
snapshot = snapshot_system.capture_snapshot(
    components=components,
    metadata={"trigger": "manual"}
)

# Serialize
json_data = snapshot_system.serialize_json(snapshot)

# Restore from snapshot
snapshot_system.restore_snapshot(snapshot, components)

# Rollback
restored = snapshot_system.rollback(components, steps=2)
```

---

## Usage Examples

### Basic Routing

```python
import asyncio
from src.unified.unified_router import (
    UnifiedRouter, UnifiedRequest, RoutingStrategy
)

async def main():
    router = UnifiedRouter()

    request = UnifiedRequest(
        query="Explain async/await in Python",
        strategy=RoutingStrategy.BALANCED
    )

    response = await router.route(request)

    print(f"Model: {response.model}")
    print(f"Response: {response.result}")
    print(f"Quality: {response.quality_score:.2f}")
    print(f"Latency: {response.latency:.3f}s")

asyncio.run(main())
```

---

### Quality-Focused Routing

```python
request = UnifiedRequest(
    query="Write a complex algorithm",
    strategy=RoutingStrategy.QUALITY_FOCUSED,
    context="Need high-quality implementation"
)

response = await router.route(request)
# Uses Eagle ELO to select highest-quality model
```

---

### Cost-Aware Routing

```python
request = UnifiedRequest(
    query="Simple factual question",
    strategy=RoutingStrategy.COST_AWARE,
    max_cost=0.1  # Low budget
)

response = await router.route(request)
# Uses CARROT to find cost-optimal model
```

---

### Cascade Routing

```python
request = UnifiedRequest(
    query="Complex reasoning task",
    strategy=RoutingStrategy.CASCADE
)

response = await router.route(request)
# Tries fast models first, escalates if needed
```

---

### Priority Handling

```python
from src.v2.batching_layer import Priority

urgent_request = UnifiedRequest(
    query="Critical production issue",
    priority=Priority.URGENT,
    strategy=RoutingStrategy.QUALITY_FOCUSED
)

response = await router.route(urgent_request)
# Processed immediately, skips batch queue
```

---

### Concurrent Processing

```python
async def process_batch():
    router = UnifiedRouter()

    queries = [
        "Query 1",
        "Query 2",
        "Query 3"
    ]

    # Process concurrently
    tasks = []
    for query in queries:
        request = UnifiedRequest(query=query)
        tasks.append(router.route(request))

    responses = await asyncio.gather(*tasks)

    for i, response in enumerate(responses):
        print(f"Query {i+1}: {response.model}")

asyncio.run(process_batch())
```

---

### Statistics Monitoring

```python
# After processing requests
stats = router.get_stats()

# Unified router stats
print(f"Requests: {stats['unified']['total_requests']}")
print(f"Success rate: {stats['unified']['successful_requests'] / stats['unified']['total_requests']:.2%}")

# Context manager stats
hit_rate = stats['context_v2']['cache_hits'] / (stats['context_v2']['cache_hits'] + stats['context_v2']['cache_misses'])
print(f"Cache hit rate: {hit_rate:.2%}")

# Quality check stats
if stats['quality']['total_checks'] > 0:
    pass_rate = stats['quality']['passed'] / stats['quality']['total_checks']
    print(f"Quality pass rate: {pass_rate:.2%}")
```

---

### State Management

```python
# Capture state before critical operation
components = router.get_all_components()
snapshot = router.snapshot.capture_snapshot(components)

# Perform operation
try:
    response = await router.route(complex_request)
except Exception as e:
    # Rollback on failure
    router.snapshot.restore_snapshot(snapshot, components)
    print("Rolled back to previous state")
```

---

## Error Handling

```python
from src.unified.unified_router import UnifiedRouter, UnifiedRequest

async def safe_route():
    router = UnifiedRouter()

    try:
        request = UnifiedRequest(query="Test query")
        response = await router.route(request)

        if not response.passed_quality_check:
            print(f"Warning: Quality check failed")
            print(f"Quality score: {response.quality_score}")

        return response

    except Exception as e:
        print(f"Routing failed: {e}")
        stats = router.get_stats()
        print(f"Failed requests: {stats['unified']['failed_requests']}")
        raise
```

---

## Advanced Configuration

### Custom Quality Thresholds

```python
router = UnifiedRouter(enable_quality_check=True)

# Configure thresholds
router.quality.focus_threshold = 7.0      # Stricter focus requirement
router.quality.result_threshold = 8.0     # Stricter result quality
router.quality.require_fact_verification = True
```

### Cache Configuration

```python
router = UnifiedRouter()

# Adjust cache size
router.context_v2.max_cache_size = 2000  # Larger cache

# Clear cache
router.context_v2.clear_cache()
```

### Reset Statistics

```python
# Reset all component statistics
router.reset_stats()

# Verify reset
stats = router.get_stats()
assert stats['unified']['total_requests'] == 0
```

---

**Version**: 2.0.0
**Last Updated**: 2025
