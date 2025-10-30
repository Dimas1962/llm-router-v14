# Unified Router v2.0 - Architecture Overview

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Component Architecture](#component-architecture)
3. [v1.4 Components (8)](#v14-components)
4. [v2.0 Components (10)](#v20-components)
5. [Data Flow](#data-flow)
6. [Integration Points](#integration-points)
7. [Design Patterns](#design-patterns)
8. [Scalability](#scalability)

---

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT APPLICATION                          │
│                          (MindRest, etc.)                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      UNIFIED ROUTER v2.0                           │
│                         (21 Components)                            │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    REQUEST PIPELINE                           │  │
│  │  1. Runtime Adaptation    → System load assessment          │  │
│  │  2. Context Sizing        → Token budget calculation         │  │
│  │  3. Context Assembly      → Context retrieval & caching      │  │
│  │  4. Model Routing         → Strategy-based selection         │  │
│  │  5. Execution             → LLM API call                     │  │
│  │  6. Quality Check         → Verification & retry             │  │
│  │  7. State Snapshot        → System state capture             │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        LLM PROVIDERS                               │
│   OpenAI │ Anthropic │ Cohere │ DeepSeek │ Qwen │ Others          │
└─────────────────────────────────────────────────────────────────────┘
```

### Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                        │
│  • UnifiedRouter (main interface)                           │
│  • UnifiedRequest/Response (DTOs)                          │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│                   ORCHESTRATION LAYER                       │
│  • Pipeline coordination                                    │
│  • Strategy selection                                       │
│  • Error handling & retry                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│                   COMPONENT LAYER                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   v1.4       │  │    v2.0      │  │  Utilities   │     │
│  │  8 comps     │  │  10 comps    │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────┐
│                    INFRASTRUCTURE LAYER                     │
│  • FAISS (vector search)                                    │
│  • Redis (caching)                                          │
│  • Prometheus (metrics)                                     │
│  • File system (snapshots)                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### Component Dependency Graph

```
                    ┌──────────────────┐
                    │ UnifiedRouter    │
                    │  (Orchestrator)  │
                    └────────┬─────────┘
                             │
        ┏━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━┓
        ▼                                          ▼
┌──────────────────┐                    ┌──────────────────┐
│  v1.4 Components │                    │  v2.0 Components │
│   (8 modules)    │                    │   (10 modules)   │
└────────┬─────────┘                    └────────┬─────────┘
         │                                       │
    ┌────┴────┐                            ┌────┴────┐
    │         │                            │         │
┌───▼──┐  ┌──▼───┐                    ┌───▼──┐  ┌──▼───┐
│Eagle │  │CARROT│                    │Ctx   │  │Self  │
│ELO   │  │      │                    │Mgr   │  │Check │
└──────┘  └──────┘                    └──────┘  └──────┘
    │         │                            │         │
┌───▼──┐  ┌──▼───┐                    ┌───▼──┐  ┌──▼───┐
│Memory│  │Cascade│                   │Runtime│ │Ctx   │
│      │  │Router│                    │Adapt  │  │Sizing│
└──────┘  └──────┘                    └──────┘  └──────┘
```

### Component Categories

| Category | v1.4 | v2.0 | Purpose |
|----------|------|------|---------|
| **Core Routing** | RouterCore | - | Central routing logic |
| **Model Selection** | Eagle, CARROT, Cascade | RuntimeAdapter | Model/strategy selection |
| **Memory** | AssociativeMemory, EpisodicMemory | - | Query caching, history |
| **Context Management** | ContextManager | ContextManager, DynamicContextSizer, HierarchicalPruner, RecursiveCompressor | Context assembly & optimization |
| **Quality** | - | SelfCheckSystem | Quality verification |
| **Multi-Turn** | MultiRoundRouter | - | Conversation handling |
| **Optimization** | - | BatchingLayer | Throughput optimization |
| **Code Analysis** | - | ASTAnalyzer | Code structure parsing |
| **Environment** | - | EnvironmentPrompter | System-aware prompting |
| **State Management** | - | StateSnapshotSystem | State capture/restore |
| **Monitoring** | PerformanceTracker | - | Metrics & telemetry |

---

## v1.4 Components

### 1. RouterCore
**Purpose**: Central routing orchestration

**Key Features**:
- Hybrid Eagle + CARROT routing
- Memory integration
- Context management
- User session tracking

**API**:
```python
async def route(
    query: str,
    session_history: List[Message] = None,
    budget: float = None,
    user_id: str = None
) -> RoutingResult
```

**Decision Logic**:
1. Load user context from memory
2. Consult Eagle ELO for quality model
3. Consult CARROT for cost-optimal model
4. Balance quality vs cost based on query complexity
5. Return routing decision with confidence

---

### 2. EagleELO
**Purpose**: Quality-focused model ranking via ELO ratings

**Key Features**:
- ELO rating system (starts at 1500)
- K-factor adaptation (32 default)
- Win/loss tracking
- Dynamic rating updates

**Algorithm**:
```
Expected = 1 / (1 + 10^((Rating_B - Rating_A) / 400))
New_Rating = Old_Rating + K × (Actual - Expected)
```

**API**:
```python
def get_best_model(
    query: str,
    k: int = 5,
    filter_task_type: str = None,
    exclude_models: List[str] = None
) -> Tuple[str, float]

def update_rating(
    winner_model: str,
    loser_model: str,
    actual_result: float = 1.0
)
```

**Use Cases**:
- Quality-focused routing (RoutingStrategy.QUALITY_FOCUSED)
- A/B testing with quality tracking
- Model performance evolution

---

### 3. CARROT
**Purpose**: Cost-aware routing optimization (Pareto frontier)

**Key Features**:
- Quality prediction (0-1 scale)
- Cost prediction (time + tokens)
- Pareto optimal model selection
- Budget constraint handling

**Algorithm**:
```
Score = Quality / Cost  (within budget)
Select: max(Score) where Cost ≤ Budget
```

**API**:
```python
def select(
    query: str,
    budget: float = 1.0,
    model_ids: List[str] = None,
    task_type: str = None,
    complexity: float = 0.5,
    context_size: int = 0
) -> Tuple[str, Dict[str, float]]
```

**Predictors**:
- **Quality**: Text complexity, query type, model capabilities
- **Cost**: Token count × price + latency penalty

---

### 4. AssociativeMemory
**Purpose**: FAISS-based semantic query caching

**Key Features**:
- Sentence-Transformers embeddings (384-dim)
- FAISS IndexFlatL2 for similarity search
- Automatic cache invalidation
- Semantic deduplication

**Architecture**:
```
Query → Encoder → Embedding (384-dim) → FAISS Search → Cached Results
                                            ↓
                                      If miss: Execute & Store
```

**API**:
```python
def search(
    query: str,
    k: int = 3,
    threshold: float = 0.8
) -> List[Dict[str, Any]]

def store(
    query: str,
    result: str,
    model: str,
    metadata: Dict[str, Any] = None
)
```

**Performance**:
- ~5ms lookup time
- 80%+ cache hit rate (after warmup)
- 1000+ entries capacity

---

### 5. EpisodicMemory
**Purpose**: Historical routing decision tracking

**Key Features**:
- Episode-based storage
- Success/failure tracking
- Pattern analysis
- Query similarity search

**Data Structure**:
```python
Episode = {
    'timestamp': datetime,
    'query': str,
    'query_embedding': np.ndarray,
    'selected_model': str,
    'confidence': float,
    'success': bool,
    'task_type': str,
    'complexity': float,
    'metadata': dict
}
```

**API**:
```python
def add_episode(
    query: str,
    selected_model: str,
    confidence: float,
    success: bool,
    task_type: str,
    complexity: float,
    metadata: Dict[str, Any] = None
)

def search_similar(
    query: str,
    k: int = 5
) -> List[Episode]

def get_successful_patterns(
    task_type: str = None,
    min_confidence: float = 0.8
) -> Dict[str, Any]
```

---

### 6. CascadeRouter
**Purpose**: Multi-tier routing with escalation

**Key Features**:
- 3-tier model hierarchy (fast → balanced → powerful)
- Confidence-based escalation
- Automatic retry with better models
- Cost tracking per tier

**Tiers**:
1. **FAST** (Tier 1): qwen2.5-coder-7b, deepseek-coder-6.7b
2. **BALANCED** (Tier 2): claude-3-haiku, gpt-3.5-turbo
3. **POWERFUL** (Tier 3): gpt-4, claude-3-opus, qwen3-next-80b

**Decision Logic**:
```python
1. Start with FAST tier
2. If confidence < threshold (0.8):
   → Escalate to BALANCED
3. If still < threshold:
   → Escalate to POWERFUL
4. Return result with escalation metadata
```

**API**:
```python
def route(
    query: str,
    task_type: str = "general",
    complexity: float = 0.5,
    budget: float = None,
    context_size: int = 0
) -> CascadeResult
```

---

### 7. MultiRoundRouter
**Purpose**: Multi-turn conversation handling

**Key Features**:
- Session management (up to 1000 concurrent)
- Context accumulation
- Model consistency tracking
- Conversation history compression

**Session Lifecycle**:
```
Create → Add messages → Route → Update context → Repeat
   ↓
After N rounds or timeout: Archive & cleanup
```

**API**:
```python
def create_session(
    user_id: str,
    initial_context: str = None
) -> str  # session_id

def add_message(
    session_id: str,
    role: str,
    content: str
)

async def route_turn(
    session_id: str,
    query: str
) -> RoutingResult
```

---

### 8. PerformanceTracker
**Purpose**: Metrics collection and monitoring

**Key Features**:
- Prometheus metrics export
- Request tracking
- Latency histograms
- Error rate monitoring

**Metrics Exposed**:
- `router_requests_total` (counter)
- `router_requests_failed` (counter)
- `router_latency_seconds` (histogram)
- `router_cache_hits` (counter)
- `router_model_selections` (counter by model)

---

## v2.0 Components

### 1. ContextManager (v2)
**Purpose**: Advanced context assembly with intelligent caching

**Improvements over v1**:
- LRU cache with configurable size
- Fragment-based assembly
- Priority-based retrieval
- Cache hit/miss statistics

**API**:
```python
async def get_context(
    query: str,
    max_tokens: int = 8000,
    complexity: float = 0.5
) -> ContextResponse

def clear_cache()
def get_statistics() -> Dict[str, Any]
```

**Cache Strategy**:
- Hash query + max_tokens as key
- Store assembled context
- TTL: 1 hour
- Max entries: 1000

---

### 2. RuntimeAdapter
**Purpose**: Dynamic strategy adaptation based on system load

**Load Levels**:
- **LOW** (< 30% CPU): Quality-focused routing
- **NORMAL** (30-70%): User-requested strategy
- **HIGH** (70-90%): Cost-aware fallback
- **CRITICAL** (> 90%): Emergency throttling

**Metrics Monitored**:
```python
SystemMetrics = {
    'cpu_percent': float,      # 0-100
    'memory_percent': float,   # 0-100
    'disk_io': float,         # ops/sec
    'network_io': float,      # MB/sec
    'active_requests': int,
    'queue_depth': int
}
```

**API**:
```python
def measure_system_load() -> SystemMetrics

async def adapt_strategy(
    request: UnifiedRequest,
    metrics: SystemMetrics
) -> RoutingStrategy
```

---

### 3. SelfCheckSystem
**Purpose**: Automated quality verification with smart retry

**Check Types**:
1. **Focus Score** (0-10): Query relevance
2. **Result Score** (0-10): Output quality
3. **Fact Verification**: Context consistency

**Scoring Heuristics**:
```python
Focus Score = {
    10: All query aspects addressed
    7-9: Mostly relevant
    4-6: Partially relevant
    0-3: Off-topic or irrelevant
}

Result Score = {
    10: Excellent quality, complete
    7-9: Good quality, minor issues
    4-6: Acceptable, notable issues
    0-3: Poor quality or incomplete
}
```

**API**:
```python
def check(
    query: str,
    result: str,
    context: str = None
) -> SelfCheckResult

# SelfCheckResult
{
    'focus_score': float,      # 0-10
    'result_score': float,     # 0-10
    'fact_verified': bool,
    'passed': bool,            # Overall pass/fail
    'issues': List[str],       # Detected problems
    'metadata': dict
}
```

**Retry Logic**:
```
1. Check quality
2. If failed and retries < 3:
   → Select better model (e.g., GPT-3.5 → GPT-4)
   → Re-execute request
   → Re-check quality
3. Return best result
```

---

### 4. DynamicContextSizer
**Purpose**: Intelligent token budget allocation

**Strategies**:
- **MINIMAL**: 25% of model context window
- **BALANCED**: 50% of model context window
- **MAXIMAL**: 80% of model context window
- **ADAPTIVE**: Dynamic based on query complexity

**Budget Calculation**:
```python
base_budget = model_max_tokens × strategy_factor
adjusted = base_budget × (1 + complexity_factor)
reserved = adjusted × 0.9  # 10% safety margin
```

**API**:
```python
def calculate_budget(
    query: str,
    model_name: str,
    complexity: float = 0.5,
    history: List[Message] = None,
    strategy: ContextSizingStrategy = ADAPTIVE
) -> ContextBudget

# ContextBudget
{
    'total_tokens': int,
    'query_tokens': int,
    'context_tokens': int,
    'response_reserve': int,
    'strategy_used': str
}
```

---

### 5. HierarchicalPruner
**Purpose**: Multi-level context compression

**Pruning Levels**:
1. **L1 - Aggressive** (80% reduction): Keep only critical info
2. **L2 - Moderate** (50% reduction): Balance detail and size
3. **L3 - Light** (20% reduction): Remove only redundancy

**Techniques**:
- Sentence importance scoring
- Redundancy elimination
- Keyword preservation
- Structure-aware compression

**API**:
```python
def prune(
    text: str,
    target_tokens: int,
    level: PruningLevel = MODERATE,
    preserve_keywords: List[str] = None
) -> PruningResult

# PruningResult
{
    'pruned_text': str,
    'original_tokens': int,
    'pruned_tokens': int,
    'reduction_ratio': float,
    'preserved_keywords': List[str],
    'level_used': str
}
```

---

### 6. BatchingLayer
**Purpose**: Request batching for throughput optimization

**Features**:
- Priority-based queueing (URGENT, HIGH, NORMAL, LOW)
- Dynamic batch sizing
- Timeout handling
- Throughput monitoring

**Batch Strategy**:
```
Collect requests until:
  1. Batch size reaches target (e.g., 10)
  OR
  2. Timeout expires (e.g., 100ms)
  OR
  3. High-priority request arrives

Then: Process batch concurrently
```

**API**:
```python
async def add_request(
    request: UnifiedRequest,
    priority: Priority = NORMAL
) -> str  # batch_id

async def process_batch(
    batch_id: str
) -> List[UnifiedResponse]

def get_stats() -> Dict[str, Any]
```

---

### 7. ASTAnalyzer
**Purpose**: Code structure analysis for coding queries

**Capabilities**:
- Python/JavaScript/Go/Rust parsing
- Function/class extraction
- Dependency analysis
- Complexity metrics

**Use Cases**:
- Code review routing
- Bug fix context assembly
- Refactoring suggestions
- Test generation

**API**:
```python
def analyze(
    code: str,
    language: str = "python"
) -> ASTAnalysis

# ASTAnalysis
{
    'language': str,
    'functions': List[Function],
    'classes': List[Class],
    'imports': List[str],
    'complexity': int,
    'lines_of_code': int,
    'issues': List[str]
}
```

---

### 8. RecursiveCompressor
**Purpose**: Advanced text compression with structure preservation

**Algorithm**:
```
1. Parse document structure (sections, paragraphs)
2. Score each segment by importance
3. Recursively compress low-importance segments
4. Preserve high-importance content
5. Reconstruct compressed document
```

**Compression Levels**:
- **Light**: Summarize only verbose sections
- **Medium**: Aggressive sentence reduction
- **Heavy**: Extreme compression with outline extraction

**API**:
```python
def compress(
    text: str,
    target_size: int,
    level: CompressionLevel = MEDIUM,
    preserve_structure: bool = True
) -> CompressionResult

# CompressionResult
{
    'compressed_text': str,
    'original_size': int,
    'compressed_size': int,
    'compression_ratio': float,
    'quality_score': float  # 0-1
}
```

---

### 9. EnvironmentPrompter
**Purpose**: System-aware prompt augmentation

**Context Sources**:
- OS version and architecture
- Python version
- Available system resources
- Installed packages
- Working directory
- Git repository info

**Prompt Augmentation**:
```python
Original: "Debug this Python code"

Augmented: """
Debug this Python code

System Environment:
- OS: macOS 14.0 (arm64)
- Python: 3.11.5
- Available packages: numpy, pandas, requests
- Working dir: /Users/home/project
- Git: main branch, 5 uncommitted changes
"""
```

**API**:
```python
def get_context_summary() -> str
def add_environment_context(prompt: str) -> str
def get_stats() -> Dict[str, Any]
```

---

### 10. StateSnapshotSystem
**Purpose**: Complete system state capture and restoration

**Snapshot Contents**:
- All component states
- Configuration parameters
- Statistics counters
- Cache contents (optional)
- Memory state

**Use Cases**:
- Debugging routing issues
- A/B testing with rollback
- System migration
- Disaster recovery

**Snapshot Format**:
```python
SystemSnapshot = {
    'version': str,
    'timestamp': datetime,
    'components': {
        'component_name': {
            'stats': dict,
            'config': dict,
            'metadata': dict
        },
        ...
    },
    'metadata': dict
}
```

**API**:
```python
def capture_snapshot(
    components: Dict[str, Any],
    metadata: Dict[str, Any] = None
) -> SystemSnapshot

def restore_snapshot(
    snapshot: SystemSnapshot,
    components: Dict[str, Any]
) -> bool

def rollback(
    components: Dict[str, Any],
    steps: int = 1
) -> Optional[SystemSnapshot]

# Serialization
def serialize_json(snapshot: SystemSnapshot) -> str
def compress_snapshot(snapshot: SystemSnapshot) -> bytes
```

---

## Data Flow

### Request Processing Pipeline

```
1. REQUEST ARRIVAL
   ↓
   UnifiedRequest {query, context, strategy, priority}
   ↓
2. RUNTIME ADAPTATION
   ↓
   SystemMetrics → Load assessment → Strategy adjustment
   ↓
3. CONTEXT SIZING
   ↓
   Query complexity → Token budget calculation
   ↓
4. CONTEXT ASSEMBLY
   ↓
   Memory search → Fragment retrieval → Cache check → Assembly
   ↓
5. MODEL ROUTING
   ↓
   ┌─────────────────────┬─────────────────────┬─────────────────────┐
   │  QUALITY_FOCUSED    │    COST_AWARE       │     CASCADE         │
   │  Eagle ELO          │    CARROT           │    Tier routing     │
   └─────────────────────┴─────────────────────┴─────────────────────┘
   ↓
6. EXECUTION
   ↓
   LLM API call → Response generation
   ↓
7. QUALITY CHECK
   ↓
   SelfCheck → Pass? → Return
                 ↓ Fail
              Retry with better model
   ↓
8. STATE SNAPSHOT
   ↓
   Capture system state (optional)
   ↓
9. RESPONSE DELIVERY
   ↓
   UnifiedResponse {model, result, quality_score, confidence}
```

### Memory Data Flow

```
Query Arrival
    ↓
    ├─→ AssociativeMemory.search()
    │       ↓
    │   Embedding generation
    │       ↓
    │   FAISS similarity search
    │       ↓
    │   Cache hit? → Return cached result
    │       ↓ miss
    ├─→ EpisodicMemory.search_similar()
    │       ↓
    │   Find historical successful patterns
    │       ↓
    │   Influence routing decision
    │       ↓
    └─→ Execute new request
            ↓
        Store in both memories
```

---

## Integration Points

### External Integrations

1. **LLM Providers**
   - OpenAI (GPT-3.5, GPT-4)
   - Anthropic (Claude)
   - DeepSeek, Qwen, Cohere

2. **Storage**
   - FAISS (vector database)
   - Redis (optional caching)
   - File system (snapshots)

3. **Monitoring**
   - Prometheus (metrics)
   - Grafana (visualization)

### Internal Integrations

- **v1.4 ↔ v2.0 Bridge**:
  - ContextManager v1 + v2 coexistence
  - Shared memory layer
  - Unified metrics collection

---

## Design Patterns

### 1. Strategy Pattern
**Used in**: Routing strategies (QUALITY_FOCUSED, COST_AWARE, CASCADE, BALANCED)

### 2. Pipeline Pattern
**Used in**: Request processing (7-step pipeline)

### 3. Cache-Aside Pattern
**Used in**: AssociativeMemory, ContextManager v2

### 4. Circuit Breaker Pattern
**Used in**: RuntimeAdapter (load-based throttling)

### 5. Observer Pattern
**Used in**: PerformanceTracker (metrics collection)

### 6. Memento Pattern
**Used in**: StateSnapshotSystem (state capture/restore)

### 7. Adapter Pattern
**Used in**: RuntimeAdapter (system metrics abstraction)

### 8. Facade Pattern
**Used in**: UnifiedRouter (unified interface to 21 components)

---

## Scalability

### Horizontal Scaling
- **Stateless design**: No shared state between instances
- **Load balancing**: Round-robin or least-connections
- **Distributed caching**: Redis for shared memory

### Vertical Scaling
- **Concurrency**: asyncio for concurrent requests
- **Batching**: Throughput optimization via BatchingLayer
- **Memory efficiency**: Pruning and compression

### Performance Targets
- **Throughput**: 10+ QPS per instance
- **Latency**: < 200ms (p50), < 500ms (p95)
- **Cache hit rate**: 80%+ after warmup
- **Memory**: < 2GB per instance

---

**Version**: 2.0.0
**Last Updated**: 2025
**Status**: Production Ready
