# LLM Router v1.4 - Complete Specification

## Executive Summary

**Router v1.4** - интеллектуальная система маршрутизации запросов между 6 локальными MLX моделями с динамическим выбором на основе Eagle ELO, CARROT cost-awareness, и контекстного анализа.

### Key Features
- 🎯 **Training-free**: Eagle ELO без переобучения
- 💰 **Cost-aware**: CARROT dual prediction (quality + cost)
- ⚡ **Fast routing**: Cascade для простых задач (80% improvement)
- 🧠 **Context-aware**: Dynamic sizing + decay monitoring
- 📊 **Self-learning**: Associative memory (linear O(n))
- 🎨 **Personalized**: Episodic memory для пользователей

### Performance Targets
- Routing latency: <100ms (95th percentile)
- Accuracy: >85% правильных выборов
- Cost reduction: 30-40% vs always-best
- Token efficiency: 25-40% improvement

---

## Architecture Overview
```
┌─────────────────────────────────────────────────────────┐
│                    ROUTER v1.4                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input Layer                                            │
│  ├── Query Analysis                                    │
│  ├── Context Analysis                                  │
│  └── Constraint Extraction                             │
│                                                         │
│  Intelligence Layer                                     │
│  ├── Eagle Scoring (Global + Local ELO)               │
│  ├── CARROT Prediction (Cost + Quality)               │
│  ├── Context Manager (Sizing + Decay)                 │
│  └── Cascade Filter (Simple task detection)           │
│                                                         │
│  Memory Layer                                           │
│  ├── Associative Memory (routing history)             │
│  ├── Vector DB (similar queries)                      │
│  └── Episodic Memory (user patterns)                  │
│                                                         │
│  Model Selection Layer                                  │
│  ├── Candidate Filtering                               │
│  ├── Multi-criteria Ranking                            │
│  └── Final Selection                                   │
│                                                         │
│  Output Layer                                           │
│  ├── Selected Model                                    │
│  ├── Confidence Score                                  │
│  └── Metadata (reasoning, alternatives)               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Eagle ELO System

**Purpose**: Training-free model ranking через Elo ratings

**Components**:
- **Global ELO**: Общий рейтинг моделей (bootstrap from benchmarks)
- **Local ELO**: Context-specific рейтинги (per task type)
- **Associative Memory**: Fast similarity search O(n)

**Algorithm**:
```python
def eagle_score(query: str, model: str) -> float:
    """
    Eagle scoring: 50% global + 50% local
    """
    # Global ELO
    global_score = self.global_elo[model]
    
    # Local ELO (from similar queries)
    query_emb = self.embed(query)
    similar = self.associative_memory.search(query_emb, k=20)
    local_score = self.calculate_local_elo(similar, model)
    
    # Combine
    return 0.5 * global_score + 0.5 * local_score
```

**Initial ELO Values** (from benchmarks):
```python
GLOBAL_ELO = {
    'glm-4-9b': 1850,      # High (quality + 1M context)
    'qwen3-next-80b': 1900, # Highest (reasoning)
    'qwen3-coder-30b': 1700, # Medium (overlap)
    'deepseek-coder-16b': 1750, # Conditional
    'qwen2.5-coder-7b': 1650  # Fast but lower quality
}
```

---

### 2. CARROT Cost-Quality Prediction

**Purpose**: Budget-aware routing с dual prediction

**Components**:
- **Quality Predictor**: Оценка качества ответа модели
- **Cost Predictor**: Оценка стоимости (tokens + time)
- **Pareto Optimizer**: Баланс quality vs cost

**Algorithm**:
```python
def carrot_select(query: str, candidates: List[str], 
                  budget: float = None) -> str:
    """
    CARROT: Select best quality within budget
    """
    predictions = {}
    for model in candidates:
        quality = self.quality_predictor.predict(query, model)
        cost = self.cost_predictor.predict(query, model)
        predictions[model] = {'quality': quality, 'cost': cost}
    
    if budget:
        # Filter by budget
        valid = {m: p for m, p in predictions.items() 
                if p['cost'] <= budget}
        if valid:
            return max(valid, key=lambda m: valid[m]['quality'])
        else:
            # Budget exceeded: cheapest option
            return min(predictions, key=lambda m: predictions[m]['cost'])
    else:
        # No budget: best quality
        return max(predictions, key=lambda m: predictions[m]['quality'])
```

---

### 3. Context Manager

**Purpose**: Динамическое управление контекстом

**Components**:
- **Dynamic Context Sizing**: Adaptive context window
- **Decay Monitoring**: Отслеживание длины сессии
- **Progressive Building**: Поэтапная сборка контекста

**Algorithm**:
```python
class ContextManager:
    def analyze_context(self, query: str, 
                       session_history: List[str]) -> Dict:
        """
        Анализ контекста запроса
        """
        # Total context size
        total_size = len(query) + sum(len(h) for h in session_history)
        
        # Complexity estimation
        complexity = self.estimate_complexity(query)
        
        # Required context window
        if complexity < 0.3:
            required = 8_000   # Simple
        elif complexity < 0.7:
            required = 32_000  # Medium
        else:
            required = 128_000 # Complex
        
        # Decay risk
        decay_risk = self.estimate_decay_risk(total_size)
        
        return {
            'total_size': total_size,
            'required_window': required,
            'complexity': complexity,
            'decay_risk': decay_risk
        }
    
    def estimate_decay_risk(self, context_size: int) -> float:
        """
        Context decay risk estimation
        """
        if context_size < 32_000:
            return 0.0  # Low
        elif context_size < 64_000:
            return 0.3  # Medium
        elif context_size < 128_000:
            return 0.5  # High
        else:
            return 0.8  # Critical
```

---

### 4. Cascade Routing

**Purpose**: Fast-path для простых задач (80% improvement)

**Logic**:
```python
def cascade_route(self, query: str, 
                  complexity: float) -> Optional[str]:
    """
    Cascade: Simple tasks → fast models
    """
    if complexity < 0.3 and self.is_simple_pattern(query):
        # Confidence threshold
        if self.confidence > 0.8:
            return 'qwen2.5-coder-7b'  # Speed demon
    
    # Not simple enough: full routing
    return None
```

---

### 5. Associative Memory

**Purpose**: Fast retrieval похожих routing событий (O(n))

**Implementation**:
```python
class AssociativeMemory:
    """
    Memory-Augmented routing history
    Linear complexity O(n) not O(n²)
    """
    
    def __init__(self, dimensions: int = 1536):
        self.memory = []  # List of (embedding, metadata)
        self.index = faiss.IndexFlatL2(dimensions)
    
    async def store(self, embedding: np.ndarray, 
                   metadata: Dict):
        """
        Store routing event
        """
        self.memory.append(metadata)
        self.index.add(embedding.reshape(1, -1))
    
    async def search(self, query_embedding: np.ndarray, 
                     k: int = 20) -> List[Dict]:
        """
        Fast similarity search O(log n) with FAISS
        """
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), k
        )
        
        return [self.memory[i] for i in indices[0]]
```

---

### 6. Episodic Memory (Optional)

**Purpose**: Персонализация на основе user patterns

**Implementation**:
```python
class EpisodicMemory:
    """
    User-specific routing patterns
    """
    
    def learn_pattern(self, user_id: str, 
                     routing_event: Dict):
        """
        Learn from user's routing choices
        """
        profile = self.user_profiles.get(user_id, {})
        
        task_type = routing_event['task_type']
        model = routing_event['model']
        satisfaction = routing_event['satisfaction']
        
        # Update preferences
        if task_type not in profile:
            profile[task_type] = {}
        
        if model not in profile[task_type]:
            profile[task_type][model] = {
                'count': 0,
                'satisfaction': []
            }
        
        profile[task_type][model]['count'] += 1
        profile[task_type][model]['satisfaction'].append(satisfaction)
        
        self.user_profiles[user_id] = profile
    
    def predict_preference(self, user_id: str, 
                          task_type: str) -> Optional[str]:
        """
        Predict user's preferred model
        """
        profile = self.user_profiles.get(user_id)
        if not profile or task_type not in profile:
            return None
        
        preferences = profile[task_type]
        
        # Best model by satisfaction
        best = max(
            preferences.items(),
            key=lambda x: np.mean(x[1]['satisfaction'])
        )
        
        return best[0]
```

---

## Model Configuration

### 6 MLX Models
```python
MODELS = {
    'glm-4-9b': {
        'name': 'GLM-4-9B-Chat-1M-BF16',
        'size': '18.99GB',
        'context': 1_000_000,  # 1M tokens!
        'speed': 45,  # tok/s
        'quality': 0.87,  # vs GPT-4o
        'tier': 'primary',
        'use_cases': ['coding', 'refactoring', 'bug_fixing'],
        'frequency': 0.75  # 75% usage
    },
    
    'qwen3-next-80b': {
        'name': 'Qwen3-Next-80B-A3B-Thinking',
        'size': '60GB',
        'context': 200_000,
        'speed': 12,
        'quality': 0.90,
        'tier': 'reasoning',
        'use_cases': ['architecture', 'complex_reasoning', 'thinking'],
        'frequency': 0.18  # 18% usage
    },
    
    'qwen3-coder-30b': {
        'name': 'Qwen3-Coder-30B-A3B',
        'size': '20GB',
        'context': 128_000,
        'speed': 22,
        'quality': 0.80,
        'tier': 'specialist',
        'use_cases': ['large_refactoring'],
        'frequency': 0.07  # 7% usage (niche)
    },
    
    'deepseek-coder-16b': {
        'name': 'DeepSeek-Coder-V2-Lite',
        'size': '10GB',
        'context': 64_000,
        'speed': 32,
        'quality': 0.77,  # Python
        'quality_rust': 0.90,  # Rust/Go!
        'tier': 'multilang',
        'use_cases': ['rust', 'go', 'kotlin', 'exotic_langs'],
        'frequency': 0.08  # 8% usage (conditional)
    },
    
    'qwen2.5-coder-7b': {
        'name': 'Qwen2.5-Coder-7B',
        'size': '5GB',
        'context': 32_000,
        'speed': 60,
        'quality': 0.70,
        'tier': 'fast',
        'use_cases': ['quick_snippets', 'real_time'],
        'frequency': 0.12  # 12% usage
    }
}
```

---

## API Design

### Main Routing Function
```python
async def route(
    query: str,
    session_history: List[str] = None,
    budget: float = None,
    user_id: str = None
) -> RoutingResult:
    """
    Main routing function
    
    Args:
        query: User query/task
        session_history: Previous messages in session
        budget: Optional budget constraint
        user_id: Optional user ID for personalization
    
    Returns:
        RoutingResult with selected model + metadata
    """
```

### RoutingResult Schema
```python
@dataclass
class RoutingResult:
    model: str  # Selected model ID
    confidence: float  # 0.0 - 1.0
    reasoning: str  # Why this model was chosen
    alternatives: List[Tuple[str, float]]  # [(model, score)]
    metadata: Dict[str, Any]  # Additional info
```

---

## Implementation Guide

### Phase 1: Core Routing (Week 1-2)

**Components**:
1. Model configuration
2. Task classification
3. Complexity estimation
4. Basic routing logic

**Deliverables**:
- `router_core.py`: Main routing logic
- `models.py`: Model configurations
- `classifiers.py`: Task/complexity classifiers

---

### Phase 2: Eagle ELO (Week 3)

**Components**:
1. Global ELO initialization
2. Associative memory setup
3. Local ELO calculation
4. ELO update mechanism

**Deliverables**:
- `eagle.py`: Eagle ELO system
- `memory.py`: Associative memory

---

### Phase 3: CARROT (Week 4)

**Components**:
1. Quality predictor
2. Cost predictor
3. Budget-aware selection
4. Pareto optimization

**Deliverables**:
- `carrot.py`: CARROT system
- `predictors.py`: Quality/cost predictors

---

### Phase 4: Context Management (Week 5)

**Components**:
1. Context analyzer
2. Dynamic sizing
3. Decay monitoring
4. Progressive building

**Deliverables**:
- `context_manager.py`: Context management

---

### Phase 5: Integration & Testing (Week 6)

**Components**:
1. Full system integration
2. End-to-end testing
3. Performance optimization
4. Documentation

**Deliverables**:
- `router_v14.py`: Integrated system
- `tests/`: Test suite
- `docs/`: Documentation

---

### Phase 6: Optional Features (Week 7+)

**Components**:
1. Episodic memory
2. Cascade routing
3. Advanced monitoring
4. UI/API endpoints

---

## Deployment Instructions

### Requirements
```bash
# Python 3.10+
python >= 3.10

# Core dependencies
numpy >= 1.24
faiss-cpu >= 1.7  # or faiss-gpu
sentence-transformers >= 2.2
pydantic >= 2.0
fastapi >= 0.100  # for API

# ML dependencies
scikit-learn >= 1.3
torch >= 2.0  # for embeddings
```

### Installation
```bash
# 1. Clone repository
git clone <repo>
cd llm-router-v14

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize
python -m router.init
```

### Configuration
```yaml
# config.yaml
models:
  mlx_path: "/path/to/mlx/models"
  lm_studio_port: 1234

routing:
  default_budget: null  # No budget constraint
  enable_cascade: true
  enable_episodic: false  # Optional

memory:
  vector_dimensions: 1536
  max_history: 100000
  storage_path: "./data/memory"

api:
  host: "0.0.0.0"
  port: 8000
  enable_cors: true
```

### Running
```bash
# Start router service
python -m router.server

# Or use as library
from router import RouterV14

router = RouterV14(config_path="config.yaml")
result = await router.route("Write a Python function to sort a list")
print(f"Selected: {result.model} (confidence: {result.confidence})")
```

---

## Testing Strategy

### Unit Tests
```python
# tests/test_eagle.py
def test_global_elo_initialization():
    """Test Global ELO bootstrap"""
    eagle = EagleELO()
    assert eagle.global_elo['qwen3-next-80b'] == 1900
    assert eagle.global_elo['qwen2.5-coder-7b'] == 1650

# tests/test_context.py
def test_dynamic_context_sizing():
    """Test context window estimation"""
    ctx_mgr = ContextManager()
    result = ctx_mgr.analyze_context(
        query="Simple hello world",
        session_history=[]
    )
    assert result['required_window'] == 8_000
```

### Integration Tests
```python
# tests/test_integration.py
@pytest.mark.asyncio
async def test_end_to_end_routing():
    """Test complete routing flow"""
    router = RouterV14()
    
    # Simple query → fast model
    result = await router.route("print hello world")
    assert result.model == 'qwen2.5-coder-7b'
    
    # Complex reasoning → 80B
    result = await router.route(
        "Design a scalable microservices architecture"
    )
    assert result.model == 'qwen3-next-80b'
    
    # Huge context → GLM-4
    result = await router.route(
        query="Refactor this",
        session_history=['...' * 10000]  # Large history
    )
    assert result.model == 'glm-4-9b'
```

### Performance Tests
```python
# tests/test_performance.py
def test_routing_latency():
    """Test routing speed"""
    router = RouterV14()
    
    start = time.time()
    result = router.route_sync("test query")
    latency = time.time() - start
    
    assert latency < 0.1  # <100ms target
```

---

## Monitoring & Metrics

### Key Metrics
```python
METRICS = {
    # Routing performance
    'routing_latency_ms': Histogram,
    'routing_accuracy': Gauge,  # % correct choices
    
    # Model usage
    'model_selection_count': Counter,  # per model
    'model_satisfaction_score': Gauge,  # per model
    
    # Cost efficiency
    'cost_savings_percent': Gauge,
    'token_efficiency': Gauge,
    
    # Context management
    'avg_context_size': Gauge,
    'decay_events': Counter,
    
    # Memory
    'associative_memory_size': Gauge,
    'search_latency_ms': Histogram
}
```

### Logging
```python
# Structured logging for analysis
logger.info(
    "routing_decision",
    query_hash=hash(query),
    selected_model=result.model,
    confidence=result.confidence,
    eagle_score=scores['eagle'],
    carrot_quality=scores['quality'],
    carrot_cost=scores['cost'],
    context_size=context['total_size'],
    latency_ms=latency
)
```

---

## Future Enhancements

### v1.5 (Q2 2025)
- [ ] Multi-round routing (Router-R1 integration)
- [ ] Advanced cascade (multiple tiers)
- [ ] Real-time ELO calibration
- [ ] Web dashboard

### v2.0 (Q3 2025)
- [ ] Cloud model support (Claude, GPT-4)
- [ ] Hybrid routing (local + cloud)
- [ ] Advanced cost optimization
- [ ] Distributed routing

---

## References

### Research Papers
1. "Eagle: Training-free Routing" (Oct 2025)
2. "CARROT: Cost-Aware Routing" (Oct 2025)
3. "MemLong: Memory-Augmented LLM" (Aug 2024)
4. "Router-R1: Multi-Round Routing" (Oct 2025)

### Tools & Libraries
- LM Studio (MLX backend)
- FAISS (vector search)
- sentence-transformers (embeddings)
- FastAPI (API server)

---

## Contact & Support

**Project Lead**: Дмитрий  
**Hardware**: Mac Studio M4 Max (128GB RAM)  
**Models**: 6 MLX models (113.99GB total)

---

## Appendix: Decision Tree
```
Input Query
    │
    ├─→ Complexity < 0.3 & Simple Pattern?
    │   └─→ YES: Cascade → qwen2.5-coder-7b (confidence > 0.8)
    │
    ├─→ Requires Reasoning? (complexity > 0.8)
    │   └─→ YES: qwen3-next-80b (Thinking mode)
    │
    ├─→ Language in [Rust, Go, Kotlin]?
    │   └─→ YES: deepseek-coder-16b (338 languages)
    │
    ├─→ Context Size > 200K?
    │   └─→ YES: glm-4-9b (1M context)
    │
    ├─→ Decay Risk > 0.7?
    │   └─→ YES: Force GLM-4 or Qwen3-80B
    │
    ├─→ Budget Constraint?
    │   └─→ YES: CARROT selection (quality vs cost)
    │
    └─→ DEFAULT: Eagle scoring (Global + Local ELO)
        └─→ Select highest score
```

---

**END OF SPECIFICATION**

**Version**: 1.4  
**Date**: October 29, 2025  
**Status**: Ready for Implementation 🚀