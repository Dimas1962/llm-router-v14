# LLM Router v1.4 - Phase 1 ✅

Intelligent routing system for 6 local MLX models with dynamic selection based on task analysis.

## Status: Phase 1 Complete

**Implementation**: Core routing with task classification and complexity estimation
**Tests**: 41/41 passed (100%)
**Performance**: <100ms routing latency ✅

---

## Quick Start

```bash
# 1. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run tests
pytest tests/test_core.py -v

# 3. Use the router
python demo.py
```

## Usage Example

```python
from router import RouterCore

router = RouterCore()

# Simple task → fast model
result = router.route_sync("print hello world")
print(f"Selected: {result.model}")  # qwen2.5-coder-7b or glm-4-9b

# Complex reasoning → 80B thinking model
result = router.route_sync("design a microservices architecture")
print(f"Selected: {result.model}")  # qwen3-next-80b

# Rust code → language specialist
result = router.route_sync("write a Rust async function")
print(f"Selected: {result.model}")  # deepseek-coder-16b

# Large context → 1M context model
result = router.route_sync("refactor this", session_history=[...])
print(f"Selected: {result.model}")  # glm-4-9b
```

---

## Architecture

### 5 MLX Models Configured

| Model | Context | Speed | Quality | Tier | Use Cases |
|-------|---------|-------|---------|------|-----------|
| **glm-4-9b** | 1M | 45 tok/s | 0.87 | Primary | Coding, large context |
| **qwen3-next-80b** | 200K | 12 tok/s | 0.90 | Reasoning | Architecture, thinking |
| **qwen3-coder-30b** | 128K | 22 tok/s | 0.80 | Specialist | Multi-file refactoring |
| **deepseek-coder-16b** | 64K | 32 tok/s | 0.77-0.90 | Multilang | Rust, Go, Kotlin, etc. |
| **qwen2.5-coder-7b** | 32K | 60 tok/s | 0.70 | Fast | Quick snippets |

### Routing Decision Tree

```
Input Query
    │
    ├─→ Complexity < 0.3 & Simple Pattern?
    │   └─→ YES: qwen2.5-coder-7b (cascade)
    │
    ├─→ Requires Reasoning? (complexity > 0.7)
    │   └─→ YES: qwen3-next-80b
    │
    ├─→ Language in [Rust, Go, Kotlin]?
    │   └─→ YES: deepseek-coder-16b
    │
    ├─→ Context Size > 200K?
    │   └─→ YES: glm-4-9b (1M context)
    │
    ├─→ Decay Risk > 0.7?
    │   └─→ YES: glm-4-9b or qwen3-next-80b
    │
    └─→ DEFAULT: ELO-based scoring
```

---

## Features Implemented

### ✅ Phase 1 (Complete)
- [x] Model configuration system
- [x] Task classification (10 types)
- [x] Complexity estimation
- [x] Language detection
- [x] Context analysis
- [x] Decay risk detection
- [x] Basic routing logic
- [x] Comprehensive test suite

### 🔜 Future Phases
- [ ] **Phase 2**: Eagle ELO + Associative Memory
- [ ] **Phase 3**: CARROT Cost-Quality Prediction
- [ ] **Phase 4**: Advanced Context Management
- [ ] **Phase 5**: Integration & API Endpoints
- [ ] **Phase 6**: Episodic Memory & Cascade Optimization

---

## Test Results

```
✅ 41/41 tests passed
⚡ Routing latency: <100ms (target met)
🎯 Coverage: Models, classifiers, routing, context analysis
```

### Test Categories
- Model configuration validation
- Task classification accuracy
- Complexity estimation
- Language detection
- Routing decision logic
- Context analysis & decay risk
- Performance benchmarks
- Async support

---

## Project Structure

```
llm-router-v14/
├── router/                 # Core router package
│   ├── __init__.py        # Package exports
│   ├── models.py          # Model configurations + ELO
│   ├── classifiers.py     # Task classification + complexity
│   └── router_core.py     # Main RouterCore class
├── tests/                  # Test suite
│   ├── __init__.py
│   └── test_core.py       # 41 comprehensive tests
├── venv/                   # Virtual environment
├── requirements.txt        # Dependencies
├── SPEC.md                 # Full specification
└── README.md               # This file
```

---

## API Reference

### RouterCore

Main routing class with sync/async support.

**Methods:**
- `route(query, session_history, budget, user_id)` - Async routing
- `route_sync(query, session_history, budget, user_id)` - Sync routing
- `get_model_info(model_id)` - Get model details
- `list_models()` - List all available models

### RoutingResult

Result object with routing decision.

**Attributes:**
- `model: str` - Selected model ID
- `confidence: float` - Confidence score (0.0-1.0)
- `reasoning: str` - Human-readable explanation
- `alternatives: List[Tuple[str, float]]` - Alternative models
- `metadata: Dict` - Additional routing info

---

## Performance

- **Routing Latency**: <100ms (Phase 1 target met)
- **Accuracy**: High confidence in specialized routing (language, reasoning)
- **Memory**: Minimal overhead (no ML models loaded)

---

## Next Steps

1. **Integrate with LM Studio** - Connect to actual MLX models
2. **Implement Phase 2** - Eagle ELO system with learning
3. **Add CARROT** - Cost-aware routing with budget constraints
4. **Build API** - FastAPI endpoints for production use
5. **Add Monitoring** - Track routing decisions and model performance

---

## Contributing

Phase 1 is complete and ready for testing. Future phases will add:
- Training-free learning (Eagle ELO)
- Cost optimization (CARROT)
- User personalization (Episodic memory)
- Advanced cascade routing

---

## License

Internal project - Mac Studio M4 Max setup

**Hardware**: Mac Studio M4 Max (128GB RAM)
**Models**: 5 MLX models (113.99GB total)
**Version**: 1.4.0
**Status**: Phase 1 Complete ✅
