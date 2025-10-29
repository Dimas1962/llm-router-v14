from router import RouterCore, ContextManager

print("🎯 Testing Phase 4: Advanced Context Management\n")

# Test 1: Router with Context Manager
print("1️⃣ Router with Context Management:")
router = RouterCore(
    enable_eagle=True,
    enable_memory=True,
    enable_carrot=True,
    enable_context_manager=True
)

# Small context
result = router.route_sync("simple query")
print(f"   Small context: {result.model}")
print(f"   Decay risk: {result.metadata.get('decay_risk', 'unknown')}")

# Large context
large_history = [f"message {i}" for i in range(100)]
result = router.route_sync("continue", session_history=large_history)
print(f"\n   Large context (100 msgs): {result.model}")
print(f"   Decay risk: {result.metadata.get('decay_risk')}")
print(f"   Strategy: {result.metadata.get('routing_strategy')}")

# Test 2: Context Manager Analysis
print(f"\n2️⃣ Context Analysis:")
ctx_mgr = ContextManager()
analysis = ctx_mgr.analyze_context("query", large_history)
print(f"   Estimated tokens: {analysis.estimated_tokens:,}")
print(f"   Decay risk: {analysis.decay_risk.level}")
print(f"   Recommended model: {analysis.recommended_model or 'any'}")

# Test 3: Context Optimization
print(f"\n3️⃣ Context Optimization:")
optimized = ctx_mgr.optimize_context("query", large_history, "qwen3-next-80b")
print(f"   Original: {len(large_history)} messages")
print(f"   Optimized: {len(optimized['optimized_history'])} messages")
print(f"   Strategy: {optimized['strategy']}")

print("\n✅ Phase 4 test completed!")
