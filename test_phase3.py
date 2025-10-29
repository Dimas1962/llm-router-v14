from router import RouterCore, CARROT

print("🥕 Testing Phase 3: CARROT Cost-Aware Routing\n")

# Test 1: Direct CARROT usage
print("1️⃣ Direct CARROT Testing:")
carrot = CARROT()

# No budget - best quality
model, pred = carrot.select("write a complex algorithm")
print(f"   No budget: {model}")
print(f"   Quality: {pred['quality']:.3f}, Cost: {pred['cost']:.2f}")

# With budget
model, pred = carrot.select("write a function", budget=10.0)
print(f"   Budget=10: {model}")
print(f"   Quality: {pred['quality']:.3f}, Cost: {pred['cost']:.2f}")

# Pareto frontier
print(f"\n2️⃣ Pareto Frontier (Quality vs Cost):")
pareto = carrot.get_pareto_frontier("write a function")
for model_id, quality, cost in pareto[:3]:
    print(f"   {model_id}: Q={quality:.3f}, C={cost:.2f}")

# Test 2: Router with budget
print(f"\n3️⃣ Router with CARROT Budget:")
router = RouterCore(enable_carrot=True, enable_eagle=True, enable_memory=True)

# With budget
result = router.route_sync("write a Python function", budget=100.0)
print(f"   Budget=100: {result.model}")
print(f"   Strategy: {result.metadata.get('routing_strategy')}")
print(f"   Predicted cost: {result.metadata.get('predicted_cost', 0):.2f}")

# Without budget (uses Eagle)
result = router.route_sync("write a Python function")
print(f"\n   No budget: {result.model}")
print(f"   Strategy: {result.metadata.get('routing_strategy')}")

# Test 3: Budget recommendation
print(f"\n4️⃣ Budget Recommendation:")
budget = carrot.recommend_budget("complex algorithm", quality_target=0.85)
print(f"   For quality 0.85: budget = {budget:.2f}")

print("\n✅ Phase 3 (CARROT) test completed!")
