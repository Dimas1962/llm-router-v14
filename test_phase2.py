from router import RouterCore

print("🧪 Testing Phase 2: Eagle ELO + Memory\n")

router = RouterCore(enable_eagle=True, enable_memory=True)

queries = [
    "write Python sorting function",
    "explain quantum computing", 
    "fix Rust bug",
    "analyze large dataset"
]

for i, query in enumerate(queries, 1):
    result = router.route_sync(query)
    print(f"{i}. Query: {query}")
    print(f"   → Model: {result.model}")
    print(f"   → Strategy: {result.metadata.get('routing_strategy', 'N/A')}")
    
    # Извлекаем task_type и complexity из metadata
    task_type = result.metadata.get('task_type', 'general')
    complexity = result.metadata.get('complexity', 0.5)
    
    # Правильный вызов provide_feedback с всеми параметрами
    router.provide_feedback(
        query=query,
        selected_model=result.model,
        success=True,
        task_type=task_type,
        complexity=complexity
    )
    print()

stats = router.get_eagle_stats()
print(f"📊 Eagle Stats:")
print(f"   Top model: {stats['eagle']['top_model']}")
print(f"   Memory size: {stats['memory']['size']}")
print(f"   Global ELO ratings:")
for model, rating in stats['eagle']['global_elo'].items():
    print(f"      {model}: {rating}")
