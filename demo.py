#!/usr/bin/env python3
"""
LLM Router v1.4 - Interactive Demo
Phase 1: Core Routing
"""

from router import RouterCore


def main():
    print("\n🤖 LLM Router v1.4 - Phase 1 Demo")
    print("=" * 70)
    print("\nTesting routing decisions across different query types...\n")

    router = RouterCore()

    # Test scenarios
    test_cases = [
        {
            "label": "Simple Task (Cascade)",
            "query": "write a simple function to print hello world",
            "expected": "qwen2.5-coder-7b or glm-4-9b"
        },
        {
            "label": "Complex Reasoning",
            "query": "design a scalable microservices architecture with event sourcing",
            "expected": "qwen3-next-80b"
        },
        {
            "label": "Rust Specialist",
            "query": "write a Rust function with async/await and error handling",
            "expected": "deepseek-coder-16b"
        },
        {
            "label": "Go Specialist",
            "query": "create a Go HTTP server with middleware",
            "expected": "deepseek-coder-16b"
        },
        {
            "label": "Large Context",
            "query": "refactor the entire codebase",
            "expected": "glm-4-9b (1M context)"
        },
        {
            "label": "Bug Fixing",
            "query": "fix the memory leak in the connection pool",
            "expected": "any primary model"
        },
        {
            "label": "Architecture Design",
            "query": "explain the tradeoffs between monolithic and microservices",
            "expected": "qwen3-next-80b"
        },
        {
            "label": "Python Coding",
            "query": "write a Python decorator for caching function results",
            "expected": "glm-4-9b or qwen3-coder-30b"
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"📝 Test {i}: {test['label']}")
        print(f"   Query: \"{test['query']}\"")

        result = router.route_sync(test['query'])

        print(f"   ✅ Selected: {result.model}")
        print(f"   💡 Reasoning: {result.reasoning}")
        print(f"   🎯 Confidence: {result.confidence:.1%}")
        print(f"   📊 Task Type: {result.metadata.get('task_type', 'N/A')}")
        print(f"   🔧 Strategy: {result.metadata.get('routing_strategy', 'N/A')}")

        # Show alternatives
        if result.alternatives:
            alts = ", ".join([f"{m} ({s:.0f})" for m, s in result.alternatives[:2]])
            print(f"   🔄 Alternatives: {alts}")

        print()

    print("=" * 70)
    print("✅ Demo completed successfully!\n")

    # Show model summary
    print("\n📊 Available Models Summary:")
    print("-" * 70)
    models = router.list_models()
    for model in models:
        print(f"  • {model['id']:20} | "
              f"Context: {model['context_window']:,} | "
              f"Speed: {model['speed']} tok/s | "
              f"ELO: {model['elo']}")

    print("-" * 70)
    print(f"\nTotal models: {len(models)}")
    print()


if __name__ == "__main__":
    main()
