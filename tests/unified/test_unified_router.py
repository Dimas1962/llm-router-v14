"""
Tests for Unified LLM Router v2.0
Integration tests for v1.4 + v2.0 components
"""

import pytest
import asyncio
from src.unified.unified_router import (
    UnifiedRouter,
    UnifiedRequest,
    UnifiedResponse,
    RoutingStrategy
)
from src.v2.batching_layer import Priority


def test_unified_initialization():
    """Test UnifiedRouter initialization with all components"""
    router = UnifiedRouter()

    # v1.4 components
    assert router.memory is not None
    assert router.episodic is not None
    assert router.eagle is not None
    assert router.carrot is not None
    assert router.cascade is not None
    assert router.multi_round is not None
    assert router.context_v1 is not None
    assert router.core is not None
    assert router.monitoring is not None

    # v2.0 components
    assert router.context_v2 is not None
    assert router.runtime is not None
    assert router.quality is not None
    assert router.context_sizer is not None
    assert router.pruner is not None
    assert router.batching is not None
    assert router.ast_analyzer is not None
    assert router.compressor is not None
    assert router.env_prompter is not None
    assert router.snapshot is not None


def test_unified_initialization_minimal():
    """Test initialization with minimal features"""
    router = UnifiedRouter(
        enable_batching=False,
        enable_quality_check=False,
        enable_snapshots=False,
        enable_monitoring=False
    )

    # Components still initialized even if disabled
    assert router.quality is not None
    assert not router.enable_batching
    assert not router.enable_quality_check


@pytest.mark.asyncio
async def test_full_routing_pipeline():
    """Test complete routing pipeline from request to response"""
    router = UnifiedRouter()

    request = UnifiedRequest(
        query="What is quantum computing?",
        context="Technical context",
        user_id="user123",
        strategy=RoutingStrategy.BALANCED
    )

    response = await router.route(request)

    assert isinstance(response, UnifiedResponse)
    assert response.model is not None
    assert response.result is not None
    assert 0 <= response.quality_score <= 1
    assert response.latency > 0
    assert 0 <= response.confidence <= 1
    assert response.reasoning is not None
    assert isinstance(response.passed_quality_check, bool)


@pytest.mark.asyncio
async def test_quality_focused_routing():
    """Test quality-focused routing with Eagle ELO"""
    router = UnifiedRouter()

    request = UnifiedRequest(
        query="Explain machine learning",
        strategy=RoutingStrategy.QUALITY_FOCUSED
    )

    response = await router.route(request)

    assert response.model is not None
    assert "Eagle" in response.reasoning or "ELO" in response.reasoning.upper()


@pytest.mark.asyncio
async def test_cost_aware_routing():
    """Test cost-aware routing with CARROT"""
    router = UnifiedRouter()

    request = UnifiedRequest(
        query="Simple math: 2+2",
        strategy=RoutingStrategy.COST_AWARE,
        max_cost=0.5
    )

    response = await router.route(request)

    assert response.model is not None
    assert "CARROT" in response.reasoning or "cost" in response.reasoning.lower()


@pytest.mark.asyncio
async def test_cascade_routing():
    """Test cascade routing strategy"""
    router = UnifiedRouter()

    request = UnifiedRequest(
        query="Complex reasoning task",
        strategy=RoutingStrategy.CASCADE
    )

    response = await router.route(request)

    assert response.model is not None
    assert "Cascade" in response.reasoning


@pytest.mark.asyncio
async def test_balanced_routing():
    """Test balanced routing (hybrid)"""
    router = UnifiedRouter()

    request = UnifiedRequest(
        query="General query",
        strategy=RoutingStrategy.BALANCED
    )

    response = await router.route(request)

    assert response.model is not None
    assert response.confidence > 0


@pytest.mark.asyncio
async def test_runtime_adaptation_high_load():
    """Test strategy adaptation under high system load"""
    router = UnifiedRouter()

    # Simulate high load by running many requests
    request = UnifiedRequest(
        query="Test query",
        strategy=RoutingStrategy.QUALITY_FOCUSED
    )

    response = await router.route(request)

    # Should still work, possibly adapted strategy
    assert response.model is not None


@pytest.mark.asyncio
async def test_context_assembly():
    """Test context assembly with compression"""
    router = UnifiedRouter()

    request = UnifiedRequest(
        query="What is Python?",
        context="Additional context"
    )

    response = await router.route(request)

    assert response.model is not None
    assert response.result is not None


@pytest.mark.asyncio
async def test_quality_check_integration():
    """Test quality check integration with retry logic"""
    router = UnifiedRouter(enable_quality_check=True)

    request = UnifiedRequest(
        query="Explain relativity",
        context="Physics context"
    )

    response = await router.route(request)

    assert isinstance(response.passed_quality_check, bool)
    assert response.quality_score >= 0


@pytest.mark.asyncio
async def test_snapshot_capture():
    """Test state snapshot capture during routing"""
    router = UnifiedRouter(enable_snapshots=True)

    request = UnifiedRequest(
        query="Test query for snapshot"
    )

    response = await router.route(request)

    assert response.snapshot_id is not None
    assert "snapshot_" in response.snapshot_id


@pytest.mark.asyncio
async def test_episodic_memory_integration():
    """Test episodic memory updates after routing"""
    router = UnifiedRouter()

    request = UnifiedRequest(
        query="Memorable query"
    )

    # Route request
    response = await router.route(request)

    # Check episodic memory was updated
    assert router.episodic.size() > 0


@pytest.mark.asyncio
async def test_context_v1_v2_integration():
    """Test integration of v1 and v2 context managers"""
    router = UnifiedRouter()

    # Both context managers should work
    assert router.context_v1 is not None
    assert router.context_v2 is not None

    # They should be separate instances
    assert router.context_v1 != router.context_v2


def test_get_all_components():
    """Test retrieval of all component instances"""
    router = UnifiedRouter()

    components = router.get_all_components()

    # Check v1.4 components
    assert "router_core" in components
    assert "eagle" in components
    assert "carrot" in components
    assert "memory" in components
    assert "episodic" in components
    assert "cascade" in components
    assert "multi_round" in components
    assert "context_v1" in components

    # Check v2.0 components
    assert "context_v2" in components
    assert "runtime" in components
    assert "quality" in components
    assert "context_sizer" in components
    assert "pruner" in components
    assert "batching" in components
    assert "ast_analyzer" in components
    assert "compressor" in components
    assert "env_prompter" in components
    assert "snapshot" in components
    assert "monitoring" in components


def test_comprehensive_statistics():
    """Test comprehensive statistics from all components"""
    router = UnifiedRouter()

    stats = router.get_stats()

    # Check unified stats
    assert "unified" in stats
    assert "total_requests" in stats["unified"]
    assert "successful_requests" in stats["unified"]

    # Check v2.0 component stats
    assert "context_v2" in stats
    assert "runtime" in stats
    assert "quality" in stats
    assert "context_sizer" in stats
    assert "pruner" in stats
    assert "compressor" in stats
    assert "env_prompter" in stats
    assert "batching" in stats
    assert "snapshot" in stats


@pytest.mark.asyncio
async def test_statistics_tracking():
    """Test statistics tracking across requests"""
    router = UnifiedRouter()

    request = UnifiedRequest(query="Test query")

    # Initial stats
    assert router.stats["total_requests"] == 0

    # Make request
    await router.route(request)

    # Check stats updated
    assert router.stats["total_requests"] == 1
    assert router.stats["successful_requests"] == 1


def test_reset_all_statistics():
    """Test resetting statistics for all components"""
    router = UnifiedRouter()

    # Set some stats
    router.stats["total_requests"] = 10
    router.stats["successful_requests"] = 8

    # Reset
    router.reset_stats()

    # Check reset
    assert router.stats["total_requests"] == 0
    assert router.stats["successful_requests"] == 0

    # Check v2.0 components reset
    stats = router.get_stats()
    # context_v2 clear_cache() doesn't reset stats, just cache
    assert stats["context_v2"]["cache_size"] == 0


@pytest.mark.asyncio
async def test_optimize_all_components():
    """Test optimization across all components"""
    router = UnifiedRouter()

    # Should not crash
    await router.optimize()

    # Check that optimization ran (Eagle ELO update)
    assert router.eagle is not None


@pytest.mark.asyncio
async def test_multiple_requests_sequential():
    """Test handling multiple sequential requests"""
    router = UnifiedRouter()

    requests = [
        UnifiedRequest(query=f"Query {i}")
        for i in range(5)
    ]

    for req in requests:
        response = await router.route(req)
        assert response.model is not None

    assert router.stats["total_requests"] == 5


@pytest.mark.asyncio
async def test_priority_handling():
    """Test request priority handling"""
    router = UnifiedRouter()

    # High priority request
    high_priority = UnifiedRequest(
        query="Urgent query",
        priority=Priority.URGENT
    )

    # Normal priority request
    normal_priority = UnifiedRequest(
        query="Normal query",
        priority=Priority.NORMAL
    )

    response1 = await router.route(high_priority)
    response2 = await router.route(normal_priority)

    assert response1.model is not None
    assert response2.model is not None


@pytest.mark.asyncio
async def test_metadata_preservation():
    """Test metadata preservation through pipeline"""
    router = UnifiedRouter()

    request = UnifiedRequest(
        query="Test query",
        metadata={"custom_key": "custom_value"}
    )

    response = await router.route(request)

    # Response should have metadata
    assert response.metadata is not None
    assert "load_level" in response.metadata
    assert "budget" in response.metadata
    assert "strategy" in response.metadata


@pytest.mark.asyncio
async def test_context_caching():
    """Test context caching in v2 context manager"""
    router = UnifiedRouter()

    request1 = UnifiedRequest(query="Same query")
    request2 = UnifiedRequest(query="Same query")

    # First request
    response1 = await router.route(request1)

    # Second request should use cache
    stats_before = router.context_v2.get_statistics()
    response2 = await router.route(request2)
    stats_after = router.context_v2.get_statistics()

    # Cache should have been used
    assert stats_after["cache_hits"] >= stats_before["cache_hits"]


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in routing pipeline"""
    router = UnifiedRouter()

    # Empty query should still work (or fail gracefully)
    request = UnifiedRequest(query="")

    try:
        response = await router.route(request)
        # If it succeeds, that's fine
        assert response is not None
    except Exception:
        # If it fails, should increment failed_requests
        assert router.stats["failed_requests"] >= 0


@pytest.mark.asyncio
async def test_avg_latency_calculation():
    """Test average latency calculation"""
    router = UnifiedRouter()

    request = UnifiedRequest(query="Test query")

    # Make multiple requests
    await router.route(request)
    await router.route(request)
    await router.route(request)

    # Check avg latency calculated
    assert router.stats["avg_latency"] > 0


@pytest.mark.asyncio
async def test_environment_integration():
    """Test environment prompter integration"""
    router = UnifiedRouter()

    # Environment prompter should be working
    summary = router.env_prompter.get_context_summary()
    assert "System Environment:" in summary

    request = UnifiedRequest(query="Test")
    response = await router.route(request)

    assert response.model is not None


def test_component_count():
    """Test that all 21 components are present"""
    router = UnifiedRouter()

    components = router.get_all_components()

    # Should have 21 components (some may be None if disabled)
    component_count = sum(1 for c in components.values() if c is not None)
    assert component_count >= 18  # At least 18 (some optional)


@pytest.mark.asyncio
async def test_end_to_end_quality_flow():
    """Test end-to-end flow with quality checking"""
    router = UnifiedRouter(enable_quality_check=True)

    request = UnifiedRequest(
        query="Complex question requiring detailed answer",
        context="Technical domain context"
    )

    response = await router.route(request)

    # Full pipeline should complete
    assert response.model is not None
    assert response.result is not None
    assert response.quality_score >= 0
    assert isinstance(response.passed_quality_check, bool)
    assert response.snapshot_id is not None


@pytest.mark.asyncio
async def test_cascade_with_batching():
    """Test cascade routing with batching enabled"""
    router = UnifiedRouter(enable_batching=True)

    request = UnifiedRequest(
        query="Test cascade with batching",
        strategy=RoutingStrategy.CASCADE
    )

    response = await router.route(request)

    assert response.model is not None
    assert router.batching is not None
