import pytest
import asyncio
import sys
sys.path.insert(0, 'src/v2')

from context_manager import (
    ContextManager,
    ContextFragment,
    ContextResponse,
    LRUCache
)


@pytest.fixture
def manager():
    return ContextManager(cache_size=100)


@pytest.mark.asyncio
async def test_initialization(manager):
    """Test 1: Basic initialization"""
    assert manager.cache.capacity == 100
    assert manager.default_budget == 8000
    assert manager.enable_deduplication is True


@pytest.mark.asyncio
async def test_get_context_basic(manager):
    """Test 2: Basic context retrieval"""
    response = await manager.get_context("test query")
    
    assert isinstance(response, ContextResponse)
    assert response.context is not None
    assert response.total_tokens > 0


@pytest.mark.asyncio
async def test_cache_hit(manager):
    """Test 3: Cache hit on second request"""
    query = "same query"
    
    # First request - cache miss
    response1 = await manager.get_context(query)
    assert response1.cache_hit is False
    
    # Second request - cache hit
    response2 = await manager.get_context(query)
    assert response2.cache_hit is True


@pytest.mark.asyncio
async def test_batch_requests(manager):
    """Test 4: Batch processing"""
    queries = ["query1", "query2", "query3"]
    responses = await manager.batch_requests(queries)
    
    assert len(responses) == 3
    assert all(isinstance(r, ContextResponse) for r in responses)


@pytest.mark.asyncio
async def test_budget_limit(manager):
    """Test 5: Budget constraint"""
    response = await manager.get_context("test", max_tokens=100)
    
    assert response.total_tokens <= 100


@pytest.mark.asyncio
async def test_deduplication(manager):
    """Test 6: Fragment deduplication"""
    fragments = [
        ContextFragment("same", 0.9, 10),
        ContextFragment("same", 0.8, 10),
        ContextFragment("different", 0.7, 10)
    ]
    
    unique = manager._deduplicate_fragments(fragments)
    assert len(unique) == 2


@pytest.mark.asyncio
async def test_statistics_tracking(manager):
    """Test 7: Statistics tracked"""
    await manager.get_context("test1")
    await manager.get_context("test2")
    
    stats = manager.get_statistics()
    assert stats['requests'] == 2


def test_lru_cache_basic():
    """Test 8: LRU cache works"""
    cache = LRUCache(capacity=2)
    
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"


def test_lru_cache_eviction():
    """Test 9: LRU cache evicts oldest"""
    cache = LRUCache(capacity=2)
    
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")  # Should evict key1
    
    assert cache.get("key1") is None
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"


@pytest.mark.asyncio
async def test_clear_cache(manager):
    """Test 10: Cache can be cleared"""
    await manager.get_context("test")
    manager.clear_cache()
    
    stats = manager.get_statistics()
    assert stats['cache_size'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])