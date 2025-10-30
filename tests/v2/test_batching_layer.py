"""
Tests for Batching Layer (Component 6)
"""

import pytest
import asyncio
from src.v2.batching_layer import (
    BatchingLayer,
    Priority,
    FlushStrategy,
    BatchRequest,
    BatchResult
)


def test_initialization():
    """Test BatchingLayer initialization"""
    layer = BatchingLayer()

    assert layer.max_batch_size == 32
    assert layer.max_wait_time == 0.5
    assert layer.min_batch_size == 4
    assert layer.flush_strategy == FlushStrategy.ADAPTIVE
    assert layer.enable_priority is True
    assert layer.stats["total_requests"] == 0


def test_sync_request_adding():
    """Test adding requests synchronously"""
    layer = BatchingLayer()

    layer.add_request_sync("req1", {"query": "test1"}, Priority.NORMAL)
    layer.add_request_sync("req2", {"query": "test2"}, Priority.URGENT)
    layer.add_request_sync("req3", {"query": "test3"}, Priority.LOW)

    assert layer.stats["total_requests"] == 3
    assert layer.stats["urgent_requests"] == 1
    assert layer.stats["normal_requests"] == 1
    assert layer.stats["low_requests"] == 1


def test_priority_queues():
    """Test priority queue segregation"""
    layer = BatchingLayer(enable_priority=True)

    layer.add_request_sync("urgent1", "data", Priority.URGENT)
    layer.add_request_sync("normal1", "data", Priority.NORMAL)
    layer.add_request_sync("low1", "data", Priority.LOW)

    queue_sizes = layer.get_queue_sizes()

    assert queue_sizes["urgent"] == 1
    assert queue_sizes["normal"] == 1
    assert queue_sizes["low"] == 1
    assert queue_sizes["total"] == 3


def test_priority_disabled():
    """Test with priority disabled"""
    layer = BatchingLayer(enable_priority=False)

    layer.add_request_sync("req1", "data", Priority.URGENT)
    layer.add_request_sync("req2", "data", Priority.LOW)

    queue_sizes = layer.get_queue_sizes()

    # All go to normal queue when priority disabled
    assert queue_sizes["normal"] == 2
    assert queue_sizes["urgent"] == 0
    assert queue_sizes["low"] == 0


def test_batch_creation_priority_order():
    """Test batch creation respects priority order"""
    layer = BatchingLayer(max_batch_size=5)

    # Add in mixed order
    layer.add_request_sync("low1", "data", Priority.LOW)
    layer.add_request_sync("urgent1", "data", Priority.URGENT)
    layer.add_request_sync("normal1", "data", Priority.NORMAL)
    layer.add_request_sync("urgent2", "data", Priority.URGENT)
    layer.add_request_sync("low2", "data", Priority.LOW)

    batch = layer._create_batch()

    # Should be in order: urgent, urgent, normal, low, low
    assert len(batch) == 5
    assert batch[0].priority == Priority.URGENT
    assert batch[1].priority == Priority.URGENT
    assert batch[2].priority == Priority.NORMAL
    assert batch[3].priority == Priority.LOW
    assert batch[4].priority == Priority.LOW


def test_flush_sync():
    """Test synchronous batch flushing"""
    layer = BatchingLayer(max_batch_size=3)

    layer.add_request_sync("req1", "data1")
    layer.add_request_sync("req2", "data2")
    layer.add_request_sync("req3", "data3")

    batch = layer.flush_sync()

    assert len(batch) == 3
    assert layer.stats["total_batches"] == 1
    assert layer.get_queue_sizes()["total"] == 0


def test_max_batch_size_limit():
    """Test max batch size limit"""
    layer = BatchingLayer(max_batch_size=5)

    # Add 10 requests
    for i in range(10):
        layer.add_request_sync(f"req{i}", f"data{i}")

    batch = layer.flush_sync()

    # Should only take 5 (max batch size)
    assert len(batch) == 5
    assert layer.get_queue_sizes()["total"] == 5


@pytest.mark.asyncio
async def test_async_request_processing():
    """Test async request processing"""
    layer = BatchingLayer(
        max_batch_size=3,
        max_wait_time=0.1,
        flush_strategy=FlushStrategy.SIZE
    )

    # Mock processor
    async def processor(batch):
        await asyncio.sleep(0.01)
        return [f"result_{req.id}" for req in batch]

    # Start processing
    await layer.start_processing(processor)

    # Add requests and wait for results
    results = await asyncio.gather(
        layer.add_request("req1", "data1"),
        layer.add_request("req2", "data2"),
        layer.add_request("req3", "data3")
    )

    await layer.stop_processing()

    assert len(results) == 3
    assert results[0] == "result_req1"
    assert results[1] == "result_req2"
    assert results[2] == "result_req3"


@pytest.mark.asyncio
async def test_flush_strategy_size():
    """Test SIZE flush strategy"""
    layer = BatchingLayer(
        max_batch_size=3,
        flush_strategy=FlushStrategy.SIZE
    )

    async def processor(batch):
        return [f"result_{req.id}" for req in batch]

    await layer.start_processing(processor)

    # Add 3 requests - should trigger size-based flush
    task1 = asyncio.create_task(layer.add_request("req1", "data1"))
    task2 = asyncio.create_task(layer.add_request("req2", "data2"))
    task3 = asyncio.create_task(layer.add_request("req3", "data3"))

    results = await asyncio.gather(task1, task2, task3)
    await layer.stop_processing()

    assert len(results) == 3
    assert layer.stats["total_batches"] >= 1


@pytest.mark.asyncio
async def test_flush_strategy_time():
    """Test TIME flush strategy"""
    layer = BatchingLayer(
        max_batch_size=10,
        max_wait_time=0.1,
        min_batch_size=2,
        flush_strategy=FlushStrategy.TIME
    )

    async def processor(batch):
        return [f"result_{req.id}" for req in batch]

    await layer.start_processing(processor)

    # Add 2 requests (meets min_batch_size)
    task1 = asyncio.create_task(layer.add_request("req1", "data1"))
    task2 = asyncio.create_task(layer.add_request("req2", "data2"))

    # Wait for time-based flush
    results = await asyncio.gather(task1, task2)
    await layer.stop_processing()

    assert len(results) == 2


@pytest.mark.asyncio
async def test_flush_strategy_priority():
    """Test PRIORITY flush strategy"""
    layer = BatchingLayer(
        max_batch_size=10,
        flush_strategy=FlushStrategy.PRIORITY
    )

    async def processor(batch):
        await asyncio.sleep(0.01)
        return [f"result_{req.id}" for req in batch]

    await layer.start_processing(processor)

    # Add urgent request - should trigger immediate flush
    result = await layer.add_request("urgent1", "data", Priority.URGENT)
    await layer.stop_processing()

    assert result == "result_urgent1"


def test_statistics_tracking():
    """Test statistics tracking"""
    layer = BatchingLayer()

    layer.add_request_sync("req1", "data", Priority.URGENT)
    layer.add_request_sync("req2", "data", Priority.NORMAL)
    layer.add_request_sync("req3", "data", Priority.LOW)

    batch1 = layer.flush_sync()

    layer.add_request_sync("req4", "data")
    layer.add_request_sync("req5", "data")

    batch2 = layer.flush_sync()

    stats = layer.get_stats()

    assert stats["total_requests"] == 5
    assert stats["total_batches"] == 2
    assert stats["urgent_requests"] == 1
    assert stats["normal_requests"] == 3
    assert stats["low_requests"] == 1
    assert stats["avg_batch_size"] == 2.5  # (3 + 2) / 2


def test_optimize_batch_size():
    """Test batch size optimization"""
    layer = BatchingLayer()

    # Set some throughput stats
    layer.stats["avg_throughput"] = 100.0  # 100 req/sec

    # Optimize for 0.1s latency
    optimal_size = layer.optimize_batch_size(target_latency=0.1)

    assert optimal_size >= layer.min_batch_size
    assert optimal_size <= layer.max_batch_size


def test_queue_clearing():
    """Test queue clearing"""
    layer = BatchingLayer()

    layer.add_request_sync("req1", "data", Priority.URGENT)
    layer.add_request_sync("req2", "data", Priority.NORMAL)
    layer.add_request_sync("req3", "data", Priority.LOW)

    assert layer.get_queue_sizes()["total"] == 3

    layer.clear_queues()

    assert layer.get_queue_sizes()["total"] == 0
    assert layer.get_queue_sizes()["urgent"] == 0
    assert layer.get_queue_sizes()["normal"] == 0
    assert layer.get_queue_sizes()["low"] == 0


def test_stats_reset():
    """Test statistics reset"""
    layer = BatchingLayer()

    layer.add_request_sync("req1", "data")
    layer.flush_sync()

    assert layer.stats["total_requests"] == 1
    assert layer.stats["total_batches"] == 1

    layer.reset_stats()

    assert layer.stats["total_requests"] == 0
    assert layer.stats["total_batches"] == 0
    assert layer.stats["avg_batch_size"] == 0.0


def test_empty_flush():
    """Test flushing with empty queues"""
    layer = BatchingLayer()

    batch = layer.flush_sync()

    assert len(batch) == 0
    assert layer.stats["total_batches"] == 0


def test_partial_batch():
    """Test batch with fewer items than max size"""
    layer = BatchingLayer(max_batch_size=10)

    layer.add_request_sync("req1", "data")
    layer.add_request_sync("req2", "data")

    batch = layer.flush_sync()

    assert len(batch) == 2
    assert layer.stats["total_batches"] == 1
    assert layer.stats["avg_batch_size"] == 2.0


@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test concurrent request handling"""
    layer = BatchingLayer(
        max_batch_size=5,
        max_wait_time=0.1,
        flush_strategy=FlushStrategy.SIZE
    )

    async def processor(batch):
        await asyncio.sleep(0.01)
        return [f"result_{req.id}" for req in batch]

    await layer.start_processing(processor)

    # Add many concurrent requests
    tasks = [
        layer.add_request(f"req{i}", f"data{i}")
        for i in range(15)
    ]

    results = await asyncio.gather(*tasks)
    await layer.stop_processing()

    assert len(results) == 15
    assert layer.stats["total_requests"] == 15
    assert layer.stats["total_batches"] >= 3  # 15 requests / 5 max = 3 batches


@pytest.mark.asyncio
async def test_adaptive_flush_strategy():
    """Test ADAPTIVE flush strategy"""
    layer = BatchingLayer(
        max_batch_size=5,
        max_wait_time=0.1,
        min_batch_size=2,
        flush_strategy=FlushStrategy.ADAPTIVE
    )

    async def processor(batch):
        return [f"result_{req.id}" for req in batch]

    await layer.start_processing(processor)

    # Add urgent - should flush immediately
    urgent_result = await layer.add_request("urgent", "data", Priority.URGENT)

    # Add multiple normal - should flush on size
    tasks = [
        layer.add_request(f"req{i}", "data")
        for i in range(5)
    ]
    results = await asyncio.gather(*tasks)

    await layer.stop_processing()

    assert urgent_result == "result_urgent"
    assert len(results) == 5
