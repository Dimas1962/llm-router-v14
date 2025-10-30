"""
Batching Layer - Component 6
Request batching with priority queues and async processing
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Awaitable
from enum import Enum
from collections import deque


class Priority(Enum):
    """Request priority levels"""
    URGENT = 3
    NORMAL = 2
    LOW = 1


class FlushStrategy(Enum):
    """Batch flush strategies"""
    SIZE = "size"          # Flush when batch reaches size limit
    TIME = "time"          # Flush after timeout
    PRIORITY = "priority"  # Flush urgent requests immediately
    ADAPTIVE = "adaptive"  # Adapt based on throughput


@dataclass
class BatchRequest:
    """A batched request"""
    id: str
    data: Any
    priority: Priority
    timestamp: float = field(default_factory=time.time)
    future: Optional[asyncio.Future] = None


@dataclass
class BatchResult:
    """Result of batch processing"""
    batch_id: str
    requests_processed: int
    processing_time: float
    throughput: float  # requests per second
    batch_size: int
    wait_times: List[float]


class BatchingLayer:
    """
    Request Batching Layer

    Features:
    - Dynamic batch sizing
    - Priority queues (urgent/normal/low)
    - Timeout handling
    - Multiple flush strategies
    - Async batch processing
    - Throughput optimization
    - Statistics tracking
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_time: float = 0.5,  # seconds
        min_batch_size: int = 4,
        flush_strategy: FlushStrategy = FlushStrategy.ADAPTIVE,
        enable_priority: bool = True
    ):
        """
        Initialize Batching Layer

        Args:
            max_batch_size: Maximum requests per batch
            max_wait_time: Maximum wait time before flush
            min_batch_size: Minimum batch size for efficiency
            flush_strategy: Strategy for flushing batches
            enable_priority: Enable priority queue handling
        """
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.min_batch_size = min_batch_size
        self.flush_strategy = flush_strategy
        self.enable_priority = enable_priority

        # Priority queues
        self.urgent_queue: deque = deque()
        self.normal_queue: deque = deque()
        self.low_queue: deque = deque()

        # Processing state
        self._processing_task: Optional[asyncio.Task] = None
        self._running = False
        self._batch_counter = 0

        # Statistics
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "avg_batch_size": 0.0,
            "avg_wait_time": 0.0,
            "avg_throughput": 0.0,
            "urgent_requests": 0,
            "normal_requests": 0,
            "low_requests": 0
        }
        self._total_batch_size = 0
        self._total_wait_time = 0.0
        self._total_throughput = 0.0

    async def add_request(
        self,
        request_id: str,
        data: Any,
        priority: Priority = Priority.NORMAL
    ) -> Any:
        """
        Add request to batch queue

        Args:
            request_id: Unique request identifier
            data: Request data
            priority: Request priority

        Returns:
            Result from batch processing
        """
        # Create request with future
        future = asyncio.Future()
        request = BatchRequest(
            id=request_id,
            data=data,
            priority=priority,
            future=future
        )

        # Add to appropriate queue
        if self.enable_priority:
            if priority == Priority.URGENT:
                self.urgent_queue.append(request)
                self.stats["urgent_requests"] += 1
            elif priority == Priority.NORMAL:
                self.normal_queue.append(request)
                self.stats["normal_requests"] += 1
            else:  # LOW
                self.low_queue.append(request)
                self.stats["low_requests"] += 1
        else:
            self.normal_queue.append(request)
            self.stats["normal_requests"] += 1

        self.stats["total_requests"] += 1

        # Wait for result
        return await future

    def add_request_sync(
        self,
        request_id: str,
        data: Any,
        priority: Priority = Priority.NORMAL
    ):
        """
        Add request synchronously (for testing)

        Args:
            request_id: Unique request identifier
            data: Request data
            priority: Request priority
        """
        request = BatchRequest(
            id=request_id,
            data=data,
            priority=priority,
            future=None
        )

        if self.enable_priority:
            if priority == Priority.URGENT:
                self.urgent_queue.append(request)
                self.stats["urgent_requests"] += 1
            elif priority == Priority.NORMAL:
                self.normal_queue.append(request)
                self.stats["normal_requests"] += 1
            else:
                self.low_queue.append(request)
                self.stats["low_requests"] += 1
        else:
            self.normal_queue.append(request)
            self.stats["normal_requests"] += 1

        self.stats["total_requests"] += 1

    async def start_processing(
        self,
        processor: Callable[[List[BatchRequest]], Awaitable[List[Any]]]
    ):
        """
        Start async batch processing

        Args:
            processor: Async function to process batches
        """
        if self._running:
            return

        self._running = True
        self._processing_task = asyncio.create_task(
            self._processing_loop(processor)
        )

    async def stop_processing(self):
        """Stop batch processing"""
        self._running = False
        if self._processing_task:
            await self._processing_task
            self._processing_task = None

    async def _processing_loop(
        self,
        processor: Callable[[List[BatchRequest]], Awaitable[List[Any]]]
    ):
        """
        Main processing loop

        Args:
            processor: Async function to process batches
        """
        last_flush_time = time.time()

        while self._running:
            current_time = time.time()
            time_since_flush = current_time - last_flush_time

            # Check if we should flush
            should_flush = False

            if self.flush_strategy == FlushStrategy.SIZE:
                # Flush when batch reaches max size
                should_flush = self._get_total_queue_size() >= self.max_batch_size

            elif self.flush_strategy == FlushStrategy.TIME:
                # Flush after timeout
                should_flush = (
                    time_since_flush >= self.max_wait_time and
                    self._get_total_queue_size() >= self.min_batch_size
                )

            elif self.flush_strategy == FlushStrategy.PRIORITY:
                # Flush if urgent requests present or timeout
                should_flush = (
                    len(self.urgent_queue) > 0 or
                    (time_since_flush >= self.max_wait_time and
                     self._get_total_queue_size() >= self.min_batch_size)
                )

            elif self.flush_strategy == FlushStrategy.ADAPTIVE:
                # Adaptive: combine size and time
                queue_size = self._get_total_queue_size()
                should_flush = (
                    queue_size >= self.max_batch_size or
                    (time_since_flush >= self.max_wait_time and
                     queue_size >= self.min_batch_size) or
                    len(self.urgent_queue) > 0
                )

            if should_flush and self._get_total_queue_size() > 0:
                # Create and process batch
                batch = self._create_batch()
                await self._process_batch(batch, processor)
                last_flush_time = time.time()

            # Small sleep to prevent busy waiting
            await asyncio.sleep(0.01)

    def _get_total_queue_size(self) -> int:
        """Get total size across all queues"""
        return (
            len(self.urgent_queue) +
            len(self.normal_queue) +
            len(self.low_queue)
        )

    def _create_batch(self) -> List[BatchRequest]:
        """
        Create batch from queues (priority order)

        Returns:
            List of BatchRequest objects
        """
        batch = []

        # Priority order: urgent -> normal -> low
        while len(batch) < self.max_batch_size and len(self.urgent_queue) > 0:
            batch.append(self.urgent_queue.popleft())

        while len(batch) < self.max_batch_size and len(self.normal_queue) > 0:
            batch.append(self.normal_queue.popleft())

        while len(batch) < self.max_batch_size and len(self.low_queue) > 0:
            batch.append(self.low_queue.popleft())

        return batch

    async def _process_batch(
        self,
        batch: List[BatchRequest],
        processor: Callable[[List[BatchRequest]], Awaitable[List[Any]]]
    ):
        """
        Process a batch of requests

        Args:
            batch: List of requests to process
            processor: Async processing function
        """
        if not batch:
            return

        start_time = time.time()

        # Process batch
        try:
            results = await processor(batch)

            # Set results on futures
            for request, result in zip(batch, results):
                if request.future and not request.future.done():
                    request.future.set_result(result)

        except Exception as e:
            # Set exception on futures
            for request in batch:
                if request.future and not request.future.done():
                    request.future.set_exception(e)

        # Calculate statistics
        processing_time = time.time() - start_time
        throughput = len(batch) / processing_time if processing_time > 0 else 0

        wait_times = [start_time - req.timestamp for req in batch]

        # Update stats
        self._batch_counter += 1
        self.stats["total_batches"] += 1
        self._total_batch_size += len(batch)
        self._total_wait_time += sum(wait_times)
        self._total_throughput += throughput

        self.stats["avg_batch_size"] = self._total_batch_size / self.stats["total_batches"]
        self.stats["avg_wait_time"] = self._total_wait_time / self.stats["total_requests"]
        self.stats["avg_throughput"] = self._total_throughput / self.stats["total_batches"]

    def flush_sync(self) -> List[BatchRequest]:
        """
        Flush current batch synchronously

        Returns:
            List of requests in the flushed batch
        """
        batch = self._create_batch()

        if batch:
            self._batch_counter += 1
            self.stats["total_batches"] += 1
            self._total_batch_size += len(batch)
            self.stats["avg_batch_size"] = self._total_batch_size / self.stats["total_batches"]

        return batch

    def get_queue_sizes(self) -> Dict[str, int]:
        """
        Get current queue sizes

        Returns:
            Dictionary with queue sizes
        """
        return {
            "urgent": len(self.urgent_queue),
            "normal": len(self.normal_queue),
            "low": len(self.low_queue),
            "total": self._get_total_queue_size()
        }

    def optimize_batch_size(self, target_latency: float) -> int:
        """
        Optimize batch size for target latency

        Args:
            target_latency: Target processing latency

        Returns:
            Recommended batch size
        """
        if self.stats["avg_throughput"] == 0:
            return self.max_batch_size

        # Calculate optimal size based on throughput and latency
        optimal_size = int(self.stats["avg_throughput"] * target_latency)

        # Constrain to limits
        optimal_size = max(self.min_batch_size, optimal_size)
        optimal_size = min(self.max_batch_size, optimal_size)

        return optimal_size

    def get_stats(self) -> Dict[str, Any]:
        """
        Get batching statistics

        Returns:
            Statistics dictionary
        """
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "avg_batch_size": 0.0,
            "avg_wait_time": 0.0,
            "avg_throughput": 0.0,
            "urgent_requests": 0,
            "normal_requests": 0,
            "low_requests": 0
        }
        self._total_batch_size = 0
        self._total_wait_time = 0.0
        self._total_throughput = 0.0
        self._batch_counter = 0

    def clear_queues(self):
        """Clear all queues"""
        self.urgent_queue.clear()
        self.normal_queue.clear()
        self.low_queue.clear()
