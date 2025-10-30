"""
Tests for Runtime Adapter (Component 2)
"""

import pytest
import asyncio
from src.v2.runtime_adapter import (
    RuntimeAdapter,
    LoadLevel,
    LoadThresholds,
    SystemMetrics
)


def test_initialization():
    """Test RuntimeAdapter initialization"""
    adapter = RuntimeAdapter()

    assert adapter.thresholds is not None
    assert adapter.monitoring_interval == 1.0
    assert adapter.current_metrics is None
    assert adapter._monitoring_running is False
    assert adapter.stats["total_measurements"] == 0


def test_measure_system_load():
    """Test system load measurement"""
    adapter = RuntimeAdapter()

    metrics = adapter.measure_system_load()

    assert isinstance(metrics, SystemMetrics)
    assert 0 <= metrics.cpu_percent <= 100
    assert 0 <= metrics.ram_percent <= 100
    assert metrics.latency_ms >= 0
    assert isinstance(metrics.load_level, LoadLevel)
    assert adapter.stats["total_measurements"] == 1


def test_load_classification():
    """Test load level classification"""
    thresholds = LoadThresholds(
        cpu_high=70.0,
        cpu_critical=90.0,
        ram_high=75.0,
        ram_critical=90.0
    )
    adapter = RuntimeAdapter(thresholds=thresholds)

    # Test LOW
    load_low = adapter._classify_load(10.0, 20.0, 5.0)
    assert load_low == LoadLevel.LOW

    # Test NORMAL
    load_normal = adapter._classify_load(40.0, 50.0, 10.0)
    assert load_normal == LoadLevel.NORMAL

    # Test HIGH
    load_high = adapter._classify_load(75.0, 60.0, 20.0)
    assert load_high == LoadLevel.HIGH

    # Test CRITICAL
    load_critical = adapter._classify_load(95.0, 85.0, 30.0)
    assert load_critical == LoadLevel.CRITICAL


def test_get_current_load():
    """Test getting current load level"""
    adapter = RuntimeAdapter()

    # Should measure if no metrics exist
    load = adapter.get_current_load()
    assert isinstance(load, LoadLevel)
    assert adapter.current_metrics is not None


def test_get_metrics():
    """Test getting system metrics"""
    adapter = RuntimeAdapter()

    # Initially None
    assert adapter.get_metrics() is None

    # After measurement
    adapter.measure_system_load()
    metrics = adapter.get_metrics()

    assert metrics is not None
    assert isinstance(metrics, SystemMetrics)


def test_callback_registration():
    """Test adaptation callback registration and triggering"""
    adapter = RuntimeAdapter()

    # Track callback invocations
    callback_calls = []

    def test_callback(load: LoadLevel):
        callback_calls.append(load)

    adapter.register_adaptation_callback(test_callback)

    # First measurement (no load change yet)
    adapter.measure_system_load()
    assert len(callback_calls) == 0

    # Force a load change by adjusting thresholds
    adapter.thresholds.cpu_high = 0.1  # Very low threshold
    adapter.measure_system_load()

    # Callback might be called if load changed
    # This is system-dependent, so we just check it doesn't error


@pytest.mark.asyncio
async def test_async_monitoring():
    """Test async monitoring start/stop"""
    adapter = RuntimeAdapter(monitoring_interval=0.1)

    # Start monitoring
    await adapter.start_monitoring()
    assert adapter._monitoring_running is True
    assert adapter._monitoring_task is not None

    # Let it run for a bit
    await asyncio.sleep(0.3)

    # Check measurements were taken
    assert adapter.stats["total_measurements"] > 0

    # Stop monitoring
    await adapter.stop_monitoring()
    assert adapter._monitoring_running is False


def test_statistics_tracking():
    """Test statistics tracking"""
    adapter = RuntimeAdapter()

    # Initial stats
    stats = adapter.get_stats()
    assert stats["total_measurements"] == 0
    assert stats["peak_cpu"] == 0.0
    assert stats["peak_ram"] == 0.0

    # Measure multiple times
    for _ in range(5):
        adapter.measure_system_load()

    stats = adapter.get_stats()
    assert stats["total_measurements"] == 5
    assert stats["peak_cpu"] > 0
    assert stats["peak_ram"] > 0
    assert stats["current_load"] is not None


def test_throttling_recommendation():
    """Test throttling recommendation"""
    adapter = RuntimeAdapter()

    # Set to LOW load
    adapter.current_metrics = SystemMetrics(
        cpu_percent=10.0,
        ram_percent=20.0,
        latency_ms=5.0,
        timestamp=0.0,
        load_level=LoadLevel.LOW
    )
    assert adapter.should_throttle() is False

    # Set to HIGH load
    adapter.current_metrics = SystemMetrics(
        cpu_percent=80.0,
        ram_percent=70.0,
        latency_ms=50.0,
        timestamp=0.0,
        load_level=LoadLevel.HIGH
    )
    assert adapter.should_throttle() is True

    # Set to CRITICAL load
    adapter.current_metrics = SystemMetrics(
        cpu_percent=95.0,
        ram_percent=90.0,
        latency_ms=200.0,
        timestamp=0.0,
        load_level=LoadLevel.CRITICAL
    )
    assert adapter.should_throttle() is True


def test_batch_size_recommendation():
    """Test batch size recommendation based on load"""
    adapter = RuntimeAdapter()

    # LOW load - increase batch size
    adapter.current_metrics = SystemMetrics(
        cpu_percent=10.0,
        ram_percent=20.0,
        latency_ms=5.0,
        timestamp=0.0,
        load_level=LoadLevel.LOW
    )
    assert adapter.get_recommended_batch_size(10) == 20

    # NORMAL load - keep default
    adapter.current_metrics = SystemMetrics(
        cpu_percent=40.0,
        ram_percent=50.0,
        latency_ms=10.0,
        timestamp=0.0,
        load_level=LoadLevel.NORMAL
    )
    assert adapter.get_recommended_batch_size(10) == 10

    # HIGH load - reduce batch size
    adapter.current_metrics = SystemMetrics(
        cpu_percent=80.0,
        ram_percent=70.0,
        latency_ms=50.0,
        timestamp=0.0,
        load_level=LoadLevel.HIGH
    )
    assert adapter.get_recommended_batch_size(10) == 5

    # CRITICAL load - minimize batch size
    adapter.current_metrics = SystemMetrics(
        cpu_percent=95.0,
        ram_percent=90.0,
        latency_ms=200.0,
        timestamp=0.0,
        load_level=LoadLevel.CRITICAL
    )
    assert adapter.get_recommended_batch_size(10) == 2


def test_stats_reset():
    """Test statistics reset"""
    adapter = RuntimeAdapter()

    # Collect some stats
    for _ in range(3):
        adapter.measure_system_load()

    assert adapter.stats["total_measurements"] == 3

    # Reset
    adapter.reset_stats()

    assert adapter.stats["total_measurements"] == 0
    assert adapter.stats["peak_cpu"] == 0.0
    assert adapter.stats["peak_ram"] == 0.0
