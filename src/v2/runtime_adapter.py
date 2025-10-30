"""
Runtime Adapter - Component 2
Dynamic system resource monitoring and load classification
"""

import asyncio
import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
import psutil


class LoadLevel(Enum):
    """System load classification levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SystemMetrics:
    """Current system metrics"""
    cpu_percent: float
    ram_percent: float
    latency_ms: float
    timestamp: float
    load_level: LoadLevel


@dataclass
class LoadThresholds:
    """Thresholds for load classification"""
    cpu_high: float = 70.0
    cpu_critical: float = 90.0
    ram_high: float = 75.0
    ram_critical: float = 90.0
    latency_high: float = 100.0
    latency_critical: float = 500.0


class RuntimeAdapter:
    """
    Runtime Adapter for dynamic system monitoring and adaptation

    Features:
    - Real-time CPU/RAM monitoring via psutil
    - Latency tracking
    - Load level classification (LOW/NORMAL/HIGH/CRITICAL)
    - Async monitoring loop
    - Strategy adaptation callbacks
    """

    def __init__(
        self,
        thresholds: Optional[LoadThresholds] = None,
        monitoring_interval: float = 1.0,
        enable_auto_monitoring: bool = False
    ):
        """
        Initialize Runtime Adapter

        Args:
            thresholds: Custom load thresholds
            monitoring_interval: Seconds between monitoring updates
            enable_auto_monitoring: Start monitoring loop automatically
        """
        self.thresholds = thresholds or LoadThresholds()
        self.monitoring_interval = monitoring_interval

        # Current metrics
        self.current_metrics: Optional[SystemMetrics] = None
        self._last_check_time: float = 0

        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_running = False

        # Strategy adaptation callbacks
        self._adaptation_callbacks: list[Callable[[LoadLevel], None]] = []

        # Statistics
        self.stats = {
            "total_measurements": 0,
            "load_changes": 0,
            "peak_cpu": 0.0,
            "peak_ram": 0.0,
            "peak_latency": 0.0
        }

        # Auto-start monitoring if enabled
        if enable_auto_monitoring:
            # Note: This will only work if called from an async context
            # For testing, we'll manually start the monitoring
            pass

    def measure_system_load(self) -> SystemMetrics:
        """
        Measure current system load

        Returns:
            SystemMetrics with current CPU, RAM, latency, and load level
        """
        # Measure latency (time to get metrics)
        start_time = time.time()

        # Get CPU and RAM usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        ram_percent = psutil.virtual_memory().percent

        latency_ms = (time.time() - start_time) * 1000

        # Classify load level
        load_level = self._classify_load(cpu_percent, ram_percent, latency_ms)

        # Create metrics
        metrics = SystemMetrics(
            cpu_percent=cpu_percent,
            ram_percent=ram_percent,
            latency_ms=latency_ms,
            timestamp=time.time(),
            load_level=load_level
        )

        # Update current metrics
        old_load = self.current_metrics.load_level if self.current_metrics else None
        self.current_metrics = metrics
        self._last_check_time = metrics.timestamp

        # Update statistics
        self.stats["total_measurements"] += 1
        self.stats["peak_cpu"] = max(self.stats["peak_cpu"], cpu_percent)
        self.stats["peak_ram"] = max(self.stats["peak_ram"], ram_percent)
        self.stats["peak_latency"] = max(self.stats["peak_latency"], latency_ms)

        # Track load changes
        if old_load and old_load != load_level:
            self.stats["load_changes"] += 1
            # Trigger adaptation callbacks
            self._trigger_adaptation(load_level)

        return metrics

    def _classify_load(
        self,
        cpu_percent: float,
        ram_percent: float,
        latency_ms: float
    ) -> LoadLevel:
        """
        Classify system load based on metrics

        Args:
            cpu_percent: CPU usage percentage
            ram_percent: RAM usage percentage
            latency_ms: Measurement latency in milliseconds

        Returns:
            LoadLevel classification
        """
        # CRITICAL: Any metric exceeds critical threshold
        if (cpu_percent >= self.thresholds.cpu_critical or
            ram_percent >= self.thresholds.ram_critical or
            latency_ms >= self.thresholds.latency_critical):
            return LoadLevel.CRITICAL

        # HIGH: Any metric exceeds high threshold
        if (cpu_percent >= self.thresholds.cpu_high or
            ram_percent >= self.thresholds.ram_high or
            latency_ms >= self.thresholds.latency_high):
            return LoadLevel.HIGH

        # NORMAL: Moderate usage
        if cpu_percent >= 30.0 or ram_percent >= 40.0:
            return LoadLevel.NORMAL

        # LOW: Minimal usage
        return LoadLevel.LOW

    def get_current_load(self) -> LoadLevel:
        """
        Get current load level (measures if needed)

        Returns:
            Current LoadLevel
        """
        if not self.current_metrics:
            self.measure_system_load()
        return self.current_metrics.load_level

    def get_metrics(self) -> Optional[SystemMetrics]:
        """
        Get current system metrics

        Returns:
            Latest SystemMetrics or None if no measurements taken
        """
        return self.current_metrics

    def register_adaptation_callback(
        self,
        callback: Callable[[LoadLevel], None]
    ):
        """
        Register a callback for load level changes

        Args:
            callback: Function to call when load level changes
        """
        self._adaptation_callbacks.append(callback)

    def _trigger_adaptation(self, new_load: LoadLevel):
        """
        Trigger adaptation callbacks

        Args:
            new_load: New load level
        """
        for callback in self._adaptation_callbacks:
            try:
                callback(new_load)
            except Exception:
                # Don't let callback errors break the adapter
                pass

    async def start_monitoring(self):
        """
        Start async monitoring loop
        """
        if self._monitoring_running:
            return

        self._monitoring_running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self):
        """
        Stop async monitoring loop
        """
        self._monitoring_running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

    async def _monitoring_loop(self):
        """
        Continuous monitoring loop
        """
        while self._monitoring_running:
            try:
                self.measure_system_load()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                # Continue monitoring even if measurement fails
                await asyncio.sleep(self.monitoring_interval)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get monitoring statistics

        Returns:
            Dictionary with statistics
        """
        return {
            **self.stats,
            "current_load": self.current_metrics.load_level.value if self.current_metrics else None,
            "monitoring_active": self._monitoring_running
        }

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "total_measurements": 0,
            "load_changes": 0,
            "peak_cpu": 0.0,
            "peak_ram": 0.0,
            "peak_latency": 0.0
        }

    def should_throttle(self) -> bool:
        """
        Determine if operations should be throttled

        Returns:
            True if system is under HIGH or CRITICAL load
        """
        load = self.get_current_load()
        return load in (LoadLevel.HIGH, LoadLevel.CRITICAL)

    def get_recommended_batch_size(self, default: int = 10) -> int:
        """
        Get recommended batch size based on current load

        Args:
            default: Default batch size

        Returns:
            Recommended batch size
        """
        load = self.get_current_load()

        if load == LoadLevel.CRITICAL:
            return max(1, default // 4)
        elif load == LoadLevel.HIGH:
            return max(1, default // 2)
        elif load == LoadLevel.LOW:
            return default * 2
        else:  # NORMAL
            return default
