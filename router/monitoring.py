"""
Production Monitoring with Prometheus Metrics
Phase 6: Advanced Features
"""

import logging
import time
from typing import Dict, Any, Optional

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, CollectorRegistry, generate_latest,
        CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Histogram = Gauge = CollectorRegistry = None


logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """
    Prometheus metrics for router monitoring

    Tracks:
    - Routing latency (histogram)
    - Model usage (counter)
    - Error rates (counter)
    - Eagle ELO progression (gauge)
    - CARROT cost tracking (counter)
    """

    def __init__(self, registry: Optional[Any] = None):
        """
        Initialize Prometheus metrics

        Args:
            registry: Optional Prometheus registry (creates new if None)
        """
        if not PROMETHEUS_AVAILABLE:
            logger.warning(
                "Prometheus client not available. Install prometheus-client "
                "for metrics support."
            )
            self.enabled = False
            return

        self.enabled = True
        self.registry = registry or CollectorRegistry()

        # Routing latency histogram (ms)
        self.routing_latency = Histogram(
            'router_latency_milliseconds',
            'Routing decision latency in milliseconds',
            buckets=(5, 10, 25, 50, 75, 100, 250, 500, 1000),
            registry=self.registry
        )

        # Model selection counter
        self.model_selections = Counter(
            'router_model_selections_total',
            'Total number of model selections',
            ['model'],
            registry=self.registry
        )

        # Error counter
        self.routing_errors = Counter(
            'router_errors_total',
            'Total number of routing errors',
            ['error_type'],
            registry=self.registry
        )

        # Success counter
        self.routing_successes = Counter(
            'router_successes_total',
            'Total number of successful routings',
            registry=self.registry
        )

        # Eagle ELO gauge
        self.eagle_elo = Gauge(
            'router_eagle_elo',
            'Current Eagle ELO rating for models',
            ['model'],
            registry=self.registry
        )

        # CARROT cost counter
        self.carrot_cost = Counter(
            'router_carrot_cost_total',
            'Total accumulated cost from CARROT',
            ['model'],
            registry=self.registry
        )

        # Feedback counter
        self.feedback_count = Counter(
            'router_feedback_total',
            'Total feedback submissions',
            ['success'],
            registry=self.registry
        )

        # Memory size gauge
        self.memory_size = Gauge(
            'router_memory_size',
            'Current size of associative memory',
            registry=self.registry
        )

        # Active sessions gauge
        self.active_sessions = Gauge(
            'router_active_sessions',
            'Number of active multi-round sessions',
            registry=self.registry
        )

        logger.info("PrometheusMetrics initialized")

    def record_routing_latency(self, latency_ms: float):
        """Record routing latency in milliseconds"""
        if self.enabled:
            self.routing_latency.observe(latency_ms)

    def record_model_selection(self, model: str):
        """Record a model selection"""
        if self.enabled:
            self.model_selections.labels(model=model).inc()

    def record_error(self, error_type: str):
        """Record an error"""
        if self.enabled:
            self.routing_errors.labels(error_type=error_type).inc()

    def record_success(self):
        """Record a successful routing"""
        if self.enabled:
            self.routing_successes.inc()

    def update_eagle_elo(self, model: str, elo: float):
        """Update Eagle ELO for a model"""
        if self.enabled:
            self.eagle_elo.labels(model=model).set(elo)

    def record_carrot_cost(self, model: str, cost: float):
        """Record CARROT cost"""
        if self.enabled:
            self.carrot_cost.labels(model=model).inc(cost)

    def record_feedback(self, success: bool):
        """Record feedback submission"""
        if self.enabled:
            self.feedback_count.labels(success=str(success)).inc()

    def update_memory_size(self, size: int):
        """Update memory size"""
        if self.enabled:
            self.memory_size.set(size)

    def update_active_sessions(self, count: int):
        """Update active session count"""
        if self.enabled:
            self.active_sessions.set(count)

    def generate_metrics(self) -> bytes:
        """
        Generate Prometheus metrics in text format

        Returns:
            Metrics in Prometheus text format
        """
        if not self.enabled:
            return b"# Prometheus metrics not available\n"

        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """Get Prometheus content type"""
        if self.enabled:
            return CONTENT_TYPE_LATEST
        return "text/plain"


class PerformanceTracker:
    """
    Tracks performance metrics without Prometheus

    Fallback for when Prometheus is not available
    """

    def __init__(self):
        """Initialize performance tracker"""
        self.metrics = {
            "routing_count": 0,
            "success_count": 0,
            "error_count": 0,
            "total_latency_ms": 0.0,
            "model_selections": {},
            "feedback_count": 0
        }

        self.latency_samples = []

        logger.info("PerformanceTracker initialized (fallback mode)")

    def record_routing(self, latency_ms: float, model: str, success: bool):
        """Record a routing decision"""
        self.metrics["routing_count"] += 1

        if success:
            self.metrics["success_count"] += 1
        else:
            self.metrics["error_count"] += 1

        self.metrics["total_latency_ms"] += latency_ms
        self.latency_samples.append(latency_ms)

        # Keep last 1000 samples
        if len(self.latency_samples) > 1000:
            self.latency_samples = self.latency_samples[-1000:]

        # Track model selections
        if model not in self.metrics["model_selections"]:
            self.metrics["model_selections"][model] = 0
        self.metrics["model_selections"][model] += 1

    def record_feedback(self):
        """Record feedback submission"""
        self.metrics["feedback_count"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        count = self.metrics["routing_count"]

        if count == 0:
            return {
                "total_routings": 0,
                "success_rate": 0.0,
                "avg_latency_ms": 0.0
            }

        # Calculate percentiles
        percentiles = self._calculate_percentiles()

        return {
            "total_routings": count,
            "success_count": self.metrics["success_count"],
            "error_count": self.metrics["error_count"],
            "success_rate": self.metrics["success_count"] / count,
            "avg_latency_ms": self.metrics["total_latency_ms"] / count,
            "p50_latency_ms": percentiles.get("p50", 0.0),
            "p95_latency_ms": percentiles.get("p95", 0.0),
            "p99_latency_ms": percentiles.get("p99", 0.0),
            "model_selections": self.metrics["model_selections"],
            "feedback_count": self.metrics["feedback_count"]
        }

    def _calculate_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentiles"""
        if not self.latency_samples:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        sorted_samples = sorted(self.latency_samples)
        n = len(sorted_samples)

        def percentile(p):
            k = (n - 1) * p
            f = int(k)
            c = k - f
            if f + 1 < n:
                return sorted_samples[f] * (1 - c) + sorted_samples[f + 1] * c
            return sorted_samples[f]

        return {
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99)
        }

    def reset(self):
        """Reset all metrics"""
        self.metrics = {
            "routing_count": 0,
            "success_count": 0,
            "error_count": 0,
            "total_latency_ms": 0.0,
            "model_selections": {},
            "feedback_count": 0
        }
        self.latency_samples = []
        logger.info("Performance metrics reset")
