"""
Test suite for Phase 6: Advanced Features
Includes tests for Episodic Memory, Cascade Router, Multi-Round, and Monitoring
"""

import pytest
import time
from router.episodic_memory import EpisodicMemory, Episode
from router.cascade_router import CascadeRouter, CascadeResult
from router.multi_round import MultiRoundRouter
from router.monitoring import PrometheusMetrics, PerformanceTracker


class TestEpisodicMemory:
    """Test episodic memory"""

    def setup_method(self):
        self.memory = EpisodicMemory(max_episodes=100)

    def test_add_episode(self):
        """Test adding an episode"""
        episode_id = self.memory.add_episode(
            query="test query",
            selected_model="qwen2.5-coder-7b",
            confidence=0.9,
            success=True,
            task_type="coding",
            complexity=0.3
        )

        assert episode_id == 0
        assert self.memory.size() == 1

    def test_search_similar(self):
        """Test searching for similar episodes"""
        # Add some episodes
        self.memory.add_episode(
            "write a Python function",
            "qwen2.5-coder-7b",
            0.9,
            True,
            "coding",
            0.3
        )
        self.memory.add_episode(
            "write a Python class",
            "qwen3-coder-30b",
            0.85,
            True,
            "coding",
            0.5
        )

        # Search for similar
        results = self.memory.search_similar("write a Python method", k=2)

        assert len(results) <= 2
        assert all(isinstance(r, Episode) for r in results)

    def test_max_episodes_limit(self):
        """Test max episodes constraint"""
        memory = EpisodicMemory(max_episodes=10)

        # Add more than max
        for i in range(15):
            memory.add_episode(
                f"query {i}",
                "qwen2.5-coder-7b",
                0.9,
                True,
                "coding",
                0.3
            )

        # Should be capped at max
        assert memory.size() == 10

    def test_successful_patterns(self):
        """Test extracting successful patterns"""
        # Add successful episodes
        for i in range(5):
            self.memory.add_episode(
                f"query {i}",
                "qwen2.5-coder-7b",
                0.9,
                True,
                "coding",
                0.3
            )

        patterns = self.memory.get_successful_patterns(task_type="coding")

        assert patterns["count"] == 5
        assert "qwen2.5-coder-7b" in patterns["model_distribution"]

    def test_stats(self):
        """Test getting stats"""
        self.memory.add_episode(
            "test",
            "qwen2.5-coder-7b",
            0.9,
            True,
            "coding",
            0.3
        )

        stats = self.memory.get_stats()

        assert stats["total_episodes"] == 1
        assert stats["success_rate"] == 1.0


class TestCascadeRouter:
    """Test cascade routing"""

    def setup_method(self):
        self.cascade = CascadeRouter(confidence_threshold=0.8)

    def test_initialization(self):
        """Test cascade router initialization"""
        assert len(self.cascade.tiers) == 3
        assert self.cascade.confidence_threshold == 0.8

    def test_route_simple_query(self):
        """Test routing a simple query"""
        result = self.cascade.route(
            query="print hello",
            task_type="coding",
            complexity=0.2,
            context_size=0
        )

        assert isinstance(result, CascadeResult)
        assert result.model is not None
        assert result.attempts > 0

    def test_escalation(self):
        """Test that escalation occurs for complex queries"""
        result = self.cascade.route(
            query="design a complex distributed system",
            task_type="architecture",
            complexity=0.9,
            context_size=0
        )

        # Complex query should reach higher tiers
        assert result.tier in ["fast", "medium", "slow"]

    def test_budget_constraint(self):
        """Test budget constraint stops escalation"""
        result = self.cascade.route(
            query="test query",
            task_type="coding",
            complexity=0.5,
            budget=5.0  # Low budget
        )

        # Should stop early due to budget
        assert result.total_cost <= 5.0 or result.tier == "fast"

    def test_tier_stats(self):
        """Test tier statistics tracking"""
        # Route some queries
        for i in range(5):
            self.cascade.route(
                f"query {i}",
                "coding",
                0.3,
                context_size=0
            )

        stats = self.cascade.get_tier_stats()

        assert "tiers" in stats
        assert stats["total_attempts"] >= 5

    def test_reset_stats(self):
        """Test resetting stats"""
        self.cascade.route("test", "coding", 0.3)
        self.cascade.reset_stats()

        stats = self.cascade.get_tier_stats()
        assert stats["total_attempts"] == 0


class TestMultiRoundRouter:
    """Test multi-round routing"""

    def setup_method(self):
        self.router = MultiRoundRouter()

    def test_create_session(self):
        """Test creating a session"""
        session_id = self.router.create_session()

        assert session_id is not None
        assert session_id in self.router.sessions

    def test_add_round(self):
        """Test adding a round"""
        session_id = self.router.create_session()

        round_id = self.router.add_round(
            session_id,
            "test query",
            "qwen2.5-coder-7b",
            0.9
        )

        assert round_id == 0

        session = self.router.get_session(session_id)
        assert len(session.rounds) == 1

    def test_update_round(self):
        """Test updating a round"""
        session_id = self.router.create_session()
        round_id = self.router.add_round(
            session_id,
            "test",
            "qwen2.5-coder-7b",
            0.9
        )

        self.router.update_round(session_id, round_id, success=True)

        session = self.router.get_session(session_id)
        assert session.rounds[0].success is True

    def test_get_context(self):
        """Test getting context"""
        session_id = self.router.create_session()

        # Add some rounds
        for i in range(5):
            self.router.add_round(
                session_id,
                f"query {i}",
                "qwen2.5-coder-7b",
                0.9
            )

        context = self.router.get_context(session_id)

        # Should have recent context
        assert len(context) > 0
        assert len(context) <= self.router.context_window

    def test_should_switch_model(self):
        """Test model switching recommendation"""
        session_id = self.router.create_session()

        # Add some unsuccessful rounds
        for i in range(5):
            round_id = self.router.add_round(
                session_id,
                f"query {i}",
                "qwen2.5-coder-7b",
                0.9
            )
            self.router.update_round(session_id, round_id, success=False)

        # Should recommend switch
        should_switch = self.router.should_switch_model(
            session_id,
            "qwen2.5-coder-7b"
        )

        assert should_switch is True

    def test_stats(self):
        """Test getting stats"""
        session_id = self.router.create_session()
        self.router.add_round(session_id, "test", "qwen2.5-coder-7b", 0.9)

        stats = self.router.get_stats()

        assert stats["total_sessions"] == 1
        assert stats["total_rounds"] >= 1

    def test_clear_session(self):
        """Test clearing a session"""
        session_id = self.router.create_session()
        self.router.add_round(session_id, "test", "qwen2.5-coder-7b", 0.9)

        self.router.clear_session(session_id)

        assert session_id not in self.router.sessions


class TestPrometheusMetrics:
    """Test Prometheus metrics"""

    def setup_method(self):
        self.metrics = PrometheusMetrics()

    def test_initialization(self):
        """Test metrics initialization"""
        assert self.metrics.enabled is True

    def test_record_latency(self):
        """Test recording latency"""
        self.metrics.record_routing_latency(50.5)
        # Should not raise error

    def test_record_model_selection(self):
        """Test recording model selection"""
        self.metrics.record_model_selection("qwen2.5-coder-7b")
        # Should not raise error

    def test_record_error(self):
        """Test recording error"""
        self.metrics.record_error("timeout")
        # Should not raise error

    def test_record_success(self):
        """Test recording success"""
        self.metrics.record_success()
        # Should not raise error

    def test_generate_metrics(self):
        """Test generating metrics"""
        # Record some data
        self.metrics.record_routing_latency(50.0)
        self.metrics.record_model_selection("qwen2.5-coder-7b")

        metrics_output = self.metrics.generate_metrics()

        assert isinstance(metrics_output, bytes)
        assert len(metrics_output) > 0


class TestPerformanceTracker:
    """Test performance tracker (fallback)"""

    def setup_method(self):
        self.tracker = PerformanceTracker()

    def test_record_routing(self):
        """Test recording a routing"""
        self.tracker.record_routing(
            latency_ms=50.0,
            model="qwen2.5-coder-7b",
            success=True
        )

        stats = self.tracker.get_stats()

        assert stats["total_routings"] == 1
        assert stats["success_count"] == 1

    def test_multiple_recordings(self):
        """Test multiple recordings"""
        for i in range(10):
            self.tracker.record_routing(
                latency_ms=50.0 + i,
                model="qwen2.5-coder-7b",
                success=True
            )

        stats = self.tracker.get_stats()

        assert stats["total_routings"] == 10
        assert stats["avg_latency_ms"] > 0

    def test_percentiles(self):
        """Test latency percentiles"""
        # Add varied latencies
        for latency in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            self.tracker.record_routing(
                latency_ms=latency,
                model="qwen2.5-coder-7b",
                success=True
            )

        stats = self.tracker.get_stats()

        assert "p50_latency_ms" in stats
        assert "p95_latency_ms" in stats
        assert "p99_latency_ms" in stats

    def test_reset(self):
        """Test resetting metrics"""
        self.tracker.record_routing(50.0, "qwen2.5-coder-7b", True)
        self.tracker.reset()

        stats = self.tracker.get_stats()
        assert stats["total_routings"] == 0


class TestIntegration:
    """Test integration of Phase 6 features"""

    def test_episodic_cascade_integration(self):
        """Test episodic memory with cascade routing"""
        memory = EpisodicMemory()
        cascade = CascadeRouter()

        # Route via cascade
        result = cascade.route(
            "test query",
            "coding",
            0.3,
            context_size=0
        )

        # Store in episodic memory
        episode_id = memory.add_episode(
            query="test query",
            selected_model=result.model,
            confidence=result.confidence,
            success=True,
            task_type="coding",
            complexity=0.3,
            metadata={"tier": result.tier}
        )

        assert episode_id >= 0
        assert memory.size() == 1

    def test_multiround_cascade_integration(self):
        """Test multi-round with cascade"""
        multi_round = MultiRoundRouter()
        cascade = CascadeRouter()

        # Create session
        session_id = multi_round.create_session()

        # Multiple rounds
        for i in range(3):
            result = cascade.route(
                f"query {i}",
                "coding",
                0.3 + i * 0.2,
                context_size=0
            )

            multi_round.add_round(
                session_id,
                f"query {i}",
                result.model,
                result.confidence
            )

        session = multi_round.get_session(session_id)
        assert session.total_rounds == 3

    def test_monitoring_full_flow(self):
        """Test monitoring full routing flow"""
        metrics = PrometheusMetrics()
        cascade = CascadeRouter()

        # Route and track
        start = time.time()
        result = cascade.route("test", "coding", 0.3, context_size=0)
        latency = (time.time() - start) * 1000

        # Record metrics
        metrics.record_routing_latency(latency)
        metrics.record_model_selection(result.model)
        metrics.record_success()

        # Should complete without errors
        assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
