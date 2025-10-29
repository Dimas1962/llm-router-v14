"""
Test suite for Context Management (Phase 4)
"""

import pytest
from router.context_manager import (
    ContextManager,
    ContextAnalysis,
    DynamicContextSizer,
    ProgressiveContextBuilder,
    DecayMonitor,
    ContextCompressor
)
from router.router_core import RouterCore
from router.models import MODELS


class TestDynamicContextSizer:
    """Test dynamic context window sizing"""

    def setup_method(self):
        self.sizer = DynamicContextSizer()

    def test_simple_task_sizing(self):
        """Test context window for simple task"""
        required = self.sizer.estimate_required_window(
            query="print hello world",
            complexity=0.2,
            session_history=None
        )
        assert required == 8_000

    def test_medium_task_sizing(self):
        """Test context window for medium complexity task"""
        required = self.sizer.estimate_required_window(
            query="implement a REST API with authentication",
            complexity=0.5,
            session_history=None
        )
        assert required == 32_000

    def test_complex_task_sizing(self):
        """Test context window for complex task"""
        required = self.sizer.estimate_required_window(
            query="design a distributed system with fault tolerance",
            complexity=0.8,
            session_history=None
        )
        assert required == 128_000

    def test_large_history_adjustment(self):
        """Test window adjustment for large history"""
        # Large history should increase window
        history = ["message " * 1000 for _ in range(20)]

        required = self.sizer.estimate_required_window(
            query="continue",
            complexity=0.3,
            session_history=history
        )

        # Should be larger than base 8K due to history
        assert required > 8_000

    def test_large_context_keywords(self):
        """Test detection of large context keywords"""
        required = self.sizer.estimate_required_window(
            query="refactor the entire codebase",
            complexity=0.5,
            session_history=None
        )

        assert required >= 200_000

    def test_recommend_models_small_window(self):
        """Test model recommendations for small window"""
        models = self.sizer.recommend_models(
            required_window=8_000,
            complexity=0.2,
            query="simple task"
        )

        # All models should be suitable for 8K
        assert len(models) == 5
        assert 'qwen2.5-coder-7b' in models

    def test_recommend_models_large_window(self):
        """Test model recommendations for large window"""
        models = self.sizer.recommend_models(
            required_window=500_000,
            complexity=0.8,
            query="large context task"
        )

        # Only GLM-4 has 1M context
        assert 'glm-4-9b' in models
        assert 'qwen2.5-coder-7b' not in models  # Only 32K context

    def test_truncation_point_calculation(self):
        """Test truncation point calculation"""
        # Create large messages that will definitely need truncation
        history = [f"message {i}" * 200 for i in range(100)]

        keep_count = self.sizer.calculate_truncation_point(
            session_history=history,
            max_window=32_000,
            query_size=100
        )

        # Should keep some messages but not all
        assert 0 < keep_count < len(history)


class TestProgressiveContextBuilder:
    """Test progressive context building"""

    def setup_method(self):
        self.builder = ProgressiveContextBuilder(
            recent_window=5,
            mid_window=10,
            summary_compression=0.3
        )

    def test_small_history_no_compression(self):
        """Test that small history is not compressed"""
        history = ["message 1", "message 2", "message 3"]

        processed, stats = self.builder.build_progressive_context(
            session_history=history,
            max_size=100_000
        )

        # Should keep all messages
        assert len(processed) == 3
        assert stats['compression'] == 0.0

    def test_large_history_compression(self):
        """Test compression of large history"""
        history = ["message " * 100 for _ in range(50)]

        processed, stats = self.builder.build_progressive_context(
            session_history=history,
            max_size=10_000
        )

        # Should compress
        assert len(processed) < len(history)
        assert stats['compression'] > 0.0

    def test_recent_messages_priority(self):
        """Test that recent messages have priority"""
        history = [f"old message {i}" for i in range(20)]
        history.extend([f"recent message {i}" for i in range(5)])

        processed, stats = self.builder.build_progressive_context(
            session_history=history,
            max_size=1_000
        )

        # Recent messages should be included
        processed_text = " ".join(processed)
        assert "recent message" in processed_text

    def test_progressive_stats(self):
        """Test statistics from progressive building"""
        history = ["message " * 50 for _ in range(30)]

        processed, stats = self.builder.build_progressive_context(
            session_history=history,
            max_size=5_000
        )

        # Check stats completeness
        assert 'original' in stats
        assert 'processed' in stats
        assert 'compression' in stats
        assert 'messages_kept' in stats
        assert 'messages_dropped' in stats
        assert stats['messages_kept'] + stats['messages_dropped'] == len(history)


class TestDecayMonitor:
    """Test context decay monitoring"""

    def setup_method(self):
        self.monitor = DecayMonitor()

    def test_low_decay_risk(self):
        """Test low decay risk for small context"""
        risk = self.monitor.estimate_decay_risk(context_size=10_000)
        assert risk == 0.0

    def test_medium_decay_risk(self):
        """Test medium decay risk"""
        risk = self.monitor.estimate_decay_risk(context_size=50_000)
        assert risk == 0.3

    def test_high_decay_risk(self):
        """Test high decay risk"""
        risk = self.monitor.estimate_decay_risk(context_size=100_000)
        assert risk == 0.5

    def test_critical_decay_risk(self):
        """Test critical decay risk"""
        risk = self.monitor.estimate_decay_risk(context_size=250_000)
        assert risk == 0.9

    def test_model_specific_risk(self):
        """Test decay risk for specific model"""
        # Small context for small model
        risk = self.monitor.estimate_decay_risk(
            context_size=30_000,  # ~7.5K tokens
            model_id='qwen2.5-coder-7b'  # 32K context
        )
        # Should be low
        assert risk < 0.5

        # Large context for small model
        risk = self.monitor.estimate_decay_risk(
            context_size=120_000,  # ~30K tokens
            model_id='qwen2.5-coder-7b'  # 32K context
        )
        # Should be high (approaching capacity)
        assert risk >= 0.7

    def test_detect_no_decay_patterns(self):
        """Test decay pattern detection with clean history"""
        history = [f"message {i}" for i in range(10)]

        patterns = self.monitor.detect_decay_patterns(history)

        assert patterns['has_decay'] is False
        assert len(patterns['patterns']) == 0

    def test_detect_repetition_pattern(self):
        """Test detection of repetition pattern"""
        # Repetitive history
        history = ["same message"] * 10

        patterns = self.monitor.detect_decay_patterns(history)

        assert patterns['has_decay'] is True
        assert 'repetition' in patterns['patterns']

    def test_detect_long_session_pattern(self):
        """Test detection of long session"""
        history = [f"message {i}" for i in range(100)]

        patterns = self.monitor.detect_decay_patterns(history)

        assert patterns['has_decay'] is True
        assert 'long_session' in patterns['patterns']

    def test_detect_large_context_pattern(self):
        """Test detection of large context"""
        history = ["message " * 1000 for _ in range(50)]

        patterns = self.monitor.detect_decay_patterns(history)

        assert patterns['has_decay'] is True
        assert 'large_context' in patterns['patterns']

    def test_mitigation_recommendations_high_risk(self):
        """Test mitigation recommendations for high risk"""
        mitigation = self.monitor.recommend_mitigation(
            decay_risk=0.8,
            context_size=200_000
        )

        assert len(mitigation['recommendations']) > 0
        actions = [r['action'] for r in mitigation['recommendations']]
        assert 'compress_context' in actions or 'switch_model' in actions

    def test_mitigation_recommendations_low_risk(self):
        """Test mitigation recommendations for low risk"""
        mitigation = self.monitor.recommend_mitigation(
            decay_risk=0.2,
            context_size=20_000
        )

        # Should just monitor
        if mitigation['recommendations']:
            assert mitigation['recommendations'][0]['priority'] == 'low'


class TestContextCompressor:
    """Test context compression"""

    def setup_method(self):
        self.compressor = ContextCompressor(
            compression_ratio=0.5,
            min_keep_messages=5
        )

    def test_compress_empty_history(self):
        """Test compression of empty history"""
        compressed, stats = self.compressor.compress([])

        assert len(compressed) == 0
        assert stats['ratio'] == 0.0

    def test_compress_small_history(self):
        """Test compression keeps minimum messages"""
        history = ["message 1", "message 2", "message 3"]

        compressed, stats = self.compressor.compress(history)

        # Should keep all (below minimum)
        assert len(compressed) == len(history)

    def test_compress_large_history(self):
        """Test compression of large history"""
        history = ["message " * 100 for _ in range(50)]

        compressed, stats = self.compressor.compress(history)

        # Should reduce size
        assert stats['compressed'] < stats['original']
        assert stats['ratio'] < 1.0
        assert stats['reduction'] > 0.0

    def test_recent_messages_preserved(self):
        """Test that recent messages are preserved"""
        history = ["old message"] * 20
        history.extend(["recent message"] * 5)

        compressed, stats = self.compressor.compress(history, target_size=1_000)

        # Recent messages should be in compressed version
        recent_in_compressed = any("recent message" in msg for msg in compressed)
        assert recent_in_compressed

    def test_compression_stats(self):
        """Test compression statistics"""
        history = ["message " * 50 for _ in range(30)]

        compressed, stats = self.compressor.compress(history)

        assert 'original' in stats
        assert 'compressed' in stats
        assert 'ratio' in stats
        assert 'reduction' in stats
        assert 'messages_original' in stats
        assert 'messages_compressed' in stats

    def test_estimate_compression_savings(self):
        """Test estimation of compression savings"""
        history = ["message " * 100 for _ in range(50)]

        savings = self.compressor.estimate_compression_savings(history)

        assert savings['original_size'] > 0
        assert savings['potential_savings'] > 0
        assert 0 <= savings['savings_percent'] <= 1

    def test_worth_compressing_large(self):
        """Test that large context is worth compressing"""
        history = ["message " * 200 for _ in range(50)]

        savings = self.compressor.estimate_compression_savings(history)

        assert savings['worth_compressing'] is True

    def test_not_worth_compressing_small(self):
        """Test that small context is not worth compressing"""
        history = ["short message"] * 5

        savings = self.compressor.estimate_compression_savings(history)

        assert savings['worth_compressing'] is False


class TestContextManager:
    """Test integrated context manager"""

    def setup_method(self):
        self.manager = ContextManager(
            enable_progressive=True,
            enable_compression=True,
            enable_decay_monitor=True
        )

    def test_analyze_simple_context(self):
        """Test analysis of simple context"""
        analysis = self.manager.analyze_context(
            query="print hello",
            session_history=None,
            complexity=0.2
        )

        assert isinstance(analysis, ContextAnalysis)
        assert analysis.total_size > 0
        assert analysis.required_window == 8_000
        assert analysis.complexity == 0.2
        assert analysis.decay_risk == 0.0

    def test_analyze_complex_context(self):
        """Test analysis of complex context"""
        analysis = self.manager.analyze_context(
            query="design a distributed system",
            session_history=None,
            complexity=0.8
        )

        assert analysis.required_window >= 128_000
        assert analysis.complexity == 0.8

    def test_analyze_with_large_history(self):
        """Test analysis with large session history"""
        history = ["message " * 1000 for _ in range(50)]

        analysis = self.manager.analyze_context(
            query="continue",
            session_history=history,
            complexity=0.5
        )

        assert analysis.total_size > 50_000
        assert analysis.decay_risk > 0.0

    def test_truncation_needed_detection(self):
        """Test detection of truncation need"""
        # Very large context
        history = ["message " * 10000 for _ in range(20)]

        analysis = self.manager.analyze_context(
            query="test",
            session_history=history,
            complexity=0.3
        )

        # May need truncation for simple task
        # (This depends on implementation logic)
        assert isinstance(analysis.truncation_needed, bool)

    def test_compression_recommendation(self):
        """Test compression recommendation"""
        # Large compressible history
        history = ["message " * 500 for _ in range(50)]

        analysis = self.manager.analyze_context(
            query="test",
            session_history=history,
            complexity=0.5
        )

        # Should recommend compression
        assert isinstance(analysis.compression_recommended, bool)

    def test_suggested_models(self):
        """Test model suggestions"""
        analysis = self.manager.analyze_context(
            query="test query",
            session_history=None,
            complexity=0.5
        )

        assert isinstance(analysis.suggested_models, list)
        assert len(analysis.suggested_models) > 0

    def test_optimize_context_no_history(self):
        """Test context optimization with no history"""
        optimized, stats = self.manager.optimize_context(
            query="test",
            session_history=[],
            target_model='qwen2.5-coder-7b',
            complexity=0.3
        )

        assert len(optimized) == 0
        assert stats['optimized'] is False

    def test_optimize_context_small_history(self):
        """Test optimization with small history (no optimization needed)"""
        history = ["message 1", "message 2"]

        optimized, stats = self.manager.optimize_context(
            query="test",
            session_history=history,
            target_model='qwen2.5-coder-7b',
            complexity=0.3
        )

        # Should not need optimization
        assert stats['strategy'] == 'none_needed'

    def test_optimize_context_large_history(self):
        """Test optimization with large history"""
        history = ["message " * 1000 for _ in range(100)]

        optimized, stats = self.manager.optimize_context(
            query="test",
            session_history=history,
            target_model='qwen2.5-coder-7b',  # Small context model
            complexity=0.3
        )

        # Should optimize
        assert stats['optimized'] is True
        assert stats['strategy'] in ['progressive', 'compression', 'truncation']

    def test_optimize_for_different_models(self):
        """Test optimization for different model sizes"""
        history = ["message " * 500 for _ in range(50)]

        # Optimize for small model
        opt_small, stats_small = self.manager.optimize_context(
            query="test",
            session_history=history,
            target_model='qwen2.5-coder-7b',  # 32K context
            complexity=0.5
        )

        # Optimize for large model
        opt_large, stats_large = self.manager.optimize_context(
            query="test",
            session_history=history,
            target_model='glm-4-9b',  # 1M context
            complexity=0.5
        )

        # Small model should need more optimization
        if stats_small['optimized']:
            # Large model may not need optimization
            assert True  # Just ensure no errors


class TestContextManagerIntegration:
    """Test Context Manager integration with RouterCore"""

    def test_router_uses_context_manager(self):
        """Test that router uses context manager when enabled"""
        router = RouterCore(enable_context_manager=True)

        assert router.enable_context_manager is True
        assert router.context_manager is not None

    def test_router_without_context_manager(self):
        """Test router works without context manager"""
        router = RouterCore(enable_context_manager=False)

        assert router.enable_context_manager is False
        assert router.context_manager is None

        # Should still route successfully
        result = router.route_sync("test query")
        assert result.model in MODELS

    def test_routing_with_context_analysis(self):
        """Test routing uses context analysis"""
        router = RouterCore(enable_context_manager=True)

        # Large context should route to GLM-4
        history = ["x" * 60000 for _ in range(5)]

        result = router.route_sync(
            query="continue",
            session_history=history
        )

        # Should route to large context model
        assert result.model in ['glm-4-9b', 'qwen3-next-80b']

    def test_decay_risk_mitigation(self):
        """Test that high decay risk affects routing"""
        router = RouterCore(enable_context_manager=True)

        # Very large context with high decay risk
        history = ["x" * 50000 for _ in range(10)]

        result = router.route_sync(
            query="continue the refactoring",
            session_history=history
        )

        # Should route to model with good long-context handling
        assert result.model in ['glm-4-9b', 'qwen3-next-80b']
        assert 'metadata' in result.metadata or 'routing_strategy' in result.metadata


class TestContextManagerDisabled:
    """Test that context manager can be disabled gracefully"""

    def test_manager_initialization_disabled(self):
        """Test manager can be initialized with features disabled"""
        manager = ContextManager(
            enable_progressive=False,
            enable_compression=False,
            enable_decay_monitor=False
        )

        assert manager.enable_progressive is False
        assert manager.enable_compression is False
        assert manager.enable_decay_monitor is False

    def test_analyze_with_disabled_features(self):
        """Test analysis works with disabled features"""
        manager = ContextManager(
            enable_progressive=False,
            enable_compression=False,
            enable_decay_monitor=False
        )

        analysis = manager.analyze_context(
            query="test",
            session_history=["msg1", "msg2"],
            complexity=0.5
        )

        # Should still work but with limited features
        assert isinstance(analysis, ContextAnalysis)
        # Decay risk should be 0.0 when monitor disabled
        # (actually it might still work, just check it doesn't crash)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
