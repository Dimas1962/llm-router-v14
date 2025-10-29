"""
Test suite for Router Core (Phase 1)
"""

import pytest
from router.router_core import RouterCore, RoutingResult
from router.models import MODELS, GLOBAL_ELO, get_model_config
from router.classifiers import TaskClassifier, ComplexityEstimator, TaskType


class TestModels:
    """Test model configuration"""

    def test_all_models_configured(self):
        """Test that all 6 models are configured"""
        assert len(MODELS) == 5  # Note: 5 models in config (not 6)
        expected_models = [
            'glm-4-9b',
            'qwen3-next-80b',
            'qwen3-coder-30b',
            'deepseek-coder-16b',
            'qwen2.5-coder-7b'
        ]
        for model_id in expected_models:
            assert model_id in MODELS

    def test_global_elo_initialization(self):
        """Test Global ELO bootstrap values"""
        assert GLOBAL_ELO['qwen3-next-80b'] == 1900  # Highest
        assert GLOBAL_ELO['glm-4-9b'] == 1850
        assert GLOBAL_ELO['qwen2.5-coder-7b'] == 1650  # Lowest

    def test_model_config_completeness(self):
        """Test that all models have complete configuration"""
        for model_id, config in MODELS.items():
            assert config.name
            assert config.size
            assert config.context > 0
            assert config.speed > 0
            assert 0 <= config.quality <= 1
            assert config.tier
            assert len(config.use_cases) > 0
            assert 0 <= config.frequency <= 1

    def test_context_windows(self):
        """Test context window sizes"""
        assert MODELS['glm-4-9b'].context == 1_000_000  # 1M
        assert MODELS['qwen3-next-80b'].context == 200_000
        assert MODELS['qwen2.5-coder-7b'].context == 32_000


class TestTaskClassifier:
    """Test task classification"""

    def setup_method(self):
        self.classifier = TaskClassifier()

    def test_simple_task_detection(self):
        """Test simple task pattern detection"""
        result = self.classifier.classify("write a simple function to print hello world")
        assert result.task_type == TaskType.QUICK_SNIPPET
        assert result.is_simple_pattern
        assert result.complexity < 0.4

    def test_architecture_task_detection(self):
        """Test architecture task detection"""
        result = self.classifier.classify(
            "design a scalable microservices architecture for e-commerce"
        )
        assert result.task_type == TaskType.ARCHITECTURE
        assert result.requires_reasoning
        assert result.complexity > 0.6

    def test_bug_fixing_detection(self):
        """Test bug fixing task detection"""
        result = self.classifier.classify("fix the bug in the authentication module")
        assert result.task_type == TaskType.BUG_FIXING

    def test_refactoring_detection(self):
        """Test refactoring task detection"""
        result = self.classifier.classify("refactor this code to improve readability")
        assert result.task_type == TaskType.REFACTORING

    def test_language_detection_python(self):
        """Test Python language detection"""
        result = self.classifier.classify("write a Python function to sort a list")
        assert result.detected_language == 'python'

    def test_language_detection_rust(self):
        """Test Rust language detection"""
        result = self.classifier.classify("create a Rust function with async/await")
        assert result.detected_language == 'rust'

    def test_language_detection_go(self):
        """Test Go language detection"""
        result = self.classifier.classify("implement a Go HTTP server")
        assert result.detected_language == 'go'

    def test_complexity_estimation_simple(self):
        """Test complexity estimation for simple tasks"""
        result = self.classifier.classify("print hello world")
        assert result.complexity < 0.3

    def test_complexity_estimation_complex(self):
        """Test complexity estimation for complex tasks"""
        result = self.classifier.classify(
            "design and implement a distributed cache with consistency guarantees"
        )
        assert result.complexity > 0.7

    def test_large_context_detection(self):
        """Test large context requirement detection"""
        result = self.classifier.classify(
            "refactor the entire codebase to use dependency injection"
        )
        assert result.requires_large_context


class TestComplexityEstimator:
    """Test complexity estimation"""

    def test_overall_complexity_simple(self):
        """Test overall complexity for simple query"""
        scores = ComplexityEstimator.estimate("print hello")
        assert scores['overall'] < 0.4

    def test_overall_complexity_complex(self):
        """Test overall complexity for complex query"""
        scores = ComplexityEstimator.estimate(
            "design a distributed system with fault tolerance and consistency"
        )
        assert scores['overall'] > 0.6

    def test_technical_complexity(self):
        """Test technical complexity detection"""
        scores = ComplexityEstimator.estimate(
            "implement a concurrent lock-free data structure with memory barriers"
        )
        assert scores['technical'] > 0.4

    def test_cognitive_complexity(self):
        """Test cognitive complexity detection"""
        scores = ComplexityEstimator.estimate(
            "explain why this approach is better and compare alternatives"
        )
        assert scores['cognitive'] > 0.3

    def test_context_complexity(self):
        """Test context complexity with session history"""
        history = ["message " * 1000 for _ in range(50)]  # Large history
        scores = ComplexityEstimator.estimate("continue", history)
        assert scores['context'] > 0.5


class TestRoutingCore:
    """Test main routing logic"""

    @pytest.fixture
    def router(self):
        return RouterCore()

    def test_simple_query_routing(self, router):
        """Test routing for simple query → fast model"""
        result = router.route_sync("write a simple function to print hello world")

        assert isinstance(result, RoutingResult)
        assert result.model == 'qwen2.5-coder-7b'  # Fast model
        assert result.confidence > 0.7
        assert 'cascade' in result.reasoning.lower() or 'simple' in result.reasoning.lower()

    def test_complex_reasoning_routing(self, router):
        """Test routing for complex reasoning → 80B model"""
        result = router.route_sync(
            "design a scalable microservices architecture with event sourcing"
        )

        assert result.model == 'qwen3-next-80b'  # Reasoning model
        assert result.confidence > 0.7
        assert 'reasoning' in result.reasoning.lower()

    def test_rust_language_routing(self, router):
        """Test routing for Rust → DeepSeek specialist"""
        result = router.route_sync(
            "write a Rust function to handle async file I/O"
        )

        assert result.model == 'deepseek-coder-16b'  # Rust specialist
        assert 'rust' in result.reasoning.lower() or 'specialist' in result.reasoning.lower()

    def test_go_language_routing(self, router):
        """Test routing for Go → DeepSeek specialist"""
        result = router.route_sync("create a Go HTTP handler with middleware")

        assert result.model == 'deepseek-coder-16b'
        assert 'go' in result.reasoning.lower() or 'specialist' in result.reasoning.lower()

    def test_large_context_routing(self, router):
        """Test routing for large context → GLM-4"""
        # Simulate large session history
        history = ["message " * 10000 for _ in range(10)]

        result = router.route_sync(
            "continue refactoring based on the previous discussion",
            session_history=history
        )

        assert result.model == 'glm-4-9b'  # 1M context model
        assert 'context' in result.reasoning.lower()

    def test_multi_file_routing(self, router):
        """Test routing for multi-file refactoring"""
        result = router.route_sync(
            "refactor all files in the project to use the new API"
        )

        # Should route to either qwen3-coder-30b, glm-4-9b, or qwen3-next-80b (with Eagle)
        # Phase 2: Eagle may select qwen3-next-80b based on ELO scores
        assert result.model in ['qwen3-coder-30b', 'glm-4-9b', 'qwen3-next-80b']

    def test_alternatives_provided(self, router):
        """Test that alternatives are provided"""
        result = router.route_sync("write a Python sorting function")

        assert len(result.alternatives) > 0
        assert all(isinstance(alt, tuple) for alt in result.alternatives)
        assert all(len(alt) == 2 for alt in result.alternatives)

    def test_metadata_completeness(self, router):
        """Test that metadata is complete"""
        result = router.route_sync("write a function")

        assert 'task_type' in result.metadata
        assert 'routing_strategy' in result.metadata

    def test_confidence_range(self, router):
        """Test that confidence is in valid range"""
        result = router.route_sync("test query")

        assert 0.0 <= result.confidence <= 1.0

    def test_decay_risk_detection(self, router):
        """Test decay risk detection and mitigation"""
        # Very large context
        history = ["x" * 50000 for _ in range(5)]  # >200K chars

        result = router.route_sync("continue", session_history=history)

        # Should route to model with good long-context handling
        assert result.model in ['glm-4-9b', 'qwen3-next-80b']


class TestContextAnalysis:
    """Test context analysis"""

    @pytest.fixture
    def router(self):
        return RouterCore()

    def test_context_size_calculation(self, router):
        """Test context size calculation"""
        query = "test query"
        history = ["message 1", "message 2"]

        context_info = router._analyze_context(query, history)

        expected_size = len(query) + len("message 1") + len("message 2")
        assert context_info['total_size'] == expected_size

    def test_required_window_simple(self, router):
        """Test required window for simple query"""
        context_info = router._analyze_context("print hello", None)
        assert context_info['required_window'] == 8_000

    def test_required_window_complex(self, router):
        """Test required window for complex query"""
        context_info = router._analyze_context(
            "design a complex distributed system with high availability",
            None
        )
        assert context_info['required_window'] >= 32_000

    def test_decay_risk_low(self, router):
        """Test low decay risk"""
        assert router._estimate_decay_risk(10_000) == 0.0

    def test_decay_risk_medium(self, router):
        """Test medium decay risk"""
        assert router._estimate_decay_risk(50_000) == 0.3

    def test_decay_risk_high(self, router):
        """Test high decay risk"""
        assert router._estimate_decay_risk(100_000) == 0.5

    def test_decay_risk_critical(self, router):
        """Test critical decay risk"""
        assert router._estimate_decay_risk(200_000) == 0.8


class TestUtilityMethods:
    """Test utility methods"""

    @pytest.fixture
    def router(self):
        return RouterCore()

    def test_get_model_info(self, router):
        """Test getting model info"""
        info = router.get_model_info('glm-4-9b')

        assert info['id'] == 'glm-4-9b'
        assert info['name'] == 'GLM-4-9B-Chat-1M-BF16'
        assert info['context_window'] == 1_000_000
        assert info['elo'] == 1850

    def test_list_models(self, router):
        """Test listing all models"""
        models = router.list_models()

        assert len(models) == 5
        assert all('id' in m for m in models)
        assert all('name' in m for m in models)
        assert all('elo' in m for m in models)


@pytest.mark.asyncio
class TestAsyncRouting:
    """Test async routing"""

    async def test_async_route(self):
        """Test async routing"""
        router = RouterCore()
        result = await router.route("write a Python function")

        assert isinstance(result, RoutingResult)
        assert result.model in MODELS

    async def test_async_with_history(self):
        """Test async routing with session history"""
        router = RouterCore()
        history = ["previous message 1", "previous message 2"]

        result = await router.route(
            "continue with the implementation",
            session_history=history
        )

        assert isinstance(result, RoutingResult)


# Performance tests
class TestPerformance:
    """Test routing performance"""

    @pytest.fixture
    def router(self):
        return RouterCore()

    def test_routing_latency(self, router):
        """Test that routing is fast (<100ms target)"""
        import time

        start = time.time()
        router.route_sync("test query")
        latency = time.time() - start

        # Phase 1 target: <100ms
        # Should be much faster without full ELO/CARROT
        assert latency < 0.1  # 100ms


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
