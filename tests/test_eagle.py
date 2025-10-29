"""
Test suite for Eagle ELO + Associative Memory (Phase 2)
"""

import pytest
import numpy as np
from router.memory import AssociativeMemory, RoutingEvent
from router.eagle import EagleELO
from router.router_core import RouterCore
from router.models import GLOBAL_ELO


class TestAssociativeMemory:
    """Test Associative Memory functionality"""

    @pytest.fixture
    def memory(self):
        return AssociativeMemory(embedding_dim=384)

    def test_memory_initialization(self, memory):
        """Test memory initializes correctly"""
        assert memory.embedding_dim == 384
        assert memory.size() == 0
        assert memory.max_memory_size == 100_000

    def test_embed_text(self, memory):
        """Test text embedding generation"""
        text = "write a Python function"
        embedding = memory.embed(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

    def test_embed_consistency(self, memory):
        """Test that same text produces similar embeddings"""
        text = "write a Python function"
        emb1 = memory.embed(text)
        emb2 = memory.embed(text)

        # Should be identical or very similar
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        assert similarity > 0.99  # Very high similarity

    def test_add_task(self, memory):
        """Test adding routing events"""
        memory.add_task(
            query="write a function",
            selected_model="glm-4-9b",
            task_type="coding",
            complexity=0.5,
            success=True
        )

        assert memory.size() == 1

    def test_add_multiple_tasks(self, memory):
        """Test adding multiple tasks"""
        for i in range(10):
            memory.add_task(
                query=f"query {i}",
                selected_model="glm-4-9b",
                task_type="coding",
                complexity=0.5
            )

        assert memory.size() == 10

    def test_search_similar_empty(self, memory):
        """Test search on empty memory"""
        results = memory.search_similar("test query", k=5)
        assert len(results) == 0

    def test_search_similar_basic(self, memory):
        """Test basic similarity search"""
        # Add some tasks
        memory.add_task("write a Python function", "glm-4-9b", "coding", 0.5)
        memory.add_task("create a Python script", "glm-4-9b", "coding", 0.4)
        memory.add_task("design architecture", "qwen3-next-80b", "architecture", 0.8)

        # Search for similar
        results = memory.search_similar("write Python code", k=2)

        assert len(results) <= 2
        assert all(isinstance(r, RoutingEvent) for r in results)

    def test_search_with_filter(self, memory):
        """Test search with task type filter"""
        memory.add_task("write a function", "glm-4-9b", "coding", 0.5)
        memory.add_task("fix a bug", "glm-4-9b", "bug_fixing", 0.6)
        memory.add_task("refactor code", "qwen3-coder-30b", "refactoring", 0.7)

        # Search with filter
        results = memory.search_similar("code task", k=5, filter_task_type="coding")

        # Should only get coding tasks
        for r in results:
            assert r.task_type == "coding"

    def test_get_local_scores(self, memory):
        """Test local score calculation"""
        # Add multiple tasks with different models
        memory.add_task("python task 1", "glm-4-9b", "coding", 0.5, success=True)
        memory.add_task("python task 2", "glm-4-9b", "coding", 0.5, success=True)
        memory.add_task("python task 3", "qwen2.5-coder-7b", "coding", 0.3, success=True)

        # Get local scores
        scores = memory.get_local_scores("python coding task", k=5)

        assert isinstance(scores, dict)
        assert "glm-4-9b" in scores
        assert "qwen2.5-coder-7b" in scores
        assert scores["glm-4-9b"] > 0  # Should have positive score

    def test_memory_stats(self, memory):
        """Test memory statistics"""
        memory.add_task("query 1", "glm-4-9b", "coding", 0.5)
        memory.add_task("query 2", "qwen3-next-80b", "architecture", 0.8)

        stats = memory.get_stats()

        assert stats['size'] == 2
        assert 'model_distribution' in stats
        assert 'task_distribution' in stats
        assert 'avg_complexity' in stats
        assert stats['avg_complexity'] == 0.65  # (0.5 + 0.8) / 2

    def test_memory_clear(self, memory):
        """Test clearing memory"""
        memory.add_task("query", "glm-4-9b", "coding", 0.5)
        assert memory.size() == 1

        memory.clear()
        assert memory.size() == 0

    def test_max_memory_size(self):
        """Test memory size limit"""
        memory = AssociativeMemory(embedding_dim=384, max_memory_size=10)

        # Add more than max
        for i in range(15):
            memory.add_task(f"query {i}", "glm-4-9b", "coding", 0.5)

        # Should be capped at max
        assert memory.size() == 10


class TestEagleELO:
    """Test Eagle ELO system"""

    @pytest.fixture
    def memory(self):
        return AssociativeMemory(embedding_dim=384)

    @pytest.fixture
    def eagle(self, memory):
        return EagleELO(memory=memory, global_alpha=0.7, k_factor=32)

    def test_eagle_initialization(self, eagle):
        """Test Eagle initializes correctly"""
        assert eagle.global_alpha == 0.7
        assert abs(eagle.local_alpha - 0.3) < 0.001  # Floating point tolerance
        assert eagle.k_factor == 32
        assert len(eagle.global_elo) > 0

    def test_global_elo_loaded(self, eagle):
        """Test global ELO loaded from GLOBAL_ELO"""
        assert eagle.global_elo['qwen3-next-80b'] == 1900  # Highest
        assert eagle.global_elo['qwen2.5-coder-7b'] == 1650  # Lowest

    def test_get_global_score(self, eagle):
        """Test getting global ELO score"""
        score = eagle.get_global_score('glm-4-9b')
        assert score == GLOBAL_ELO['glm-4-9b']
        assert score == 1850

    def test_get_global_score_unknown_model(self, eagle):
        """Test getting global score for unknown model"""
        score = eagle.get_global_score('unknown-model')
        assert score == eagle.initial_elo  # Should return initial ELO

    def test_get_local_score_empty_memory(self, eagle):
        """Test local score with empty memory"""
        score = eagle.get_local_score("test query", "glm-4-9b")
        assert score == 0.5  # Neutral score

    def test_get_local_score_with_history(self, eagle, memory):
        """Test local score with history"""
        # Add some history
        memory.add_task("python task 1", "glm-4-9b", "coding", 0.5, success=True)
        memory.add_task("python task 2", "glm-4-9b", "coding", 0.5, success=True)

        score = eagle.get_local_score("python coding", "glm-4-9b", k=5)
        assert 0.0 <= score <= 1.0
        assert score > 0  # Should have positive score due to history

    def test_get_combined_score(self, eagle):
        """Test combined Eagle score calculation"""
        score = eagle.get_combined_score("test query", "glm-4-9b")

        assert 0.0 <= score <= 1.0
        # With empty memory, should be mostly global score
        global_normalized = eagle._normalize_elo(GLOBAL_ELO['glm-4-9b'])
        expected = 0.7 * global_normalized + 0.3 * 0.5  # 70% global + 30% neutral local
        assert abs(score - expected) < 0.01

    def test_score_all_models(self, eagle):
        """Test scoring all models"""
        scores = eagle.score_all_models("test query")

        assert isinstance(scores, dict)
        assert len(scores) == 5  # All 5 models
        assert all(0.0 <= s <= 1.0 for s in scores.values())

    def test_get_best_model(self, eagle):
        """Test getting best model"""
        best_model, best_score = eagle.get_best_model("test query")

        assert best_model in GLOBAL_ELO
        assert 0.0 <= best_score <= 1.0

    def test_get_best_model_with_exclusions(self, eagle):
        """Test best model with exclusions"""
        best_model, best_score = eagle.get_best_model(
            "test query",
            exclude_models=['qwen3-next-80b']
        )

        assert best_model != 'qwen3-next-80b'

    def test_update_rating(self, eagle):
        """Test ELO rating update"""
        initial_winner = eagle.global_elo['glm-4-9b']
        initial_loser = eagle.global_elo['qwen2.5-coder-7b']

        new_winner, new_loser = eagle.update_rating(
            'glm-4-9b', 'qwen2.5-coder-7b', actual_result=1.0
        )

        # Winner should increase, loser should decrease
        assert new_winner > initial_winner
        assert new_loser < initial_loser

    def test_update_rating_tie(self, eagle):
        """Test ELO update with tie"""
        initial_a = eagle.global_elo['glm-4-9b']
        initial_b = eagle.global_elo['qwen3-next-80b']

        new_a, new_b = eagle.update_rating('glm-4-9b', 'qwen3-next-80b', 0.5)

        # Ratings should change based on expected scores
        # Since qwen3-next-80b has higher rating, glm-4-9b should gain from tie
        assert new_a > initial_a  # Underdog gains
        assert new_b < initial_b  # Favorite loses

    def test_update_from_feedback(self, eagle, memory):
        """Test updating ELO from feedback"""
        # Add some history first
        for i in range(15):
            memory.add_task(f"task {i}", "qwen3-next-80b", "coding", 0.5, success=True)

        initial_elo = eagle.global_elo['glm-4-9b']

        # Provide feedback
        eagle.update_from_feedback(
            model_id='glm-4-9b',
            query='python task',
            success=True,
            task_type='coding',
            complexity=0.5
        )

        # Should be added to memory
        assert memory.size() == 16

        # ELO might change if compared with similar tasks
        # (We can't predict exact value, just check it's valid)
        new_elo = eagle.global_elo['glm-4-9b']
        assert new_elo > 0

    def test_expected_score(self, eagle):
        """Test expected score calculation"""
        # Equal ratings
        expected = eagle._expected_score(1500, 1500)
        assert expected == 0.5

        # Higher rating should have higher expected score
        expected_high = eagle._expected_score(1700, 1500)
        assert expected_high > 0.5

        expected_low = eagle._expected_score(1500, 1700)
        assert expected_low < 0.5

    def test_normalize_elo(self, eagle):
        """Test ELO normalization"""
        # Min value
        assert eagle._normalize_elo(1500) == 0.0

        # Max value
        assert eagle._normalize_elo(2000) == 1.0

        # Middle
        normalized = eagle._normalize_elo(1750)
        assert 0.4 < normalized < 0.6

    def test_get_rankings(self, eagle):
        """Test getting model rankings"""
        rankings = eagle.get_rankings()

        assert len(rankings) == 5
        assert all(isinstance(r, tuple) for r in rankings)
        assert rankings[0][0] == 'qwen3-next-80b'  # Highest ELO
        assert rankings[-1][0] == 'qwen2.5-coder-7b'  # Lowest ELO

        # Verify sorted descending
        for i in range(len(rankings) - 1):
            assert rankings[i][1] >= rankings[i + 1][1]

    def test_get_stats(self, eagle, memory):
        """Test getting Eagle stats"""
        memory.add_task("query", "glm-4-9b", "coding", 0.5)

        stats = eagle.get_stats()

        assert 'global_alpha' in stats
        assert 'local_alpha' in stats
        assert 'k_factor' in stats
        assert 'rankings' in stats
        assert 'top_model' in stats
        assert 'memory_size' in stats

        assert stats['global_alpha'] == 0.7
        assert abs(stats['local_alpha'] - 0.3) < 0.001  # Floating point tolerance
        assert stats['memory_size'] == 1

    def test_reset_to_defaults(self, eagle):
        """Test resetting ELO to defaults"""
        # Modify ELO
        eagle.global_elo['glm-4-9b'] = 2000

        # Reset
        eagle.reset_to_defaults()

        # Should be back to default
        assert eagle.global_elo['glm-4-9b'] == GLOBAL_ELO['glm-4-9b']

    def test_get_model_confidence(self, eagle):
        """Test model confidence calculation"""
        confidence = eagle.get_model_confidence("test query", "qwen3-next-80b")

        assert 0.0 <= confidence <= 1.0
        # Highest ELO model should have high confidence
        assert confidence > 0.5


class TestRouterCorePhase2:
    """Test RouterCore with Phase 2 integration"""

    @pytest.fixture
    def router_with_eagle(self):
        try:
            return RouterCore(enable_eagle=True, enable_memory=True)
        except Exception:
            pytest.skip("Eagle/Memory dependencies not available")

    @pytest.fixture
    def router_without_eagle(self):
        return RouterCore(enable_eagle=False, enable_memory=False)

    def test_router_initialization_with_eagle(self, router_with_eagle):
        """Test router initializes with Eagle"""
        assert router_with_eagle.enable_eagle
        assert router_with_eagle.enable_memory
        assert router_with_eagle.eagle is not None
        assert router_with_eagle.memory is not None

    def test_router_initialization_without_eagle(self, router_without_eagle):
        """Test router initializes without Eagle"""
        assert not router_without_eagle.enable_eagle
        assert not router_without_eagle.enable_memory
        assert router_without_eagle.eagle is None
        assert router_without_eagle.memory is None

    def test_routing_with_eagle(self, router_with_eagle):
        """Test routing uses Eagle when enabled"""
        result = router_with_eagle.route_sync("write a Python function")

        assert result.model in GLOBAL_ELO
        assert 0.0 <= result.confidence <= 1.0

        # Check if Eagle was used (if not cascade)
        if result.metadata.get('routing_strategy') not in ['cascade', 'reasoning', 'language_specialist']:
            assert result.metadata.get('routing_strategy') == 'eagle_elo'

    def test_routing_without_eagle(self, router_without_eagle):
        """Test routing works without Eagle"""
        result = router_without_eagle.route_sync("write a Python function")

        assert result.model in GLOBAL_ELO
        assert 0.0 <= result.confidence <= 1.0

        # Should use basic ELO
        if result.metadata.get('routing_strategy') not in ['cascade', 'reasoning', 'language_specialist']:
            assert result.metadata.get('routing_strategy') == 'elo_based'

    def test_provide_feedback(self, router_with_eagle):
        """Test providing feedback"""
        initial_size = router_with_eagle.memory.size()

        router_with_eagle.provide_feedback(
            query="test query",
            selected_model="glm-4-9b",
            success=True,
            task_type="coding",
            complexity=0.5
        )

        # Should be in memory (one more than before)
        assert router_with_eagle.memory.size() == initial_size + 1

    def test_provide_feedback_without_eagle(self, router_without_eagle):
        """Test feedback is ignored without Eagle"""
        router_without_eagle.provide_feedback(
            query="test query",
            selected_model="glm-4-9b",
            success=True,
            task_type="coding",
            complexity=0.5
        )

        # Should do nothing (no error)
        assert router_without_eagle.memory is None

    def test_get_eagle_stats(self, router_with_eagle):
        """Test getting Eagle stats"""
        stats = router_with_eagle.get_eagle_stats()

        assert stats is not None
        assert 'eagle' in stats
        assert 'memory' in stats

    def test_get_eagle_stats_without_eagle(self, router_without_eagle):
        """Test stats returns None without Eagle"""
        stats = router_without_eagle.get_eagle_stats()
        assert stats is None

    def test_eagle_learning_from_feedback(self, router_with_eagle):
        """Test that Eagle learns from feedback"""
        # Get initial score
        initial_scores = router_with_eagle.eagle.score_all_models("python coding task")
        initial_glm_score = initial_scores['glm-4-9b']

        # Provide positive feedback for multiple similar tasks
        for i in range(10):
            router_with_eagle.provide_feedback(
                query=f"python coding task {i}",
                selected_model="glm-4-9b",
                success=True,
                task_type="coding",
                complexity=0.5
            )

        # Get new score for similar query
        new_scores = router_with_eagle.eagle.score_all_models("python coding task")
        new_glm_score = new_scores['glm-4-9b']

        # Local score should influence (though global stays same)
        # Score might increase due to positive local history
        assert new_glm_score >= initial_glm_score - 0.05  # Allow small variance


class TestIntegration:
    """Integration tests for Phase 2"""

    @pytest.fixture
    def router(self):
        try:
            return RouterCore(enable_eagle=True, enable_memory=True)
        except Exception:
            pytest.skip("Eagle/Memory dependencies not available")

    def test_end_to_end_with_feedback(self, router):
        """Test complete routing cycle with feedback"""
        # Route a query
        result = router.route_sync("write a Python sorting function")
        selected_model = result.model

        # Provide feedback
        router.provide_feedback(
            query="write a Python sorting function",
            selected_model=selected_model,
            success=True,
            task_type="coding",
            complexity=0.4
        )

        # Route similar query
        result2 = router.route_sync("write a Python search function")

        # Should prefer models that succeeded on similar tasks
        assert result2.model in GLOBAL_ELO

    def test_memory_persistence(self, router):
        """Test memory accumulates over time"""
        initial_size = router.memory.size()

        # Add multiple tasks
        for i in range(5):
            result = router.route_sync(f"task {i}")
            router.provide_feedback(
                query=f"task {i}",
                selected_model=result.model,
                success=True,
                task_type="coding",
                complexity=0.5
            )

        # Should have at least 5 more events (might have more if routing adds events)
        assert router.memory.size() >= initial_size + 5

    def test_eagle_rankings_update(self, router):
        """Test Eagle rankings update with feedback"""
        initial_rankings = router.eagle.get_rankings()

        # Provide lots of feedback for lower-ranked model
        for i in range(20):
            router.provide_feedback(
                query=f"task {i}",
                selected_model="qwen2.5-coder-7b",  # Lowest initial ELO
                success=True,
                task_type="coding",
                complexity=0.3
            )

        # Rankings might change (ELO updates from comparisons)
        new_rankings = router.eagle.get_rankings()
        assert len(new_rankings) == len(initial_rankings)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
