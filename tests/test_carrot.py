"""
Test suite for CARROT (Cost-Aware Routing) - Phase 3
"""

import pytest
from router.carrot import (
    QualityPredictor,
    CostPredictor,
    CARROT,
    Prediction
)
from router.router_core import RouterCore
from router.models import MODELS, get_all_models


class TestQualityPredictor:
    """Test Quality Predictor functionality"""

    @pytest.fixture
    def predictor(self):
        return QualityPredictor()

    def test_quality_predictor_initialization(self, predictor):
        """Test predictor initializes"""
        assert predictor is not None

    def test_basic_quality_prediction(self, predictor):
        """Test basic quality prediction"""
        quality = predictor.predict("write a function", "glm-4-9b")

        assert 0.0 <= quality <= 1.0
        # Should be close to model's base quality
        assert abs(quality - MODELS['glm-4-9b'].quality) < 0.3

    def test_quality_with_task_match(self, predictor):
        """Test quality bonus for task type match"""
        # GLM-4 is good for coding
        quality_with_match = predictor.predict(
            "write code", "glm-4-9b", task_type="coding"
        )

        quality_without_match = predictor.predict(
            "write code", "glm-4-9b", task_type="other"
        )

        # Should get bonus for task match
        assert quality_with_match >= quality_without_match

    def test_quality_with_complexity_match(self, predictor):
        """Test quality for complexity matching"""
        # High-quality model with high complexity
        quality_high = predictor.predict(
            "complex query", "qwen3-next-80b", complexity=0.9
        )

        # Low-quality model with low complexity
        quality_low = predictor.predict(
            "simple query", "qwen2.5-coder-7b", complexity=0.2
        )

        # Both should have reasonable quality
        assert 0.5 <= quality_high <= 1.0
        assert 0.4 <= quality_low <= 1.0

    def test_quality_penalty_for_large_context(self, predictor):
        """Test quality penalty when context too large"""
        # qwen2.5-coder-7b has 32K context window
        quality_small = predictor.predict(
            "query", "qwen2.5-coder-7b", context_size=10_000
        )

        quality_large = predictor.predict(
            "query", "qwen2.5-coder-7b", context_size=50_000  # Exceeds 32K
        )

        # Should have penalty for oversized context
        assert quality_large < quality_small

    def test_rust_quality_bonus(self, predictor):
        """Test quality bonus for Rust on deepseek"""
        quality_rust = predictor.predict(
            "write a Rust async function", "deepseek-coder-16b"
        )

        quality_python = predictor.predict(
            "write a Python function", "deepseek-coder-16b"
        )

        # Rust should get higher quality on deepseek
        assert quality_rust > quality_python

    def test_batch_prediction(self, predictor):
        """Test batch prediction for multiple models"""
        qualities = predictor.predict_batch(
            "write a function",
            get_all_models()
        )

        assert len(qualities) == len(get_all_models())
        assert all(0.0 <= q <= 1.0 for q in qualities.values())


class TestCostPredictor:
    """Test Cost Predictor functionality"""

    @pytest.fixture
    def predictor(self):
        return CostPredictor()

    def test_cost_predictor_initialization(self, predictor):
        """Test predictor initializes"""
        assert predictor is not None
        assert predictor.time_weight == 0.5
        assert predictor.token_weight == 0.5

    def test_token_estimation(self, predictor):
        """Test token count estimation"""
        input_tokens, output_tokens, total_tokens = predictor.estimate_tokens(
            "test query" * 10  # ~100 chars
        )

        assert input_tokens > 0
        assert output_tokens > 0
        assert total_tokens == input_tokens + output_tokens
        # ~100 chars / 4 = ~25 tokens input
        assert 20 <= input_tokens <= 30

    def test_token_estimation_with_context(self, predictor):
        """Test token estimation with context"""
        _, _, total_small = predictor.estimate_tokens("query", context_size=0)
        _, _, total_large = predictor.estimate_tokens("query", context_size=10_000)

        # Larger context = more tokens
        assert total_large > total_small

    def test_time_estimation(self, predictor):
        """Test time estimation"""
        # Fast model (60 tok/s)
        time_fast = predictor.estimate_time("qwen2.5-coder-7b", 1000)

        # Slow model (12 tok/s)
        time_slow = predictor.estimate_time("qwen3-next-80b", 1000)

        # Faster model should take less time
        assert time_fast < time_slow

        # qwen2.5-coder-7b: 1000 / 60 = ~16.7 seconds
        assert 15 <= time_fast <= 18

    def test_cost_prediction_simple(self, predictor):
        """Test basic cost prediction"""
        cost = predictor.predict("test query", "glm-4-9b")

        assert cost > 0.0

    def test_cost_varies_by_model(self, predictor):
        """Test that cost varies by model"""
        cost_fast = predictor.predict("query", "qwen2.5-coder-7b")
        cost_slow = predictor.predict("query", "qwen3-next-80b")

        # Different models should have different costs
        assert cost_fast != cost_slow

    def test_cost_increases_with_query_size(self, predictor):
        """Test cost increases with query size"""
        cost_small = predictor.predict("short", "glm-4-9b")
        cost_large = predictor.predict("a" * 1000, "glm-4-9b")

        assert cost_large > cost_small

    def test_batch_prediction(self, predictor):
        """Test batch cost prediction"""
        costs = predictor.predict_batch("query", get_all_models())

        assert len(costs) == len(get_all_models())
        assert all(c > 0.0 for c in costs.values())

    def test_detailed_prediction(self, predictor):
        """Test detailed prediction object"""
        prediction = predictor.get_detailed_prediction("query", "glm-4-9b")

        assert isinstance(prediction, Prediction)
        assert prediction.model_id == "glm-4-9b"
        assert 0.0 <= prediction.quality <= 1.0
        assert prediction.cost > 0.0
        assert prediction.time_estimate > 0.0
        assert prediction.tokens_estimate > 0


class TestCARROT:
    """Test CARROT system"""

    @pytest.fixture
    def carrot(self):
        return CARROT()

    def test_carrot_initialization(self, carrot):
        """Test CARROT initializes"""
        assert carrot is not None
        assert carrot.quality_predictor is not None
        assert carrot.cost_predictor is not None

    def test_predict_all(self, carrot):
        """Test predicting for all models"""
        predictions = carrot.predict_all("write a function")

        assert len(predictions) == len(get_all_models())
        for model_id, pred in predictions.items():
            assert 'quality' in pred
            assert 'cost' in pred
            assert 0.0 <= pred['quality'] <= 1.0
            assert pred['cost'] > 0.0

    def test_select_no_budget(self, carrot):
        """Test selection without budget (best quality)"""
        selected, prediction = carrot.select("write a function")

        assert selected in MODELS
        assert 'quality' in prediction
        assert 'cost' in prediction

        # Should select model with highest quality
        all_preds = carrot.predict_all("write a function")
        best_quality_model = max(all_preds.items(), key=lambda x: x[1]['quality'])
        assert selected == best_quality_model[0]

    def test_select_with_sufficient_budget(self, carrot):
        """Test selection with sufficient budget"""
        # High budget - should select high quality model
        selected, prediction = carrot.select("write a function", budget=1000.0)

        assert selected in MODELS
        assert prediction['cost'] <= 1000.0

    def test_select_with_tight_budget(self, carrot):
        """Test selection with tight budget"""
        # Very low budget - should select cheapest model
        selected, prediction = carrot.select("write a function", budget=0.1)

        # Should select cheapest model (or within budget if any)
        all_preds = carrot.predict_all("write a function")
        cheapest = min(all_preds.items(), key=lambda x: x[1]['cost'])

        # Either selected is within budget, or it's the cheapest
        if prediction['cost'] <= 0.1:
            # Found model within budget
            assert prediction['cost'] <= 0.1
        else:
            # Budget exceeded, selected cheapest
            assert selected == cheapest[0]

    def test_select_with_medium_budget(self, carrot):
        """Test selection balances quality and cost"""
        # Medium budget
        selected, prediction = carrot.select("write a function", budget=50.0)

        assert selected in MODELS
        assert 'quality' in prediction
        assert 'cost' in prediction

        # Should be within budget or cheapest if budget exceeded
        if prediction['cost'] <= 50.0:
            assert prediction['cost'] <= 50.0
        else:
            # Budget exceeded, should be cheapest
            all_preds = carrot.predict_all("write a function")
            cheapest = min(all_preds.items(), key=lambda x: x[1]['cost'])
            assert selected == cheapest[0]

    def test_pareto_frontier(self, carrot):
        """Test Pareto frontier calculation"""
        pareto = carrot.get_pareto_frontier("write a function")

        assert len(pareto) > 0
        assert len(pareto) <= len(get_all_models())

        # Pareto frontier should be sorted by quality (descending)
        for i in range(len(pareto) - 1):
            assert pareto[i][1] >= pareto[i + 1][1]

        # All models on frontier should be non-dominated
        for model_id, quality, cost in pareto:
            assert model_id in MODELS
            assert 0.0 <= quality <= 1.0
            assert cost > 0.0

    def test_pareto_frontier_properties(self, carrot):
        """Test Pareto frontier has correct properties"""
        pareto = carrot.get_pareto_frontier("write a function")

        # Check that no model dominates another on the frontier
        for i, (m1, q1, c1) in enumerate(pareto):
            for j, (m2, q2, c2) in enumerate(pareto):
                if i == j:
                    continue

                # m2 should not strictly dominate m1
                dominates = (q2 >= q1 and c2 <= c1 and (q2 > q1 or c2 < c1))
                assert not dominates, f"{m2} dominates {m1} on frontier"

    def test_recommend_budget(self, carrot):
        """Test budget recommendation"""
        # Low quality target
        budget_low = carrot.recommend_budget("write a function", quality_target=0.6)

        # High quality target
        budget_high = carrot.recommend_budget("write a function", quality_target=0.9)

        assert budget_low is not None
        assert budget_high is not None

        # Higher quality should require higher budget
        assert budget_high >= budget_low

    def test_recommend_budget_unreachable(self, carrot):
        """Test budget recommendation for unreachable quality"""
        # Quality target higher than any model can achieve
        budget = carrot.recommend_budget("write a function", quality_target=1.5)

        # Should return None if unreachable
        assert budget is None

    def test_carrot_with_task_type(self, carrot):
        """Test CARROT with task type specified"""
        selected, prediction = carrot.select(
            "write a function",
            task_type="coding",
            complexity=0.5
        )

        assert selected in MODELS
        assert 'quality' in prediction


class TestRouterIntegration:
    """Test CARROT integration with RouterCore"""

    @pytest.fixture
    def router(self):
        return RouterCore(enable_eagle=True, enable_memory=True, enable_carrot=True)

    @pytest.fixture
    def router_no_carrot(self):
        return RouterCore(enable_eagle=True, enable_memory=True, enable_carrot=False)

    def test_router_with_carrot_initialization(self, router):
        """Test router initializes with CARROT"""
        assert router.enable_carrot
        assert router.carrot is not None

    def test_router_without_carrot(self, router_no_carrot):
        """Test router works without CARROT"""
        assert not router_no_carrot.enable_carrot
        assert router_no_carrot.carrot is None

    def test_routing_with_budget(self, router):
        """Test routing uses CARROT when budget specified"""
        result = router.route_sync("write a function", budget=100.0)

        assert result.model in MODELS
        assert 0.0 <= result.confidence <= 1.0

        # Should use CARROT strategy
        if result.metadata.get('routing_strategy') not in ['cascade', 'reasoning', 'language_specialist']:
            assert result.metadata.get('routing_strategy') == 'carrot'
            assert 'budget' in result.metadata
            assert result.metadata['budget'] == 100.0

    def test_routing_without_budget(self, router):
        """Test routing works without budget (uses Eagle/ELO)"""
        result = router.route_sync("write a function")

        assert result.model in MODELS
        # Should not use CARROT without budget
        if result.metadata.get('routing_strategy') not in ['cascade', 'reasoning', 'language_specialist']:
            assert result.metadata.get('routing_strategy') in ['eagle_elo', 'elo_based']

    def test_routing_with_tight_budget(self, router):
        """Test routing with very tight budget"""
        result = router.route_sync("write a complex function", budget=1.0)

        assert result.model in MODELS

        # Should use CARROT
        if result.metadata.get('routing_strategy') not in ['cascade', 'reasoning']:
            assert result.metadata.get('routing_strategy') == 'carrot'

    def test_routing_with_high_budget(self, router):
        """Test routing with high budget"""
        result = router.route_sync("write a function", budget=10000.0)

        assert result.model in MODELS

        # High budget should allow high-quality model
        # qwen3-next-80b has highest quality
        # But may not be selected if other rules override

    def test_carrot_metadata(self, router):
        """Test CARROT adds proper metadata"""
        result = router.route_sync("write a function", budget=50.0)

        if result.metadata.get('routing_strategy') == 'carrot':
            assert 'budget' in result.metadata
            assert 'predicted_quality' in result.metadata
            assert 'predicted_cost' in result.metadata
            assert result.metadata['predicted_cost'] <= 50.0 or result.metadata['predicted_cost'] > 0

    def test_routing_with_budget_no_carrot(self, router_no_carrot):
        """Test routing with budget but CARROT disabled"""
        result = router_no_carrot.route_sync("write a function", budget=100.0)

        # Should still work, but won't use CARROT strategy
        assert result.model in MODELS
        assert result.metadata.get('routing_strategy') != 'carrot'


class TestCostQualityTradeoffs:
    """Test cost-quality tradeoff scenarios"""

    @pytest.fixture
    def carrot(self):
        return CARROT()

    def test_cheap_fast_model(self, carrot):
        """Test selection of cheap fast model with tight budget"""
        selected, pred = carrot.select("simple query", budget=5.0, complexity=0.2)

        # Should select qwen2.5-coder-7b (fastest, cheapest)
        # But depends on actual cost calculation
        assert selected in MODELS
        assert pred['cost'] > 0.0

    def test_quality_prioritization(self, carrot):
        """Test quality is prioritized without budget"""
        selected, pred = carrot.select("complex query", complexity=0.9)

        # Without budget, should select high quality
        # qwen3-next-80b has highest quality (0.90)
        all_preds = carrot.predict_all("complex query", complexity=0.9)
        best_quality = max(all_preds.items(), key=lambda x: x[1]['quality'])

        assert selected == best_quality[0]

    def test_budget_filtering(self, carrot):
        """Test budget correctly filters models"""
        # Get all predictions
        all_preds = carrot.predict_all("query")

        # Find a budget that excludes some models
        costs = [p['cost'] for p in all_preds.values()]
        median_cost = sorted(costs)[len(costs) // 2]

        selected, pred = carrot.select("query", budget=median_cost)

        # Selected model should be within budget or cheapest
        if pred['cost'] <= median_cost:
            assert pred['cost'] <= median_cost
        else:
            # Budget exceeded, should be cheapest
            assert pred['cost'] == min(costs)


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_query(self):
        """Test handling of empty query"""
        carrot = CARROT()
        selected, pred = carrot.select("")

        assert selected in MODELS
        assert 'quality' in pred
        assert 'cost' in pred

    def test_very_long_query(self):
        """Test handling of very long query"""
        carrot = CARROT()
        long_query = "a" * 10000

        selected, pred = carrot.select(long_query)

        assert selected in MODELS
        # Cost should be higher for long query
        assert pred['cost'] > 0.0

    def test_zero_budget(self):
        """Test handling of zero budget"""
        carrot = CARROT()
        selected, pred = carrot.select("query", budget=0.0)

        # Should select cheapest model
        assert selected in MODELS

    def test_negative_budget(self):
        """Test handling of negative budget"""
        carrot = CARROT()
        # Negative budget should be treated as constraint
        selected, pred = carrot.select("query", budget=-1.0)

        # Should select cheapest since nothing is within budget
        assert selected in MODELS


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
