"""
Tests for Dynamic Context Sizing (Component 4)
"""

import pytest
from src.v2.context_sizing import (
    DynamicContextSizer,
    ContextBudget,
    ContextSizingStrategy,
    ModelFamily,
    MODEL_SPECS
)


def test_initialization():
    """Test DynamicContextSizer initialization"""
    sizer = DynamicContextSizer()

    assert sizer.default_strategy == ContextSizingStrategy.BALANCED
    assert sizer.response_reserve_ratio == 0.3
    assert sizer.min_history_messages == 2
    assert sizer.stats["total_budgets"] == 0


def test_basic_budget_calculation():
    """Test basic budget calculation"""
    sizer = DynamicContextSizer()

    budget = sizer.calculate_budget(
        query="What is machine learning?",
        model_name="gpt-3.5-turbo",
        complexity=0.5
    )

    assert isinstance(budget, ContextBudget)
    assert budget.total_tokens > 0
    assert budget.query_tokens > 0
    assert budget.reserved_tokens > 0
    assert budget.available_tokens >= 0


def test_token_estimation():
    """Test token estimation"""
    sizer = DynamicContextSizer()

    # Get model spec
    model_spec = sizer._get_model_spec("gpt-3.5-turbo")

    # Estimate tokens for short text
    short_text = "Hello"
    short_tokens = sizer._estimate_tokens(short_text, model_spec)
    assert short_tokens > 0

    # Estimate tokens for long text
    long_text = "This is a much longer text with more words and complexity" * 10
    long_tokens = sizer._estimate_tokens(long_text, model_spec)
    assert long_tokens > short_tokens


def test_model_spec_lookup():
    """Test model specification lookup"""
    sizer = DynamicContextSizer()

    # Exact match
    gpt_spec = sizer._get_model_spec("gpt-4")
    assert gpt_spec.name == "gpt-4"
    assert gpt_spec.family == ModelFamily.GPT

    # Partial match
    claude_spec = sizer._get_model_spec("claude")
    assert claude_spec.family == ModelFamily.CLAUDE

    # Unknown model (should default)
    unknown_spec = sizer._get_model_spec("unknown-model")
    assert unknown_spec is not None


def test_aggressive_strategy():
    """Test AGGRESSIVE sizing strategy"""
    sizer = DynamicContextSizer()

    budget = sizer.calculate_budget(
        query="Test query",
        model_name="gpt-4",
        complexity=0.8,
        strategy=ContextSizingStrategy.AGGRESSIVE
    )

    # Should use high percentage of context
    model_spec = MODEL_SPECS["gpt-4"]
    utilization = budget.total_tokens / model_spec.max_context
    assert utilization >= 0.9  # Should be ~95%


def test_conservative_strategy():
    """Test CONSERVATIVE sizing strategy"""
    sizer = DynamicContextSizer()

    budget = sizer.calculate_budget(
        query="Test query",
        model_name="gpt-4",
        complexity=0.3,
        strategy=ContextSizingStrategy.CONSERVATIVE
    )

    # Should use low percentage of context
    model_spec = MODEL_SPECS["gpt-4"]
    utilization = budget.total_tokens / model_spec.max_context
    assert utilization <= 0.6  # Should be ~50%


def test_adaptive_strategy():
    """Test ADAPTIVE sizing strategy"""
    sizer = DynamicContextSizer()

    # Low complexity - smaller budget
    low_budget = sizer.calculate_budget(
        query="Test",
        model_name="gpt-4",
        complexity=0.2,
        strategy=ContextSizingStrategy.ADAPTIVE
    )

    # High complexity - larger budget
    high_budget = sizer.calculate_budget(
        query="Test",
        model_name="gpt-4",
        complexity=0.9,
        strategy=ContextSizingStrategy.ADAPTIVE
    )

    # Higher complexity should get more tokens
    assert high_budget.total_tokens > low_budget.total_tokens


def test_history_sizing():
    """Test history token calculation"""
    sizer = DynamicContextSizer()

    history = [
        "Message 1",
        "Message 2",
        "Message 3",
        "Message 4",
        "Message 5"
    ]

    # With history
    budget_with_history = sizer.calculate_budget(
        query="Test query",
        model_name="gpt-3.5-turbo",
        complexity=0.5,
        history=history
    )

    # Without history
    budget_no_history = sizer.calculate_budget(
        query="Test query",
        model_name="gpt-3.5-turbo",
        complexity=0.5,
        history=None
    )

    # With history should have more history tokens
    assert budget_with_history.history_tokens > budget_no_history.history_tokens


def test_complexity_impact():
    """Test complexity impact on budget"""
    sizer = DynamicContextSizer()

    # Simple task
    simple_budget = sizer.calculate_budget(
        query="What is 2+2?",
        model_name="gpt-3.5-turbo",
        complexity=0.1,
        strategy=ContextSizingStrategy.ADAPTIVE
    )

    # Complex task
    complex_budget = sizer.calculate_budget(
        query="Explain quantum computing",
        model_name="gpt-3.5-turbo",
        complexity=0.9,
        strategy=ContextSizingStrategy.ADAPTIVE
    )

    # Complex task should get more budget
    assert complex_budget.total_tokens > simple_budget.total_tokens


def test_model_family_optimization():
    """Test model family-specific optimization"""
    sizer = DynamicContextSizer()

    # Create base budget
    base_budget = ContextBudget(
        total_tokens=4000,
        query_tokens=100,
        history_tokens=500,
        reserved_tokens=1000,
        available_tokens=2400
    )

    # Optimize for Claude (should increase reserved)
    claude_budget = sizer.optimize_for_model(ModelFamily.CLAUDE, base_budget)
    assert claude_budget.reserved_tokens > base_budget.reserved_tokens

    # Optimize for Llama (should decrease reserved)
    llama_budget = sizer.optimize_for_model(ModelFamily.LLAMA, base_budget)
    assert llama_budget.reserved_tokens < base_budget.reserved_tokens


def test_statistics_tracking():
    """Test statistics tracking"""
    sizer = DynamicContextSizer()

    # Calculate multiple budgets
    for i in range(5):
        sizer.calculate_budget(
            query=f"Query {i}",
            model_name="gpt-4",
            complexity=0.5
        )

    stats = sizer.get_stats()

    assert stats["total_budgets"] == 5
    assert 0 <= stats["avg_utilization"] <= 1.0
    assert "strategy_usage" in stats


def test_strategy_recommendation():
    """Test strategy recommendation"""
    sizer = DynamicContextSizer()

    # Simple task, short history -> CONSERVATIVE
    rec1 = sizer.recommend_strategy(
        complexity=0.2,
        history_length=3,
        model_context_size=4096
    )
    assert rec1 == ContextSizingStrategy.CONSERVATIVE

    # Complex task, large context -> AGGRESSIVE
    rec2 = sizer.recommend_strategy(
        complexity=0.8,
        history_length=10,
        model_context_size=100000
    )
    assert rec2 == ContextSizingStrategy.AGGRESSIVE

    # Medium complexity -> ADAPTIVE
    rec3 = sizer.recommend_strategy(
        complexity=0.5,
        history_length=5,
        model_context_size=8192
    )
    assert rec3 == ContextSizingStrategy.ADAPTIVE


def test_multiple_models():
    """Test budget calculation for different models"""
    sizer = DynamicContextSizer()

    models = ["gpt-4", "claude-3", "llama-2-7b", "qwen2.5-coder-7b"]
    budgets = {}

    for model in models:
        budget = sizer.calculate_budget(
            query="Test query",
            model_name=model,
            complexity=0.5
        )
        budgets[model] = budget

    # Claude should have largest context
    assert budgets["claude-3"].total_tokens > budgets["gpt-4"].total_tokens
    assert budgets["claude-3"].total_tokens > budgets["llama-2-7b"].total_tokens


def test_stats_reset():
    """Test statistics reset"""
    sizer = DynamicContextSizer()

    # Generate some stats
    sizer.calculate_budget("test", "gpt-4", 0.5)
    sizer.calculate_budget("test", "gpt-4", 0.5)

    assert sizer.stats["total_budgets"] == 2

    # Reset
    sizer.reset_stats()

    assert sizer.stats["total_budgets"] == 0
    assert sizer.stats["avg_utilization"] == 0.0


def test_reserved_tokens():
    """Test response reserve ratio"""
    # Custom reserve ratio
    sizer = DynamicContextSizer(response_reserve_ratio=0.4)

    budget = sizer.calculate_budget(
        query="Test",
        model_name="gpt-4",
        complexity=0.5
    )

    # Reserve is calculated from max_context, not strategy-adjusted total
    model_spec = MODEL_SPECS["gpt-4"]
    expected_reserve = int(model_spec.max_context * 0.4)

    # Check reserved tokens match the ratio applied to max_context
    assert budget.reserved_tokens == expected_reserve

    # Verify it's being used in the calculation
    assert budget.available_tokens == (
        budget.total_tokens - budget.reserved_tokens -
        budget.query_tokens - budget.history_tokens
    )


def test_empty_query():
    """Test budget with empty query"""
    sizer = DynamicContextSizer()

    budget = sizer.calculate_budget(
        query="",
        model_name="gpt-4",
        complexity=0.5
    )

    # Should still work, query tokens should be minimal
    assert budget.query_tokens >= 0
    assert budget.total_tokens > 0


def test_very_long_query():
    """Test budget with very long query"""
    sizer = DynamicContextSizer()

    long_query = "This is a test query. " * 200

    budget = sizer.calculate_budget(
        query=long_query,
        model_name="gpt-4",
        complexity=0.5
    )

    # Query tokens should be significant
    assert budget.query_tokens > 100
    assert budget.available_tokens >= 0


def test_history_size_limits():
    """Test history size with different strategies"""
    sizer = DynamicContextSizer(min_history_messages=3)

    history = [f"Message {i}" for i in range(20)]

    # Conservative should use fewer messages
    size_conservative = sizer._get_history_size(
        len(history),
        0.5,
        ContextSizingStrategy.CONSERVATIVE
    )

    # Aggressive should use more messages
    size_aggressive = sizer._get_history_size(
        len(history),
        0.5,
        ContextSizingStrategy.AGGRESSIVE
    )

    assert size_aggressive > size_conservative
    assert size_conservative >= sizer.min_history_messages
