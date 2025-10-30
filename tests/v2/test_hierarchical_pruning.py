"""
Tests for Hierarchical Pruning (Component 5)
"""

import pytest
from src.v2.hierarchical_pruning import (
    HierarchicalPruner,
    ContentItem,
    Priority,
    PruningResult
)


def test_initialization():
    """Test HierarchicalPruner initialization"""
    pruner = HierarchicalPruner()

    assert pruner.target_reduction == 0.25
    assert pruner.preserve_critical is True
    assert pruner.aggressive_mode is False
    assert pruner.stats["total_prunings"] == 0


def test_basic_pruning():
    """Test basic content pruning"""
    pruner = HierarchicalPruner(target_reduction=0.5)

    items = [
        ContentItem("Critical content", Priority.CRITICAL, 100),
        ContentItem("High priority", Priority.HIGH, 100),
        ContentItem("Medium priority", Priority.MEDIUM, 100),
        ContentItem("Low priority", Priority.LOW, 100),
    ]

    result = pruner.prune(items)

    assert isinstance(result, PruningResult)
    assert result.original_size == 400
    assert result.final_size <= 200  # ~50% reduction
    assert len(result.pruned_items) < len(items)
    assert len(result.removed_items) > 0


def test_critical_preservation():
    """Test that CRITICAL items are preserved"""
    pruner = HierarchicalPruner(preserve_critical=True, target_reduction=0.8)

    items = [
        ContentItem("Critical 1", Priority.CRITICAL, 100),
        ContentItem("Critical 2", Priority.CRITICAL, 100),
        ContentItem("Low 1", Priority.LOW, 100),
        ContentItem("Low 2", Priority.LOW, 100),
    ]

    result = pruner.prune(items)

    # All CRITICAL items should be preserved
    critical_remaining = [
        item for item in result.pruned_items
        if item.priority == Priority.CRITICAL
    ]

    assert len(critical_remaining) == 2


def test_priority_based_removal():
    """Test removing items by priority"""
    pruner = HierarchicalPruner()

    items = [
        ContentItem("Critical", Priority.CRITICAL, 100),
        ContentItem("High", Priority.HIGH, 100),
        ContentItem("Medium", Priority.MEDIUM, 100),
        ContentItem("Low", Priority.LOW, 100),
    ]

    # Remove all LOW and MEDIUM priority items
    result = pruner.prune_by_priority(
        items,
        remove_priorities=[Priority.LOW, Priority.MEDIUM]
    )

    assert len(result.pruned_items) == 2
    assert len(result.removed_items) == 2
    assert result.reduction_ratio == 0.5


def test_target_size_pruning():
    """Test pruning to specific target size"""
    pruner = HierarchicalPruner()

    items = [
        ContentItem("Item 1", Priority.LOW, 100),
        ContentItem("Item 2", Priority.LOW, 100),
        ContentItem("Item 3", Priority.MEDIUM, 100),
        ContentItem("Item 4", Priority.HIGH, 100),
    ]

    # Prune to 250 characters
    result = pruner.prune(items, target_size=250)

    assert result.final_size <= 250
    assert result.original_size == 400


def test_recursive_pruning():
    """Test recursive pruning"""
    pruner = HierarchicalPruner()

    items = [
        ContentItem("Critical", Priority.CRITICAL, 100),
        ContentItem("High 1", Priority.HIGH, 100),
        ContentItem("High 2", Priority.HIGH, 100),
        ContentItem("Medium 1", Priority.MEDIUM, 100),
        ContentItem("Medium 2", Priority.MEDIUM, 100),
        ContentItem("Low 1", Priority.LOW, 100),
        ContentItem("Low 2", Priority.LOW, 100),
        ContentItem("Low 3", Priority.LOW, 100),
    ]

    # Recursive prune to 300
    result = pruner.recursive_prune(items, target_size=300)

    assert result.final_size <= 300
    assert result.items_removed > 0
    # CRITICAL should be preserved
    assert any(item.priority == Priority.CRITICAL for item in result.pruned_items)


def test_memory_optimization():
    """Test default memory optimization (25% reduction)"""
    pruner = HierarchicalPruner()

    items = [
        ContentItem(f"Item {i}", Priority.LOW, 100)
        for i in range(10)
    ]

    result = pruner.optimize_memory(items)

    # Should achieve ~25% reduction
    assert 0.20 <= result.reduction_ratio <= 0.30


def test_aggressive_mode():
    """Test aggressive pruning mode"""
    pruner_normal = HierarchicalPruner(aggressive_mode=False)
    pruner_aggressive = HierarchicalPruner(aggressive_mode=True)

    items = [
        ContentItem("Small", Priority.LOW, 10),
        ContentItem("Large", Priority.LOW, 100),
    ]

    # Aggressive mode should prefer removing larger items
    result_aggressive = pruner_aggressive.prune(items, target_size=50)

    # Should remove the large item first
    assert result_aggressive.items_removed >= 1


def test_statistics_tracking():
    """Test pruning statistics"""
    pruner = HierarchicalPruner()

    items1 = [ContentItem(f"Item {i}", Priority.LOW, 50) for i in range(5)]
    items2 = [ContentItem(f"Item {i}", Priority.MEDIUM, 50) for i in range(5)]

    pruner.prune(items1)
    pruner.prune(items2)

    stats = pruner.get_stats()

    assert stats["total_prunings"] == 2
    assert stats["total_items_removed"] > 0
    assert stats["total_size_reduced"] > 0
    assert 0 <= stats["avg_reduction_ratio"] <= 1


def test_content_analysis():
    """Test content distribution analysis"""
    pruner = HierarchicalPruner()

    items = [
        ContentItem("Critical", Priority.CRITICAL, 100),
        ContentItem("High 1", Priority.HIGH, 50),
        ContentItem("High 2", Priority.HIGH, 50),
        ContentItem("Medium", Priority.MEDIUM, 75),
        ContentItem("Low 1", Priority.LOW, 25),
        ContentItem("Low 2", Priority.LOW, 25),
        ContentItem("Low 3", Priority.LOW, 25),
    ]

    analysis = pruner.analyze_content(items)

    assert analysis["total_items"] == 7
    assert analysis["total_size"] == 350
    assert "priority_distribution" in analysis
    assert "CRITICAL" in analysis["priority_distribution"]
    assert analysis["priority_distribution"]["CRITICAL"]["count"] == 1


def test_empty_content():
    """Test pruning empty content"""
    pruner = HierarchicalPruner()

    result = pruner.prune([])

    assert result.original_size == 0
    assert result.final_size == 0
    assert result.reduction_ratio == 0.0
    assert len(result.pruned_items) == 0


def test_no_removable_items():
    """Test pruning when all items are CRITICAL"""
    pruner = HierarchicalPruner(preserve_critical=True)

    items = [
        ContentItem("Critical 1", Priority.CRITICAL, 100),
        ContentItem("Critical 2", Priority.CRITICAL, 100),
        ContentItem("Critical 3", Priority.CRITICAL, 100),
    ]

    # Try to prune to tiny size
    result = pruner.prune(items, target_size=50)

    # Should not remove any CRITICAL items
    assert len(result.pruned_items) == 3
    assert result.items_removed == 0


def test_min_priority_filter():
    """Test pruning with minimum priority filter"""
    pruner = HierarchicalPruner(preserve_critical=False)

    items = [
        ContentItem("Critical", Priority.CRITICAL, 100),
        ContentItem("High", Priority.HIGH, 100),
        ContentItem("Medium", Priority.MEDIUM, 100),
        ContentItem("Low", Priority.LOW, 100),
    ]

    # Only allow removing LOW priority
    result = pruner.prune(
        items,
        target_size=200,
        min_priority=Priority.LOW
    )

    # Should only remove LOW items
    removed_priorities = [item.priority for item in result.removed_items]
    assert all(p == Priority.LOW for p in removed_priorities)


def test_stats_reset():
    """Test statistics reset"""
    pruner = HierarchicalPruner()

    items = [ContentItem(f"Item {i}", Priority.LOW, 50) for i in range(5)]
    pruner.prune(items)

    assert pruner.stats["total_prunings"] == 1

    pruner.reset_stats()

    assert pruner.stats["total_prunings"] == 0
    assert pruner.stats["total_items_removed"] == 0
    assert pruner.stats["avg_reduction_ratio"] == 0.0


def test_reduction_ratio_calculation():
    """Test reduction ratio is calculated correctly"""
    pruner = HierarchicalPruner(target_reduction=0.5)

    items = [
        ContentItem(f"Item {i}", Priority.LOW, 100)
        for i in range(10)
    ]

    result = pruner.prune(items)

    # Reduction ratio should be approximately 0.5
    expected_ratio = (result.original_size - result.final_size) / result.original_size
    assert abs(result.reduction_ratio - expected_ratio) < 0.01


def test_multiple_priority_levels():
    """Test pruning with all priority levels"""
    pruner = HierarchicalPruner()

    items = []
    for priority in [Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.LOW]:
        for i in range(3):
            items.append(ContentItem(f"{priority.name} {i}", priority, 50))

    # Total: 12 items, 600 size
    result = pruner.prune(items, target_size=300)

    # Should reduce by ~50%
    assert result.final_size <= 300
    # CRITICAL should remain
    assert any(item.priority == Priority.CRITICAL for item in result.pruned_items)


def test_analysis_empty_content():
    """Test content analysis with empty list"""
    pruner = HierarchicalPruner()

    analysis = pruner.analyze_content([])

    assert analysis["total_items"] == 0
    assert analysis["total_size"] == 0
    assert analysis["avg_item_size"] == 0
