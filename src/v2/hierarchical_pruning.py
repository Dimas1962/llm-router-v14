"""
Hierarchical Pruning - Component 5
Priority-based context reduction with smart content preservation
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


class Priority(Enum):
    """Content priority levels"""
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1


@dataclass
class ContentItem:
    """A piece of content with priority"""
    content: str
    priority: Priority
    size: int  # Size in characters/tokens
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PruningResult:
    """Result of pruning operation"""
    pruned_items: List[ContentItem]
    removed_items: List[ContentItem]
    original_size: int
    final_size: int
    reduction_ratio: float
    items_removed: int


class HierarchicalPruner:
    """
    Hierarchical Context Pruning System

    Features:
    - Priority-based content reduction
    - Hierarchical importance structure (Critical/High/Medium/Low)
    - Recursive removal of low-priority content
    - Critical fragment preservation
    - Smart reduction to target size
    - Memory optimization (~25% reduction target)
    """

    def __init__(
        self,
        target_reduction: float = 0.25,  # 25% reduction target
        preserve_critical: bool = True,
        aggressive_mode: bool = False
    ):
        """
        Initialize Hierarchical Pruner

        Args:
            target_reduction: Target reduction ratio (0.0-1.0)
            preserve_critical: Always preserve CRITICAL priority items
            aggressive_mode: More aggressive pruning
        """
        self.target_reduction = target_reduction
        self.preserve_critical = preserve_critical
        self.aggressive_mode = aggressive_mode

        # Statistics
        self.stats = {
            "total_prunings": 0,
            "total_items_removed": 0,
            "total_size_reduced": 0,
            "avg_reduction_ratio": 0.0
        }
        self._total_reduction = 0.0

    def prune(
        self,
        items: List[ContentItem],
        target_size: Optional[int] = None,
        min_priority: Optional[Priority] = None
    ) -> PruningResult:
        """
        Prune content items to target size

        Args:
            items: List of ContentItem objects
            target_size: Optional target size (if None, uses target_reduction)
            min_priority: Optional minimum priority to keep

        Returns:
            PruningResult with pruned content and statistics
        """
        if not items:
            return PruningResult(
                pruned_items=[],
                removed_items=[],
                original_size=0,
                final_size=0,
                reduction_ratio=0.0,
                items_removed=0
            )

        # Calculate original size
        original_size = sum(item.size for item in items)

        # Determine target size
        if target_size is None:
            target_size = int(original_size * (1 - self.target_reduction))

        # Start with all items
        remaining_items = items.copy()
        removed_items = []

        # Iteratively remove lowest priority items
        current_size = original_size

        while current_size > target_size and remaining_items:
            # Find removable item with lowest priority
            removable_item = self._find_lowest_priority_item(
                remaining_items,
                min_priority
            )

            if removable_item is None:
                # No more removable items
                break

            # Remove the item
            remaining_items.remove(removable_item)
            removed_items.append(removable_item)
            current_size -= removable_item.size

        # Calculate statistics
        final_size = sum(item.size for item in remaining_items)
        reduction_ratio = (original_size - final_size) / original_size if original_size > 0 else 0.0

        result = PruningResult(
            pruned_items=remaining_items,
            removed_items=removed_items,
            original_size=original_size,
            final_size=final_size,
            reduction_ratio=reduction_ratio,
            items_removed=len(removed_items)
        )

        # Update statistics
        self._update_stats(result)

        return result

    def _find_lowest_priority_item(
        self,
        items: List[ContentItem],
        min_priority: Optional[Priority] = None
    ) -> Optional[ContentItem]:
        """
        Find the item with lowest removable priority

        Args:
            items: List of items to search
            min_priority: Minimum priority to consider for removal

        Returns:
            Lowest priority item or None
        """
        if not items:
            return None

        # Filter by constraints
        removable_items = []
        for item in items:
            # Skip CRITICAL if preserve_critical is True
            if self.preserve_critical and item.priority == Priority.CRITICAL:
                continue

            # Skip if above min_priority
            if min_priority and item.priority.value > min_priority.value:
                continue

            removable_items.append(item)

        if not removable_items:
            return None

        # Find item with lowest priority
        # If priorities are equal, prefer larger items in aggressive mode
        lowest_item = min(
            removable_items,
            key=lambda x: (
                x.priority.value,
                -x.size if self.aggressive_mode else x.size
            )
        )

        return lowest_item

    def prune_by_priority(
        self,
        items: List[ContentItem],
        remove_priorities: List[Priority]
    ) -> PruningResult:
        """
        Remove all items with specified priorities

        Args:
            items: List of ContentItem objects
            remove_priorities: Priorities to remove

        Returns:
            PruningResult
        """
        original_size = sum(item.size for item in items)

        # Filter items
        remaining_items = [
            item for item in items
            if item.priority not in remove_priorities
        ]

        removed_items = [
            item for item in items
            if item.priority in remove_priorities
        ]

        final_size = sum(item.size for item in remaining_items)
        reduction_ratio = (original_size - final_size) / original_size if original_size > 0 else 0.0

        result = PruningResult(
            pruned_items=remaining_items,
            removed_items=removed_items,
            original_size=original_size,
            final_size=final_size,
            reduction_ratio=reduction_ratio,
            items_removed=len(removed_items)
        )

        self._update_stats(result)

        return result

    def recursive_prune(
        self,
        items: List[ContentItem],
        target_size: int,
        max_iterations: int = 10
    ) -> PruningResult:
        """
        Recursively prune until target size is reached

        Args:
            items: List of ContentItem objects
            target_size: Target size to achieve
            max_iterations: Maximum pruning iterations

        Returns:
            PruningResult
        """
        current_items = items.copy()
        all_removed_items = []
        original_size = sum(item.size for item in items)

        for iteration in range(max_iterations):
            current_size = sum(item.size for item in current_items)

            if current_size <= target_size:
                break

            # Prune one level
            # Start with LOW priority, then MEDIUM, then HIGH
            priorities_to_try = [Priority.LOW, Priority.MEDIUM, Priority.HIGH]

            pruned = False
            for min_priority in priorities_to_try:
                # Try to remove items below this priority
                removable = [
                    item for item in current_items
                    if item.priority.value <= min_priority.value
                ]

                if not removable:
                    continue

                if self.preserve_critical:
                    removable = [
                        item for item in removable
                        if item.priority != Priority.CRITICAL
                    ]

                if removable:
                    # Remove lowest priority item
                    item_to_remove = min(
                        removable,
                        key=lambda x: (x.priority.value, -x.size if self.aggressive_mode else x.size)
                    )

                    current_items.remove(item_to_remove)
                    all_removed_items.append(item_to_remove)
                    pruned = True
                    break

            if not pruned:
                # Can't prune anymore
                break

        final_size = sum(item.size for item in current_items)
        reduction_ratio = (original_size - final_size) / original_size if original_size > 0 else 0.0

        result = PruningResult(
            pruned_items=current_items,
            removed_items=all_removed_items,
            original_size=original_size,
            final_size=final_size,
            reduction_ratio=reduction_ratio,
            items_removed=len(all_removed_items)
        )

        self._update_stats(result)

        return result

    def optimize_memory(
        self,
        items: List[ContentItem]
    ) -> PruningResult:
        """
        Optimize memory usage with default 25% reduction

        Args:
            items: List of ContentItem objects

        Returns:
            PruningResult with ~25% reduction
        """
        return self.prune(items)

    def _update_stats(self, result: PruningResult):
        """
        Update pruning statistics

        Args:
            result: PruningResult to record
        """
        self.stats["total_prunings"] += 1
        self.stats["total_items_removed"] += result.items_removed
        self.stats["total_size_reduced"] += (result.original_size - result.final_size)

        self._total_reduction += result.reduction_ratio
        self.stats["avg_reduction_ratio"] = self._total_reduction / self.stats["total_prunings"]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pruning statistics

        Returns:
            Statistics dictionary
        """
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "total_prunings": 0,
            "total_items_removed": 0,
            "total_size_reduced": 0,
            "avg_reduction_ratio": 0.0
        }
        self._total_reduction = 0.0

    def analyze_content(
        self,
        items: List[ContentItem]
    ) -> Dict[str, Any]:
        """
        Analyze content distribution

        Args:
            items: List of ContentItem objects

        Returns:
            Analysis dictionary
        """
        if not items:
            return {
                "total_items": 0,
                "total_size": 0,
                "priority_distribution": {},
                "avg_item_size": 0
            }

        total_size = sum(item.size for item in items)
        priority_counts = {
            "CRITICAL": 0,
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0
        }
        priority_sizes = {
            "CRITICAL": 0,
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0
        }

        for item in items:
            priority_name = item.priority.name
            priority_counts[priority_name] += 1
            priority_sizes[priority_name] += item.size

        return {
            "total_items": len(items),
            "total_size": total_size,
            "avg_item_size": total_size / len(items),
            "priority_distribution": {
                priority: {
                    "count": priority_counts[priority],
                    "size": priority_sizes[priority],
                    "percentage": (priority_sizes[priority] / total_size * 100) if total_size > 0 else 0
                }
                for priority in priority_counts.keys()
            }
        }
