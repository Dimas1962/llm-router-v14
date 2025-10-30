"""
Dynamic Context Sizing - Component 4
Adaptive context budget calculation and optimization
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class ModelFamily(Enum):
    """LLM model families"""
    GPT = "gpt"
    CLAUDE = "claude"
    LLAMA = "llama"
    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"


@dataclass
class ModelSpec:
    """Model specifications"""
    name: str
    family: ModelFamily
    max_context: int  # Maximum context window
    avg_tokens_per_char: float  # Average tokens per character


@dataclass
class ContextBudget:
    """Context budget allocation"""
    total_tokens: int
    query_tokens: int
    history_tokens: int
    reserved_tokens: int  # Reserved for response
    available_tokens: int


class ContextSizingStrategy(Enum):
    """Budget optimization strategies"""
    AGGRESSIVE = "aggressive"  # Maximum context usage
    BALANCED = "balanced"      # Balanced approach
    CONSERVATIVE = "conservative"  # Minimal context usage
    ADAPTIVE = "adaptive"      # Adapts based on complexity


# Model specifications
MODEL_SPECS = {
    "gpt-4": ModelSpec("gpt-4", ModelFamily.GPT, 8192, 0.25),
    "gpt-3.5-turbo": ModelSpec("gpt-3.5-turbo", ModelFamily.GPT, 4096, 0.25),
    "claude-3": ModelSpec("claude-3", ModelFamily.CLAUDE, 200000, 0.24),
    "claude-2": ModelSpec("claude-2", ModelFamily.CLAUDE, 100000, 0.24),
    "llama-2-7b": ModelSpec("llama-2-7b", ModelFamily.LLAMA, 4096, 0.28),
    "llama-2-13b": ModelSpec("llama-2-13b", ModelFamily.LLAMA, 4096, 0.28),
    "qwen2.5-coder-7b": ModelSpec("qwen2.5-coder-7b", ModelFamily.QWEN, 32768, 0.26),
    "qwen3-coder-30b": ModelSpec("qwen3-coder-30b", ModelFamily.QWEN, 32768, 0.26),
    "deepseek-coder-16b": ModelSpec("deepseek-coder-16b", ModelFamily.DEEPSEEK, 16384, 0.27),
    "gemini-pro": ModelSpec("gemini-pro", ModelFamily.GEMINI, 32768, 0.25),
}


class DynamicContextSizer:
    """
    Dynamic Context Sizing System

    Features:
    - Dynamic budget calculation based on complexity
    - Model-specific adaptation
    - History length management
    - Token estimation
    - Multiple optimization strategies
    """

    def __init__(
        self,
        default_strategy: ContextSizingStrategy = ContextSizingStrategy.BALANCED,
        response_reserve_ratio: float = 0.3,  # 30% reserved for response
        min_history_messages: int = 2
    ):
        """
        Initialize Dynamic Context Sizer

        Args:
            default_strategy: Default optimization strategy
            response_reserve_ratio: Ratio of context to reserve for response
            min_history_messages: Minimum history messages to include
        """
        self.default_strategy = default_strategy
        self.response_reserve_ratio = response_reserve_ratio
        self.min_history_messages = min_history_messages

        # Statistics
        self.stats = {
            "total_budgets": 0,
            "avg_utilization": 0.0,
            "strategy_usage": {}
        }
        self._total_utilization = 0.0

    def calculate_budget(
        self,
        query: str,
        model_name: str,
        complexity: float,
        history: Optional[List[str]] = None,
        strategy: Optional[ContextSizingStrategy] = None
    ) -> ContextBudget:
        """
        Calculate context budget

        Args:
            query: User query
            model_name: Target model name
            complexity: Task complexity (0.0-1.0)
            history: Optional conversation history
            strategy: Optional strategy override

        Returns:
            ContextBudget with token allocations
        """
        strategy = strategy or self.default_strategy

        # Get model spec
        model_spec = self._get_model_spec(model_name)

        # Estimate tokens
        query_tokens = self._estimate_tokens(query, model_spec)
        history_tokens = self._estimate_history_tokens(
            history or [],
            model_spec,
            complexity,
            strategy
        )

        # Calculate total available
        max_context = model_spec.max_context
        reserved_tokens = int(max_context * self.response_reserve_ratio)

        # Apply strategy-based adjustments
        total_tokens = self._apply_strategy(
            max_context,
            complexity,
            strategy
        )

        # Ensure we don't exceed limits
        available_tokens = total_tokens - reserved_tokens - query_tokens - history_tokens
        available_tokens = max(0, available_tokens)

        budget = ContextBudget(
            total_tokens=total_tokens,
            query_tokens=query_tokens,
            history_tokens=history_tokens,
            reserved_tokens=reserved_tokens,
            available_tokens=available_tokens
        )

        # Update statistics
        self._update_stats(budget, max_context, strategy)

        return budget

    def _get_model_spec(self, model_name: str) -> ModelSpec:
        """
        Get model specification

        Args:
            model_name: Model name

        Returns:
            ModelSpec for the model
        """
        # Try exact match first
        if model_name in MODEL_SPECS:
            return MODEL_SPECS[model_name]

        # Try partial match
        for spec_name, spec in MODEL_SPECS.items():
            if spec_name.startswith(model_name) or model_name.startswith(spec_name):
                return spec

        # Default to GPT-3.5 spec if unknown
        return MODEL_SPECS["gpt-3.5-turbo"]

    def _estimate_tokens(self, text: str, model_spec: ModelSpec) -> int:
        """
        Estimate token count for text

        Args:
            text: Input text
            model_spec: Model specification

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # Simple estimation: chars * avg_tokens_per_char
        estimated = int(len(text) * model_spec.avg_tokens_per_char)

        # Add overhead for formatting
        overhead = 10
        return estimated + overhead

    def _estimate_history_tokens(
        self,
        history: List[str],
        model_spec: ModelSpec,
        complexity: float,
        strategy: ContextSizingStrategy
    ) -> int:
        """
        Estimate tokens needed for history

        Args:
            history: Conversation history
            model_spec: Model specification
            complexity: Task complexity
            strategy: Sizing strategy

        Returns:
            Estimated history tokens
        """
        if not history:
            return 0

        # Determine how many messages to include
        num_messages = self._get_history_size(
            len(history),
            complexity,
            strategy
        )

        # Take most recent messages
        recent_history = history[-num_messages:] if num_messages > 0 else []

        # Estimate tokens
        total_tokens = sum(
            self._estimate_tokens(msg, model_spec)
            for msg in recent_history
        )

        return total_tokens

    def _get_history_size(
        self,
        total_messages: int,
        complexity: float,
        strategy: ContextSizingStrategy
    ) -> int:
        """
        Determine optimal history size

        Args:
            total_messages: Total available messages
            complexity: Task complexity
            strategy: Sizing strategy

        Returns:
            Number of messages to include
        """
        if total_messages == 0:
            return 0

        # Base size depends on strategy
        if strategy == ContextSizingStrategy.AGGRESSIVE:
            base_ratio = 1.0  # Use all history
        elif strategy == ContextSizingStrategy.CONSERVATIVE:
            base_ratio = 0.3  # Use minimal history
        elif strategy == ContextSizingStrategy.ADAPTIVE:
            # Scale with complexity
            base_ratio = 0.5 + (complexity * 0.4)
        else:  # BALANCED
            base_ratio = 0.6

        # Calculate size
        size = int(total_messages * base_ratio)

        # Ensure minimum
        size = max(self.min_history_messages, size)

        # Don't exceed total
        size = min(size, total_messages)

        return size

    def _apply_strategy(
        self,
        max_context: int,
        complexity: float,
        strategy: ContextSizingStrategy
    ) -> int:
        """
        Apply strategy-based adjustments to context size

        Args:
            max_context: Maximum context window
            complexity: Task complexity
            strategy: Sizing strategy

        Returns:
            Adjusted context size
        """
        if strategy == ContextSizingStrategy.AGGRESSIVE:
            # Use 95% of max context
            return int(max_context * 0.95)

        elif strategy == ContextSizingStrategy.CONSERVATIVE:
            # Use 50% of max context
            return int(max_context * 0.50)

        elif strategy == ContextSizingStrategy.ADAPTIVE:
            # Scale from 60% to 90% based on complexity
            ratio = 0.60 + (complexity * 0.30)
            return int(max_context * ratio)

        else:  # BALANCED
            # Use 75% of max context
            return int(max_context * 0.75)

    def _update_stats(
        self,
        budget: ContextBudget,
        max_context: int,
        strategy: ContextSizingStrategy
    ):
        """
        Update statistics

        Args:
            budget: Calculated budget
            max_context: Maximum context
            strategy: Used strategy
        """
        self.stats["total_budgets"] += 1

        # Track utilization
        utilization = budget.total_tokens / max_context
        self._total_utilization += utilization
        self.stats["avg_utilization"] = self._total_utilization / self.stats["total_budgets"]

        # Track strategy usage
        strategy_name = strategy.value
        if strategy_name not in self.stats["strategy_usage"]:
            self.stats["strategy_usage"][strategy_name] = 0
        self.stats["strategy_usage"][strategy_name] += 1

    def optimize_for_model(
        self,
        model_family: ModelFamily,
        base_budget: ContextBudget
    ) -> ContextBudget:
        """
        Optimize budget for specific model family

        Args:
            model_family: Target model family
            base_budget: Base budget to optimize

        Returns:
            Optimized ContextBudget
        """
        # Model-specific optimizations
        if model_family == ModelFamily.CLAUDE:
            # Claude handles large contexts well
            reserved = int(base_budget.reserved_tokens * 1.2)
        elif model_family == ModelFamily.LLAMA:
            # Llama prefers smaller contexts
            reserved = int(base_budget.reserved_tokens * 0.8)
        elif model_family == ModelFamily.GPT:
            # GPT balanced approach
            reserved = base_budget.reserved_tokens
        else:
            # Default
            reserved = base_budget.reserved_tokens

        # Recalculate available
        available = (
            base_budget.total_tokens -
            reserved -
            base_budget.query_tokens -
            base_budget.history_tokens
        )
        available = max(0, available)

        return ContextBudget(
            total_tokens=base_budget.total_tokens,
            query_tokens=base_budget.query_tokens,
            history_tokens=base_budget.history_tokens,
            reserved_tokens=reserved,
            available_tokens=available
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get sizing statistics

        Returns:
            Statistics dictionary
        """
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "total_budgets": 0,
            "avg_utilization": 0.0,
            "strategy_usage": {}
        }
        self._total_utilization = 0.0

    def recommend_strategy(
        self,
        complexity: float,
        history_length: int,
        model_context_size: int
    ) -> ContextSizingStrategy:
        """
        Recommend optimal strategy

        Args:
            complexity: Task complexity
            history_length: Length of conversation history
            model_context_size: Model's maximum context

        Returns:
            Recommended ContextSizingStrategy
        """
        # For simple tasks with short history, use conservative
        if complexity < 0.3 and history_length < 5:
            return ContextSizingStrategy.CONSERVATIVE

        # For complex tasks with large context window, use aggressive
        if complexity > 0.7 and model_context_size > 32000:
            return ContextSizingStrategy.AGGRESSIVE

        # For varying complexity, use adaptive
        if 0.4 <= complexity <= 0.7:
            return ContextSizingStrategy.ADAPTIVE

        # Default to balanced
        return ContextSizingStrategy.BALANCED
