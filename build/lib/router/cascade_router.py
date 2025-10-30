"""
Cascade Router - Multi-tier cascading with confidence-based escalation
Phase 6: Advanced Features
"""

import logging
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from .models import MODELS, get_model_config


logger = logging.getLogger(__name__)


@dataclass
class CascadeTier:
    """Configuration for a cascade tier"""
    name: str
    model_ids: List[str]
    confidence_threshold: float
    cost_multiplier: float


@dataclass
class CascadeResult:
    """Result from cascade routing"""
    model: str
    confidence: float
    tier: str
    escalated: bool
    attempts: int
    total_cost: float
    reasoning: str
    metadata: Dict[str, Any]


class CascadeRouter:
    """
    Multi-tier cascade router with confidence-based escalation

    Tiers:
    - Fast (7B): qwen2.5-coder-7b
    - Medium (30B): qwen3-coder-30b, deepseek-coder-16b
    - Slow (80B): qwen3-next-80b, glm-4-9b

    Flow: Fast → Medium → Slow
    Escalates if confidence < threshold (default 0.8)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.8,
        max_retries: int = 2,
        enable_cost_aware: bool = True
    ):
        """
        Initialize cascade router

        Args:
            confidence_threshold: Minimum confidence to stop escalation
            max_retries: Maximum retry attempts per tier
            enable_cost_aware: Enable CARROT cost tracking
        """
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries
        self.enable_cost_aware = enable_cost_aware

        # Define tiers
        self.tiers = [
            CascadeTier(
                name="fast",
                model_ids=["qwen2.5-coder-7b"],
                confidence_threshold=confidence_threshold,
                cost_multiplier=1.0
            ),
            CascadeTier(
                name="medium",
                model_ids=["qwen3-coder-30b", "deepseek-coder-16b"],
                confidence_threshold=confidence_threshold,
                cost_multiplier=2.0
            ),
            CascadeTier(
                name="slow",
                model_ids=["qwen3-next-80b", "glm-4-9b"],
                confidence_threshold=1.0,  # Always stop at highest tier
                cost_multiplier=4.0
            )
        ]

        # Performance tracking
        self.tier_stats = {
            tier.name: {
                "attempts": 0,
                "successes": 0,
                "escalations": 0,
                "avg_confidence": 0.0
            }
            for tier in self.tiers
        }

        logger.info(
            f"CascadeRouter initialized: threshold={confidence_threshold}, "
            f"max_retries={max_retries}"
        )

    def route(
        self,
        query: str,
        task_type: str,
        complexity: float,
        budget: Optional[float] = None,
        context_size: int = 0
    ) -> CascadeResult:
        """
        Route query through cascade tiers

        Args:
            query: The query to route
            task_type: Task type
            complexity: Query complexity
            budget: Optional budget constraint
            context_size: Context size

        Returns:
            CascadeResult with selected model and metadata
        """
        total_cost = 0.0
        attempts = 0
        escalated = False

        for tier_idx, tier in enumerate(self.tiers):
            # Update stats
            self.tier_stats[tier.name]["attempts"] += 1
            attempts += 1

            # Select best model from tier
            selected_model, confidence = self._select_from_tier(
                tier, query, task_type, complexity, context_size
            )

            # Estimate cost
            tier_cost = self._estimate_tier_cost(
                tier, query, context_size
            )
            total_cost += tier_cost

            # Check if we should stop
            should_stop = confidence >= tier.confidence_threshold

            # Check budget constraint
            if budget and total_cost > budget:
                # Budget exceeded, use current selection
                should_stop = True
                logger.info(
                    f"Budget exceeded ({total_cost:.2f} > {budget:.2f}), "
                    f"stopping at tier {tier.name}"
                )

            # Final tier always stops
            if tier_idx == len(self.tiers) - 1:
                should_stop = True

            if should_stop:
                # Success at this tier
                self.tier_stats[tier.name]["successes"] += 1

                reasoning = (
                    f"Cascade routing: stopped at {tier.name} tier "
                    f"(confidence={confidence:.3f}, threshold={tier.confidence_threshold:.3f})"
                )

                if escalated:
                    reasoning += f", escalated through {attempts - 1} tier(s)"

                return CascadeResult(
                    model=selected_model,
                    confidence=confidence,
                    tier=tier.name,
                    escalated=escalated,
                    attempts=attempts,
                    total_cost=total_cost,
                    reasoning=reasoning,
                    metadata={
                        "task_type": task_type,
                        "complexity": complexity,
                        "budget": budget,
                        "tier_costs": total_cost
                    }
                )

            # Need to escalate
            escalated = True
            self.tier_stats[tier.name]["escalations"] += 1

            logger.debug(
                f"Escalating from {tier.name} tier "
                f"(confidence={confidence:.3f} < threshold={tier.confidence_threshold:.3f})"
            )

        # Fallback: should never reach here due to final tier always stopping
        # But included for safety
        final_tier = self.tiers[-1]
        final_model = final_tier.model_ids[0]

        return CascadeResult(
            model=final_model,
            confidence=0.5,
            tier="fallback",
            escalated=True,
            attempts=attempts,
            total_cost=total_cost,
            reasoning="Cascade fallback: reached end of tiers",
            metadata={"task_type": task_type, "complexity": complexity}
        )

    def _select_from_tier(
        self,
        tier: CascadeTier,
        query: str,
        task_type: str,
        complexity: float,
        context_size: int
    ) -> Tuple[str, float]:
        """
        Select best model from tier

        Args:
            tier: The tier to select from
            query: Query
            task_type: Task type
            complexity: Complexity
            context_size: Context size

        Returns:
            Tuple of (model_id, confidence)
        """
        # Simple selection: pick first available model
        # In production, could use more sophisticated logic

        # Check for language specialization
        if "rust" in query.lower() or "go" in query.lower():
            if "deepseek-coder-16b" in tier.model_ids:
                return "deepseek-coder-16b", 0.9

        # Check for large context
        if context_size > 128_000:
            if "glm-4-9b" in tier.model_ids:
                return "glm-4-9b", 0.95

        # Check complexity match
        if complexity > 0.7:
            # High complexity: prefer larger models
            for model_id in reversed(tier.model_ids):
                config = get_model_config(model_id)
                if config.quality >= 0.85:
                    return model_id, 0.85
        elif complexity < 0.3:
            # Low complexity: prefer faster models
            for model_id in tier.model_ids:
                config = get_model_config(model_id)
                if config.speed > 40:
                    return model_id, 0.90

        # Default: first model in tier with base confidence
        selected = tier.model_ids[0]
        base_confidence = 0.75 + (complexity * 0.1)  # 0.75-0.85 range

        return selected, min(base_confidence, 0.95)

    def _estimate_tier_cost(
        self,
        tier: CascadeTier,
        query: str,
        context_size: int
    ) -> float:
        """
        Estimate cost for using this tier

        Args:
            tier: Tier
            query: Query
            context_size: Context size

        Returns:
            Estimated cost
        """
        # Simple cost model: base cost * tier multiplier
        base_cost = (len(query) + context_size) / 1000.0  # Per 1K chars
        return base_cost * tier.cost_multiplier

    def get_tier_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all tiers"""
        return {
            "tiers": self.tier_stats,
            "total_attempts": sum(
                stats["attempts"] for stats in self.tier_stats.values()
            ),
            "total_escalations": sum(
                stats["escalations"] for stats in self.tier_stats.values()
            ),
            "escalation_rate": self._calculate_escalation_rate()
        }

    def _calculate_escalation_rate(self) -> float:
        """Calculate overall escalation rate"""
        total_attempts = sum(
            stats["attempts"] for stats in self.tier_stats.values()
        )
        total_escalations = sum(
            stats["escalations"] for stats in self.tier_stats.values()
        )

        if total_attempts == 0:
            return 0.0

        return total_escalations / total_attempts

    def reset_stats(self):
        """Reset performance statistics"""
        for tier_name in self.tier_stats:
            self.tier_stats[tier_name] = {
                "attempts": 0,
                "successes": 0,
                "escalations": 0,
                "avg_confidence": 0.0
            }
        logger.info("Cascade router stats reset")
