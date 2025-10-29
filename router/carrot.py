"""
CARROT: Cost-Aware Routing with Routing Optimization Techniques
Phase 3: Budget-aware routing with dual prediction (quality + cost)
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .models import MODELS, get_model_config, get_all_models


logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Prediction result for a model"""
    model_id: str
    quality: float  # 0.0 - 1.0
    cost: float  # Estimated cost (arbitrary units)
    time_estimate: float  # Estimated time in seconds
    tokens_estimate: int  # Estimated tokens
    confidence: float = 0.8  # Prediction confidence


class QualityPredictor:
    """
    Predicts quality of model response for a given query

    Quality factors:
    - Model's base quality rating
    - Task type match with model use cases
    - Complexity match with model capability
    - Context window requirements
    """

    def __init__(self):
        """Initialize quality predictor"""
        logger.info("Quality predictor initialized")

    def predict(
        self,
        query: str,
        model_id: str,
        task_type: Optional[str] = None,
        complexity: float = 0.5,
        context_size: int = 0
    ) -> float:
        """
        Predict quality score for model on query

        Args:
            query: The user query
            model_id: Model to predict for
            task_type: Optional task type
            complexity: Query complexity (0.0-1.0)
            context_size: Context size in tokens

        Returns:
            Quality score (0.0-1.0)
        """
        config = get_model_config(model_id)

        # Start with model's base quality
        quality = config.quality

        # Bonus for task type match
        if task_type and task_type in config.use_cases:
            quality += 0.05

        # Bonus/penalty for complexity match
        # High-quality models better for complex tasks
        complexity_match = abs(config.quality - complexity)
        if complexity_match < 0.2:
            quality += 0.05
        elif complexity_match > 0.5:
            quality -= 0.10

        # Penalty if context too large for model
        if context_size > config.context:
            quality -= 0.20

        # Bonus for deep-seek on Rust/Go
        if hasattr(config, 'quality_rust') and config.quality_rust:
            # Check if query mentions specialized languages
            if any(lang in query.lower() for lang in ['rust', 'go', 'kotlin']):
                quality = config.quality_rust

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, quality))

    def predict_batch(
        self,
        query: str,
        model_ids: List[str],
        task_type: Optional[str] = None,
        complexity: float = 0.5,
        context_size: int = 0
    ) -> Dict[str, float]:
        """
        Predict quality for multiple models

        Args:
            query: The user query
            model_ids: List of model IDs
            task_type: Optional task type
            complexity: Query complexity
            context_size: Context size

        Returns:
            Dict mapping model_id to quality score
        """
        return {
            model_id: self.predict(query, model_id, task_type, complexity, context_size)
            for model_id in model_ids
        }


class CostPredictor:
    """
    Predicts cost of using a model for a query

    Cost factors:
    - Token count (input + estimated output)
    - Time cost (based on model speed)
    - Model size (larger = more expensive)
    - Context window usage
    """

    def __init__(self, time_weight: float = 0.5, token_weight: float = 0.5):
        """
        Initialize cost predictor

        Args:
            time_weight: Weight for time cost (0.0-1.0)
            token_weight: Weight for token cost (0.0-1.0)
        """
        self.time_weight = time_weight
        self.token_weight = token_weight
        logger.info(
            f"Cost predictor initialized: time_weight={time_weight}, "
            f"token_weight={token_weight}"
        )

    def estimate_tokens(
        self,
        query: str,
        context_size: int = 0,
        output_multiplier: float = 1.5
    ) -> Tuple[int, int, int]:
        """
        Estimate token counts

        Args:
            query: The user query
            context_size: Context size
            output_multiplier: Output tokens = input * multiplier

        Returns:
            Tuple of (input_tokens, output_tokens, total_tokens)
        """
        # Simple estimation: ~4 chars per token
        input_tokens = (len(query) + context_size) // 4

        # Estimate output tokens (usually similar to input or less)
        output_tokens = int(input_tokens * output_multiplier)

        total_tokens = input_tokens + output_tokens

        return input_tokens, output_tokens, total_tokens

    def estimate_time(
        self,
        model_id: str,
        total_tokens: int
    ) -> float:
        """
        Estimate time in seconds

        Args:
            model_id: Model to estimate for
            total_tokens: Total tokens to process

        Returns:
            Estimated time in seconds
        """
        config = get_model_config(model_id)

        # Time = tokens / speed (tokens per second)
        time_seconds = total_tokens / config.speed

        return time_seconds

    def predict(
        self,
        query: str,
        model_id: str,
        context_size: int = 0,
        output_multiplier: float = 1.5
    ) -> float:
        """
        Predict cost for model on query

        Args:
            query: The user query
            model_id: Model to predict for
            context_size: Context size in tokens
            output_multiplier: Output size multiplier

        Returns:
            Cost score (arbitrary units, higher = more expensive)
        """
        config = get_model_config(model_id)

        # Estimate tokens
        _, _, total_tokens = self.estimate_tokens(query, context_size, output_multiplier)

        # Estimate time
        time_seconds = self.estimate_time(model_id, total_tokens)

        # Token cost (normalized by context window)
        # Larger context windows are more "expensive"
        token_cost = total_tokens * (config.context / 100_000)

        # Time cost (seconds)
        time_cost = time_seconds

        # Combined cost
        cost = (self.token_weight * token_cost +
                self.time_weight * time_cost)

        return cost

    def predict_batch(
        self,
        query: str,
        model_ids: List[str],
        context_size: int = 0,
        output_multiplier: float = 1.5
    ) -> Dict[str, float]:
        """
        Predict cost for multiple models

        Args:
            query: The user query
            model_ids: List of model IDs
            context_size: Context size
            output_multiplier: Output size multiplier

        Returns:
            Dict mapping model_id to cost
        """
        return {
            model_id: self.predict(query, model_id, context_size, output_multiplier)
            for model_id in model_ids
        }

    def get_detailed_prediction(
        self,
        query: str,
        model_id: str,
        context_size: int = 0,
        output_multiplier: float = 1.5
    ) -> Prediction:
        """
        Get detailed prediction with all components

        Args:
            query: The user query
            model_id: Model to predict for
            context_size: Context size
            output_multiplier: Output size multiplier

        Returns:
            Prediction object with all details
        """
        # Estimate tokens
        _, _, total_tokens = self.estimate_tokens(query, context_size, output_multiplier)

        # Estimate time
        time_seconds = self.estimate_time(model_id, total_tokens)

        # Calculate cost
        cost = self.predict(query, model_id, context_size, output_multiplier)

        # Quality (placeholder - would use QualityPredictor in practice)
        quality = get_model_config(model_id).quality

        return Prediction(
            model_id=model_id,
            quality=quality,
            cost=cost,
            time_estimate=time_seconds,
            tokens_estimate=total_tokens
        )


class CARROT:
    """
    CARROT: Cost-Aware Routing with Routing Optimization Techniques

    Combines quality and cost predictions for budget-aware model selection
    """

    def __init__(
        self,
        quality_predictor: Optional[QualityPredictor] = None,
        cost_predictor: Optional[CostPredictor] = None
    ):
        """
        Initialize CARROT system

        Args:
            quality_predictor: Quality predictor (creates default if None)
            cost_predictor: Cost predictor (creates default if None)
        """
        self.quality_predictor = quality_predictor or QualityPredictor()
        self.cost_predictor = cost_predictor or CostPredictor()

        logger.info("CARROT system initialized")

    def predict_all(
        self,
        query: str,
        model_ids: Optional[List[str]] = None,
        task_type: Optional[str] = None,
        complexity: float = 0.5,
        context_size: int = 0
    ) -> Dict[str, Dict[str, float]]:
        """
        Predict quality and cost for all models

        Args:
            query: The user query
            model_ids: List of model IDs (all if None)
            task_type: Optional task type
            complexity: Query complexity
            context_size: Context size

        Returns:
            Dict mapping model_id to {'quality': float, 'cost': float}
        """
        if model_ids is None:
            model_ids = get_all_models()

        predictions = {}

        for model_id in model_ids:
            quality = self.quality_predictor.predict(
                query, model_id, task_type, complexity, context_size
            )
            cost = self.cost_predictor.predict(query, model_id, context_size)

            predictions[model_id] = {
                'quality': quality,
                'cost': cost
            }

        return predictions

    def select(
        self,
        query: str,
        budget: Optional[float] = None,
        model_ids: Optional[List[str]] = None,
        task_type: Optional[str] = None,
        complexity: float = 0.5,
        context_size: int = 0
    ) -> Tuple[str, Dict[str, float]]:
        """
        Select best model based on CARROT algorithm

        Args:
            query: The user query
            budget: Optional budget constraint
            model_ids: Candidate models (all if None)
            task_type: Optional task type
            complexity: Query complexity
            context_size: Context size

        Returns:
            Tuple of (selected_model_id, prediction_dict)
        """
        # Get predictions for all candidates
        predictions = self.predict_all(
            query, model_ids, task_type, complexity, context_size
        )

        if not predictions:
            raise ValueError("No candidate models available")

        # CARROT selection logic
        if budget is not None:
            # Filter by budget
            valid_models = {
                model_id: pred
                for model_id, pred in predictions.items()
                if pred['cost'] <= budget
            }

            if valid_models:
                # Select best quality within budget
                best_model = max(
                    valid_models.items(),
                    key=lambda x: x[1]['quality']
                )
                selected = best_model[0]

                logger.info(
                    f"CARROT selected {selected} within budget {budget:.2f}: "
                    f"quality={best_model[1]['quality']:.3f}, "
                    f"cost={best_model[1]['cost']:.3f}"
                )
            else:
                # Budget exceeded: select cheapest option
                cheapest = min(predictions.items(), key=lambda x: x[1]['cost'])
                selected = cheapest[0]

                logger.warning(
                    f"CARROT budget exceeded ({budget:.2f}), "
                    f"selecting cheapest: {selected} (cost={cheapest[1]['cost']:.3f})"
                )
        else:
            # No budget: select best quality
            best_model = max(predictions.items(), key=lambda x: x[1]['quality'])
            selected = best_model[0]

            logger.info(
                f"CARROT selected {selected} (no budget): "
                f"quality={best_model[1]['quality']:.3f}, "
                f"cost={best_model[1]['cost']:.3f}"
            )

        return selected, predictions[selected]

    def get_pareto_frontier(
        self,
        query: str,
        model_ids: Optional[List[str]] = None,
        task_type: Optional[str] = None,
        complexity: float = 0.5,
        context_size: int = 0
    ) -> List[Tuple[str, float, float]]:
        """
        Get Pareto frontier of models (quality vs cost trade-off)

        Models on the Pareto frontier are non-dominated:
        no other model is both cheaper AND higher quality

        Args:
            query: The user query
            model_ids: Candidate models
            task_type: Optional task type
            complexity: Query complexity
            context_size: Context size

        Returns:
            List of (model_id, quality, cost) on Pareto frontier,
            sorted by quality (descending)
        """
        predictions = self.predict_all(
            query, model_ids, task_type, complexity, context_size
        )

        # Convert to list of (model_id, quality, cost)
        models = [
            (model_id, pred['quality'], pred['cost'])
            for model_id, pred in predictions.items()
        ]

        # Find Pareto frontier
        pareto = []

        for model_id, quality, cost in models:
            # Check if this model is dominated by any other
            is_dominated = False

            for other_id, other_quality, other_cost in models:
                if other_id == model_id:
                    continue

                # Other model dominates if it has better/equal quality AND lower/equal cost
                # (with at least one strict improvement)
                if (other_quality >= quality and other_cost <= cost and
                        (other_quality > quality or other_cost < cost)):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto.append((model_id, quality, cost))

        # Sort by quality (descending)
        pareto.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Pareto frontier has {len(pareto)} models: {[m[0] for m in pareto]}")

        return pareto

    def recommend_budget(
        self,
        query: str,
        quality_target: float = 0.8,
        model_ids: Optional[List[str]] = None,
        task_type: Optional[str] = None,
        complexity: float = 0.5,
        context_size: int = 0
    ) -> Optional[float]:
        """
        Recommend minimum budget to achieve target quality

        Args:
            query: The user query
            quality_target: Target quality level
            model_ids: Candidate models
            task_type: Optional task type
            complexity: Query complexity
            context_size: Context size

        Returns:
            Recommended budget, or None if target unreachable
        """
        predictions = self.predict_all(
            query, model_ids, task_type, complexity, context_size
        )

        # Find models that meet quality target
        valid_models = [
            (model_id, pred['cost'])
            for model_id, pred in predictions.items()
            if pred['quality'] >= quality_target
        ]

        if not valid_models:
            logger.warning(f"No models meet quality target {quality_target}")
            return None

        # Return cost of cheapest model that meets target
        recommended = min(valid_models, key=lambda x: x[1])

        logger.info(
            f"Recommended budget for quality {quality_target}: "
            f"{recommended[1]:.2f} (model: {recommended[0]})"
        )

        return recommended[1]
