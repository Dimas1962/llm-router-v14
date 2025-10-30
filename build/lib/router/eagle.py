"""
Eagle ELO System - Phase 2
Training-free model ranking using ELO ratings
Combines global ratings with local context-specific scores
"""

import logging
from typing import Dict, List, Optional, Tuple
import copy

from .models import GLOBAL_ELO, get_all_models
from .memory import AssociativeMemory


logger = logging.getLogger(__name__)


class EagleELO:
    """
    Eagle ELO: Training-free routing using ELO ratings

    Combines:
    - Global ELO: General model performance (from benchmarks)
    - Local ELO: Context-specific performance (from memory)

    Scoring formula: score = α * global + (1-α) * local
    Default: α = 0.7 (70% global, 30% local)
    """

    def __init__(
        self,
        memory: AssociativeMemory,
        global_alpha: float = 0.7,  # Weight for global score
        k_factor: int = 32,  # ELO update factor
        initial_elo: int = 1500  # Base ELO for new models
    ):
        """
        Initialize Eagle ELO system

        Args:
            memory: AssociativeMemory instance for local scoring
            global_alpha: Weight for global ELO (0.0-1.0), default 0.7
            k_factor: ELO K-factor for rating updates, default 32
            initial_elo: Base ELO rating for new models, default 1500
        """
        self.memory = memory
        self.global_alpha = global_alpha
        self.local_alpha = 1.0 - global_alpha
        self.k_factor = k_factor
        self.initial_elo = initial_elo

        # Initialize global ELO from models.py
        self.global_elo = copy.deepcopy(GLOBAL_ELO)

        # Track ELO history for analysis
        self.elo_history: List[Dict[str, int]] = []

        logger.info(
            f"Eagle ELO initialized: α={global_alpha:.2f} "
            f"(global={global_alpha:.0%}, local={self.local_alpha:.0%}), K={k_factor}"
        )

    def get_global_score(self, model_id: str) -> float:
        """
        Get global ELO score for a model

        Args:
            model_id: Model identifier

        Returns:
            Global ELO rating
        """
        return float(self.global_elo.get(model_id, self.initial_elo))

    def get_local_score(
        self,
        query: str,
        model_id: str,
        k: int = 5,
        filter_task_type: Optional[str] = None
    ) -> float:
        """
        Get local ELO score based on similar queries

        Args:
            query: Query to score for
            model_id: Model to score
            k: Number of similar queries to consider
            filter_task_type: Optional task type filter

        Returns:
            Local score (0.0-1.0 normalized)
        """
        # Get local scores from memory
        local_scores = self.memory.get_local_scores(
            query, k=k, filter_task_type=filter_task_type
        )

        if not local_scores:
            # No local data: return neutral score
            return 0.5

        # Get score for this model (0.0 if not in local data)
        model_score = local_scores.get(model_id, 0.0)

        return model_score

    def get_combined_score(
        self,
        query: str,
        model_id: str,
        k: int = 5,
        filter_task_type: Optional[str] = None
    ) -> float:
        """
        Get combined Eagle score (global + local)

        Formula: score = α * global + (1-α) * local
        Where α = global_alpha (default 0.7)

        Args:
            query: Query to score for
            model_id: Model to score
            k: Number of similar queries for local scoring
            filter_task_type: Optional task type filter

        Returns:
            Combined Eagle score
        """
        # Get global score (normalized to 0-1)
        global_score = self.get_global_score(model_id)
        global_normalized = self._normalize_elo(global_score)

        # Get local score (already 0-1)
        local_score = self.get_local_score(query, model_id, k, filter_task_type)

        # Combine
        combined = (self.global_alpha * global_normalized +
                   self.local_alpha * local_score)

        logger.debug(
            f"Eagle score for {model_id}: "
            f"global={global_normalized:.3f}, local={local_score:.3f}, "
            f"combined={combined:.3f}"
        )

        return combined

    def score_all_models(
        self,
        query: str,
        k: int = 5,
        filter_task_type: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Score all available models for a query

        Args:
            query: Query to score for
            k: Number of similar queries for local scoring
            filter_task_type: Optional task type filter

        Returns:
            Dict mapping model_id to Eagle score
        """
        scores = {}

        for model_id in get_all_models():
            scores[model_id] = self.get_combined_score(
                query, model_id, k, filter_task_type
            )

        return scores

    def get_best_model(
        self,
        query: str,
        k: int = 5,
        filter_task_type: Optional[str] = None,
        exclude_models: Optional[List[str]] = None
    ) -> Tuple[str, float]:
        """
        Get best model based on Eagle scoring

        Args:
            query: Query to route
            k: Number of similar queries for local scoring
            filter_task_type: Optional task type filter
            exclude_models: Optional list of models to exclude

        Returns:
            Tuple of (best_model_id, score)
        """
        scores = self.score_all_models(query, k, filter_task_type)

        # Filter excluded models
        if exclude_models:
            scores = {m: s for m, s in scores.items() if m not in exclude_models}

        if not scores:
            raise ValueError("No models available after filtering")

        # Get best model
        best_model = max(scores.items(), key=lambda x: x[1])

        logger.info(
            f"Eagle best model: {best_model[0]} (score={best_model[1]:.3f})"
        )

        return best_model

    def update_rating(
        self,
        winner_model: str,
        loser_model: str,
        actual_result: float = 1.0
    ) -> Tuple[float, float]:
        """
        Update ELO ratings based on comparison outcome

        Uses standard ELO formula:
        new_rating = old_rating + K * (actual - expected)

        Args:
            winner_model: Model that performed better
            loser_model: Model that performed worse
            actual_result: Actual outcome (1.0=winner won, 0.5=tie, 0.0=loser won)

        Returns:
            Tuple of (winner_new_rating, loser_new_rating)
        """
        # Get current ratings
        winner_rating = self.global_elo.get(winner_model, self.initial_elo)
        loser_rating = self.global_elo.get(loser_model, self.initial_elo)

        # Calculate expected scores
        winner_expected = self._expected_score(winner_rating, loser_rating)
        loser_expected = 1.0 - winner_expected

        # Update ratings
        winner_new = winner_rating + self.k_factor * (actual_result - winner_expected)
        loser_new = loser_rating + self.k_factor * ((1.0 - actual_result) - loser_expected)

        # Store new ratings
        self.global_elo[winner_model] = winner_new
        self.global_elo[loser_model] = loser_new

        # Log update
        logger.info(
            f"ELO update: {winner_model} {winner_rating:.0f}→{winner_new:.0f}, "
            f"{loser_model} {loser_rating:.0f}→{loser_new:.0f}"
        )

        # Record history
        self.elo_history.append(copy.deepcopy(self.global_elo))

        return winner_new, loser_new

    def update_from_feedback(
        self,
        model_id: str,
        query: str,
        success: bool,
        task_type: str,
        complexity: float
    ) -> None:
        """
        Update ELO based on task success feedback

        Args:
            model_id: Model that was used
            query: Query that was routed
            success: Whether the routing was successful
            task_type: Type of task
            complexity: Complexity score
        """
        # Add to memory with success feedback
        self.memory.add_task(
            query=query,
            selected_model=model_id,
            task_type=task_type,
            complexity=complexity,
            success=success
        )

        # If we have enough history, update ELO against other models
        if self.memory.size() > 10:
            # Get similar tasks
            similar = self.memory.search_similar(query, k=5)

            # Compare with other models used for similar tasks
            for event in similar:
                if event.selected_model != model_id:
                    # Compare success rates
                    if success and not event.success:
                        # This model won
                        self.update_rating(model_id, event.selected_model, 1.0)
                    elif not success and event.success:
                        # This model lost
                        self.update_rating(event.selected_model, model_id, 1.0)
                    elif success == event.success:
                        # Tie
                        self.update_rating(model_id, event.selected_model, 0.5)

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for model A vs model B

        Formula: E_A = 1 / (1 + 10^((R_B - R_A)/400))

        Args:
            rating_a: ELO rating of model A
            rating_b: ELO rating of model B

        Returns:
            Expected score for model A (0.0-1.0)
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def _normalize_elo(self, elo: float, min_elo: float = 1500, max_elo: float = 2000) -> float:
        """
        Normalize ELO rating to 0-1 range

        Args:
            elo: ELO rating to normalize
            min_elo: Minimum expected ELO
            max_elo: Maximum expected ELO

        Returns:
            Normalized score (0.0-1.0)
        """
        normalized = (elo - min_elo) / (max_elo - min_elo)
        return max(0.0, min(1.0, normalized))  # Clamp to [0, 1]

    def get_rankings(self) -> List[Tuple[str, float]]:
        """
        Get current model rankings sorted by global ELO

        Returns:
            List of (model_id, elo_rating) tuples, sorted descending
        """
        rankings = list(self.global_elo.items())
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def get_stats(self) -> Dict[str, any]:
        """
        Get Eagle ELO statistics

        Returns:
            Dict with statistics
        """
        rankings = self.get_rankings()

        return {
            'global_alpha': self.global_alpha,
            'local_alpha': self.local_alpha,
            'k_factor': self.k_factor,
            'rankings': rankings,
            'top_model': rankings[0] if rankings else None,
            'elo_range': (min(r[1] for r in rankings), max(r[1] for r in rankings)) if rankings else (0, 0),
            'memory_size': self.memory.size(),
            'history_length': len(self.elo_history)
        }

    def reset_to_defaults(self) -> None:
        """Reset global ELO to default values from models.py"""
        self.global_elo = copy.deepcopy(GLOBAL_ELO)
        self.elo_history.clear()
        logger.info("ELO ratings reset to defaults")

    def get_model_confidence(
        self,
        query: str,
        model_id: str,
        k: int = 5
    ) -> float:
        """
        Get confidence in model selection based on Eagle score

        Args:
            query: Query to evaluate
            model_id: Model to get confidence for
            k: Number of similar queries to consider

        Returns:
            Confidence score (0.0-1.0)
        """
        # Get all model scores
        all_scores = self.score_all_models(query, k)

        if not all_scores:
            return 0.5  # Neutral confidence

        # Get this model's score
        model_score = all_scores.get(model_id, 0.0)

        # Get score range
        max_score = max(all_scores.values())
        min_score = min(all_scores.values())

        if max_score == min_score:
            return 0.5  # All models equal

        # Confidence based on how close to max score
        # If this is the best model, confidence = 1.0
        # If this is the worst model, confidence = 0.0
        confidence = (model_score - min_score) / (max_score - min_score)

        return confidence
