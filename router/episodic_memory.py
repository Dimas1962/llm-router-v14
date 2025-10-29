"""
Episodic Memory - Long-term routing episode storage and retrieval
Phase 6: Advanced Features
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None
    SentenceTransformer = None


logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """A routing episode"""
    query: str
    selected_model: str
    confidence: float
    success: bool
    timestamp: datetime
    task_type: str
    complexity: float
    metadata: Dict[str, Any]
    episode_id: int


class EpisodicMemory:
    """
    Long-term episodic memory for routing decisions

    Stores up to 10K episodes with semantic search via FAISS
    Extracts patterns from successful routes
    """

    def __init__(
        self,
        max_episodes: int = 10000,
        embedding_dim: int = 384,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize episodic memory

        Args:
            max_episodes: Maximum number of episodes to retain
            embedding_dim: Dimension of embeddings
            model_name: Sentence transformer model name
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS and sentence-transformers required for episodic memory")

        self.max_episodes = max_episodes
        self.embedding_dim = embedding_dim
        self.episodes: List[Episode] = []
        self.episode_counter = 0

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(embedding_dim)

        # Initialize sentence transformer
        self.encoder = SentenceTransformer(model_name)

        logger.info(
            f"EpisodicMemory initialized: max={max_episodes}, dim={embedding_dim}"
        )

    def add_episode(
        self,
        query: str,
        selected_model: str,
        confidence: float,
        success: bool,
        task_type: str,
        complexity: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a new episode

        Args:
            query: The query
            selected_model: Model that was selected
            confidence: Routing confidence
            success: Whether routing was successful
            task_type: Type of task
            complexity: Task complexity
            metadata: Additional metadata

        Returns:
            Episode ID
        """
        # Create episode
        episode = Episode(
            query=query,
            selected_model=selected_model,
            confidence=confidence,
            success=success,
            timestamp=datetime.now(),
            task_type=task_type,
            complexity=complexity,
            metadata=metadata or {},
            episode_id=self.episode_counter
        )

        # Encode query
        embedding = self.encoder.encode([query])[0]

        # Add to index
        self.index.add(np.array([embedding], dtype=np.float32))

        # Add to episodes list
        self.episodes.append(episode)
        self.episode_counter += 1

        # Maintain max size
        if len(self.episodes) > self.max_episodes:
            # Remove oldest episode
            removed_episode = self.episodes.pop(0)
            logger.debug(f"Removed oldest episode {removed_episode.episode_id}")

        logger.debug(f"Added episode {episode.episode_id}: {query[:50]}...")

        return episode.episode_id

    def search_similar(
        self,
        query: str,
        k: int = 5,
        filter_success: bool = True
    ) -> List[Episode]:
        """
        Search for similar episodes

        Args:
            query: Query to search for
            k: Number of results
            filter_success: Only return successful episodes

        Returns:
            List of similar episodes
        """
        if len(self.episodes) == 0:
            return []

        # Encode query
        query_embedding = self.encoder.encode([query])[0]

        # Search
        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32),
            min(k * 2, len(self.episodes))  # Get more results for filtering
        )

        # Get episodes
        results = []
        for idx in indices[0]:
            if idx < len(self.episodes):
                episode = self.episodes[idx]

                # Filter by success if requested
                if filter_success and not episode.success:
                    continue

                results.append(episode)

                if len(results) >= k:
                    break

        return results

    def get_successful_patterns(
        self,
        task_type: Optional[str] = None,
        min_confidence: float = 0.8
    ) -> Dict[str, Any]:
        """
        Extract patterns from successful episodes

        Args:
            task_type: Filter by task type
            min_confidence: Minimum confidence threshold

        Returns:
            Dict with pattern statistics
        """
        # Filter successful episodes
        successful = [
            ep for ep in self.episodes
            if ep.success and ep.confidence >= min_confidence
        ]

        if task_type:
            successful = [ep for ep in successful if ep.task_type == task_type]

        if not successful:
            return {"count": 0, "patterns": {}}

        # Count model selections
        model_counts = {}
        for ep in successful:
            model_counts[ep.selected_model] = model_counts.get(ep.selected_model, 0) + 1

        # Calculate average metrics
        avg_confidence = sum(ep.confidence for ep in successful) / len(successful)
        avg_complexity = sum(ep.complexity for ep in successful) / len(successful)

        return {
            "count": len(successful),
            "model_distribution": model_counts,
            "avg_confidence": avg_confidence,
            "avg_complexity": avg_complexity,
            "success_rate": 1.0  # By definition
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get episodic memory statistics"""
        if not self.episodes:
            return {
                "total_episodes": 0,
                "success_rate": 0.0,
                "avg_confidence": 0.0
            }

        successful = sum(1 for ep in self.episodes if ep.success)
        total = len(self.episodes)

        return {
            "total_episodes": total,
            "success_rate": successful / total if total > 0 else 0.0,
            "avg_confidence": sum(ep.confidence for ep in self.episodes) / total,
            "max_episodes": self.max_episodes,
            "capacity_used": total / self.max_episodes
        }

    def size(self) -> int:
        """Get number of episodes"""
        return len(self.episodes)

    def clear(self):
        """Clear all episodes"""
        self.episodes.clear()
        self.index.reset()
        self.episode_counter = 0
        logger.info("Episodic memory cleared")
