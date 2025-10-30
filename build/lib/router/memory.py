"""
Associative Memory - Phase 2
Fast retrieval of similar routing events using FAISS + sentence-transformers
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


logger = logging.getLogger(__name__)


@dataclass
class RoutingEvent:
    """Stores a routing decision event"""
    query: str
    query_embedding: np.ndarray
    selected_model: str
    task_type: str
    complexity: float
    success: Optional[bool] = None  # Feedback on routing quality
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: __import__('time').time())


class AssociativeMemory:
    """
    Memory-Augmented routing history with FAISS vector search

    Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings
    FAISS IndexFlatL2 for fast similarity search
    """

    def __init__(
        self,
        embedding_dim: int = 384,  # all-MiniLM-L6-v2 dimension
        model_name: str = "all-MiniLM-L6-v2",
        max_memory_size: int = 100_000
    ):
        """
        Initialize associative memory

        Args:
            embedding_dim: Dimension of embeddings (384 for all-MiniLM-L6-v2)
            model_name: Sentence transformer model name
            max_memory_size: Maximum number of events to store
        """
        self.embedding_dim = embedding_dim
        self.max_memory_size = max_memory_size
        self.memory: List[RoutingEvent] = []

        # Initialize FAISS index
        if faiss is None:
            logger.warning("FAISS not installed. Memory will work in degraded mode.")
            self.index = None
        else:
            self.index = faiss.IndexFlatL2(embedding_dim)
            logger.info(f"FAISS index initialized with dimension {embedding_dim}")

        # Initialize sentence transformer
        if SentenceTransformer is None:
            logger.warning("sentence-transformers not installed. Using random embeddings.")
            self.encoder = None
        else:
            logger.info(f"Loading sentence transformer: {model_name}")
            self.encoder = SentenceTransformer(model_name)
            logger.info("Sentence transformer loaded successfully")

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for text

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as numpy array
        """
        if self.encoder is None:
            # Fallback: random embeddings (for testing without dependencies)
            logger.warning("Using random embeddings (encoder not available)")
            np.random.seed(hash(text) % (2**32))
            return np.random.randn(self.embedding_dim).astype('float32')

        embedding = self.encoder.encode(text, convert_to_numpy=True)
        return embedding.astype('float32')

    def add_task(
        self,
        query: str,
        selected_model: str,
        task_type: str,
        complexity: float,
        success: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a routing event to memory

        Args:
            query: The user query
            selected_model: Model that was selected
            task_type: Type of task
            complexity: Complexity score
            success: Whether routing was successful (optional)
            metadata: Additional metadata
        """
        # Generate embedding
        query_embedding = self.embed(query)

        # Create routing event
        event = RoutingEvent(
            query=query,
            query_embedding=query_embedding,
            selected_model=selected_model,
            task_type=task_type,
            complexity=complexity,
            success=success,
            metadata=metadata or {}
        )

        # Add to memory
        self.memory.append(event)

        # Add to FAISS index
        if self.index is not None:
            self.index.add(query_embedding.reshape(1, -1))

        # Enforce max memory size
        if len(self.memory) > self.max_memory_size:
            # Remove oldest entries
            remove_count = len(self.memory) - self.max_memory_size
            self.memory = self.memory[remove_count:]

            # Rebuild FAISS index
            if self.index is not None:
                self._rebuild_index()

            logger.info(f"Memory size exceeded. Removed {remove_count} oldest entries.")

        logger.debug(
            f"Added routing event: model={selected_model}, "
            f"task={task_type}, complexity={complexity:.2f}"
        )

    def search_similar(
        self,
        query: str,
        k: int = 5,
        filter_task_type: Optional[str] = None
    ) -> List[RoutingEvent]:
        """
        Search for similar queries in memory

        Args:
            query: Query to search for
            k: Number of similar queries to return
            filter_task_type: Optional filter by task type

        Returns:
            List of similar routing events
        """
        if len(self.memory) == 0:
            return []

        # Generate query embedding
        query_embedding = self.embed(query)

        if self.index is None:
            # Fallback: linear search with cosine similarity
            return self._linear_search(query_embedding, k, filter_task_type)

        # FAISS search
        search_k = min(k * 3, len(self.memory))  # Search more for filtering
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), search_k
        )

        # Get events
        similar_events = []
        for idx in indices[0]:
            if idx < len(self.memory):
                event = self.memory[idx]

                # Apply filter if specified
                if filter_task_type is None or event.task_type == filter_task_type:
                    similar_events.append(event)

                # Stop if we have enough
                if len(similar_events) >= k:
                    break

        logger.debug(f"Found {len(similar_events)} similar events for query")
        return similar_events

    def get_local_scores(
        self,
        query: str,
        k: int = 5,
        filter_task_type: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get local ELO scores for models based on similar queries

        Args:
            query: Query to score models for
            k: Number of similar queries to consider
            filter_task_type: Optional filter by task type

        Returns:
            Dict mapping model_id to local score
        """
        similar_events = self.search_similar(query, k, filter_task_type)

        if not similar_events:
            return {}

        # Count model selections
        model_counts = {}
        model_successes = {}

        for event in similar_events:
            model = event.selected_model

            if model not in model_counts:
                model_counts[model] = 0
                model_successes[model] = 0

            model_counts[model] += 1

            # If we have success feedback, use it
            if event.success is not None and event.success:
                model_successes[model] += 1

        # Calculate local scores
        # Score = frequency * success_rate
        local_scores = {}
        for model, count in model_counts.items():
            frequency = count / len(similar_events)

            if event.success is not None:
                success_rate = model_successes[model] / count if count > 0 else 0.5
            else:
                success_rate = 1.0  # Assume success if no feedback

            # Local score: normalized frequency weighted by success
            local_scores[model] = frequency * success_rate

        return local_scores

    def _linear_search(
        self,
        query_embedding: np.ndarray,
        k: int,
        filter_task_type: Optional[str] = None
    ) -> List[RoutingEvent]:
        """
        Fallback linear search using cosine similarity
        """
        similarities = []

        for i, event in enumerate(self.memory):
            # Apply filter
            if filter_task_type is not None and event.task_type != filter_task_type:
                continue

            # Cosine similarity
            sim = np.dot(query_embedding, event.query_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(event.query_embedding)
            )
            similarities.append((sim, i))

        # Sort by similarity (descending)
        similarities.sort(reverse=True, key=lambda x: x[0])

        # Return top k
        top_k_indices = [idx for _, idx in similarities[:k]]
        return [self.memory[idx] for idx in top_k_indices]

    def _rebuild_index(self) -> None:
        """Rebuild FAISS index from scratch"""
        if self.index is None or len(self.memory) == 0:
            return

        logger.info("Rebuilding FAISS index...")
        self.index = faiss.IndexFlatL2(self.embedding_dim)

        # Add all embeddings
        embeddings = np.vstack([event.query_embedding for event in self.memory])
        self.index.add(embeddings)

        logger.info(f"FAISS index rebuilt with {len(self.memory)} entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        if len(self.memory) == 0:
            return {
                'size': 0,
                'model_distribution': {},
                'task_distribution': {},
                'avg_complexity': 0.0
            }

        # Model distribution
        model_counts = {}
        for event in self.memory:
            model_counts[event.selected_model] = model_counts.get(event.selected_model, 0) + 1

        # Task distribution
        task_counts = {}
        for event in self.memory:
            task_counts[event.task_type] = task_counts.get(event.task_type, 0) + 1

        # Average complexity
        avg_complexity = sum(event.complexity for event in self.memory) / len(self.memory)

        return {
            'size': len(self.memory),
            'model_distribution': model_counts,
            'task_distribution': task_counts,
            'avg_complexity': avg_complexity,
            'max_size': self.max_memory_size
        }

    def clear(self) -> None:
        """Clear all memory"""
        self.memory.clear()
        if self.index is not None:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        logger.info("Memory cleared")

    def size(self) -> int:
        """Get current memory size"""
        return len(self.memory)
