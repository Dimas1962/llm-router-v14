"""
Phase 4: Advanced Context Management
Dynamic context sizing, progressive building, decay monitoring, and compression
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

from .models import MODELS, get_model_config


logger = logging.getLogger(__name__)


@dataclass
class ContextAnalysis:
    """Result of context analysis"""
    total_size: int  # Total context size in characters
    required_window: int  # Estimated required context window
    complexity: float  # Query complexity (0.0-1.0)
    decay_risk: float  # Context decay risk (0.0-1.0)
    truncation_needed: bool  # Whether truncation is needed
    compression_recommended: bool  # Whether compression is recommended
    suggested_models: List[str]  # Models suitable for this context
    metadata: Dict  # Additional metadata


class DynamicContextSizer:
    """
    Dynamically adjusts context window based on task requirements

    Strategies:
    - Simple tasks: 8K tokens
    - Medium complexity: 32K tokens
    - High complexity: 128K tokens
    - Very large context: 200K+ tokens
    """

    def __init__(self):
        """Initialize dynamic context sizer"""
        logger.info("DynamicContextSizer initialized")

    def estimate_required_window(
        self,
        query: str,
        complexity: float,
        session_history: Optional[List[str]] = None
    ) -> int:
        """
        Estimate required context window size

        Args:
            query: The user query
            complexity: Query complexity (0.0-1.0)
            session_history: Session history

        Returns:
            Required context window size in tokens
        """
        # Base requirement from complexity
        if complexity < 0.3:
            base_window = 8_000  # Simple tasks
        elif complexity < 0.7:
            base_window = 32_000  # Medium tasks
        else:
            base_window = 128_000  # Complex tasks

        # Adjust based on session history
        if session_history:
            history_size = sum(len(msg) for msg in session_history)
            # ~4 chars per token
            history_tokens = history_size // 4

            # Add buffer for history
            if history_tokens > 10_000:
                base_window = max(base_window, history_tokens * 2)

        # Check for large context keywords
        large_context_keywords = [
            'entire', 'all files', 'whole project', 'complete codebase',
            'refactor everything', 'analyze all'
        ]

        if any(keyword in query.lower() for keyword in large_context_keywords):
            base_window = max(base_window, 200_000)

        logger.debug(
            f"Estimated required window: {base_window:,} tokens "
            f"(complexity={complexity:.2f})"
        )

        return base_window

    def recommend_models(
        self,
        required_window: int,
        complexity: float,
        query: str
    ) -> List[str]:
        """
        Recommend models based on context requirements

        Args:
            required_window: Required context window
            complexity: Query complexity
            query: The user query

        Returns:
            List of suitable model IDs
        """
        suitable = []

        for model_id, config in MODELS.items():
            # Must have sufficient context window
            if config.context >= required_window:
                suitable.append(model_id)

        # Sort by quality (prefer better models)
        suitable.sort(key=lambda m: get_model_config(m).quality, reverse=True)

        logger.debug(
            f"Recommended models for {required_window:,} token window: {suitable}"
        )

        return suitable

    def calculate_truncation_point(
        self,
        session_history: List[str],
        max_window: int,
        query_size: int
    ) -> int:
        """
        Calculate how many history messages to keep

        Args:
            session_history: Session history
            max_window: Maximum context window
            query_size: Size of current query

        Returns:
            Number of messages to keep from history
        """
        # Reserve space for query and response
        available = (max_window * 4) - query_size - 2000  # ~500 tokens buffer

        if available <= 0:
            return 0

        # Count from most recent
        cumulative = 0
        keep_count = 0

        for i in range(len(session_history) - 1, -1, -1):
            msg_size = len(session_history[i])
            if cumulative + msg_size <= available:
                cumulative += msg_size
                keep_count += 1
            else:
                break

        return keep_count


class ProgressiveContextBuilder:
    """
    Builds context incrementally for large sessions

    Strategies:
    - Recent messages: Full inclusion
    - Mid-range messages: Summary inclusion
    - Old messages: Key points only
    """

    def __init__(
        self,
        recent_window: int = 5,
        mid_window: int = 10,
        summary_compression: float = 0.3
    ):
        """
        Initialize progressive context builder

        Args:
            recent_window: Number of recent messages to keep full
            mid_window: Number of mid-range messages to keep
            summary_compression: Compression ratio for summaries
        """
        self.recent_window = recent_window
        self.mid_window = mid_window
        self.summary_compression = summary_compression

        logger.info(
            f"ProgressiveContextBuilder initialized: "
            f"recent={recent_window}, mid={mid_window}, "
            f"compression={summary_compression}"
        )

    def build_progressive_context(
        self,
        session_history: List[str],
        max_size: int
    ) -> Tuple[List[str], Dict[str, int]]:
        """
        Build progressive context with intelligent truncation

        Args:
            session_history: Full session history
            max_size: Maximum context size in characters

        Returns:
            Tuple of (processed_history, stats)
        """
        if not session_history:
            return [], {'original': 0, 'processed': 0, 'compression': 0.0}

        total = len(session_history)

        # Recent messages: keep full
        recent = session_history[-self.recent_window:]

        # Mid-range messages: keep but may summarize
        mid_start = max(0, total - self.recent_window - self.mid_window)
        mid_end = total - self.recent_window
        mid_range = session_history[mid_start:mid_end] if mid_end > mid_start else []

        # Old messages: summarize or drop
        old = session_history[:mid_start] if mid_start > 0 else []

        # Build context
        processed = []
        current_size = 0

        # Add old messages (summarized)
        if old:
            summary = self._summarize_messages(old)
            if len(summary) + current_size < max_size:
                processed.append(f"[Summary of {len(old)} messages: {summary}]")
                current_size += len(summary)

        # Add mid-range messages
        for msg in mid_range:
            if current_size + len(msg) < max_size * 0.7:  # Reserve 30% for recent
                processed.append(msg)
                current_size += len(msg)

        # Add recent messages (highest priority)
        for msg in recent:
            if current_size + len(msg) < max_size:
                processed.append(msg)
                current_size += len(msg)

        # Calculate stats
        original_size = sum(len(msg) for msg in session_history)
        processed_size = sum(len(msg) for msg in processed)
        compression = 1.0 - (processed_size / original_size) if original_size > 0 else 0.0

        stats = {
            'original': original_size,
            'processed': processed_size,
            'compression': compression,
            'messages_kept': len(processed),
            'messages_dropped': total - len(processed)
        }

        logger.debug(
            f"Progressive context built: {total} → {len(processed)} messages, "
            f"compression={compression:.1%}"
        )

        return processed, stats

    def _summarize_messages(self, messages: List[str]) -> str:
        """
        Create summary of messages

        Args:
            messages: Messages to summarize

        Returns:
            Summary text
        """
        # Simple summarization: extract key phrases
        # In production, could use LLM for better summaries

        total_length = sum(len(msg) for msg in messages)
        target_length = int(total_length * self.summary_compression)

        # For now, just truncate combined text
        combined = " ".join(messages)

        if len(combined) > target_length:
            summary = combined[:target_length] + "..."
        else:
            summary = combined

        return summary


class DecayMonitor:
    """
    Enhanced decay risk detection and prevention

    Monitors:
    - Context size vs model capacity
    - Long-range dependency tracking
    - Attention decay patterns
    """

    def __init__(self):
        """Initialize decay monitor"""
        self.decay_thresholds = {
            'low': 32_000,      # <32K chars: low risk
            'medium': 64_000,   # 32-64K: medium risk
            'high': 128_000,    # 64-128K: high risk
            'critical': 200_000  # >128K: critical risk
        }

        logger.info("DecayMonitor initialized")

    def estimate_decay_risk(
        self,
        context_size: int,
        model_id: Optional[str] = None
    ) -> float:
        """
        Estimate context decay risk

        Args:
            context_size: Total context size in characters
            model_id: Optional model ID to check against

        Returns:
            Decay risk score (0.0-1.0)
        """
        # Character to token approximation (~4 chars per token)
        token_estimate = context_size // 4

        # Base risk from absolute size
        if token_estimate < self.decay_thresholds['low'] // 4:
            base_risk = 0.0
        elif token_estimate < self.decay_thresholds['medium'] // 4:
            base_risk = 0.3
        elif token_estimate < self.decay_thresholds['high'] // 4:
            base_risk = 0.5
        elif token_estimate < self.decay_thresholds['critical'] // 4:
            base_risk = 0.7
        else:
            base_risk = 0.9

        # Adjust for specific model capacity
        if model_id:
            config = get_model_config(model_id)
            utilization = token_estimate / config.context

            # Risk increases as we approach capacity
            if utilization > 0.9:
                base_risk = max(base_risk, 0.9)
            elif utilization > 0.7:
                base_risk = max(base_risk, 0.7)
            elif utilization > 0.5:
                base_risk = max(base_risk, 0.5)

        return min(1.0, base_risk)

    def detect_decay_patterns(
        self,
        session_history: List[str]
    ) -> Dict[str, any]:
        """
        Detect patterns indicating context decay

        Args:
            session_history: Session history

        Returns:
            Dict with decay pattern analysis
        """
        if not session_history:
            return {
                'has_decay': False,
                'patterns': [],
                'severity': 0.0
            }

        patterns = []
        severity = 0.0

        # Check for repetition (sign of context loss)
        recent = session_history[-5:] if len(session_history) >= 5 else session_history
        unique_recent = set(recent)

        if len(unique_recent) < len(recent) * 0.5:
            patterns.append('repetition')
            severity += 0.3

        # Check for very long session
        if len(session_history) > 50:
            patterns.append('long_session')
            severity += 0.2

        # Check for large total size
        total_size = sum(len(msg) for msg in session_history)
        if total_size > 100_000:
            patterns.append('large_context')
            severity += 0.3

        has_decay = len(patterns) > 0
        severity = min(1.0, severity)

        return {
            'has_decay': has_decay,
            'patterns': patterns,
            'severity': severity,
            'message_count': len(session_history),
            'total_size': total_size
        }

    def recommend_mitigation(
        self,
        decay_risk: float,
        context_size: int
    ) -> Dict[str, any]:
        """
        Recommend decay mitigation strategies

        Args:
            decay_risk: Current decay risk
            context_size: Context size

        Returns:
            Dict with mitigation recommendations
        """
        recommendations = []

        if decay_risk > 0.7:
            recommendations.append({
                'action': 'compress_context',
                'priority': 'high',
                'description': 'Compress or summarize context immediately'
            })
            recommendations.append({
                'action': 'switch_model',
                'priority': 'high',
                'description': 'Switch to model with larger context window'
            })
        elif decay_risk > 0.5:
            recommendations.append({
                'action': 'progressive_build',
                'priority': 'medium',
                'description': 'Use progressive context building'
            })
        elif decay_risk > 0.3:
            recommendations.append({
                'action': 'monitor',
                'priority': 'low',
                'description': 'Continue monitoring, no action needed yet'
            })

        return {
            'decay_risk': decay_risk,
            'context_size': context_size,
            'recommendations': recommendations
        }


class ContextCompressor:
    """
    Compress/summarize large contexts efficiently

    Strategies:
    - Sliding window: Keep recent + summary of old
    - Key extraction: Extract important information
    - Semantic compression: Remove redundancy
    """

    def __init__(
        self,
        compression_ratio: float = 0.5,
        min_keep_messages: int = 5
    ):
        """
        Initialize context compressor

        Args:
            compression_ratio: Target compression ratio
            min_keep_messages: Minimum messages to keep full
        """
        self.compression_ratio = compression_ratio
        self.min_keep_messages = min_keep_messages

        logger.info(
            f"ContextCompressor initialized: ratio={compression_ratio}, "
            f"min_keep={min_keep_messages}"
        )

    def compress(
        self,
        session_history: List[str],
        target_size: Optional[int] = None
    ) -> Tuple[List[str], Dict[str, any]]:
        """
        Compress session history

        Args:
            session_history: Full session history
            target_size: Optional target size in characters

        Returns:
            Tuple of (compressed_history, compression_stats)
        """
        if not session_history:
            return [], {'original': 0, 'compressed': 0, 'ratio': 0.0}

        original_size = sum(len(msg) for msg in session_history)

        # Calculate target
        if target_size is None:
            target_size = int(original_size * self.compression_ratio)

        # Keep recent messages
        keep_count = max(self.min_keep_messages, len(session_history) // 4)
        recent = session_history[-keep_count:]
        old = session_history[:-keep_count] if len(session_history) > keep_count else []

        compressed = []
        current_size = 0

        # Compress old messages
        if old:
            summary = self._compress_messages(old, target_size // 3)
            compressed.append(f"[Compressed context: {summary}]")
            current_size += len(summary)

        # Add recent messages
        for msg in recent:
            compressed.append(msg)
            current_size += len(msg)

        compressed_size = sum(len(msg) for msg in compressed)
        ratio = compressed_size / original_size if original_size > 0 else 0.0

        stats = {
            'original': original_size,
            'compressed': compressed_size,
            'ratio': ratio,
            'messages_original': len(session_history),
            'messages_compressed': len(compressed),
            'reduction': 1.0 - ratio
        }

        logger.info(
            f"Context compressed: {len(session_history)} → {len(compressed)} messages, "
            f"size reduction: {(1.0 - ratio):.1%}"
        )

        return compressed, stats

    def _compress_messages(
        self,
        messages: List[str],
        target_length: int
    ) -> str:
        """
        Compress multiple messages into summary

        Args:
            messages: Messages to compress
            target_length: Target summary length

        Returns:
            Compressed text
        """
        # Simple compression: extract key phrases and truncate
        # In production, use LLM-based summarization

        combined = " ".join(messages)

        # Extract sentences with key programming terms
        key_terms = [
            'function', 'class', 'method', 'error', 'fix', 'implement',
            'create', 'update', 'refactor', 'optimize', 'test'
        ]

        sentences = combined.split('.')
        important = []

        for sentence in sentences:
            if any(term in sentence.lower() for term in key_terms):
                important.append(sentence.strip())

        if important:
            compressed = '. '.join(important[:10])  # Keep top 10 important sentences
        else:
            # Fallback: just truncate
            compressed = combined[:target_length]

        if len(compressed) > target_length:
            compressed = compressed[:target_length] + "..."

        return compressed

    def estimate_compression_savings(
        self,
        session_history: List[str]
    ) -> Dict[str, any]:
        """
        Estimate potential savings from compression

        Args:
            session_history: Session history

        Returns:
            Dict with savings estimates
        """
        if not session_history:
            return {
                'potential_savings': 0,
                'worth_compressing': False
            }

        original_size = sum(len(msg) for msg in session_history)
        estimated_compressed = int(original_size * self.compression_ratio)
        savings = original_size - estimated_compressed

        # Worth compressing if we save >20% and >10K chars
        worth_it = (savings > original_size * 0.2) and (savings > 10_000)

        return {
            'original_size': original_size,
            'estimated_compressed': estimated_compressed,
            'potential_savings': savings,
            'savings_percent': savings / original_size if original_size > 0 else 0.0,
            'worth_compressing': worth_it
        }


class ContextManager:
    """
    Integrated context management system

    Combines all context management strategies:
    - Dynamic sizing
    - Progressive building
    - Decay monitoring
    - Compression
    """

    def __init__(
        self,
        enable_progressive: bool = True,
        enable_compression: bool = True,
        enable_decay_monitor: bool = True
    ):
        """
        Initialize context manager

        Args:
            enable_progressive: Enable progressive context building
            enable_compression: Enable context compression
            enable_decay_monitor: Enable decay monitoring
        """
        self.sizer = DynamicContextSizer()
        self.builder = ProgressiveContextBuilder() if enable_progressive else None
        self.monitor = DecayMonitor() if enable_decay_monitor else None
        self.compressor = ContextCompressor() if enable_compression else None

        self.enable_progressive = enable_progressive
        self.enable_compression = enable_compression
        self.enable_decay_monitor = enable_decay_monitor

        logger.info(
            f"ContextManager initialized: progressive={enable_progressive}, "
            f"compression={enable_compression}, decay_monitor={enable_decay_monitor}"
        )

    def analyze_context(
        self,
        query: str,
        session_history: Optional[List[str]] = None,
        complexity: float = 0.5,
        model_id: Optional[str] = None
    ) -> ContextAnalysis:
        """
        Comprehensive context analysis

        Args:
            query: The user query
            session_history: Session history
            complexity: Query complexity
            model_id: Optional model to analyze for

        Returns:
            ContextAnalysis with all information
        """
        session_history = session_history or []

        # Calculate total size
        total_size = len(query) + sum(len(msg) for msg in session_history)

        # Estimate required window
        required_window = self.sizer.estimate_required_window(
            query, complexity, session_history
        )

        # Estimate decay risk
        if self.enable_decay_monitor:
            decay_risk = self.monitor.estimate_decay_risk(total_size, model_id)
        else:
            decay_risk = 0.0

        # Check if truncation needed
        token_estimate = total_size // 4
        truncation_needed = token_estimate > required_window

        # Check if compression recommended
        compression_recommended = False
        if self.enable_compression and session_history:
            savings = self.compressor.estimate_compression_savings(session_history)
            compression_recommended = savings['worth_compressing']

        # Get suitable models
        suggested_models = self.sizer.recommend_models(
            required_window, complexity, query
        )

        # Metadata
        metadata = {
            'token_estimate': token_estimate,
            'message_count': len(session_history),
            'query_length': len(query)
        }

        return ContextAnalysis(
            total_size=total_size,
            required_window=required_window,
            complexity=complexity,
            decay_risk=decay_risk,
            truncation_needed=truncation_needed,
            compression_recommended=compression_recommended,
            suggested_models=suggested_models,
            metadata=metadata
        )

    def optimize_context(
        self,
        query: str,
        session_history: List[str],
        target_model: str,
        complexity: float = 0.5
    ) -> Tuple[List[str], Dict[str, any]]:
        """
        Optimize context for target model

        Args:
            query: The user query
            session_history: Session history
            target_model: Target model ID
            complexity: Query complexity

        Returns:
            Tuple of (optimized_history, optimization_stats)
        """
        if not session_history:
            return [], {'optimized': False, 'reason': 'no_history'}

        config = get_model_config(target_model)
        max_tokens = config.context
        max_chars = max_tokens * 4

        # Calculate available space
        available = max_chars - len(query) - 2000  # Buffer for response

        total_size = sum(len(msg) for msg in session_history)

        stats = {
            'original_size': total_size,
            'available_space': available,
            'optimized': False,
            'strategy': None
        }

        # No optimization needed
        if total_size <= available:
            stats['optimized'] = False
            stats['strategy'] = 'none_needed'
            return session_history, stats

        # Apply optimization
        optimized = session_history

        # Try progressive building first
        if self.enable_progressive and self.builder:
            optimized, build_stats = self.builder.build_progressive_context(
                session_history, available
            )

            if build_stats['processed'] <= available:
                stats['optimized'] = True
                stats['strategy'] = 'progressive'
                stats['compression_ratio'] = build_stats['compression']
                return optimized, stats

        # Try compression
        if self.enable_compression and self.compressor:
            optimized, comp_stats = self.compressor.compress(
                session_history, available
            )

            stats['optimized'] = True
            stats['strategy'] = 'compression'
            stats['compression_ratio'] = comp_stats['reduction']
            return optimized, stats

        # Fallback: simple truncation
        keep_count = self.sizer.calculate_truncation_point(
            session_history, max_tokens, len(query)
        )

        optimized = session_history[-keep_count:] if keep_count > 0 else []
        stats['optimized'] = True
        stats['strategy'] = 'truncation'
        stats['messages_kept'] = keep_count

        return optimized, stats
