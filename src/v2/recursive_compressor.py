"""
Recursive Compressor - Component 8
Multi-level text compression with quality preservation
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from collections import Counter


class CompressionStrategy(Enum):
    """Compression strategies"""
    EXTRACTIVE = "extractive"    # Extract key sentences
    ABSTRACTIVE = "abstractive"  # Generate summaries
    HYBRID = "hybrid"            # Combine both


class CompressionLevel(Enum):
    """Compression granularity levels"""
    WORD = "word"              # Word-level compression
    SENTENCE = "sentence"      # Sentence-level compression
    PARAGRAPH = "paragraph"    # Paragraph-level compression


@dataclass
class CompressionResult:
    """Result of compression operation"""
    original_text: str
    compressed_text: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    quality_score: float
    strategy_used: CompressionStrategy
    iterations: int


class RecursiveCompressor:
    """
    Recursive Text Compressor

    Features:
    - Recursive compression to target size
    - Multiple strategies (extractive, abstractive, hybrid)
    - Multi-level compression (word, sentence, paragraph)
    - Preservation ratio for important content
    - Quality metrics
    - Iterative compression with quality checks
    """

    def __init__(
        self,
        target_ratio: float = 0.5,  # Target compression ratio
        preservation_ratio: float = 0.8,  # Preserve 80% quality
        min_quality_score: float = 0.6,
        max_iterations: int = 5
    ):
        """
        Initialize Recursive Compressor

        Args:
            target_ratio: Target compression ratio (0-1)
            preservation_ratio: Quality preservation ratio
            min_quality_score: Minimum acceptable quality
            max_iterations: Maximum compression iterations
        """
        self.target_ratio = target_ratio
        self.preservation_ratio = preservation_ratio
        self.min_quality_score = min_quality_score
        self.max_iterations = max_iterations

        # Statistics
        self.stats = {
            "total_compressions": 0,
            "avg_compression_ratio": 0.0,
            "avg_quality_score": 0.0,
            "avg_iterations": 0.0
        }
        self._total_ratio = 0.0
        self._total_quality = 0.0
        self._total_iterations = 0

        # Stop words for word-level compression
        self.stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'them', 'their', 'what', 'which', 'who',
            'when', 'where', 'why', 'how', 'very', 'just', 'really'
        }

    def compress(
        self,
        text: str,
        target_size: Optional[int] = None,
        strategy: CompressionStrategy = CompressionStrategy.HYBRID,
        level: CompressionLevel = CompressionLevel.SENTENCE
    ) -> CompressionResult:
        """
        Compress text to target size

        Args:
            text: Input text to compress
            target_size: Target size (characters), if None uses target_ratio
            strategy: Compression strategy
            level: Compression level

        Returns:
            CompressionResult
        """
        original_size = len(text)

        if target_size is None:
            target_size = int(original_size * self.target_ratio)

        # If already small enough
        if original_size <= target_size:
            return CompressionResult(
                original_text=text,
                compressed_text=text,
                original_size=original_size,
                compressed_size=original_size,
                compression_ratio=1.0,
                quality_score=1.0,
                strategy_used=strategy,
                iterations=0
            )

        # Apply strategy
        if strategy == CompressionStrategy.EXTRACTIVE:
            compressed = self._extractive_compress(text, target_size, level)
        elif strategy == CompressionStrategy.ABSTRACTIVE:
            compressed = self._abstractive_compress(text, target_size, level)
        else:  # HYBRID
            compressed = self._hybrid_compress(text, target_size, level)

        # Calculate metrics
        compressed_size = len(compressed)
        compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
        quality_score = self._calculate_quality(text, compressed)

        result = CompressionResult(
            original_text=text,
            compressed_text=compressed,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            quality_score=quality_score,
            strategy_used=strategy,
            iterations=1
        )

        # Update stats
        self._update_stats(result)

        return result

    def _extractive_compress(
        self,
        text: str,
        target_size: int,
        level: CompressionLevel
    ) -> str:
        """
        Extractive compression: select important content

        Args:
            text: Input text
            target_size: Target size
            level: Compression level

        Returns:
            Compressed text
        """
        if level == CompressionLevel.WORD:
            return self._word_level_compress(text, target_size)
        elif level == CompressionLevel.SENTENCE:
            return self._sentence_level_compress(text, target_size)
        else:  # PARAGRAPH
            return self._paragraph_level_compress(text, target_size)

    def _abstractive_compress(
        self,
        text: str,
        target_size: int,
        level: CompressionLevel
    ) -> str:
        """
        Abstractive compression: generate summaries

        Args:
            text: Input text
            target_size: Target size
            level: Compression level

        Returns:
            Compressed text
        """
        # For abstractive, we'll simulate by taking key sentences
        # and reformulating them (simplified version)
        sentences = self._split_sentences(text)

        if not sentences:
            return text[:target_size]

        # Score sentences
        scores = self._score_sentences(sentences, text)

        # Select top sentences
        sorted_sentences = sorted(
            zip(sentences, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Build compressed version
        compressed = ""
        for sentence, score in sorted_sentences:
            # Simplify sentence (remove some words)
            simplified = self._simplify_sentence(sentence)
            if len(compressed) + len(simplified) <= target_size:
                compressed += simplified + " "
            else:
                break

        result = compressed.strip()

        # Fallback: if nothing was added, just truncate
        if not result and sentences:
            result = self._simplify_sentence(sentences[0])[:target_size]

        return result

    def _hybrid_compress(
        self,
        text: str,
        target_size: int,
        level: CompressionLevel
    ) -> str:
        """
        Hybrid compression: combine extractive and abstractive

        Args:
            text: Input text
            target_size: Target size
            level: Compression level

        Returns:
            Compressed text
        """
        # Use extractive for initial reduction
        extractive_target = int(target_size * 1.2)  # Leave room for abstractive
        extractive = self._extractive_compress(text, extractive_target, level)

        # If still too large, apply abstractive
        if len(extractive) > target_size:
            abstractive = self._abstractive_compress(extractive, target_size, level)
            return abstractive

        return extractive

    def _word_level_compress(self, text: str, target_size: int) -> str:
        """Word-level compression: remove unimportant words"""
        words = text.split()

        if not words:
            return text

        # Calculate word importance (simplified)
        word_freq = Counter(w.lower().strip('.,!?;:') for w in words)
        important_words = []

        for word in words:
            word_lower = word.lower().strip('.,!?;:')
            # Keep if not stop word or high frequency
            if word_lower not in self.stop_words or word_freq[word_lower] > 2:
                important_words.append(word)

        # Build compressed text
        compressed = " ".join(important_words)

        # If still too large, truncate
        if len(compressed) > target_size:
            compressed = compressed[:target_size].rsplit(' ', 1)[0]

        return compressed

    def _sentence_level_compress(self, text: str, target_size: int) -> str:
        """Sentence-level compression: select important sentences"""
        sentences = self._split_sentences(text)

        if not sentences:
            return text[:target_size]

        # Score sentences by importance
        scores = self._score_sentences(sentences, text)

        # Sort by score
        sorted_sentences = sorted(
            zip(sentences, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Select top sentences until target size
        compressed_sentences = []
        current_size = 0

        for idx, (sentence, score) in enumerate(sorted_sentences):
            sentence_size = len(sentence)
            if current_size + sentence_size <= target_size:
                # Store sentence with original index
                orig_idx = sentences.index(sentence)
                compressed_sentences.append((sentence, score, orig_idx))
                current_size += sentence_size
            elif current_size < target_size * 0.8:
                # Try to fit partial sentence
                remaining = target_size - current_size
                if remaining > 20:  # Only if meaningful
                    orig_idx = sentences.index(sentence)
                    compressed_sentences.append((sentence[:remaining], score, orig_idx))
                break

        # Return empty if nothing selected
        if not compressed_sentences:
            return text[:target_size] if target_size > 0 else ""

        # Sort back by original order (maintain flow)
        compressed_sentences.sort(key=lambda x: x[2])

        return " ".join(s[0] for s in compressed_sentences)

    def _paragraph_level_compress(self, text: str, target_size: int) -> str:
        """Paragraph-level compression: compress paragraphs"""
        paragraphs = text.split('\n\n')

        if len(paragraphs) <= 1:
            # Fall back to sentence level
            return self._sentence_level_compress(text, target_size)

        # Score paragraphs
        scores = []
        for para in paragraphs:
            if para.strip():
                score = len(para.split())  # Simple: longer = more important
                scores.append((para, score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        # Select paragraphs
        compressed = ""
        for para, score in scores:
            if len(compressed) + len(para) <= target_size:
                compressed += para + "\n\n"
            else:
                # Compress this paragraph to fit
                remaining = target_size - len(compressed)
                if remaining > 50:
                    compressed += self._sentence_level_compress(para, remaining)
                break

        return compressed.strip()

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _score_sentences(self, sentences: List[str], full_text: str) -> List[float]:
        """Score sentence importance"""
        scores = []
        words = full_text.lower().split()
        word_freq = Counter(words)

        for sentence in sentences:
            # Score based on word frequency and position
            sentence_words = sentence.lower().split()
            score = 0.0

            for word in sentence_words:
                if word not in self.stop_words:
                    score += word_freq.get(word, 0)

            # Normalize by sentence length
            score = score / max(len(sentence_words), 1)

            scores.append(score)

        return scores

    def _simplify_sentence(self, sentence: str) -> str:
        """Simplify sentence by removing less important words"""
        words = sentence.split()
        important = []

        for word in words:
            word_lower = word.lower().strip('.,!?;:')
            # Keep non-stop words
            if word_lower not in self.stop_words or len(word_lower) > 6:
                important.append(word)

        return " ".join(important)

    def _calculate_quality(self, original: str, compressed: str) -> float:
        """
        Calculate compression quality score

        Args:
            original: Original text
            compressed: Compressed text

        Returns:
            Quality score (0-1)
        """
        if not original or not compressed:
            return 0.0

        # Extract important words from original
        original_words = set(
            w.lower().strip('.,!?;:')
            for w in original.split()
            if w.lower().strip('.,!?;:') not in self.stop_words
        )

        # Extract words from compressed
        compressed_words = set(
            w.lower().strip('.,!?;:')
            for w in compressed.split()
            if w.lower().strip('.,!?;:') not in self.stop_words
        )

        if not original_words:
            return 1.0

        # Calculate preservation ratio
        preserved = len(original_words & compressed_words)
        quality = preserved / len(original_words)

        return min(1.0, quality)

    def iterative_compress(
        self,
        text: str,
        target_size: int,
        strategy: CompressionStrategy = CompressionStrategy.HYBRID
    ) -> CompressionResult:
        """
        Iteratively compress with quality checks

        Args:
            text: Input text
            target_size: Target size
            strategy: Compression strategy

        Returns:
            CompressionResult with best quality
        """
        current_text = text
        best_result = None
        iterations = 0

        for i in range(self.max_iterations):
            iterations += 1

            # Compress
            result = self.compress(
                current_text,
                target_size,
                strategy,
                CompressionLevel.SENTENCE
            )

            # Check quality
            if result.quality_score >= self.min_quality_score:
                result.iterations = iterations
                return result

            # If quality too low but size met, try different level
            if result.compressed_size <= target_size:
                if best_result is None or result.quality_score > best_result.quality_score:
                    best_result = result
                    best_result.iterations = iterations

            # If not at target size, continue
            if result.compressed_size > target_size:
                current_text = result.compressed_text
            else:
                break

        # Return best result found
        if best_result:
            return best_result

        # Fallback: just truncate
        return CompressionResult(
            original_text=text,
            compressed_text=text[:target_size],
            original_size=len(text),
            compressed_size=min(len(text), target_size),
            compression_ratio=min(1.0, target_size / len(text)),
            quality_score=0.5,
            strategy_used=strategy,
            iterations=iterations
        )

    def compress_with_preservation(
        self,
        text: str,
        target_size: int,
        preserve_keywords: List[str]
    ) -> CompressionResult:
        """
        Compress while preserving specific keywords

        Args:
            text: Input text
            target_size: Target size
            preserve_keywords: Keywords to preserve

        Returns:
            CompressionResult
        """
        sentences = self._split_sentences(text)
        preserve_set = set(k.lower() for k in preserve_keywords)

        # Score sentences, boost those with keywords
        scores = []
        for sentence in sentences:
            base_score = len(sentence.split())
            sentence_lower = sentence.lower()

            # Boost for preserved keywords
            keyword_count = sum(1 for kw in preserve_set if kw in sentence_lower)
            score = base_score * (1 + keyword_count * 2)

            scores.append((sentence, score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        # Select sentences
        compressed = ""
        for sentence, score in scores:
            if len(compressed) + len(sentence) <= target_size:
                compressed += sentence + " "
            else:
                break

        compressed = compressed.strip()

        return CompressionResult(
            original_text=text,
            compressed_text=compressed,
            original_size=len(text),
            compressed_size=len(compressed),
            compression_ratio=len(compressed) / len(text),
            quality_score=self._calculate_quality(text, compressed),
            strategy_used=CompressionStrategy.EXTRACTIVE,
            iterations=1
        )

    def _update_stats(self, result: CompressionResult):
        """Update compression statistics"""
        self.stats["total_compressions"] += 1
        self._total_ratio += result.compression_ratio
        self._total_quality += result.quality_score
        self._total_iterations += result.iterations

        self.stats["avg_compression_ratio"] = self._total_ratio / self.stats["total_compressions"]
        self.stats["avg_quality_score"] = self._total_quality / self.stats["total_compressions"]
        self.stats["avg_iterations"] = self._total_iterations / self.stats["total_compressions"]

    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "total_compressions": 0,
            "avg_compression_ratio": 0.0,
            "avg_quality_score": 0.0,
            "avg_iterations": 0.0
        }
        self._total_ratio = 0.0
        self._total_quality = 0.0
        self._total_iterations = 0
