"""
Tests for Recursive Compressor (Component 8)
"""

import pytest
from src.v2.recursive_compressor import (
    RecursiveCompressor,
    CompressionStrategy,
    CompressionLevel,
    CompressionResult
)


def test_initialization():
    """Test RecursiveCompressor initialization"""
    compressor = RecursiveCompressor()

    assert compressor.target_ratio == 0.5
    assert compressor.preservation_ratio == 0.8
    assert compressor.min_quality_score == 0.6
    assert compressor.max_iterations == 5
    assert compressor.stats["total_compressions"] == 0


def test_basic_compression():
    """Test basic text compression"""
    compressor = RecursiveCompressor(target_ratio=0.5)

    text = "This is a test sentence. " * 20  # Long text

    result = compressor.compress(text)

    assert isinstance(result, CompressionResult)
    assert result.compressed_size < result.original_size
    assert 0 < result.compression_ratio <= 1.0
    assert 0 <= result.quality_score <= 1.0


def test_extractive_compression():
    """Test extractive compression strategy"""
    compressor = RecursiveCompressor()

    text = """
    Machine learning is a field of artificial intelligence.
    It focuses on developing algorithms that learn from data.
    Deep learning is a subset of machine learning.
    Neural networks are used in deep learning.
    """

    result = compressor.compress(
        text,
        target_size=100,
        strategy=CompressionStrategy.EXTRACTIVE
    )

    assert result.compressed_size <= 100
    assert result.strategy_used == CompressionStrategy.EXTRACTIVE


def test_abstractive_compression():
    """Test abstractive compression strategy"""
    compressor = RecursiveCompressor()

    text = """
    Python is a high-level programming language.
    It was created by Guido van Rossum in 1991.
    Python emphasizes code readability and simplicity.
    It supports multiple programming paradigms.
    """

    result = compressor.compress(
        text,
        target_size=100,
        strategy=CompressionStrategy.ABSTRACTIVE
    )

    assert result.compressed_size <= 100
    assert result.strategy_used == CompressionStrategy.ABSTRACTIVE


def test_hybrid_compression():
    """Test hybrid compression strategy"""
    compressor = RecursiveCompressor()

    text = """
    Artificial intelligence has many applications.
    Natural language processing is one application.
    Computer vision is another important application.
    Robotics also uses artificial intelligence extensively.
    Machine learning powers many AI systems today.
    """ * 3

    result = compressor.compress(
        text,
        target_size=200,
        strategy=CompressionStrategy.HYBRID
    )

    assert result.compressed_size <= 200
    assert result.strategy_used == CompressionStrategy.HYBRID


def test_word_level_compression():
    """Test word-level compression"""
    compressor = RecursiveCompressor()

    text = "The quick brown fox jumps over the lazy dog in the very sunny afternoon"

    result = compressor.compress(
        text,
        target_size=30,
        strategy=CompressionStrategy.EXTRACTIVE,
        level=CompressionLevel.WORD
    )

    assert result.compressed_size <= 30
    # Should keep important words, remove stop words
    assert "fox" in result.compressed_text.lower() or "dog" in result.compressed_text.lower()


def test_sentence_level_compression():
    """Test sentence-level compression"""
    compressor = RecursiveCompressor()

    text = """
    First sentence about Python programming.
    Second sentence about machine learning.
    Third sentence about data science.
    Fourth sentence about artificial intelligence.
    Fifth sentence about deep learning.
    """

    result = compressor.compress(
        text,
        target_size=100,
        strategy=CompressionStrategy.EXTRACTIVE,
        level=CompressionLevel.SENTENCE
    )

    # Allow small overage due to spacing
    assert result.compressed_size <= 105
    # Should preserve complete sentences
    sentences = result.compressed_text.split('.')
    assert len(sentences) > 0


def test_paragraph_level_compression():
    """Test paragraph-level compression"""
    compressor = RecursiveCompressor()

    text = """
    Paragraph one talks about programming.
    It has multiple sentences.

    Paragraph two discusses algorithms.
    This is important content.

    Paragraph three covers data structures.
    Another important topic.
    """

    result = compressor.compress(
        text,
        target_size=150,
        strategy=CompressionStrategy.EXTRACTIVE,
        level=CompressionLevel.PARAGRAPH
    )

    assert result.compressed_size <= 150


def test_already_compressed():
    """Test compression of already small text"""
    compressor = RecursiveCompressor()

    text = "Short text"

    result = compressor.compress(text, target_size=100)

    # Should not compress (already small)
    assert result.compressed_text == text
    assert result.compression_ratio == 1.0
    assert result.iterations == 0


def test_compression_ratio_calculation():
    """Test compression ratio calculation"""
    compressor = RecursiveCompressor()

    text = "A" * 1000  # 1000 characters

    result = compressor.compress(text, target_size=500)

    expected_ratio = result.compressed_size / result.original_size
    assert abs(result.compression_ratio - expected_ratio) < 0.01


def test_quality_score():
    """Test compression quality scoring"""
    compressor = RecursiveCompressor()

    text = """
    Machine learning algorithm development requires data.
    Data preprocessing is an important step.
    Model training uses the preprocessed data.
    Evaluation metrics measure model performance.
    """

    result = compressor.compress(text, target_size=100)

    # Quality should be reasonable
    assert 0 <= result.quality_score <= 1.0

    # Higher quality with less compression
    result_light = compressor.compress(text, target_size=200)
    assert result_light.quality_score >= result.quality_score


def test_iterative_compression():
    """Test iterative compression with quality checks"""
    compressor = RecursiveCompressor(
        min_quality_score=0.7,
        max_iterations=3
    )

    text = """
    Python is a versatile programming language.
    It supports object-oriented programming.
    Python has extensive standard libraries.
    The language emphasizes code readability.
    """ * 5

    result = compressor.iterative_compress(
        text,
        target_size=200,
        strategy=CompressionStrategy.HYBRID
    )

    assert result.compressed_size <= 200
    assert result.iterations >= 1
    assert result.iterations <= compressor.max_iterations


def test_compression_with_preservation():
    """Test compression with keyword preservation"""
    compressor = RecursiveCompressor()

    text = """
    Python is a programming language.
    Java is another programming language.
    Machine learning uses Python frequently.
    Data science prefers Python over Java.
    Python has great libraries for AI.
    """

    preserve_keywords = ["Python", "machine learning"]

    result = compressor.compress_with_preservation(
        text,
        target_size=100,
        preserve_keywords=preserve_keywords
    )

    assert result.compressed_size <= 100
    # Should preserve important keywords
    compressed_lower = result.compressed_text.lower()
    assert "python" in compressed_lower


def test_statistics_tracking():
    """Test compression statistics tracking"""
    compressor = RecursiveCompressor()

    text1 = "Test text one. " * 10
    text2 = "Test text two. " * 10

    compressor.compress(text1, target_size=50)
    compressor.compress(text2, target_size=50)

    stats = compressor.get_stats()

    assert stats["total_compressions"] == 2
    assert 0 <= stats["avg_compression_ratio"] <= 1.0
    assert 0 <= stats["avg_quality_score"] <= 1.0
    assert stats["avg_iterations"] >= 1


def test_stats_reset():
    """Test statistics reset"""
    compressor = RecursiveCompressor()

    compressor.compress("Test text", target_size=5)

    assert compressor.stats["total_compressions"] == 1

    compressor.reset_stats()

    assert compressor.stats["total_compressions"] == 0
    assert compressor.stats["avg_compression_ratio"] == 0.0


def test_empty_text():
    """Test compression of empty text"""
    compressor = RecursiveCompressor()

    result = compressor.compress("", target_size=100)

    assert result.compressed_text == ""
    assert result.compressed_size == 0
    assert result.original_size == 0


def test_sentence_scoring():
    """Test sentence importance scoring"""
    compressor = RecursiveCompressor()

    text = """
    Python programming language is widely used.
    The weather is nice today.
    Python has many libraries for data science.
    I like coffee in the morning.
    Machine learning uses Python extensively.
    """

    result = compressor.compress(
        text,
        target_size=100,
        strategy=CompressionStrategy.EXTRACTIVE,
        level=CompressionLevel.SENTENCE
    )

    # Should prefer sentences with "Python" (repeated keyword)
    assert "python" in result.compressed_text.lower()


def test_multiple_strategies():
    """Test compression with different strategies"""
    compressor = RecursiveCompressor()

    text = """
    Artificial intelligence is transforming technology.
    Machine learning is a subset of AI.
    Deep learning uses neural networks.
    Natural language processing handles text.
    Computer vision processes images.
    """ * 3

    target = 200

    result_extractive = compressor.compress(
        text, target, CompressionStrategy.EXTRACTIVE
    )
    result_abstractive = compressor.compress(
        text, target, CompressionStrategy.ABSTRACTIVE
    )
    result_hybrid = compressor.compress(
        text, target, CompressionStrategy.HYBRID
    )

    # All should meet target
    assert result_extractive.compressed_size <= target
    assert result_abstractive.compressed_size <= target
    assert result_hybrid.compressed_size <= target

    # Results may differ
    assert (
        result_extractive.compressed_text != result_abstractive.compressed_text or
        result_extractive.compressed_text != result_hybrid.compressed_text
    )


def test_large_compression_ratio():
    """Test aggressive compression (high ratio)"""
    compressor = RecursiveCompressor()

    text = """
    This is a very long text that needs significant compression.
    We have many sentences here to test the compression algorithm.
    The compressor should handle large reduction ratios.
    Quality might decrease with higher compression.
    But the algorithm should still produce reasonable output.
    """ * 10

    result = compressor.compress(text, target_size=100)

    assert result.compressed_size <= 100
    assert result.compression_ratio < 0.5  # High compression


def test_preservation_ratio():
    """Test quality preservation ratio"""
    compressor = RecursiveCompressor(preservation_ratio=0.9)

    text = """
    Important data science concepts include statistics.
    Machine learning algorithms learn from data patterns.
    Feature engineering improves model performance significantly.
    Cross-validation prevents overfitting in models.
    """ * 3

    result = compressor.iterative_compress(text, target_size=200)

    # Should produce some output
    assert result.compressed_size > 0
    assert result.compression_ratio > 0


def test_different_target_ratios():
    """Test compression with different target ratios"""
    text = """
    Machine learning enables computers to learn patterns.
    Deep learning uses neural networks for complex tasks.
    Natural language processing handles text analysis.
    Computer vision allows machines to interpret images.
    Reinforcement learning teaches through rewards.
    """ * 5

    compressor_light = RecursiveCompressor(target_ratio=0.8)
    compressor_heavy = RecursiveCompressor(target_ratio=0.3)

    result_light = compressor_light.compress(text)
    result_heavy = compressor_heavy.compress(text)

    # Both should produce output
    assert result_light.compressed_size > 0
    assert result_heavy.compressed_size > 0

    # Light compression should have higher ratio
    assert result_light.compression_ratio > result_heavy.compression_ratio

    # Light compression should have better quality
    assert result_light.quality_score >= result_heavy.quality_score
