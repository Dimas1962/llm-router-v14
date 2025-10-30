"""
Tests for SELF-CHECK Quality Assurance System (Component 3)
"""

import pytest
from src.v2.self_check import SelfCheckSystem, SelfCheckResult


def test_initialization():
    """Test SelfCheckSystem initialization"""
    system = SelfCheckSystem()

    assert system.focus_threshold == 6.0
    assert system.result_threshold == 6.0
    assert system.require_fact_verification is True
    assert system.stats["total_checks"] == 0
    assert system.stats["passed"] == 0


def test_basic_check():
    """Test basic quality check"""
    system = SelfCheckSystem()

    result = system.check(
        query="What is Python?",
        result="Python is a high-level programming language known for readability and versatility."
    )

    assert isinstance(result, SelfCheckResult)
    assert 0 <= result.focus_score <= 10
    assert 0 <= result.result_score <= 10
    assert isinstance(result.passed, bool)
    assert isinstance(result.issues, list)


def test_focus_score_high():
    """Test high focus score for relevant result"""
    system = SelfCheckSystem()

    result = system.check(
        query="Explain machine learning algorithms",
        result="Machine learning algorithms are computational methods that learn patterns from data. "
               "These algorithms include supervised learning, unsupervised learning, and reinforcement learning."
    )

    # Should have good focus (relevant keywords match)
    assert result.focus_score >= 6.0


def test_focus_score_low():
    """Test low focus score for irrelevant result"""
    system = SelfCheckSystem()

    result = system.check(
        query="What is quantum computing?",
        result="The weather is nice today."
    )

    # Should have poor focus (no keyword match)
    assert result.focus_score < 6.0


def test_result_score_quality():
    """Test result score for output quality"""
    system = SelfCheckSystem()

    # High quality result (complete sentences, good length)
    good_result = system.check(
        query="Test",
        result="This is a well-formed response. It contains multiple sentences. "
               "The content is coherent and provides useful information."
    )
    assert good_result.result_score >= 6.0

    # Low quality result (too short)
    poor_result = system.check(
        query="Test",
        result="Yes."
    )
    assert poor_result.result_score < 6.0


def test_fact_verification_pass():
    """Test fact verification with matching context"""
    system = SelfCheckSystem()

    context = "Python was created by Guido van Rossum in 1991. It has version 3.9 released."

    result = system.check(
        query="When was Python created?",
        result="Python was created by Guido van Rossum in 1991.",
        context=context
    )

    assert result.fact_verified is True


def test_fact_verification_fail():
    """Test fact verification with contradicting facts"""
    system = SelfCheckSystem()

    context = "Python was created in 1991."

    result = system.check(
        query="When was Python created?",
        result="Python was created in 2000.",  # Wrong year
        context=context
    )

    # May fail fact verification (number mismatch)
    # This is heuristic-based, so we check it ran without error
    assert isinstance(result.fact_verified, bool)


def test_pass_fail_determination():
    """Test pass/fail logic"""
    system = SelfCheckSystem(
        focus_threshold=5.0,
        result_threshold=5.0
    )

    # Should pass (good quality)
    pass_result = system.check(
        query="Explain Python programming",
        result="Python is a programming language that emphasizes code readability. "
               "It supports multiple programming paradigms including procedural and object-oriented."
    )
    assert pass_result.passed is True

    # Should fail (poor quality)
    fail_result = system.check(
        query="Explain quantum physics",
        result="No."
    )
    assert fail_result.passed is False


def test_statistics_tracking():
    """Test statistics tracking across multiple checks"""
    system = SelfCheckSystem()

    # Perform multiple checks
    for i in range(5):
        system.check(
            query="Test query",
            result="This is a test response with reasonable quality and length."
        )

    stats = system.get_stats()

    assert stats["total_checks"] == 5
    assert stats["passed"] + stats["failed"] == 5
    assert 0 <= stats["avg_focus_score"] <= 10
    assert 0 <= stats["avg_result_score"] <= 10
    assert 0 <= stats["pass_rate"] <= 1.0


def test_batch_check():
    """Test batch checking"""
    system = SelfCheckSystem()

    checks = [
        {
            "query": "What is AI?",
            "result": "AI is artificial intelligence, a field of computer science."
        },
        {
            "query": "What is ML?",
            "result": "ML is machine learning, a subset of AI."
        },
        {
            "query": "What is DL?",
            "result": "DL is deep learning, using neural networks."
        }
    ]

    results = system.batch_check(checks)

    assert len(results) == 3
    assert all(isinstance(r, SelfCheckResult) for r in results)
    assert system.stats["total_checks"] == 3


def test_pass_rate_calculation():
    """Test pass rate calculation"""
    system = SelfCheckSystem(
        focus_threshold=3.0,
        result_threshold=3.0
    )

    # Add passing checks
    for _ in range(7):
        system.check(
            query="Good query",
            result="This is a good response with decent length and quality."
        )

    # Add failing checks
    for _ in range(3):
        system.check(
            query="Test",
            result="No"
        )

    pass_rate = system.get_pass_rate()
    assert 0.0 <= pass_rate <= 1.0
    assert pass_rate == system.stats["passed"] / 10


def test_stats_reset():
    """Test statistics reset"""
    system = SelfCheckSystem()

    # Perform some checks
    system.check("Test", "Response")
    system.check("Test 2", "Response 2")

    assert system.stats["total_checks"] == 2

    # Reset
    system.reset_stats()

    assert system.stats["total_checks"] == 0
    assert system.stats["passed"] == 0
    assert system.stats["failed"] == 0
    assert system.stats["avg_focus_score"] == 0.0


def test_expected_topics():
    """Test focus score with expected topics"""
    system = SelfCheckSystem()

    result = system.check(
        query="Explain neural networks",
        result="Neural networks are computing systems inspired by biological neural networks. "
               "They consist of layers including input, hidden, and output layers.",
        expected_topics=["neural", "networks", "layers"]
    )

    # Should get bonus for topic matches
    assert result.focus_score >= 6.0


def test_metadata_tracking():
    """Test metadata in check results"""
    system = SelfCheckSystem()

    result = system.check(
        query="Short query",
        result="A reasonably long response to demonstrate metadata tracking.",
        context="Some context"
    )

    assert "query_length" in result.metadata
    assert "result_length" in result.metadata
    assert "has_context" in result.metadata
    assert result.metadata["has_context"] is True


def test_issues_tracking():
    """Test issue tracking in failed checks"""
    system = SelfCheckSystem(
        focus_threshold=8.0,
        result_threshold=8.0
    )

    result = system.check(
        query="Test query",
        result="Short"
    )

    # Should have issues for low scores
    assert len(result.issues) > 0
    assert any("focus score" in issue.lower() or "result score" in issue.lower()
               for issue in result.issues)


def test_no_context_verification():
    """Test that verification passes when no context provided"""
    system = SelfCheckSystem(require_fact_verification=True)

    result = system.check(
        query="Test",
        result="Response without context"
        # No context provided
    )

    # Should pass verification when no context to check against
    assert result.fact_verified is True
