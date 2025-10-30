"""
SELF-CHECK Quality Assurance System - Component 3
Automated quality verification for LLM outputs
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import re


@dataclass
class SelfCheckResult:
    """Result of a SELF-CHECK quality assessment"""
    focus_score: float  # 0-10: relevance to query
    result_score: float  # 0-10: output quality
    fact_verified: bool  # Facts match context
    passed: bool  # Overall pass/fail
    issues: List[str]  # List of detected issues
    metadata: Dict[str, Any]  # Additional metadata


class SelfCheckSystem:
    """
    SELF-CHECK Quality Assurance System

    Features:
    - Focus scoring: Measures query relevance
    - Result scoring: Assesses output quality
    - Fact verification: Validates against context
    - Pass/fail determination
    - Statistics tracking
    """

    def __init__(
        self,
        focus_threshold: float = 6.0,
        result_threshold: float = 6.0,
        require_fact_verification: bool = True
    ):
        """
        Initialize SELF-CHECK system

        Args:
            focus_threshold: Minimum focus score to pass (0-10)
            result_threshold: Minimum result score to pass (0-10)
            require_fact_verification: Whether fact verification is required to pass
        """
        self.focus_threshold = focus_threshold
        self.result_threshold = result_threshold
        self.require_fact_verification = require_fact_verification

        # Statistics
        self.stats = {
            "total_checks": 0,
            "passed": 0,
            "failed": 0,
            "avg_focus_score": 0.0,
            "avg_result_score": 0.0,
            "fact_verification_rate": 0.0
        }

        # Running totals for averages
        self._total_focus = 0.0
        self._total_result = 0.0
        self._total_verified = 0

    def check(
        self,
        query: str,
        result: str,
        context: Optional[str] = None,
        expected_topics: Optional[List[str]] = None
    ) -> SelfCheckResult:
        """
        Perform SELF-CHECK quality assessment

        Args:
            query: Original query/question
            result: Generated result/answer
            context: Optional context for fact verification
            expected_topics: Optional list of expected topics

        Returns:
            SelfCheckResult with scores and pass/fail determination
        """
        issues = []

        # 1. Focus Score: Measure query relevance
        focus_score = self._calculate_focus_score(query, result, expected_topics)
        if focus_score < self.focus_threshold:
            issues.append(f"Low focus score: {focus_score:.1f} < {self.focus_threshold}")

        # 2. Result Score: Assess output quality
        result_score = self._calculate_result_score(result)
        if result_score < self.result_threshold:
            issues.append(f"Low result score: {result_score:.1f} < {self.result_threshold}")

        # 3. Fact Verification: Check against context
        fact_verified = self._verify_facts(result, context) if context else True
        if self.require_fact_verification and not fact_verified:
            issues.append("Fact verification failed")

        # 4. Determine pass/fail
        passed = (
            focus_score >= self.focus_threshold and
            result_score >= self.result_threshold and
            (not self.require_fact_verification or fact_verified)
        )

        # Update statistics
        self._update_stats(focus_score, result_score, fact_verified, passed)

        return SelfCheckResult(
            focus_score=focus_score,
            result_score=result_score,
            fact_verified=fact_verified,
            passed=passed,
            issues=issues,
            metadata={
                "query_length": len(query),
                "result_length": len(result),
                "has_context": context is not None
            }
        )

    def _calculate_focus_score(
        self,
        query: str,
        result: str,
        expected_topics: Optional[List[str]] = None
    ) -> float:
        """
        Calculate focus score (0-10) - relevance to query

        Args:
            query: Original query
            result: Generated result
            expected_topics: Optional expected topics

        Returns:
            Focus score (0-10)
        """
        score = 5.0  # Start with neutral score

        query_lower = query.lower()
        result_lower = result.lower()

        # Extract key words from query (simple heuristic)
        query_words = set(re.findall(r'\b\w{4,}\b', query_lower))
        result_words = set(re.findall(r'\b\w{4,}\b', result_lower))

        if query_words:
            # Calculate word overlap
            overlap = len(query_words & result_words)
            overlap_ratio = overlap / len(query_words)

            # Score based on overlap (0-5 points)
            score = 5.0 + (overlap_ratio * 5.0)

        # Bonus for expected topics (if provided)
        if expected_topics:
            topic_matches = sum(
                1 for topic in expected_topics
                if topic.lower() in result_lower
            )
            topic_ratio = topic_matches / len(expected_topics)
            score = min(10.0, score + (topic_ratio * 2.0))

        # Penalty for very short results
        if len(result) < 20:
            score = max(0.0, score - 3.0)

        return min(10.0, max(0.0, score))

    def _calculate_result_score(self, result: str) -> float:
        """
        Calculate result score (0-10) - output quality

        Args:
            result: Generated result

        Returns:
            Result score (0-10)
        """
        score = 5.0  # Start with neutral score

        # Length check
        length = len(result)
        if length < 10:
            score -= 4.0
        elif length < 50:
            score -= 2.0
        elif length > 100:
            score += 1.0

        # Sentence structure (basic heuristic)
        sentences = re.split(r'[.!?]+', result)
        complete_sentences = [s for s in sentences if len(s.strip()) > 10]

        if len(complete_sentences) >= 2:
            score += 2.0
        elif len(complete_sentences) >= 1:
            score += 1.0

        # Coherence indicators
        if any(word in result.lower() for word in ['because', 'therefore', 'however', 'thus']):
            score += 1.0

        # Check for repetition (quality issue)
        words = result.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                score -= 2.0

        # Capitalization and punctuation
        if result and result[0].isupper():
            score += 0.5
        if result and result[-1] in '.!?':
            score += 0.5

        return min(10.0, max(0.0, score))

    def _verify_facts(self, result: str, context: Optional[str]) -> bool:
        """
        Verify facts against context

        Args:
            result: Generated result
            context: Reference context

        Returns:
            True if facts are verified
        """
        if not context:
            return True

        result_lower = result.lower()
        context_lower = context.lower()

        # Extract potential facts (numbers, capitalized terms)
        # Numbers
        result_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', result))
        if result_numbers:
            context_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', context))
            # Check if result numbers are subset of context numbers
            if not result_numbers.issubset(context_numbers):
                # Allow some flexibility (80% match)
                match_ratio = len(result_numbers & context_numbers) / len(result_numbers)
                if match_ratio < 0.8:
                    return False

        # Capitalized terms (potential proper nouns)
        result_terms = set(re.findall(r'\b[A-Z][a-z]+\b', result))
        if result_terms:
            # Check if terms appear in context
            verified_terms = sum(1 for term in result_terms if term.lower() in context_lower)
            verification_ratio = verified_terms / len(result_terms)

            # Require 70% of terms to be in context
            if verification_ratio < 0.7:
                return False

        return True

    def _update_stats(
        self,
        focus_score: float,
        result_score: float,
        fact_verified: bool,
        passed: bool
    ):
        """
        Update statistics

        Args:
            focus_score: Focus score
            result_score: Result score
            fact_verified: Whether facts were verified
            passed: Whether check passed
        """
        self.stats["total_checks"] += 1

        if passed:
            self.stats["passed"] += 1
        else:
            self.stats["failed"] += 1

        # Update running totals
        self._total_focus += focus_score
        self._total_result += result_score
        if fact_verified:
            self._total_verified += 1

        # Calculate averages
        total = self.stats["total_checks"]
        self.stats["avg_focus_score"] = self._total_focus / total
        self.stats["avg_result_score"] = self._total_result / total
        self.stats["fact_verification_rate"] = self._total_verified / total

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics

        Returns:
            Statistics dictionary
        """
        stats = self.stats.copy()
        if stats["total_checks"] > 0:
            stats["pass_rate"] = stats["passed"] / stats["total_checks"]
            stats["fail_rate"] = stats["failed"] / stats["total_checks"]
        else:
            stats["pass_rate"] = 0.0
            stats["fail_rate"] = 0.0

        return stats

    def reset_stats(self):
        """Reset all statistics"""
        self.stats = {
            "total_checks": 0,
            "passed": 0,
            "failed": 0,
            "avg_focus_score": 0.0,
            "avg_result_score": 0.0,
            "fact_verification_rate": 0.0
        }
        self._total_focus = 0.0
        self._total_result = 0.0
        self._total_verified = 0

    def batch_check(
        self,
        checks: List[Dict[str, Any]]
    ) -> List[SelfCheckResult]:
        """
        Perform batch quality checks

        Args:
            checks: List of check dictionaries with 'query', 'result', 'context', etc.

        Returns:
            List of SelfCheckResult objects
        """
        results = []
        for check_data in checks:
            result = self.check(
                query=check_data["query"],
                result=check_data["result"],
                context=check_data.get("context"),
                expected_topics=check_data.get("expected_topics")
            )
            results.append(result)

        return results

    def get_pass_rate(self) -> float:
        """
        Get current pass rate

        Returns:
            Pass rate (0.0-1.0)
        """
        if self.stats["total_checks"] == 0:
            return 0.0
        return self.stats["passed"] / self.stats["total_checks"]
