"""
Task Classification and Complexity Estimation
Analyzes queries to determine task type and complexity
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """Classification of task types"""
    CODING = "coding"
    REFACTORING = "refactoring"
    BUG_FIXING = "bug_fixing"
    ARCHITECTURE = "architecture"
    REASONING = "reasoning"
    QUICK_SNIPPET = "quick_snippet"
    MULTI_FILE = "multi_file"
    DOCUMENTATION = "documentation"
    REVIEW = "review"
    EXPLANATION = "explanation"


@dataclass
class TaskAnalysis:
    """Result of task classification"""
    task_type: TaskType
    complexity: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    detected_language: Optional[str]
    requires_reasoning: bool
    requires_large_context: bool
    is_simple_pattern: bool


class TaskClassifier:
    """Classifies tasks based on query content"""

    # Patterns for task type detection
    PATTERNS = {
        TaskType.ARCHITECTURE: [
            r'design\s+(?:a|the)\s+\w+',
            r'architect\w*',
            r'system\s+design',
            r'microservices',
            r'distributed\s+system',
            r'scalable',
            r'high-level\s+design',
        ],
        TaskType.REFACTORING: [
            r'refactor',
            r'restructure',
            r'improve\s+(?:the\s+)?code',
            r'clean\s+up',
            r'reorganize',
        ],
        TaskType.BUG_FIXING: [
            r'fix\s+(?:the\s+)?bug',
            r'debug',
            r'error',
            r'not\s+working',
            r'issue\s+with',
            r'problem\s+with',
        ],
        TaskType.QUICK_SNIPPET: [
            r'^write\s+(?:a\s+)?(?:simple|small|quick)\s+',
            r'^create\s+(?:a\s+)?(?:simple|small|quick)\s+',
            r'hello\s+world',
            r'print\s+',
            r'simple\s+function',
        ],
        TaskType.MULTI_FILE: [
            r'multiple\s+files',
            r'across\s+files',
            r'entire\s+codebase',
            r'project-wide',
            r'all\s+files',
        ],
        TaskType.DOCUMENTATION: [
            r'document',
            r'add\s+comments',
            r'write\s+docs',
            r'explain\s+(?:the\s+)?code',
            r'readme',
        ],
        TaskType.REVIEW: [
            r'review',
            r'analyze\s+(?:this\s+)?code',
            r'check\s+(?:this\s+)?code',
            r'what\s+does\s+this',
        ],
    }

    # Language detection patterns
    LANGUAGE_PATTERNS = {
        'python': [r'\bpython\b', r'\.py\b', r'\bdef\s+\w+', r'\bimport\s+'],
        'rust': [r'\brust\b', r'\.rs\b', r'\bfn\s+\w+', r'\blet\s+mut\s+'],
        'go': [r'\bgolang\b', r'\bgo\b', r'\.go\b', r'\bfunc\s+\w+'],
        'javascript': [r'\bjavascript\b', r'\bjs\b', r'\.js\b', r'\bconst\s+\w+\s*='],
        'typescript': [r'\btypescript\b', r'\bts\b', r'\.ts\b', r'\binterface\s+'],
        'kotlin': [r'\bkotlin\b', r'\.kt\b', r'\bfun\s+\w+'],
        'swift': [r'\bswift\b', r'\.swift\b', r'\bfunc\s+\w+'],
        'java': [r'\bjava\b', r'\.java\b', r'\bpublic\s+class\s+'],
    }

    # Complexity indicators
    COMPLEXITY_KEYWORDS = {
        'high': ['complex', 'advanced', 'sophisticated', 'distributed', 'concurrent',
                 'optimize', 'performance', 'scale', 'architecture'],
        'medium': ['implement', 'create', 'build', 'design', 'develop'],
        'low': ['simple', 'basic', 'quick', 'small', 'trivial', 'print', 'hello'],
    }

    def classify(self, query: str, session_history: List[str] = None) -> TaskAnalysis:
        """
        Classify the task based on query content

        Args:
            query: The user's query/task
            session_history: Previous messages in session (optional)

        Returns:
            TaskAnalysis with classification results
        """
        query_lower = query.lower()

        # Detect task type
        task_type = self._detect_task_type(query_lower)

        # Estimate complexity
        complexity = self._estimate_complexity(query_lower, session_history)

        # Detect programming language
        detected_language = self._detect_language(query_lower)

        # Determine if reasoning is required
        requires_reasoning = self._requires_reasoning(query_lower, complexity)

        # Check if large context is needed
        requires_large_context = self._requires_large_context(
            query, session_history
        )

        # Check for simple pattern
        is_simple_pattern = self._is_simple_pattern(query_lower, complexity)

        # Calculate confidence based on pattern matches
        confidence = self._calculate_confidence(query_lower, task_type)

        return TaskAnalysis(
            task_type=task_type,
            complexity=complexity,
            confidence=confidence,
            detected_language=detected_language,
            requires_reasoning=requires_reasoning,
            requires_large_context=requires_large_context,
            is_simple_pattern=is_simple_pattern
        )

    def _detect_task_type(self, query: str) -> TaskType:
        """Detect the type of task from query"""
        best_match = (TaskType.CODING, 0)  # default

        for task_type, patterns in self.PATTERNS.items():
            match_count = sum(
                1 for pattern in patterns
                if re.search(pattern, query, re.IGNORECASE)
            )
            if match_count > best_match[1]:
                best_match = (task_type, match_count)

        return best_match[0]

    def _detect_language(self, query: str) -> Optional[str]:
        """Detect programming language from query"""
        for lang, patterns in self.LANGUAGE_PATTERNS.items():
            if any(re.search(pattern, query, re.IGNORECASE) for pattern in patterns):
                return lang
        return None

    def _estimate_complexity(self, query: str,
                            session_history: List[str] = None) -> float:
        """
        Estimate complexity score (0.0 - 1.0)

        Factors:
        - Keywords (high/medium/low complexity words)
        - Query length
        - Technical terms
        - Session context
        """
        complexity = 0.5  # default medium

        # Check complexity keywords
        if any(word in query for word in self.COMPLEXITY_KEYWORDS['high']):
            complexity += 0.3
        elif any(word in query for word in self.COMPLEXITY_KEYWORDS['low']):
            complexity -= 0.3

        # Query length factor
        word_count = len(query.split())
        if word_count > 50:
            complexity += 0.1
        elif word_count < 10:
            complexity -= 0.1

        # Technical depth indicators
        technical_terms = [
            'algorithm', 'data structure', 'concurrent', 'async',
            'distributed', 'optimization', 'pattern', 'framework'
        ]
        tech_count = sum(1 for term in technical_terms if term in query)
        complexity += tech_count * 0.05

        # Session context factor
        if session_history and len(session_history) > 5:
            complexity += 0.1  # Longer sessions tend to be more complex

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, complexity))

    def _requires_reasoning(self, query: str, complexity: float) -> bool:
        """Determine if task requires complex reasoning"""
        reasoning_keywords = [
            'design', 'architect', 'why', 'how should', 'best approach',
            'recommend', 'compare', 'analyze', 'strategy', 'plan'
        ]

        has_reasoning_keyword = any(kw in query for kw in reasoning_keywords)
        is_complex = complexity > 0.7

        return has_reasoning_keyword or is_complex

    def _requires_large_context(self, query: str,
                               session_history: List[str] = None) -> bool:
        """Determine if task requires large context window"""
        # Calculate total context size
        total_size = len(query)
        if session_history:
            total_size += sum(len(msg) for msg in session_history)

        # Large context indicators
        large_context_keywords = [
            'entire', 'all files', 'whole project', 'codebase',
            'multiple files', 'across files'
        ]

        has_keyword = any(kw in query.lower() for kw in large_context_keywords)
        is_large_session = total_size > 100_000  # >100K chars

        return has_keyword or is_large_session

    def _is_simple_pattern(self, query: str, complexity: float) -> bool:
        """Check if query matches simple task patterns"""
        simple_patterns = [
            r'^write\s+(?:a\s+)?(?:simple|quick|small)',
            r'^print\s+',
            r'^create\s+(?:a\s+)?(?:simple|quick|small)',
            r'hello\s+world',
            r'^sort\s+(?:a\s+)?list',
            r'^reverse\s+(?:a\s+)?string',
        ]

        matches_pattern = any(
            re.match(pattern, query, re.IGNORECASE)
            for pattern in simple_patterns
        )

        return matches_pattern and complexity < 0.3

    def _calculate_confidence(self, query: str, task_type: TaskType) -> float:
        """Calculate confidence in classification"""
        if task_type not in self.PATTERNS:
            return 0.5  # default confidence

        patterns = self.PATTERNS[task_type]
        matches = sum(
            1 for pattern in patterns
            if re.search(pattern, query, re.IGNORECASE)
        )

        # More matches = higher confidence
        confidence = 0.5 + (matches * 0.15)
        return min(1.0, confidence)


class ComplexityEstimator:
    """
    Detailed complexity estimation for queries
    Provides more granular analysis than TaskClassifier
    """

    @staticmethod
    def estimate(query: str, session_history: List[str] = None) -> Dict[str, float]:
        """
        Estimate various complexity dimensions

        Returns:
            Dict with complexity scores for different dimensions
        """
        return {
            'overall': ComplexityEstimator._overall_complexity(query, session_history),
            'technical': ComplexityEstimator._technical_complexity(query),
            'cognitive': ComplexityEstimator._cognitive_complexity(query),
            'context': ComplexityEstimator._context_complexity(session_history),
        }

    @staticmethod
    def _overall_complexity(query: str, session_history: List[str] = None) -> float:
        """Overall complexity score"""
        classifier = TaskClassifier()
        return classifier._estimate_complexity(query.lower(), session_history)

    @staticmethod
    def _technical_complexity(query: str) -> float:
        """Technical depth complexity"""
        technical_indicators = [
            'algorithm', 'optimization', 'concurrent', 'async', 'parallel',
            'distributed', 'scalable', 'performance', 'thread', 'lock',
            'mutex', 'race condition', 'deadlock', 'memory', 'cache'
        ]

        count = sum(1 for term in technical_indicators if term in query.lower())
        return min(1.0, count * 0.15)

    @staticmethod
    def _cognitive_complexity(query: str) -> float:
        """Cognitive/reasoning complexity"""
        reasoning_indicators = [
            'why', 'how', 'explain', 'compare', 'analyze', 'evaluate',
            'design', 'architect', 'strategy', 'approach', 'tradeoff',
            'pros and cons', 'best practice'
        ]

        count = sum(1 for term in reasoning_indicators if term in query.lower())
        return min(1.0, count * 0.2)

    @staticmethod
    def _context_complexity(session_history: List[str] = None) -> float:
        """Context window complexity"""
        if not session_history:
            return 0.0

        total_size = sum(len(msg) for msg in session_history)

        if total_size < 10_000:
            return 0.1
        elif total_size < 50_000:
            return 0.3
        elif total_size < 100_000:
            return 0.6
        else:
            return 0.9
