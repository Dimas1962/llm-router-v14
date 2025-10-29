"""
Router Core - Main Routing Logic
Phase 1: Basic routing based on task classification and complexity
Phase 2: Eagle ELO + Associative Memory integration
Phase 3: CARROT Cost-Aware Routing
Phase 4: Advanced Context Management
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import logging

from .models import (
    MODELS, GLOBAL_ELO, LANGUAGE_SPECIALISTS,
    get_model_config, get_all_models
)
from .classifiers import TaskClassifier, ComplexityEstimator, TaskType
from .memory import AssociativeMemory
from .eagle import EagleELO
from .carrot import CARROT
from .context_manager import ContextManager


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RoutingResult:
    """Result of routing decision"""
    model: str  # Selected model ID
    confidence: float  # 0.0 - 1.0
    reasoning: str  # Why this model was chosen
    alternatives: List[Tuple[str, float]]  # [(model, score), ...]
    metadata: Dict[str, Any] = field(default_factory=dict)


class RouterCore:
    """
    Core routing logic for LLM Router v1.4
    Phase 1: Basic routing based on task analysis
    Phase 2: Eagle ELO + Associative Memory
    Phase 3: CARROT Cost-Aware Routing
    Phase 4: Advanced Context Management
    """

    def __init__(
        self,
        enable_eagle: bool = True,
        enable_memory: bool = True,
        enable_carrot: bool = True,
        enable_context_manager: bool = True
    ):
        """
        Initialize router with classifiers and model configs

        Args:
            enable_eagle: Enable Eagle ELO scoring (Phase 2)
            enable_memory: Enable Associative Memory (Phase 2)
            enable_carrot: Enable CARROT cost-aware routing (Phase 3)
            enable_context_manager: Enable advanced context management (Phase 4)
        """
        self.classifier = TaskClassifier()
        self.complexity_estimator = ComplexityEstimator()

        # Phase 2: Initialize Memory and Eagle
        self.enable_eagle = enable_eagle
        self.enable_memory = enable_memory

        if enable_memory:
            try:
                self.memory = AssociativeMemory(embedding_dim=384)
                logger.info("Associative Memory initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Memory: {e}. Running without memory.")
                self.memory = None
                self.enable_memory = False
        else:
            self.memory = None

        if enable_eagle and self.enable_memory:
            try:
                self.eagle = EagleELO(
                    memory=self.memory,
                    global_alpha=0.7,  # 70% global, 30% local
                    k_factor=32
                )
                logger.info("Eagle ELO initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Eagle: {e}. Running without Eagle.")
                self.eagle = None
                self.enable_eagle = False
        else:
            self.eagle = None
            if enable_eagle and not self.enable_memory:
                logger.warning("Eagle requires Memory. Running without Eagle.")
                self.enable_eagle = False

        # Phase 3: Initialize CARROT
        self.enable_carrot = enable_carrot

        if enable_carrot:
            try:
                self.carrot = CARROT()
                logger.info("CARROT cost-aware routing initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize CARROT: {e}. Running without CARROT.")
                self.carrot = None
                self.enable_carrot = False
        else:
            self.carrot = None

        # Phase 4: Initialize Context Manager
        self.enable_context_manager = enable_context_manager

        if enable_context_manager:
            try:
                self.context_manager = ContextManager(
                    enable_progressive=True,
                    enable_compression=True,
                    enable_decay_monitor=True
                )
                logger.info("Context Manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Context Manager: {e}. Using basic context analysis.")
                self.context_manager = None
                self.enable_context_manager = False
        else:
            self.context_manager = None

        logger.info(
            "RouterCore initialized with %d models (Eagle=%s, Memory=%s, CARROT=%s, ContextMgr=%s)",
            len(MODELS), self.enable_eagle, self.enable_memory, self.enable_carrot, self.enable_context_manager
        )

    async def route(
        self,
        query: str,
        session_history: List[str] = None,
        budget: float = None,
        user_id: str = None
    ) -> RoutingResult:
        """
        Main routing function - async version

        Args:
            query: User query/task
            session_history: Previous messages in session
            budget: Optional budget constraint (not used in Phase 1)
            user_id: Optional user ID for personalization (not used in Phase 1)

        Returns:
            RoutingResult with selected model and metadata
        """
        logger.info("Routing query: %s", query[:100])

        # Step 1: Classify the task
        task_analysis = self.classifier.classify(query, session_history)

        # Step 2: Estimate complexity
        complexity_scores = self.complexity_estimator.estimate(query, session_history)

        # Step 3: Analyze context requirements
        context_info = self._analyze_context(query, session_history)

        # Step 4: Select model based on rules
        result = self._select_model(
            query=query,
            task_analysis=task_analysis,
            complexity_scores=complexity_scores,
            context_info=context_info,
            budget=budget
        )

        logger.info(
            "Selected model: %s (confidence: %.2f, reasoning: %s)",
            result.model,
            result.confidence,
            result.reasoning[:100]
        )

        return result

    def route_sync(
        self,
        query: str,
        session_history: List[str] = None,
        budget: float = None,
        user_id: str = None
    ) -> RoutingResult:
        """
        Synchronous version of route() for non-async contexts
        """
        # For Phase 1, we just call the async version
        # In future phases, this will use proper async handling
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.route(query, session_history, budget, user_id)
        )

    def _select_model(
        self,
        query: str,
        task_analysis,
        complexity_scores: Dict[str, float],
        context_info: Dict[str, Any],
        budget: float = None
    ) -> RoutingResult:
        """
        Select best model based on task analysis and constraints

        Decision tree (from SPEC.md):
        1. Check for simple patterns (cascade)
        2. Check if reasoning required
        3. Check language specialization
        4. Check context size requirements
        5. Fall back to ELO-based selection
        """

        candidates_scores = []  # List of (model_id, score, reasoning)

        # Rule 1: Simple pattern detection (Cascade)
        if task_analysis.is_simple_pattern and task_analysis.complexity < 0.3:
            if task_analysis.confidence > 0.8:
                return RoutingResult(
                    model='qwen2.5-coder-7b',
                    confidence=task_analysis.confidence,
                    reasoning="Simple task detected → fast model (cascade routing)",
                    alternatives=self._get_alternatives('qwen2.5-coder-7b'),
                    metadata={
                        'task_type': task_analysis.task_type.value,
                        'complexity': task_analysis.complexity,
                        'routing_strategy': 'cascade'
                    }
                )

        # Rule 2: Complex reasoning required
        if task_analysis.requires_reasoning or complexity_scores['cognitive'] > 0.7:
            return RoutingResult(
                model='qwen3-next-80b',
                confidence=0.85,
                reasoning="Complex reasoning required → 80B thinking model",
                alternatives=self._get_alternatives('qwen3-next-80b'),
                metadata={
                    'task_type': task_analysis.task_type.value,
                    'complexity': complexity_scores['overall'],
                    'cognitive_complexity': complexity_scores['cognitive'],
                    'routing_strategy': 'reasoning'
                }
            )

        # Rule 3: Language specialization (Rust, Go, Kotlin, etc.)
        if task_analysis.detected_language in LANGUAGE_SPECIALISTS:
            specialist = LANGUAGE_SPECIALISTS[task_analysis.detected_language]
            return RoutingResult(
                model=specialist,
                confidence=0.90,
                reasoning=f"Language specialist for {task_analysis.detected_language} → DeepSeek",
                alternatives=self._get_alternatives(specialist),
                metadata={
                    'task_type': task_analysis.task_type.value,
                    'detected_language': task_analysis.detected_language,
                    'routing_strategy': 'language_specialist'
                }
            )

        # Rule 4: Large context requirements
        if context_info['total_size'] > 200_000 or context_info['required_window'] > 200_000:
            return RoutingResult(
                model='glm-4-9b',
                confidence=0.95,
                reasoning=f"Large context required ({context_info['total_size']:,} tokens) → GLM-4 1M",
                alternatives=self._get_alternatives('glm-4-9b'),
                metadata={
                    'task_type': task_analysis.task_type.value,
                    'context_size': context_info['total_size'],
                    'required_window': context_info['required_window'],
                    'routing_strategy': 'large_context'
                }
            )

        # Rule 5: Decay risk mitigation
        if context_info['decay_risk'] > 0.7:
            # Use models with best long-context handling
            return RoutingResult(
                model='glm-4-9b',
                confidence=0.80,
                reasoning=f"High decay risk ({context_info['decay_risk']:.1%}) → GLM-4",
                alternatives=[('qwen3-next-80b', 0.75)],
                metadata={
                    'task_type': task_analysis.task_type.value,
                    'decay_risk': context_info['decay_risk'],
                    'routing_strategy': 'decay_mitigation'
                }
            )

        # Rule 6: Multi-file refactoring
        if task_analysis.task_type == TaskType.MULTI_FILE:
            return RoutingResult(
                model='qwen3-coder-30b',
                confidence=0.75,
                reasoning="Multi-file refactoring → Qwen3 Coder 30B specialist",
                alternatives=self._get_alternatives('qwen3-coder-30b'),
                metadata={
                    'task_type': task_analysis.task_type.value,
                    'routing_strategy': 'multi_file'
                }
            )

        # Phase 3: CARROT budget-aware selection
        if budget is not None and self.enable_carrot:
            logger.info(f"Using CARROT for budget-aware routing (budget={budget})")

            selected_model, prediction = self.carrot.select(
                query=query,
                budget=budget,
                task_type=task_analysis.task_type.value,
                complexity=complexity_scores['overall'],
                context_size=context_info['total_size']
            )

            # Get all predictions for alternatives
            all_predictions = self.carrot.predict_all(
                query, None, task_analysis.task_type.value,
                complexity_scores['overall'], context_info['total_size']
            )

            # Build alternatives list
            alternatives = [
                (model_id, pred['quality'])
                for model_id, pred in all_predictions.items()
                if model_id != selected_model
            ]
            alternatives.sort(key=lambda x: x[1], reverse=True)

            return RoutingResult(
                model=selected_model,
                confidence=prediction['quality'],
                reasoning=(
                    f"CARROT selected (budget={budget:.2f}): "
                    f"quality={prediction['quality']:.3f}, "
                    f"cost={prediction['cost']:.3f}"
                ),
                alternatives=alternatives[:3],
                metadata={
                    'task_type': task_analysis.task_type.value,
                    'complexity': complexity_scores['overall'],
                    'routing_strategy': 'carrot',
                    'budget': budget,
                    'predicted_quality': prediction['quality'],
                    'predicted_cost': prediction['cost']
                }
            )

        # Default: ELO-based selection (Phase 2: use Eagle if available)
        if self.enable_eagle:
            # Phase 2: Use Eagle scoring
            eagle_scores = self._get_eagle_scores(query, task_analysis)
            candidates_scores = list(eagle_scores.items())
        else:
            # Phase 1: Score all models based on Global ELO + task fit
            for model_id in get_all_models():
                score = self._score_model(
                    model_id, task_analysis, complexity_scores, context_info
                )
                candidates_scores.append((model_id, score))

        # Sort by score
        candidates_scores.sort(key=lambda x: x[1], reverse=True)

        best_model = candidates_scores[0][0]
        best_score = candidates_scores[0][1]

        # Determine confidence and strategy
        if self.enable_eagle:
            confidence = best_score  # Eagle scores are already 0-1
            strategy = 'eagle_elo'
            reasoning = f"Eagle ELO score ({best_score:.3f}) for task type: {task_analysis.task_type.value}"
        else:
            confidence = min(0.75, best_score / 2000)  # Normalize Phase 1 ELO to 0-1
            strategy = 'elo_based'
            reasoning = f"Best ELO score ({best_score:.0f}) for task type: {task_analysis.task_type.value}"

        return RoutingResult(
            model=best_model,
            confidence=confidence,
            reasoning=reasoning,
            alternatives=[(m, s) for m, s in candidates_scores[1:4]],
            metadata={
                'task_type': task_analysis.task_type.value,
                'complexity': complexity_scores['overall'],
                'score': best_score,
                'routing_strategy': strategy
            }
        )

    def _score_model(
        self,
        model_id: str,
        task_analysis,
        complexity_scores: Dict[str, float],
        context_info: Dict[str, Any]
    ) -> float:
        """
        Score a model based on task requirements
        Phase 1: Simple scoring based on Global ELO + task fit
        """
        # Start with Global ELO
        score = GLOBAL_ELO.get(model_id, 1700)

        model_config = get_model_config(model_id)

        # Bonus for task type match
        if any(uc in model_config.use_cases for uc in [task_analysis.task_type.value]):
            score += 100

        # Penalty if context too large for model
        if context_info['required_window'] > model_config.context:
            score -= 200

        # Bonus for quality match with complexity
        quality_match = abs(model_config.quality - complexity_scores['overall'])
        if quality_match < 0.2:
            score += 50

        # Speed bonus for simple tasks
        if complexity_scores['overall'] < 0.4 and model_config.speed > 40:
            score += 75

        return score

    def _analyze_context(
        self,
        query: str,
        session_history: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze context requirements

        Returns:
            Dict with context analysis:
            - total_size: Total tokens in context
            - required_window: Estimated required context window
            - complexity: Overall complexity
            - decay_risk: Risk of context decay (0.0 - 1.0)
        """
        # Phase 4: Use advanced Context Manager if enabled
        if self.enable_context_manager and self.context_manager:
            complexity = self.complexity_estimator.estimate(query, session_history)['overall']

            analysis = self.context_manager.analyze_context(
                query=query,
                session_history=session_history,
                complexity=complexity,
                model_id=None  # No specific model yet
            )

            return {
                'total_size': analysis.total_size,
                'required_window': analysis.required_window,
                'complexity': analysis.complexity,
                'decay_risk': analysis.decay_risk,
                'truncation_needed': analysis.truncation_needed,
                'compression_recommended': analysis.compression_recommended,
                'suggested_models': analysis.suggested_models
            }

        # Fallback: Basic context analysis (Phase 1)
        # Calculate total context size
        total_size = len(query)
        if session_history:
            total_size += sum(len(msg) for msg in session_history)

        # Estimate required context window based on complexity
        complexity = self.complexity_estimator.estimate(query, session_history)['overall']

        if complexity < 0.3:
            required_window = 8_000
        elif complexity < 0.7:
            required_window = 32_000
        else:
            required_window = 128_000

        # Estimate decay risk
        decay_risk = self._estimate_decay_risk(total_size)

        return {
            'total_size': total_size,
            'required_window': required_window,
            'complexity': complexity,
            'decay_risk': decay_risk
        }

    def _estimate_decay_risk(self, context_size: int) -> float:
        """
        Estimate context decay risk based on size

        From SPEC.md:
        - < 32K: Low (0.0)
        - 32K - 64K: Medium (0.3)
        - 64K - 128K: High (0.5)
        - > 128K: Critical (0.8)
        """
        if context_size < 32_000:
            return 0.0
        elif context_size < 64_000:
            return 0.3
        elif context_size < 128_000:
            return 0.5
        else:
            return 0.8

    def _get_alternatives(self, selected_model: str) -> List[Tuple[str, float]]:
        """Get alternative models with approximate scores"""
        alternatives = []

        for model_id in get_all_models():
            if model_id != selected_model:
                # Simple scoring based on ELO
                score = GLOBAL_ELO.get(model_id, 1700)
                alternatives.append((model_id, score))

        # Sort by score and return top 3
        alternatives.sort(key=lambda x: x[1], reverse=True)
        return alternatives[:3]

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        config = get_model_config(model_id)

        return {
            'id': model_id,
            'name': config.name,
            'size': config.size,
            'context_window': config.context,
            'speed': config.speed,
            'quality': config.quality,
            'tier': config.tier,
            'use_cases': config.use_cases,
            'frequency': config.frequency,
            'elo': GLOBAL_ELO.get(model_id, 1700)
        }

    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models with their info"""
        return [self.get_model_info(mid) for mid in get_all_models()]

    def _get_eagle_scores(
        self,
        query: str,
        task_analysis,
        k: int = 5
    ) -> Dict[str, float]:
        """
        Get Eagle ELO scores for all models (Phase 2)

        Args:
            query: Query to score for
            task_analysis: Task analysis result
            k: Number of similar queries to consider

        Returns:
            Dict mapping model_id to Eagle score
        """
        if not self.enable_eagle:
            raise RuntimeError("Eagle is not enabled")

        # Use task type as filter for more relevant local scores
        filter_task_type = task_analysis.task_type.value if task_analysis else None

        # Get Eagle scores for all models
        eagle_scores = self.eagle.score_all_models(
            query, k=k, filter_task_type=filter_task_type
        )

        return eagle_scores

    def provide_feedback(
        self,
        query: str,
        selected_model: str,
        success: bool,
        task_type: str,
        complexity: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Provide feedback on routing decision to update Eagle ELO (Phase 2)

        Args:
            query: The query that was routed
            selected_model: Model that was selected
            success: Whether the routing was successful
            task_type: Type of task
            complexity: Complexity score
            metadata: Optional additional metadata
        """
        if not self.enable_eagle or not self.enable_memory:
            logger.debug("Feedback ignored: Eagle/Memory not enabled")
            return

        # Update Eagle ELO (this will also add to memory)
        self.eagle.update_from_feedback(
            model_id=selected_model,
            query=query,
            success=success,
            task_type=task_type,
            complexity=complexity
        )

        logger.info(
            f"Feedback recorded: model={selected_model}, success={success}, "
            f"memory_size={self.memory.size()}"
        )

    def get_eagle_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get Eagle ELO statistics (Phase 2)

        Returns:
            Dict with Eagle stats, or None if Eagle not enabled
        """
        if not self.enable_eagle:
            return None

        eagle_stats = self.eagle.get_stats()
        memory_stats = self.memory.get_stats() if self.enable_memory else {}

        return {
            'eagle': eagle_stats,
            'memory': memory_stats
        }
