"""
Unified LLM Router v2.0
Complete integration of v1.4 (6 phases) + v2.0 (10 components)

21 components, 429 tests
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# v1.4 imports (Phase 1-6)
from router.router_core import RouterCore, RoutingResult
from router.eagle import EagleELO
from router.carrot import CARROT
from router.memory import AssociativeMemory
from router.episodic_memory import EpisodicMemory
from router.cascade_router import CascadeRouter
from router.multi_round import MultiRoundRouter
from router.monitoring import PerformanceTracker
from router.context_manager import ContextManager as ContextManagerV1

# v2.0 imports (Component 1-10)
from src.v2.context_manager import ContextManager as ContextManagerV2
from src.v2.runtime_adapter import RuntimeAdapter, LoadLevel
from src.v2.self_check import SelfCheckSystem
from src.v2.context_sizing import DynamicContextSizer, ContextSizingStrategy
from src.v2.hierarchical_pruning import HierarchicalPruner
from src.v2.batching_layer import BatchingLayer, Priority
from src.v2.ast_analyzer import ASTAnalyzer
from src.v2.recursive_compressor import RecursiveCompressor
from src.v2.environment_prompting import EnvironmentPrompter
from src.v2.state_snapshot import StateSnapshotSystem


logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Routing strategy selection"""
    QUALITY_FOCUSED = "quality_focused"  # Eagle ELO
    COST_AWARE = "cost_aware"  # CARROT
    BALANCED = "balanced"  # Hybrid
    CASCADE = "cascade"  # Cascade routing


@dataclass
class UnifiedRequest:
    """Unified routing request"""
    query: str
    context: Optional[str] = None
    user_id: Optional[str] = None
    priority: Priority = Priority.NORMAL
    strategy: RoutingStrategy = RoutingStrategy.BALANCED
    max_cost: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedResponse:
    """Unified routing response"""
    model: str
    result: str
    quality_score: float
    cost: float
    latency: float
    confidence: float
    reasoning: str
    passed_quality_check: bool
    snapshot_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedRouter:
    """
    LLM Router v2.0 - Complete System

    Integrates:
    - v1.4: RouterCore, Eagle, CARROT, Memory, Cascade, MultiRound, Monitoring
    - v2.0: ContextManager, Runtime, SelfCheck, ContextSizing, Pruning,
            Batching, AST, Compression, Environment, StateSnapshot

    Total: 21 components
    """

    def __init__(
        self,
        enable_batching: bool = True,
        enable_quality_check: bool = True,
        enable_snapshots: bool = True,
        enable_monitoring: bool = True
    ):
        """
        Initialize Unified Router

        Args:
            enable_batching: Enable request batching
            enable_quality_check: Enable quality verification
            enable_snapshots: Enable state snapshots
            enable_monitoring: Enable monitoring/telemetry
        """
        logger.info("Initializing Unified Router v2.0...")

        # Configuration
        self.enable_batching = enable_batching
        self.enable_quality_check = enable_quality_check
        self.enable_snapshots = enable_snapshots
        self.enable_monitoring = enable_monitoring

        # v1.4 Components
        self.memory = AssociativeMemory()
        self.episodic = EpisodicMemory()
        self.eagle = EagleELO(memory=self.memory)
        self.carrot = CARROT()
        self.cascade = CascadeRouter()
        self.multi_round = MultiRoundRouter()
        self.context_v1 = ContextManagerV1()
        self.core = RouterCore(
            enable_eagle=True,
            enable_carrot=True,
            enable_memory=True
        )

        if self.enable_monitoring:
            self.monitoring = PerformanceTracker()

        # v2.0 Components
        self.context_v2 = ContextManagerV2()
        self.runtime = RuntimeAdapter()
        self.quality = SelfCheckSystem()
        self.context_sizer = DynamicContextSizer()
        self.pruner = HierarchicalPruner()
        self.compressor = RecursiveCompressor()
        self.ast_analyzer = ASTAnalyzer()
        self.env_prompter = EnvironmentPrompter()

        if self.enable_snapshots:
            self.snapshot = StateSnapshotSystem()

        if self.enable_batching:
            self.batching = BatchingLayer()

        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "quality_retries": 0,
            "avg_latency": 0.0,
            "total_cost": 0.0
        }

        logger.info("Unified Router v2.0 initialized successfully")

    async def route(
        self,
        request: UnifiedRequest
    ) -> UnifiedResponse:
        """
        Main routing pipeline

        Pipeline:
        1. Runtime adaptation (v2)
        2. Context sizing (v2)
        3. Context management (v2)
        4. Model routing (v1.4: Eagle/CARROT/Cascade)
        5. Execution
        6. Quality check (v2)
        7. State snapshot (v2)

        Args:
            request: UnifiedRequest

        Returns:
            UnifiedResponse
        """
        start_time = datetime.now()
        self.stats["total_requests"] += 1

        try:
            # Step 1: Runtime adaptation
            system_metrics = self.runtime.measure_system_load()
            adaptation = await self._adapt_strategy(request, system_metrics)

            # Step 2: Context sizing
            budget = self.context_sizer.calculate_budget(
                query=request.query,
                model_name="gpt-4",  # Default, will adjust
                complexity=0.5,  # TODO: calculate from query
                history=[],
                strategy=ContextSizingStrategy.ADAPTIVE
            )

            # Step 3: Context assembly (v2)
            # v2 ContextManager handles caching internally
            context_response = await self.context_v2.get_context(
                query=request.query,
                max_tokens=budget.total_tokens,
                complexity=0.5  # TODO: calculate from query
            )
            assembled_context = context_response.context

            # Step 4: Model routing (v1.4)
            routing_result = await self._route_model(
                request,
                adaptation["strategy"],
                assembled_context
            )

            # Step 5: Execute request
            result = await self._execute_request(
                request.query,
                routing_result.model,
                assembled_context
            )

            # Step 6: Quality check (v2)
            quality_result = None
            passed = True

            if self.enable_quality_check:
                quality_result = self.quality.check(
                    query=request.query,
                    result=result,
                    context=assembled_context
                )
                passed = quality_result.passed

                # Retry if quality check failed
                if not passed and self.stats["quality_retries"] < 3:
                    self.stats["quality_retries"] += 1
                    issues_str = ", ".join(quality_result.issues) if quality_result.issues else "No specific issues"
                    logger.warning(f"Quality check failed, retrying... ({issues_str})")

                    # Retry with better model
                    result = await self._retry_with_better_model(
                        request,
                        routing_result,
                        assembled_context
                    )

                    # Re-check quality
                    quality_result = self.quality.check(
                        query=request.query,
                        result=result,
                        context=assembled_context
                    )
                    passed = quality_result.passed

            # Step 7: State snapshot (v2)
            snapshot_id = None
            if self.enable_snapshots:
                snapshot = self.snapshot.capture_snapshot({
                    "context_v2": self.context_v2,
                    "runtime": self.runtime,
                    "quality": self.quality,
                    "context_sizer": self.context_sizer,
                    "memory": self.memory,
                    "eagle": self.eagle,
                    "carrot": self.carrot
                })
                snapshot_id = f"snapshot_{snapshot.timestamp.isoformat()}"

            # Calculate metrics
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds()

            # Update episodic memory
            overall_score = (quality_result.focus_score + quality_result.result_score) / 2 if quality_result else 0.5
            self.episodic.add_episode(
                query=request.query,
                selected_model=routing_result.model,
                confidence=routing_result.confidence,
                success=passed,
                task_type="general",
                complexity=0.5
            )

            # Update stats
            self.stats["successful_requests"] += 1
            self._update_avg_latency(latency)

            # Create response
            response = UnifiedResponse(
                model=routing_result.model,
                result=result,
                quality_score=overall_score / 10.0 if quality_result else 0.8,  # Normalize to 0-1
                cost=0.0,  # TODO: calculate actual cost
                latency=latency,
                confidence=routing_result.confidence,
                reasoning=routing_result.reasoning,
                passed_quality_check=passed,
                snapshot_id=snapshot_id,
                metadata={
                    "load_level": system_metrics.load_level.value,
                    "budget": budget.total_tokens,
                    "strategy": adaptation["strategy"].value
                }
            )

            return response

        except Exception as e:
            logger.error(f"Routing failed: {e}")
            self.stats["failed_requests"] += 1
            raise

    async def _adapt_strategy(
        self,
        request: UnifiedRequest,
        metrics
    ) -> Dict[str, Any]:
        """Adapt routing strategy based on system load"""

        # High load → cost-aware routing
        if metrics.load_level == LoadLevel.CRITICAL:
            return {
                "strategy": RoutingStrategy.COST_AWARE,
                "reason": "High system load, prioritizing efficiency"
            }

        # Low load → quality-focused routing
        elif metrics.load_level == LoadLevel.LOW:
            return {
                "strategy": RoutingStrategy.QUALITY_FOCUSED,
                "reason": "Low system load, prioritizing quality"
            }

        # Use requested strategy
        return {
            "strategy": request.strategy,
            "reason": "Using requested strategy"
        }

    async def _assemble_context(
        self,
        request: UnifiedRequest,
        max_tokens: int
    ) -> str:
        """Assemble context from memory and request"""

        context_parts = []

        # Add system context
        sys_summary = self.env_prompter.get_context_summary()
        context_parts.append(sys_summary)

        # Add request context
        if request.context:
            context_parts.append(request.context)

        # Add relevant memories
        if request.query:
            memories = self.memory.search(request.query, k=3)
            if memories:
                context_parts.append("\nRelevant memories:")
                for mem in memories:
                    context_parts.append(f"- {mem['content'][:100]}")

        # Join and compress if needed
        full_context = "\n".join(context_parts)

        if len(full_context) > max_tokens * 4:  # ~4 chars per token
            compressed = self.compressor.compress(
                full_context,
                target_size=max_tokens * 4
            )
            return compressed.compressed_text

        return full_context

    async def _route_model(
        self,
        request: UnifiedRequest,
        strategy: RoutingStrategy,
        context: str
    ) -> RoutingResult:
        """Route to best model using v1.4 routers"""

        if strategy == RoutingStrategy.QUALITY_FOCUSED:
            # Use Eagle ELO
            model, score = self.eagle.get_best_model(
                query=request.query
            )
            return RoutingResult(
                model=model,
                confidence=score / 2000,  # Normalize ELO
                reasoning="Selected by Eagle ELO",
                alternatives=[]
            )

        elif strategy == RoutingStrategy.COST_AWARE:
            # Use CARROT
            model, predictions = self.carrot.select(
                query=request.query,
                budget=request.max_cost or 1.0
            )
            return RoutingResult(
                model=model,
                confidence=predictions.get("quality", 0.8),
                reasoning="Selected by CARROT cost optimization",
                alternatives=[]
            )

        elif strategy == RoutingStrategy.CASCADE:
            # Use Cascade Router
            result = self.cascade.route(
                query=request.query,
                task_type="general",
                complexity=0.5
            )
            return RoutingResult(
                model=result.model,
                confidence=0.8,
                reasoning="Selected by Cascade Router",
                alternatives=[]
            )

        else:  # BALANCED
            # Use RouterCore (hybrid)
            return await self.core.route(
                query=request.query,
                user_id=request.user_id
            )

    async def _execute_request(
        self,
        query: str,
        model: str,
        context: str
    ) -> str:
        """
        Execute request with selected model

        NOTE: This is a mock implementation
        In production, would call actual LLM API
        """
        # Mock response
        return f"Response from {model} for query: {query[:50]}..."

    async def _retry_with_better_model(
        self,
        request: UnifiedRequest,
        original_routing: RoutingResult,
        context: str
    ) -> str:
        """Retry with a better model after quality check failure"""

        # Select better model (e.g., upgrade to GPT-4)
        better_model = "gpt-4" if original_routing.model != "gpt-4" else "claude-3-opus"

        logger.info(f"Retrying with better model: {better_model}")

        return await self._execute_request(
            request.query,
            better_model,
            context
        )

    def _update_avg_latency(self, latency: float):
        """Update average latency statistics"""
        n = self.stats["successful_requests"]
        current_avg = self.stats["avg_latency"]
        self.stats["avg_latency"] = (current_avg * (n - 1) + latency) / n

    async def optimize(self):
        """
        Run optimization across all components

        - Eagle ELO training (v1.4)
        - CARROT Pareto optimization (v1.4)
        - Batch size optimization (v2.0)
        """
        logger.info("Running system optimization...")

        # Eagle optimization
        self.eagle.update_rating(
            winner_model="gpt-4",
            loser_model="claude-3",
            actual_result=1.0
        )

        # CARROT optimization
        # (Would run Pareto frontier optimization here)

        # Batch optimization
        if self.enable_batching:
            # Adjust batch size based on throughput
            stats = self.batching.get_stats()
            if stats["avg_throughput"] < 10:
                # Increase batch size
                pass

        logger.info("Optimization complete")

    def get_all_components(self) -> Dict[str, Any]:
        """Get all component instances for snapshot/introspection"""
        return {
            # v1.4 components
            "router_core": self.core,
            "eagle": self.eagle,
            "carrot": self.carrot,
            "memory": self.memory,
            "episodic": self.episodic,
            "cascade": self.cascade,
            "multi_round": self.multi_round,
            "context_v1": self.context_v1,

            # v2.0 components
            "context_v2": self.context_v2,
            "runtime": self.runtime,
            "quality": self.quality,
            "context_sizer": self.context_sizer,
            "pruner": self.pruner,
            "batching": self.batching if self.enable_batching else None,
            "ast_analyzer": self.ast_analyzer,
            "compressor": self.compressor,
            "env_prompter": self.env_prompter,
            "snapshot": self.snapshot if self.enable_snapshots else None,
            "monitoring": self.monitoring if self.enable_monitoring else None
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components"""
        stats = {
            "unified": self.stats.copy(),
            "context_v2": self.context_v2.get_statistics(),  # v2 uses get_statistics()
            "runtime": self.runtime.get_stats(),
            "quality": self.quality.get_stats(),
            "context_sizer": self.context_sizer.get_stats(),
            "pruner": self.pruner.get_stats(),
            "compressor": self.compressor.get_stats(),
            "env_prompter": self.env_prompter.get_stats(),
        }

        if self.enable_batching:
            stats["batching"] = self.batching.get_stats()

        if self.enable_snapshots:
            stats["snapshot"] = self.snapshot.get_stats()

        return stats

    def reset_stats(self):
        """Reset all component statistics"""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "quality_retries": 0,
            "avg_latency": 0.0,
            "total_cost": 0.0
        }

        # Reset all v2.0 component stats
        self.context_v2.clear_cache()  # v2 uses clear_cache()
        self.runtime.reset_stats()
        self.quality.reset_stats()
        self.context_sizer.reset_stats()
        self.pruner.reset_stats()
        self.compressor.reset_stats()
        self.env_prompter.reset_stats()

        if self.enable_batching:
            self.batching.reset_stats()

        if self.enable_snapshots:
            self.snapshot.reset_stats()
