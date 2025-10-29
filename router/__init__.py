"""
LLM Router v1.4
Intelligent routing system for 6 local MLX models
Phase 1: Core Routing
Phase 2: Eagle ELO + Associative Memory
Phase 3: CARROT Cost-Aware Routing
"""

from .router_core import RouterCore, RoutingResult
from .models import MODELS, ModelConfig
from .memory import AssociativeMemory, RoutingEvent
from .eagle import EagleELO
from .carrot import CARROT, QualityPredictor, CostPredictor, Prediction

__version__ = "1.4.0"
__all__ = [
    "RouterCore",
    "RoutingResult",
    "MODELS",
    "ModelConfig",
    "AssociativeMemory",
    "RoutingEvent",
    "EagleELO",
    "CARROT",
    "QualityPredictor",
    "CostPredictor",
    "Prediction"
]
