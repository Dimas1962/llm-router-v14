"""
API Models - Pydantic request/response schemas
Phase 5: Production API Integration
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class RouteRequest(BaseModel):
    """Request model for routing queries"""
    query: str = Field(..., description="The query/task to route", min_length=1)
    session_history: Optional[List[str]] = Field(
        default=None,
        description="Previous messages in the session"
    )
    budget: Optional[float] = Field(
        default=None,
        description="Optional budget constraint for CARROT",
        ge=0
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Optional user ID for personalization"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Write a Python function to sort a list",
                "session_history": ["previous message 1", "previous message 2"],
                "budget": 50.0,
                "user_id": "user123"
            }
        }


class RouteResponse(BaseModel):
    """Response model for routing results"""
    model: str = Field(..., description="Selected model ID")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)", ge=0, le=1)
    reasoning: str = Field(..., description="Explanation for model selection")
    alternatives: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Alternative models with scores"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional routing metadata"
    )
    routing_time_ms: float = Field(..., description="Time taken for routing (ms)")

    class Config:
        json_schema_extra = {
            "example": {
                "model": "qwen2.5-coder-7b",
                "confidence": 0.85,
                "reasoning": "Simple task detected → fast model",
                "alternatives": [
                    {"model": "glm-4-9b", "score": 0.75},
                    {"model": "qwen3-coder-30b", "score": 0.70}
                ],
                "metadata": {
                    "task_type": "quick_snippet",
                    "complexity": 0.25,
                    "routing_strategy": "cascade"
                },
                "routing_time_ms": 45.2
            }
        }


class FeedbackRequest(BaseModel):
    """Request model for providing feedback"""
    query: str = Field(..., description="The query that was routed")
    selected_model: str = Field(..., description="Model that was selected")
    success: bool = Field(..., description="Whether routing was successful")
    task_type: str = Field(..., description="Type of task")
    complexity: float = Field(..., description="Complexity score", ge=0, le=1)
    satisfaction: Optional[float] = Field(
        default=None,
        description="User satisfaction (0.0-1.0)",
        ge=0,
        le=1
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional feedback metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Write a Python function",
                "selected_model": "qwen2.5-coder-7b",
                "success": True,
                "task_type": "coding",
                "complexity": 0.3,
                "satisfaction": 0.9,
                "metadata": {"response_time": 2.5}
            }
        }


class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""
    status: str = Field(..., description="Status of feedback processing")
    message: str = Field(..., description="Response message")
    elo_updated: bool = Field(..., description="Whether ELO was updated")
    memory_size: Optional[int] = Field(
        default=None,
        description="Current size of associative memory"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Feedback recorded successfully",
                "elo_updated": True,
                "memory_size": 152
            }
        }


class ModelInfo(BaseModel):
    """Model information schema"""
    id: str = Field(..., description="Model ID")
    name: str = Field(..., description="Full model name")
    size: str = Field(..., description="Model size")
    context_window: int = Field(..., description="Context window size")
    speed: float = Field(..., description="Speed (tokens/sec)")
    quality: float = Field(..., description="Quality score (0.0-1.0)")
    tier: str = Field(..., description="Model tier")
    use_cases: List[str] = Field(..., description="Recommended use cases")
    frequency: float = Field(..., description="Usage frequency")
    elo: int = Field(..., description="Current ELO rating")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "qwen2.5-coder-7b",
                "name": "Qwen2.5-Coder-7B",
                "size": "5GB",
                "context_window": 32000,
                "speed": 60.0,
                "quality": 0.70,
                "tier": "fast",
                "use_cases": ["quick_snippets", "real_time"],
                "frequency": 0.12,
                "elo": 1650
            }
        }


class ModelsResponse(BaseModel):
    """Response model for listing models"""
    models: List[ModelInfo] = Field(..., description="List of available models")
    total_count: int = Field(..., description="Total number of models")

    class Config:
        json_schema_extra = {
            "example": {
                "models": [
                    {
                        "id": "qwen2.5-coder-7b",
                        "name": "Qwen2.5-Coder-7B",
                        "size": "5GB",
                        "context_window": 32000,
                        "speed": 60.0,
                        "quality": 0.70,
                        "tier": "fast",
                        "use_cases": ["quick_snippets"],
                        "frequency": 0.12,
                        "elo": 1650
                    }
                ],
                "total_count": 5
            }
        }


class StatsResponse(BaseModel):
    """Response model for router statistics"""
    eagle_enabled: bool = Field(..., description="Whether Eagle ELO is enabled")
    memory_enabled: bool = Field(..., description="Whether Memory is enabled")
    carrot_enabled: bool = Field(..., description="Whether CARROT is enabled")
    context_manager_enabled: bool = Field(..., description="Whether Context Manager is enabled")

    eagle_stats: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Eagle ELO statistics"
    )
    memory_stats: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Memory statistics"
    )
    model_count: int = Field(..., description="Number of available models")

    class Config:
        json_schema_extra = {
            "example": {
                "eagle_enabled": True,
                "memory_enabled": True,
                "carrot_enabled": True,
                "context_manager_enabled": True,
                "eagle_stats": {
                    "global_elo": {
                        "qwen3-next-80b": 1900,
                        "glm-4-9b": 1850
                    }
                },
                "memory_stats": {
                    "size": 152,
                    "capacity": 100000
                },
                "model_count": 5
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Health status")
    version: str = Field(..., description="Router version")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    components: Dict[str, bool] = Field(..., description="Component health status")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.4.0",
                "uptime_seconds": 3600.5,
                "components": {
                    "router": True,
                    "eagle": True,
                    "memory": True,
                    "carrot": True,
                    "context_manager": True
                }
            }
        }


class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Query field is required",
                "details": {"field": "query"}
            }
        }
