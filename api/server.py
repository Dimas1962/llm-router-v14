"""
FastAPI REST API Server
Phase 5: Production API Integration
"""

import time
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from router import RouterCore
from .models import (
    RouteRequest, RouteResponse,
    FeedbackRequest, FeedbackResponse,
    ModelsResponse, ModelInfo,
    StatsResponse,
    HealthResponse,
    ErrorResponse
)
from .config import get_config


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
router: Optional[RouterCore] = None
start_time: float = 0


def initialize_router():
    """Initialize the router instance"""
    global router, start_time

    if router is not None:
        return

    logger.info("Starting LLM Router API...")
    config = get_config()

    try:
        router = RouterCore(
            enable_eagle=config.enable_eagle,
            enable_memory=config.enable_memory,
            enable_carrot=config.enable_carrot,
            enable_context_manager=config.enable_context_manager
        )
        start_time = time.time()
        logger.info("Router initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize router: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown
    """
    # Startup
    initialize_router()

    yield

    # Shutdown
    logger.info("Shutting down LLM Router API...")


# Create FastAPI app
config = get_config()
app = FastAPI(
    title=config.title,
    description=config.description,
    version=config.version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
if config.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=config.cors_methods,
        allow_headers=config.cors_headers,
    )


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="ValidationError",
            message="Request validation failed",
            details={"errors": exc.errors()}
        ).model_dump()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPError",
            message=exc.detail,
            details={"status_code": exc.status_code}
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message=str(exc),
            details={"type": type(exc).__name__}
        ).model_dump()
    )


# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": config.title,
        "version": config.version,
        "description": config.description,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint

    Returns server health status and component availability
    """
    global router, start_time

    if router is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Router not initialized"
        )

    uptime = time.time() - start_time

    components = {
        "router": router is not None,
        "eagle": router.enable_eagle if router else False,
        "memory": router.enable_memory if router else False,
        "carrot": router.enable_carrot if router else False,
        "context_manager": router.enable_context_manager if router else False
    }

    return HealthResponse(
        status="healthy",
        version=config.version,
        uptime_seconds=uptime,
        components=components
    )


@app.post("/route", response_model=RouteResponse, tags=["Routing"])
async def route_query(request: RouteRequest):
    """
    Route a query to the best model

    This endpoint uses all phases (1-4) to select the optimal model:
    - Phase 1: Task classification and complexity estimation
    - Phase 2: Eagle ELO scoring and associative memory
    - Phase 3: CARROT cost-aware selection (if budget provided)
    - Phase 4: Advanced context management

    Returns the selected model with confidence score and reasoning.
    """
    global router

    if router is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Router not initialized"
        )

    try:
        start = time.time()

        # Route the query
        result = await router.route(
            query=request.query,
            session_history=request.session_history,
            budget=request.budget,
            user_id=request.user_id
        )

        routing_time = (time.time() - start) * 1000  # Convert to ms

        # Format alternatives
        alternatives = [
            {"model": model_id, "score": score}
            for model_id, score in result.alternatives
        ]

        return RouteResponse(
            model=result.model,
            confidence=result.confidence,
            reasoning=result.reasoning,
            alternatives=alternatives,
            metadata=result.metadata,
            routing_time_ms=routing_time
        )

    except Exception as e:
        logger.error(f"Routing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Routing failed: {str(e)}"
        )


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback on routing decision

    This endpoint allows clients to provide feedback on routing decisions,
    which is used to update Eagle ELO ratings and improve future routing.

    Feedback helps the router learn and improve over time.
    """
    global router

    if router is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Router not initialized"
        )

    if not router.enable_eagle or not router.enable_memory:
        return FeedbackResponse(
            status="skipped",
            message="Feedback ignored: Eagle/Memory not enabled",
            elo_updated=False,
            memory_size=None
        )

    try:
        # Provide feedback to router
        router.provide_feedback(
            query=request.query,
            selected_model=request.selected_model,
            success=request.success,
            task_type=request.task_type,
            complexity=request.complexity,
            metadata=request.metadata
        )

        memory_size = router.memory.size() if router.memory else None

        return FeedbackResponse(
            status="success",
            message="Feedback recorded successfully",
            elo_updated=True,
            memory_size=memory_size
        )

    except Exception as e:
        logger.error(f"Feedback error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feedback submission failed: {str(e)}"
        )


@app.get("/models", response_model=ModelsResponse, tags=["Models"])
async def list_models():
    """
    List all available models

    Returns information about all configured models including:
    - Model specifications (size, context window, speed)
    - Quality metrics
    - Current ELO ratings
    - Use case recommendations
    """
    global router

    if router is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Router not initialized"
        )

    try:
        models_info = router.list_models()

        models = [
            ModelInfo(
                id=info['id'],
                name=info['name'],
                size=info['size'],
                context_window=info['context_window'],
                speed=info['speed'],
                quality=info['quality'],
                tier=info['tier'],
                use_cases=info['use_cases'],
                frequency=info['frequency'],
                elo=info['elo']
            )
            for info in models_info
        ]

        return ModelsResponse(
            models=models,
            total_count=len(models)
        )

    except Exception as e:
        logger.error(f"Models listing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@app.get("/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_statistics():
    """
    Get router statistics

    Returns comprehensive statistics including:
    - Component status (Eagle, Memory, CARROT, Context Manager)
    - Eagle ELO ratings
    - Memory statistics
    - Model information
    """
    global router

    if router is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Router not initialized"
        )

    try:
        # Get Eagle stats if available
        stats_data = router.get_eagle_stats() if router.enable_eagle else None

        return StatsResponse(
            eagle_enabled=router.enable_eagle,
            memory_enabled=router.enable_memory,
            carrot_enabled=router.enable_carrot,
            context_manager_enabled=router.enable_context_manager,
            eagle_stats=stats_data.get('eagle') if stats_data else None,
            memory_stats=stats_data.get('memory') if stats_data else None,
            model_count=len(router.list_models())
        )

    except Exception as e:
        logger.error(f"Statistics error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )


# Main entry point
def main():
    """Main entry point for running the server"""
    import uvicorn

    config = get_config()

    uvicorn.run(
        "api.server:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        workers=config.workers,
        log_level=config.log_level.lower()
    )


if __name__ == "__main__":
    main()
