"""
API Configuration
Phase 5: Production API Integration
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class APIConfig(BaseSettings):
    """API Server Configuration"""

    # Server settings
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    reload: bool = Field(default=False, description="Enable auto-reload")
    workers: int = Field(default=1, description="Number of worker processes")

    # CORS settings
    enable_cors: bool = Field(default=True, description="Enable CORS")
    cors_origins: list = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    cors_methods: list = Field(
        default=["*"],
        description="Allowed CORS methods"
    )
    cors_headers: list = Field(
        default=["*"],
        description="Allowed CORS headers"
    )

    # Router settings
    enable_eagle: bool = Field(default=True, description="Enable Eagle ELO")
    enable_memory: bool = Field(default=True, description="Enable Associative Memory")
    enable_carrot: bool = Field(default=True, description="Enable CARROT")
    enable_context_manager: bool = Field(default=True, description="Enable Context Manager")

    # Memory settings
    memory_embedding_dim: int = Field(default=384, description="Embedding dimension")
    memory_max_size: int = Field(default=100000, description="Max memory size")

    # Logging settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )

    # API metadata
    title: str = Field(default="LLM Router v1.4 API", description="API title")
    description: str = Field(
        default="Intelligent routing system for 5 local MLX models",
        description="API description"
    )
    version: str = Field(default="1.4.0", description="API version")

    # Rate limiting (optional, not implemented yet)
    rate_limit_enabled: bool = Field(default=False, description="Enable rate limiting")
    rate_limit_per_minute: int = Field(default=60, description="Requests per minute")

    class Config:
        env_prefix = "ROUTER_"
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global config instance
config = APIConfig()


def get_config() -> APIConfig:
    """Get API configuration"""
    return config


def update_config(**kwargs) -> APIConfig:
    """Update configuration with new values"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
