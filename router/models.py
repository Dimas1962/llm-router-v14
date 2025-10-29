"""
Model Configuration
Defines the 6 MLX models with their capabilities and characteristics
"""

from dataclasses import dataclass
from typing import List, Literal


@dataclass
class ModelConfig:
    """Configuration for a single MLX model"""
    name: str
    size: str
    context: int
    speed: int  # tokens per second
    quality: float  # 0.0 - 1.0
    tier: Literal['primary', 'reasoning', 'specialist', 'multilang', 'fast']
    use_cases: List[str]
    frequency: float  # expected usage frequency
    quality_rust: float = None  # special quality for Rust/Go (DeepSeek only)


# 6 MLX Models Configuration
MODELS = {
    'glm-4-9b': ModelConfig(
        name='GLM-4-9B-Chat-1M-BF16',
        size='18.99GB',
        context=1_000_000,  # 1M tokens!
        speed=45,
        quality=0.87,  # vs GPT-4o
        tier='primary',
        use_cases=['coding', 'refactoring', 'bug_fixing', 'large_context'],
        frequency=0.75  # 75% usage
    ),

    'qwen3-next-80b': ModelConfig(
        name='Qwen3-Next-80B-A3B-Thinking',
        size='60GB',
        context=200_000,
        speed=12,
        quality=0.90,
        tier='reasoning',
        use_cases=['architecture', 'complex_reasoning', 'thinking', 'design'],
        frequency=0.18  # 18% usage
    ),

    'qwen3-coder-30b': ModelConfig(
        name='Qwen3-Coder-30B-A3B',
        size='20GB',
        context=128_000,
        speed=22,
        quality=0.80,
        tier='specialist',
        use_cases=['large_refactoring', 'multi_file_changes'],
        frequency=0.07  # 7% usage (niche)
    ),

    'deepseek-coder-16b': ModelConfig(
        name='DeepSeek-Coder-V2-Lite',
        size='10GB',
        context=64_000,
        speed=32,
        quality=0.77,  # Python
        quality_rust=0.90,  # Rust/Go!
        tier='multilang',
        use_cases=['rust', 'go', 'kotlin', 'exotic_langs'],
        frequency=0.08  # 8% usage (conditional)
    ),

    'qwen2.5-coder-7b': ModelConfig(
        name='Qwen2.5-Coder-7B',
        size='5GB',
        context=32_000,
        speed=60,
        quality=0.70,
        tier='fast',
        use_cases=['quick_snippets', 'real_time', 'simple_tasks'],
        frequency=0.12  # 12% usage
    )
}


# Initial Global ELO ratings (from benchmarks)
GLOBAL_ELO = {
    'glm-4-9b': 1850,           # High (quality + 1M context)
    'qwen3-next-80b': 1900,     # Highest (reasoning)
    'qwen3-coder-30b': 1700,    # Medium (overlap)
    'deepseek-coder-16b': 1750, # Conditional
    'qwen2.5-coder-7b': 1650    # Fast but lower quality
}


# Language specialization mapping
LANGUAGE_SPECIALISTS = {
    'rust': 'deepseek-coder-16b',
    'go': 'deepseek-coder-16b',
    'kotlin': 'deepseek-coder-16b',
    'swift': 'deepseek-coder-16b',
    'scala': 'deepseek-coder-16b',
    'julia': 'deepseek-coder-16b',
}


def get_model_config(model_id: str) -> ModelConfig:
    """Get configuration for a specific model"""
    if model_id not in MODELS:
        raise ValueError(f"Unknown model: {model_id}")
    return MODELS[model_id]


def get_all_models() -> List[str]:
    """Get list of all available model IDs"""
    return list(MODELS.keys())


def get_models_by_tier(tier: str) -> List[str]:
    """Get models filtered by tier"""
    return [mid for mid, cfg in MODELS.items() if cfg.tier == tier]


def get_models_for_use_case(use_case: str) -> List[str]:
    """Get models that support a specific use case"""
    return [mid for mid, cfg in MODELS.items() if use_case in cfg.use_cases]
