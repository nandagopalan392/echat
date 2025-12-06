"""
Configuration for the RAG evaluation system
"""

import os
from typing import Dict, Any, Optional

class EvaluationConfig:
    """Configuration class for evaluation system"""
    
    # LLM Provider settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

    EVALUATION_MODEL = os.getenv("EVALUATION_MODEL", "gemma2:2b")
    
    # Evaluation settings
    EVALUATION_TEMPERATURE = float(os.getenv("EVALUATION_TEMPERATURE", "0.0"))
    EVALUATION_MAX_TOKENS = int(os.getenv("EVALUATION_MAX_TOKENS", "512"))
    EVALUATION_TIMEOUT = int(os.getenv("EVALUATION_TIMEOUT", "120"))
    
    # Scoring settings
    MIN_SCORE = 0
    MAX_SCORE = 10
    
    # Metric weights for overall score calculation
    METRIC_WEIGHTS = {
        "groundedness": 0.4,      # Most important - answers must be grounded
        "answer_relevance": 0.35, # Second most important - answers must be relevant
        "context_relevance": 0.25 # Supporting metric - context quality
    }
    
    # Quality thresholds
    QUALITY_THRESHOLDS = {
        "excellent": 0.8,
        "good": 0.6,
        "fair": 0.4,
        "poor": 0.2
    }
    
    # Background evaluation settings
    ENABLE_BACKGROUND_EVALUATION = os.getenv("ENABLE_BACKGROUND_EVALUATION", "true").lower() == "true"
    MAX_EVALUATION_HISTORY = int(os.getenv("MAX_EVALUATION_HISTORY", "1000"))
    
    # Celery and Redis settings
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/1")
    
    # WebSocket settings
    WEBSOCKET_TIMEOUT = int(os.getenv("WEBSOCKET_TIMEOUT", "300"))  # 5 minutes
    WEBSOCKET_PING_INTERVAL = int(os.getenv("WEBSOCKET_PING_INTERVAL", "30"))  # 30 seconds
    
    # Task execution settings
    EVALUATION_TASK_TIMEOUT = int(os.getenv("EVALUATION_TASK_TIMEOUT", "600"))  # 10 minutes
    BATCH_EVALUATION_MAX_SIZE = int(os.getenv("BATCH_EVALUATION_MAX_SIZE", "100"))
    
    # Result caching settings
    RESULT_CACHE_TIMEOUT = int(os.getenv("RESULT_CACHE_TIMEOUT", "3600"))  # 1 hour
    
    # Evaluation prompts customization
    CUSTOM_PROMPTS_ENABLED = os.getenv("CUSTOM_PROMPTS_ENABLED", "false").lower() == "true"
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """Get all configuration as a dictionary"""
        return {
            "ollama_base_url": cls.OLLAMA_BASE_URL,
            "evaluation_model": cls.EVALUATION_MODEL,
            "evaluation_temperature": cls.EVALUATION_TEMPERATURE,
            "evaluation_max_tokens": cls.EVALUATION_MAX_TOKENS,
            "evaluation_timeout": cls.EVALUATION_TIMEOUT,
            "min_score": cls.MIN_SCORE,
            "max_score": cls.MAX_SCORE,
            "metric_weights": cls.METRIC_WEIGHTS,
            "quality_thresholds": cls.QUALITY_THRESHOLDS,
            "enable_background_evaluation": cls.ENABLE_BACKGROUND_EVALUATION,
            "max_evaluation_history": cls.MAX_EVALUATION_HISTORY,
            "custom_prompts_enabled": cls.CUSTOM_PROMPTS_ENABLED,
            "celery_broker_url": cls.CELERY_BROKER_URL,
            "celery_result_backend": cls.CELERY_RESULT_BACKEND,
            "redis_url": cls.REDIS_URL,
            "websocket_timeout": cls.WEBSOCKET_TIMEOUT,
            "websocket_ping_interval": cls.WEBSOCKET_PING_INTERVAL,
            "evaluation_task_timeout": cls.EVALUATION_TASK_TIMEOUT,
            "batch_evaluation_max_size": cls.BATCH_EVALUATION_MAX_SIZE,
            "result_cache_timeout": cls.RESULT_CACHE_TIMEOUT
        }
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return status"""
        issues = []
        
        # Check if weights sum to 1.0
        weight_sum = sum(cls.METRIC_WEIGHTS.values())
        if abs(weight_sum - 1.0) > 0.01:
            issues.append(f"Metric weights sum to {weight_sum}, should be 1.0")
        
        # Check temperature range
        if not 0.0 <= cls.EVALUATION_TEMPERATURE <= 2.0:
            issues.append(f"Temperature {cls.EVALUATION_TEMPERATURE} should be between 0.0 and 2.0")
        
        # Check max tokens
        if cls.EVALUATION_MAX_TOKENS < 100:
            issues.append(f"Max tokens {cls.EVALUATION_MAX_TOKENS} seems too low")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "config": cls.get_config_dict()
        }

# Create global config instance
evaluation_config = EvaluationConfig()
