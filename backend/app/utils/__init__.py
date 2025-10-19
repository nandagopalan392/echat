"""
Utility modules for the eChat application.

This package contains reusable utility functions and helpers including:
- Model utilities (model_utils.py)
- GPU utilities (gpu_utils.py)
- RAG utilities (rag_utils.py)
"""

from .model_utils import (
    format_model_size,
    categorize_model,
    normalize_model_name,
    normalize_bge_model_name,
    normalize_nomic_model_name,
    validate_model_parameters,
    generate_compatibility_recommendations,
    is_gated_model_error,
    is_embedding_model,
    get_model_variants
)

from .gpu_utils import (
    check_gpu_memory_available,
    get_gpu_memory_info,
    configure_docling_device,
    check_model_compatibility_detailed,
    is_cuda_memory_error,
    clear_gpu_memory,
    get_optimal_device
)

from .rag_utils import (
    detect_model_provider,
    check_model_compatibility,
    create_embedding_model,
    get_model_provider_info
)

__all__ = [
    # Model utilities
    'format_model_size',
    'categorize_model',
    'normalize_model_name',
    'normalize_bge_model_name',
    'normalize_nomic_model_name',
    'validate_model_parameters',
    'generate_compatibility_recommendations',
    'is_gated_model_error',
    'is_embedding_model',
    'get_model_variants',
    
    # GPU utilities
    'check_gpu_memory_available',
    'get_gpu_memory_info',
    'configure_docling_device',
    'check_model_compatibility_detailed',
    'is_cuda_memory_error',
    'clear_gpu_memory',
    'get_optimal_device',
    
    # RAG utilities
    'detect_model_provider',
    'check_model_compatibility',
    'create_embedding_model',
    'get_model_provider_info'
]
