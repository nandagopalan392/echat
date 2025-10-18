"""
Utility modules for the eChat application.

This package contains reusable utility functions and helpers.
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

__all__ = [
    'format_model_size',
    'categorize_model',
    'normalize_model_name',
    'normalize_bge_model_name',
    'normalize_nomic_model_name',
    'validate_model_parameters',
    'generate_compatibility_recommendations',
    'is_gated_model_error',
    'is_embedding_model',
    'get_model_variants'
]
