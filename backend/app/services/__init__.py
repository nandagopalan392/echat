"""
Services Package

This package contains business logic services that handle core application functionality.
Services are independent of the HTTP layer and can be reused across different interfaces.
"""

from .model_service import ModelService

__all__ = ['ModelService']
