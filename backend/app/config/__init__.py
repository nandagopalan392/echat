"""
Configuration Module
Application configuration including chunking settings
"""

from app.config.chunking import (
    ChunkingMethod,
    ChunkingConfig,
    FileFormatSupport,
    ChunkingConfigManager,
    get_chunking_config_manager
)

# Import settings from settings.py in the same directory
from app.config.settings import Settings, settings, API_TAGS_METADATA

__all__ = [
    'ChunkingMethod',
    'ChunkingConfig',
    'FileFormatSupport',
    'ChunkingConfigManager',
    'get_chunking_config_manager',
    'Settings',
    'settings',
    'API_TAGS_METADATA'
]
