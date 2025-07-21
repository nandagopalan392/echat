"""
Document Chunking Configuration System
Handles different chunking methods and configurations for various document types
"""

import json
import os
import logging
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

class ChunkingMethod(Enum):
    """Enumeration of available chunking methods"""
    QA = "qa"               # Q&A formatted documents  
    RESUME = "resume"       # Resume documents (Enterprise edition)
    GENERAL = "general"     # General document chunking for PDF, DOCX, MD, TXT
    TABLE = "table"         # Spreadsheet/tabular data
    PRESENTATION = "presentation"  # PPT/presentation files
    PICTURE = "picture"     # Image/visual content processing
    EMAIL = "email"         # Email content chunking

@dataclass
class ChunkingConfig:
    """Configuration for document chunking"""
    method: ChunkingMethod = ChunkingMethod.QA
    chunk_token_num: int = 1000  # Token threshold for chunk size (128-8192)
    delimiter: str = "\\n!?。；！？"  # Text delimiters
    layout_recognize: str = "auto"  # Layout recognition method
    max_token: int = 8192  # Maximum tokens per chunk
    chunk_overlap: int = 200  # Overlap between chunks
    
    # Additional configuration options
    preserve_formatting: bool = True  # Whether to preserve document formatting
    extract_tables: bool = True  # Whether to extract tables separately
    extract_images: bool = False  # Whether to extract images
    language: str = "auto"  # Document language detection
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary with NaN validation"""
        import math
        
        result = {
            'method': self.method.value,
            'chunk_token_num': self.chunk_token_num,
            'max_token': self.max_token,
            'chunk_overlap': self.chunk_overlap,
            'preserve_formatting': self.preserve_formatting,
            'extract_tables': self.extract_tables,
            'extract_images': self.extract_images
        }
        
        # Ensure numeric values are not NaN
        for key, value in result.items():
            if isinstance(value, (int, float)) and math.isnan(value):
                # Set default values for NaN
                if key == 'chunk_token_num':
                    result[key] = 1000
                elif key == 'max_token':
                    result[key] = 8192
                elif key == 'chunk_overlap':
                    result[key] = 200
                else:
                    result[key] = 0
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkingConfig':
        """Create config from dictionary with validation"""
        import math
        
        # Create a copy to avoid modifying the original
        validated_data = data.copy()
        
        # Handle method conversion
        if 'method' in validated_data:
            validated_data['method'] = ChunkingMethod(validated_data['method'])
        
        # Validate and fix numeric fields
        numeric_fields = {
            'chunk_token_num': 1000,
            'max_token': 8192,
            'chunk_overlap': 200
        }
        
        for field, default_value in numeric_fields.items():
            if field in validated_data:
                value = validated_data[field]
                # Handle NaN, None, or invalid values
                if value is None or (isinstance(value, (int, float)) and math.isnan(value)):
                    validated_data[field] = default_value
                else:
                    try:
                        validated_data[field] = int(value)
                    except (ValueError, TypeError):
                        validated_data[field] = default_value
        
        # Validate boolean fields
        boolean_fields = ['preserve_formatting', 'extract_tables', 'extract_images']
        for field in boolean_fields:
            if field in validated_data:
                value = validated_data[field]
                if not isinstance(value, bool):
                    # Convert string representations to boolean
                    if isinstance(value, str):
                        validated_data[field] = value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        validated_data[field] = bool(value)
        
        return cls(**validated_data)

class FileFormatSupport:
    """File format support by chunking method"""
    
    SUPPORTED_FORMATS = {
        ChunkingMethod.QA: ['md', 'docx', 'txt', 'json', 'pdf'],
        ChunkingMethod.RESUME: ['docx', 'pdf', 'txt'],
        ChunkingMethod.GENERAL: ['pdf', 'docx', 'md', 'txt'],
        ChunkingMethod.TABLE: ['xlsx', 'xls', 'csv', 'txt'],
        ChunkingMethod.PRESENTATION: ['ppt', 'pptx'],
        ChunkingMethod.PICTURE: ['jpeg', 'jpg', 'png', 'tif', 'gif'],
        ChunkingMethod.EMAIL: ['eml', 'txt']
    }
    
    @classmethod
    def is_supported(cls, method: ChunkingMethod, file_extension: str) -> bool:
        """Check if file format is supported by chunking method"""
        supported = cls.SUPPORTED_FORMATS.get(method, [])
        return file_extension.lower() in [fmt.lower() for fmt in supported]
    
    @classmethod
    def get_supported_formats(cls, method: ChunkingMethod) -> List[str]:
        """Get supported file formats for a chunking method"""
        return cls.SUPPORTED_FORMATS.get(method, [])
    
    @classmethod
    def get_optimal_method(cls, file_extension: str) -> ChunkingMethod:
        """Get optimal chunking method for file type"""
        ext = file_extension.lower()
        
        # Mapping of file extensions to optimal methods
        optimal_methods = {
            'pdf': ChunkingMethod.GENERAL,
            'docx': ChunkingMethod.GENERAL,
            'md': ChunkingMethod.GENERAL,
            'txt': ChunkingMethod.GENERAL,
            'xlsx': ChunkingMethod.TABLE,
            'xls': ChunkingMethod.TABLE,
            'csv': ChunkingMethod.TABLE,
            'ppt': ChunkingMethod.PRESENTATION,
            'pptx': ChunkingMethod.PRESENTATION,
            'jpeg': ChunkingMethod.PICTURE,
            'jpg': ChunkingMethod.PICTURE,
            'png': ChunkingMethod.PICTURE,
            'gif': ChunkingMethod.PICTURE,
            'tif': ChunkingMethod.PICTURE,
            'eml': ChunkingMethod.EMAIL,
            'json': ChunkingMethod.QA,
            'html': ChunkingMethod.QA
        }
        
        return optimal_methods.get(ext, ChunkingMethod.GENERAL)

class ChunkingConfigManager:
    """Manager for chunking configurations"""
    
    def __init__(self, config_dir: str = "/app/data/chunking_configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.default_configs = self._load_default_configs()
        # Import here to avoid circular imports
        from chat_db import ChatDB
        self.db = ChatDB()
    
    def _load_default_configs(self) -> Dict[ChunkingMethod, ChunkingConfig]:
        """Load default configurations for each chunking method"""
        defaults = {
            ChunkingMethod.QA: ChunkingConfig(
                method=ChunkingMethod.QA,
                chunk_token_num=512,
                chunk_overlap=50,
                delimiter="\\n\\n|Q:|A:",
                max_token=2048
            ),
            ChunkingMethod.RESUME: ChunkingConfig(
                method=ChunkingMethod.RESUME,
                chunk_token_num=800,
                chunk_overlap=100,
                delimiter="\\n\\n|EXPERIENCE|EDUCATION|SKILLS",
                max_token=3072
            ),
            ChunkingMethod.GENERAL: ChunkingConfig(
                method=ChunkingMethod.GENERAL,
                chunk_token_num=1000,
                chunk_overlap=200,
                delimiter="\\n\\n|\\n|\\.|\\!|\\?",
                max_token=4096,
                layout_recognize="advanced",
                preserve_formatting=True,
                extract_tables=True
            ),
            ChunkingMethod.TABLE: ChunkingConfig(
                method=ChunkingMethod.TABLE,
                chunk_token_num=2000,
                chunk_overlap=0,
                delimiter="row_separator",
                max_token=8192,
                extract_tables=True,
                preserve_formatting=True
            ),
            ChunkingMethod.PRESENTATION: ChunkingConfig(
                method=ChunkingMethod.PRESENTATION,
                chunk_token_num=800,
                chunk_overlap=100,
                delimiter="slide_break",
                max_token=3072,
                extract_images=True
            ),
            ChunkingMethod.PICTURE: ChunkingConfig(
                method=ChunkingMethod.PICTURE,
                chunk_token_num=500,
                chunk_overlap=0,
                delimiter="image_section",
                max_token=2048,
                extract_images=True,
                layout_recognize="ocr"
            ),
            ChunkingMethod.EMAIL: ChunkingConfig(
                method=ChunkingMethod.EMAIL,
                chunk_token_num=600,
                chunk_overlap=50,
                delimiter="From:|To:|Subject:|Date:",
                max_token=2048
            )
        }
        return defaults
    
    def get_config(self, method: ChunkingMethod, user_id: Optional[str] = None) -> ChunkingConfig:
        """Get configuration for a chunking method"""
        # Try to load user-specific config from database first
        if user_id:
            try:
                config_data = self.db.get_chunking_config(user_id, method.value)
                if config_data:
                    return ChunkingConfig.from_dict(config_data)
            except Exception as e:
                logger.warning(f"Failed to load user config from database for {user_id}, {method.value}: {e}")
            
            # Fallback to file-based config for backward compatibility
            user_config_path = self.config_dir / f"{user_id}_{method.value}.json"
            if user_config_path.exists():
                try:
                    with open(user_config_path, 'r') as f:
                        config_data = json.load(f)
                    return ChunkingConfig.from_dict(config_data)
                except Exception as e:
                    logger.warning(f"Failed to load user config {user_config_path}: {e}")
        
        # Fall back to default config
        return self.default_configs.get(method, self.default_configs[ChunkingMethod.GENERAL])
    
    def save_config(self, method: ChunkingMethod, config: ChunkingConfig, user_id: Optional[str] = None):
        """Save configuration for a chunking method"""
        try:
            if user_id:
                # Save to database
                success = self.db.save_chunking_config(user_id, method.value, config.to_dict())
                if success:
                    logger.info(f"Saved chunking config to database for user {user_id}, method {method.value}")
                    return
                else:
                    logger.warning(f"Failed to save to database, falling back to file for user {user_id}")
            
            # Fallback to file-based storage
            if user_id:
                config_path = self.config_dir / f"{user_id}_{method.value}.json"
            else:
                config_path = self.config_dir / f"default_{method.value}.json"
            
            with open(config_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            logger.info(f"Saved chunking config to file {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def get_available_methods(self, file_extension: str) -> List[ChunkingMethod]:
        """Get available chunking methods for a file type"""
        available = []
        for method in ChunkingMethod:
            if FileFormatSupport.is_supported(method, file_extension):
                available.append(method)
        return available
    
    def validate_config(self, config: ChunkingConfig) -> List[str]:
        """Validate chunking configuration and return list of warnings/errors"""
        warnings = []
        
        if config.chunk_token_num < 128:
            warnings.append("chunk_token_num is very small (< 128), may result in fragmented chunks")
        
        if config.chunk_token_num > 8192:
            warnings.append("chunk_token_num is very large (> 8192), may exceed model limits")
        
        if config.chunk_overlap >= config.chunk_token_num:
            warnings.append("chunk_overlap should be smaller than chunk_token_num")
        
        if config.max_token < config.chunk_token_num:
            warnings.append("max_token should be >= chunk_token_num")
        
        return warnings

# Global instance
_chunking_config_manager = None

def get_chunking_config_manager() -> ChunkingConfigManager:
    """Get global chunking configuration manager instance"""
    global _chunking_config_manager
    if _chunking_config_manager is None:
        _chunking_config_manager = ChunkingConfigManager()
    return _chunking_config_manager
