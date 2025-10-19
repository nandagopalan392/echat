"""
Chunking Configuration Service

Business logic for managing chunking configurations.
"""
import logging
from typing import Dict, Any, List, Optional

from app.config.chunking import (
    ChunkingMethod,
    ChunkingConfig,
    FileFormatSupport,
    get_chunking_config_manager
)
from app.db.repositories import DocumentRepository

logger = logging.getLogger(__name__)


class ChunkingConfigService:
    """
    Service class for managing chunking configurations.
    
    Handles chunking method information, configuration management,
    and optimal method selection.
    """
    
    def __init__(self, document_repository: DocumentRepository):
        """
        Initialize chunking config service.
        
        Args:
            document_repository: Repository for document operations
        """
        self.document_repository = document_repository
        self.config_manager = get_chunking_config_manager()
    
    def get_available_methods(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available chunking methods with their descriptions.
        
        Returns:
            Dictionary mapping method names to their information
        """
        try:
            methods = {}
            for method in ChunkingMethod:
                methods[method.value] = {
                    'name': method.value,
                    'description': self._get_method_description(method),
                    'supported_formats': FileFormatSupport.get_supported_formats(method)
                }
            
            return methods
        except Exception as e:
            logger.error(f"Error getting chunking methods: {e}")
            raise
    
    def get_config(self, method: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get chunking configuration for a specific method.
        
        Args:
            method: Chunking method name
            user_id: Optional user ID for user-specific config
            
        Returns:
            Configuration dictionary
            
        Raises:
            ValueError: If method is invalid
        """
        try:
            chunking_method = ChunkingMethod(method)
        except ValueError:
            raise ValueError(f"Invalid chunking method: {method}")
        
        try:
            config = self.config_manager.get_config(chunking_method, user_id)
            return config.to_dict()
        except Exception as e:
            logger.error(f"Error getting chunking config for {method}: {e}")
            raise
    
    def update_config(
        self,
        method: str,
        config_data: Dict[str, Any],
        user_id: str
    ) -> tuple[Dict[str, Any], List[str]]:
        """
        Update chunking configuration for a specific method.
        
        Args:
            method: Chunking method name
            config_data: New configuration data
            user_id: User ID for user-specific config
            
        Returns:
            Tuple of (updated config dict, list of warnings)
            
        Raises:
            ValueError: If method is invalid or config is invalid
        """
        try:
            chunking_method = ChunkingMethod(method)
        except ValueError:
            raise ValueError(f"Invalid chunking method: {method}")
        
        try:
            # Create config from provided data
            config = ChunkingConfig.from_dict(config_data)
            
            # Validate configuration
            warnings = self.config_manager.validate_config(config)
            
            # Save configuration
            self.config_manager.save_config(chunking_method, config, user_id)
            
            return config.to_dict(), warnings
        except Exception as e:
            logger.error(f"Error updating chunking config for {method}: {e}")
            raise
    
    def get_optimal_method(self, file_extension: str) -> tuple[str, List[str]]:
        """
        Get optimal chunking method for a file extension.
        
        Args:
            file_extension: File extension (with or without dot)
            
        Returns:
            Tuple of (optimal method name, list of available method names)
        """
        try:
            # Remove dot if present
            ext = file_extension.lstrip('.')
            
            optimal_method = FileFormatSupport.get_optimal_method(ext)
            available_methods = self.config_manager.get_available_methods(ext)
            
            return (
                optimal_method.value,
                [method.value for method in available_methods]
            )
        except Exception as e:
            logger.error(f"Error getting optimal chunking method for {file_extension}: {e}")
            raise
    
    @staticmethod
    def _get_method_description(method: ChunkingMethod) -> str:
        """
        Get description for a chunking method.
        
        Args:
            method: ChunkingMethod enum value
            
        Returns:
            Human-readable description
        """
        descriptions = {
            ChunkingMethod.GENERAL: "General document chunking for PDF, DOCX, MD, TXT files",
            ChunkingMethod.QA: "For question-answer formatted documents",
            ChunkingMethod.RESUME: "Enterprise edition for resume documents",
            ChunkingMethod.TABLE: "For spreadsheet/tabular data",
            ChunkingMethod.PRESENTATION: "For PPT/presentation files",
            ChunkingMethod.PICTURE: "Image/visual content processing",
            ChunkingMethod.EMAIL: "Email content chunking"
        }
        return descriptions.get(method, "Custom chunking method")
