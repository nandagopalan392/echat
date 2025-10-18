"""
Base Model Provider - Abstract Base Class

Defines the interface that all model providers must implement.
This ensures consistent API across different providers (Ollama, HuggingFace, etc.)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging


class BaseModelProvider(ABC):
    """
    Abstract base class for model providers.
    
    All provider implementations (Ollama, HuggingFace, etc.) must inherit from this
    class and implement the required methods.
    """
    
    def __init__(self):
        """Initialize the provider with logger."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialize()
    
    def _initialize(self):
        """
        Optional initialization hook for subclasses.
        Override this method to perform provider-specific initialization.
        """
        pass
    
    @abstractmethod
    async def get_installed_models(self) -> List[Dict[str, Any]]:
        """
        Get list of locally installed/cached models.
        
        Returns:
            List of dictionaries containing model information:
            [
                {
                    'name': 'model-name',
                    'size': size_in_bytes or size_string,
                    'modified': timestamp,
                    'digest': model_hash,
                    ...
                }
            ]
            
        Raises:
            Exception: If unable to fetch installed models
        """
        pass
    
    @abstractmethod
    async def get_available_models(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get list of models available for download from provider.
        
        Args:
            limit: Optional maximum number of models to return
            
        Returns:
            List of dictionaries containing model information:
            [
                {
                    'name': 'model-name',
                    'description': 'model description',
                    'tags': ['tag1', 'tag2'],
                    'size': estimated_size,
                    ...
                }
            ]
            
        Raises:
            Exception: If unable to fetch available models
        """
        pass
    
    @abstractmethod
    async def download_model(
        self, 
        model_name: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Download a model from the provider.
        
        Args:
            model_name: Name/identifier of the model to download
            **kwargs: Provider-specific options (e.g., stream=True, timeout=600)
            
        Returns:
            Dictionary with download status:
            {
                'success': bool,
                'message': str,
                'model_name': str,
                'size': int (bytes downloaded),
                ...
            }
            
        Raises:
            Exception: If download fails
        """
        pass
    
    @abstractmethod
    async def check_model_exists(self, model_name: str) -> bool:
        """
        Check if a model is installed/cached locally.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model exists locally, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model details or None if not found:
            {
                'name': str,
                'size': int,
                'description': str,
                'capabilities': List[str],
                'parameters': Dict,
                ...
            }
        """
        pass
    
    async def validate_model(self, model_name: str) -> Dict[str, Any]:
        """
        Validate that a model is accessible and working.
        
        This is an optional method that providers can override.
        Default implementation checks if model exists.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'message': str,
                'details': Dict
            }
        """
        exists = await self.check_model_exists(model_name)
        return {
            'valid': exists,
            'message': f"Model {'found' if exists else 'not found'}",
            'details': {}
        }
    
    def get_provider_name(self) -> str:
        """
        Get the name of this provider.
        
        Returns:
            Provider name (e.g., 'Ollama', 'HuggingFace')
        """
        return self.__class__.__name__.replace('Provider', '')
    
    def __repr__(self) -> str:
        """String representation of provider."""
        return f"<{self.__class__.__name__}>"
