"""
Ollama Provider - Handles Ollama-specific model operations

This provider manages Ollama            try:
                from app.utils.external.ollama_scraper import get_available_ollama_models
                library_models = get_available_ollama_models(use_cache=True)dels including installation, downloading,
and metadata retrieval.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import httpx

from .base import BaseModelProvider


class OllamaProvider(BaseModelProvider):
    """
    Provider for Ollama models.
    
    Handles all Ollama-specific operations including:
    - Fetching installed models
    - Downloading new models
    - Checking model availability
    - Getting model metadata
    """
    
    def _initialize(self):
        """Initialize Ollama-specific configuration."""
        self.ollama_host = os.getenv('OLLAMA_HOST', 'http://ollama:11434')
        self.default_timeout = 30.0
        self.download_timeout = 600.0
        self.logger.info(f"Initialized OllamaProvider with host: {self.ollama_host}")
    
    async def get_installed_models(self) -> List[Dict[str, Any]]:
        """
        Get list of locally installed Ollama models.
        
        Returns:
            List of installed model dictionaries with metadata
            
        Raises:
            Exception: If unable to connect to Ollama or fetch models
        """
        try:
            async with httpx.AsyncClient(timeout=self.default_timeout) as client:
                response = await client.get(f"{self.ollama_host}/api/tags")
                
                if response.status_code != 200:
                    raise Exception(f"Ollama API returned status {response.status_code}")
                
                data = response.json()
                models = data.get('models', [])
                
                # Format models with consistent structure
                formatted_models = []
                for model in models:
                    formatted_models.append({
                        'name': model.get('name', ''),
                        'size': model.get('size', 0),
                        'modified': model.get('modified_at', ''),
                        'digest': model.get('digest', ''),
                        'details': model.get('details', {}),
                        'source': 'ollama',
                        'installed': True
                    })
                
                self.logger.info(f"Found {len(formatted_models)} installed Ollama models")
                return formatted_models
                
        except httpx.TimeoutException:
            self.logger.error(f"Timeout connecting to Ollama at {self.ollama_host}")
            raise Exception(f"Timeout connecting to Ollama")
        except httpx.RequestError as e:
            self.logger.error(f"Error connecting to Ollama: {e}")
            raise Exception(f"Could not connect to Ollama: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error fetching installed models: {e}")
            raise
    
    async def get_available_models(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get list of models available in Ollama library.
        
        Fetches models from Ollama API or uses local scraping.
        
        Args:
            limit: Optional limit on number of models to return
            
        Returns:
            List of available model dictionaries
        """
        try:
            # Import ollama scraper from app structure
            try:
                from app.utils.external.ollama_scraper import get_available_ollama_models
                available_models = get_available_ollama_models(use_cache=True)
            except ImportError:
                # Fallback: Return empty list or fetch from Ollama API directly
                self.logger.warning("ollama_scraper not available, using fallback")
                available_models = []
            
            if limit and len(available_models) > limit:
                available_models = available_models[:limit]
            
            self.logger.info(f"Found {len(available_models)} available Ollama models")
            return available_models
            
        except Exception as e:
            self.logger.error(f"Error fetching available Ollama models: {e}")
            # Return empty list instead of raising to be more resilient
            return []
    
    async def download_model(
        self, 
        model_name: str,
        stream: bool = True,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Download an Ollama model with optional progress streaming.
        
        Args:
            model_name: Name of the Ollama model to download
            stream: Whether to stream progress updates
            timeout: Download timeout in seconds (default: 600)
            
        Returns:
            Dictionary with download status and details
            
        Raises:
            Exception: If download fails
        """
        if timeout is None:
            timeout = self.download_timeout
        
        try:
            self.logger.info(f"Starting download of Ollama model: {model_name}")
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                download_response = await client.post(
                    f"{self.ollama_host}/api/pull",
                    json={"name": model_name, "stream": stream},
                    timeout=timeout
                )
                
                if download_response.status_code != 200:
                    error_msg = f"Failed to download model {model_name}: HTTP {download_response.status_code}"
                    self.logger.error(error_msg)
                    return {
                        'success': False,
                        'message': error_msg,
                        'model_name': model_name,
                        'status_code': download_response.status_code
                    }
                
                # Process streaming response
                download_complete = False
                last_status = ""
                
                if stream:
                    async for line in download_response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                status = data.get("status", "")
                                last_status = status
                                
                                if "success" in status.lower() or "pull complete" in status.lower():
                                    download_complete = True
                                    self.logger.info(f"Model {model_name}: Download completed")
                                    break
                                    
                            except json.JSONDecodeError:
                                continue
                else:
                    # Non-streaming download
                    result = download_response.json()
                    download_complete = result.get('status') == 'success'
                
                return {
                    'success': download_complete,
                    'message': f"Successfully downloaded {model_name}" if download_complete else last_status,
                    'model_name': model_name,
                    'status': last_status
                }
                
        except httpx.TimeoutException:
            error_msg = f"Timeout downloading model {model_name}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'model_name': model_name,
                'error': 'timeout'
            }
        except Exception as e:
            error_msg = f"Error downloading model {model_name}: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'model_name': model_name,
                'error': str(e)
            }
    
    async def check_model_exists(self, model_name: str) -> bool:
        """
        Check if an Ollama model is installed locally.
        
        Checks for exact match and common variants (with/without :latest tag).
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model exists, False otherwise
        """
        try:
            installed_models = await self.get_installed_models()
            installed_names = {model['name'] for model in installed_models}
            
            # Check exact match
            if model_name in installed_names:
                return True
            
            # Check common variants
            variants = [
                model_name,
                f"{model_name}:latest",
                model_name.replace(":latest", "")
            ]
            
            return any(variant in installed_names for variant in variants)
            
        except Exception as e:
            self.logger.error(f"Error checking if model exists: {e}")
            return False
    
    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific Ollama model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary or None if not found
        """
        try:
            # First check installed models
            installed_models = await self.get_installed_models()
            for model in installed_models:
                if model['name'] == model_name:
                    return model
            
            # If not installed, check available models
            available_models = await self.get_available_models()
            for model in available_models:
                if model.get('name') == model_name:
                    return {
                        **model,
                        'installed': False
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting model info for {model_name}: {e}")
            return None
    
    async def get_model_variants(self, base_model_name: str) -> List[str]:
        """
        Get all installed variants of a base model name.
        
        For example, for base_model_name='llama2', might return:
        ['llama2:latest', 'llama2:7b', 'llama2:13b']
        
        Args:
            base_model_name: Base name of the model
            
        Returns:
            List of variant names
        """
        try:
            installed_models = await self.get_installed_models()
            variants = []
            
            for model in installed_models:
                model_name = model['name']
                # Check if model name starts with base name
                if model_name.startswith(base_model_name):
                    variants.append(model_name)
            
            return variants
            
        except Exception as e:
            self.logger.error(f"Error getting model variants: {e}")
            return []
    
    async def delete_model(self, model_name: str) -> Dict[str, Any]:
        """
        Delete an installed Ollama model.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            Dictionary with deletion status
        """
        try:
            async with httpx.AsyncClient(timeout=self.default_timeout) as client:
                response = await client.delete(
                    f"{self.ollama_host}/api/delete",
                    json={"name": model_name}
                )
                
                if response.status_code == 200:
                    self.logger.info(f"Successfully deleted model: {model_name}")
                    return {
                        'success': True,
                        'message': f"Model {model_name} deleted successfully",
                        'model_name': model_name
                    }
                else:
                    error_msg = f"Failed to delete model: HTTP {response.status_code}"
                    self.logger.error(error_msg)
                    return {
                        'success': False,
                        'message': error_msg,
                        'model_name': model_name
                    }
                    
        except Exception as e:
            error_msg = f"Error deleting model {model_name}: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'model_name': model_name,
                'error': str(e)
            }
    
    async def get_all_models(self) -> Dict[str, Any]:
        """
        Get both installed and available models in a single call.
        
        Returns:
            Dictionary with installed and available models
        """
        try:
            installed = await self.get_installed_models()
            available = await self.get_available_models()
            
            return {
                'installed': installed,
                'available': available,
                'total_installed': len(installed),
                'total_available': len(available)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting all models: {e}")
            return {
                'installed': [],
                'available': [],
                'total_installed': 0,
                'total_available': 0,
                'error': str(e)
            }
