"""
Retrieval Configuration Service

Business logic for managing retrieval configurations and reranker models.
"""
import logging
import os
import json
import asyncio
import httpx
from typing import Dict, Any, List, Optional, Tuple

from app.config.retrieval import (
    RetrievalConfig,
    get_retrieval_config_manager
)
from app.db.repositories import DocumentRepository

logger = logging.getLogger(__name__)

# Global cache for download status
download_status_cache: Dict[str, Dict[str, Any]] = {}


class RetrievalConfigService:
    """
    Service class for managing retrieval configurations.
    
    Handles configuration management, reranker model availability,
    and model downloads.
    """
    
    def __init__(self, document_repository: DocumentRepository):
        """
        Initialize retrieval config service.
        
        Args:
            document_repository: Repository for document operations
        """
        self.document_repository = document_repository
        self.config_manager = get_retrieval_config_manager()
    
    def get_config(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get retrieval configuration.
        
        Args:
            user_id: Optional user ID for user-specific config
            
        Returns:
            Configuration dictionary
        """
        try:
            config = self.config_manager.get_config(user_id)
            return config.to_dict()
        except Exception as e:
            logger.error(f"Error getting retrieval config: {e}")
            raise
    
    async def update_config(
        self,
        config_data: Dict[str, Any],
        user_id: str
    ) -> Tuple[Dict[str, Any], List[str], Dict[str, Any], Optional[str]]:
        """
        Update retrieval configuration with auto-download for reranker models.
        
        Args:
            config_data: New configuration data
            user_id: User ID for user-specific config
            
        Returns:
            Tuple of (updated config dict, list of warnings, download result, optional message)
        """
        try:
            # Create config from provided data
            config = RetrievalConfig.from_dict(config_data)
            
            # Check and handle reranker model download
            download_result = {"success": True, "downloaded": False, "message": "No reranker model specified"}
            message = None
            
            if config.reranker_enabled and config.reranker_model and config.reranker_model.lower() != "none":
                logger.info(f"🎯 Checking reranker model availability: {config.reranker_model}")
                
                if config.reranker_provider == "huggingface":
                    # For HuggingFace models, start download in background
                    if config.reranker_model not in download_status_cache:
                        download_status_cache[config.reranker_model] = {
                            "downloading": True,
                            "completed": False,
                            "message": "Download starting..."
                        }
                        asyncio.create_task(self._download_huggingface_model_background(config.reranker_model))
                    download_result = {
                        "success": True,
                        "downloaded": False,
                        "message": f"HuggingFace model {config.reranker_model} download started in background"
                    }
                else:
                    # For Ollama models, download synchronously
                    download_result = await self._check_and_download_reranker_model(
                        config.reranker_model,
                        config.reranker_provider
                    )
                    
                    if not download_result["success"]:
                        logger.warning(f"⚠️ Reranker model download failed but continuing with config save: {download_result['message']}")
            
            # Validate configuration
            warnings = self.config_manager.validate_config(config)
            
            # Add download-related warning if download failed
            if not download_result["success"]:
                warnings.append(f"Reranker model download failed: {download_result['message']}")
            
            # Save configuration
            success = self.config_manager.save_config(config, user_id)
            
            if not success:
                raise Exception("Failed to save retrieval configuration")
            
            # Add success message if model was downloaded
            if download_result.get("downloaded"):
                message = f"Configuration saved and reranker model '{config.reranker_model}' downloaded successfully"
            
            return config.to_dict(), warnings, download_result, message
            
        except Exception as e:
            logger.error(f"Error updating retrieval config: {e}")
            raise
    
    async def get_available_reranker_models(self, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of available reranker models from Ollama and HuggingFace.
        
        Args:
            provider: Optional filter by provider (ollama, huggingface)
            
        Returns:
            List of reranker model information
        """
        try:
            ollama_url = os.getenv('OLLAMA_HOST', 'http://ollama:11434')
            
            from app.utils.external.ollama_scraper import get_available_ollama_models
            
            # Get locally installed models
            local_models = []
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(f"{ollama_url}/api/tags")
                    
                    if response.status_code == 200:
                        ollama_data = response.json()
                        local_models = ollama_data.get('models', [])
                except Exception as e:
                    logger.error(f"Failed to fetch models from Ollama API: {e}")
            
            # Get models from Ollama library
            try:
                library_models = get_available_ollama_models(use_cache=True)
            except Exception as e:
                logger.warning(f"Could not fetch Ollama library models: {e}")
                library_models = []
            
            # Combine local and library models
            all_ollama_models = list(local_models)
            
            # Add library models that aren't already installed locally
            local_model_names = {model.get('name', '') for model in local_models}
            for lib_model in library_models:
                if lib_model.get('name', '') not in local_model_names:
                    all_ollama_models.append({
                        'name': lib_model.get('name', ''),
                        'category': lib_model.get('category', 'llm')
                    })
            
            # Filter for reranker/embedding models
            reranker_models = []
            
            # Add "None" option first
            reranker_models.append({
                "name": "",
                "display_name": "None (Vector + Keyword)",
                "description": "Use weighted combination of vector similarity and keyword matching",
                "provider": "none"
            })
            
            # Process Ollama models
            reranker_models_found = 0
            for model in all_ollama_models:
                model_name = model.get('name', '').lower()
                original_name = model.get('name', '')
                model_category = model.get('category', 'llm')
                
                # Prioritize dedicated reranker models ONLY
                is_dedicated_reranker = (
                    model_category == 'reranker' or
                    any(keyword in model_name for keyword in [
                        'rerank', 'cross-encoder', 'qwen3-reranker', 'qwen-reranker',
                        'bce-reranker', 'bge-reranker', 'reranker', 'ranking'
                    ])
                )
                
                # Only include BGE models that are specifically rerankers
                is_bge_reranker = (
                    'bge' in model_name and
                    ('reranker' in model_name or 'rerank' in model_name)
                )
                
                if is_dedicated_reranker or is_bge_reranker:
                    reranker_models_found += 1
                    
                    display_name = original_name.replace(':', ' ').replace('-', ' ').replace('_', ' ')
                    display_name = ' '.join(word.capitalize() for word in display_name.split())
                    
                    # Add description
                    if 'qwen' in model_name and 'reranker' in model_name:
                        description = "🎯 Qwen dedicated reranking model (multilingual, high performance)"
                    elif 'bge-reranker' in model_name or ('bge' in model_name and 'reranker' in model_name):
                        description = "🎯 BGE dedicated reranking model (excellent for document ranking)"
                    elif 'bce-reranker' in model_name:
                        description = "🎯 BCE reranking model (cross-language support)"
                    elif 'cross-encoder' in model_name:
                        description = "🎯 Cross-encoder reranking model (high accuracy)"
                    else:
                        description = "🎯 Dedicated reranking model"
                    
                    is_local = original_name in local_model_names
                    if not is_local:
                        description += " (available to download)"
                    
                    reranker_models.append({
                        "name": original_name,
                        "display_name": display_name,
                        "description": description,
                        "is_local": is_local,
                        "provider": "ollama"
                    })
            
            # Add HuggingFace reranker models
            huggingface_reranker_models = self._get_huggingface_models()
            reranker_models.extend(huggingface_reranker_models)
            
            # If no ollama reranker models found, add fallback
            if reranker_models_found == 0:
                logger.warning("No reranker models found in Ollama, adding fallback Ollama models")
                fallback_models = self._get_fallback_ollama_models()
                reranker_models.extend(fallback_models)
            
            # Apply server-side filtering if provider is specified
            if provider:
                if provider == 'ollama':
                    reranker_models = [model for model in reranker_models if model.get('provider') in ['ollama', 'none']]
                elif provider == 'huggingface':
                    reranker_models = [model for model in reranker_models if model.get('provider') in ['huggingface', 'none']]
            
            return reranker_models
            
        except Exception as e:
            logger.error(f"Error getting reranker models: {e}")
            # Return basic fallback on error
            return self._get_fallback_models()
    
    def get_reranker_download_status(self, model_name: str) -> Dict[str, Any]:
        """
        Get the download status of a reranker model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with download status information
        """
        if model_name in download_status_cache:
            status = download_status_cache[model_name]
            return {
                "model_name": model_name,
                "downloading": status.get("downloading", False),
                "completed": status.get("completed", False),
                "message": status.get("message", "Unknown status")
            }
        else:
            return {
                "model_name": model_name,
                "downloading": False,
                "completed": False,
                "message": "No download status available"
            }
    
    async def _check_and_download_reranker_model(
        self,
        model_name: str,
        provider: str = "ollama"
    ) -> Dict[str, Any]:
        """Check if a reranker model is available locally, download if not"""
        if not model_name or model_name.lower() == "none":
            return {"success": True, "downloaded": False, "message": "No reranker model specified"}
        
        # For HuggingFace models, handle download with status tracking
        if provider == "huggingface":
            if model_name in download_status_cache:
                status = download_status_cache[model_name]
                return {
                    "success": True,
                    "downloaded": status.get("completed", False),
                    "message": status.get("message", "Download in progress"),
                    "downloading": status.get("downloading", False)
                }
            
            # Start download process
            try:
                logger.info(f"🔄 Downloading HuggingFace reranker model: {model_name}")
                download_status_cache[model_name] = {
                    "downloading": True,
                    "completed": False,
                    "message": "Starting download..."
                }
                
                from sentence_transformers import CrossEncoder
                
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                model = await loop.run_in_executor(None, lambda: CrossEncoder(model_name))
                
                # Update status on success
                download_status_cache[model_name] = {
                    "downloading": False,
                    "completed": True,
                    "message": f"Successfully downloaded {model_name}"
                }
                
                return {"success": True, "downloaded": True, "message": f"HuggingFace model {model_name} downloaded successfully"}
            except Exception as e:
                logger.error(f"Failed to download HuggingFace model {model_name}: {e}")
                download_status_cache[model_name] = {
                    "downloading": False,
                    "completed": False,
                    "message": f"Download failed: {str(e)}"
                }
                return {"success": False, "downloaded": False, "message": f"Failed to download HuggingFace model: {str(e)}"}
        
        # For Ollama models
        try:
            ollama_host = os.getenv('OLLAMA_HOST', 'http://ollama:11434')
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{ollama_host}/api/tags")
                if response.status_code == 200:
                    models_data = response.json()
                    available_models = {model.get('name', '') for model in models_data.get('models', [])}
                    
                    # Check for exact model name and common variants
                    model_variants = [
                        model_name,
                        f"{model_name}:latest",
                        model_name.replace(":latest", "")
                    ]
                    
                    model_found = any(variant in available_models for variant in model_variants)
                    
                    if model_found:
                        logger.info(f"🎯 Reranker model {model_name} is already available locally")
                        return {"success": True, "downloaded": False, "message": f"Model {model_name} already available"}
                    
                    # Model not found locally, download it
                    logger.info(f"🔄 Downloading reranker model: {model_name}")
                    
                    download_response = await client.post(
                        f"{ollama_host}/api/pull",
                        json={"name": model_name, "stream": True},
                        timeout=600.0
                    )
                    
                    if download_response.status_code != 200:
                        error_text = await download_response.text()
                        return {"success": False, "downloaded": False, "message": f"Failed to download: {error_text}"}
                    
                    # Process download stream
                    download_success = False
                    async for line in download_response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                status = data.get("status", "")
                                
                                if any(indicator in status.lower() for indicator in ["success", "pull complete", "already exists"]):
                                    download_success = True
                                    break
                            except json.JSONDecodeError:
                                continue
                    
                    if download_success:
                        return {"success": True, "downloaded": True, "message": f"Successfully downloaded reranker model {model_name}"}
                    else:
                        return {"success": False, "downloaded": False, "message": "Download verification failed"}
                else:
                    return {"success": False, "downloaded": False, "message": f"Failed to check available models: HTTP {response.status_code}"}
                    
        except httpx.TimeoutException:
            return {"success": False, "downloaded": False, "message": f"Timeout downloading reranker model {model_name}"}
        except Exception as e:
            return {"success": False, "downloaded": False, "message": f"Error with reranker model: {str(e)}"}
    
    async def _download_huggingface_model_background(self, model_name: str):
        """Download HuggingFace model in background"""
        try:
            logger.info(f"🔄 Starting background download of HuggingFace reranker model: {model_name}")
            
            download_status_cache[model_name] = {
                "downloading": True,
                "completed": False,
                "message": "Downloading model files...",
                "status": "downloading"
            }
            
            from sentence_transformers import CrossEncoder
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(None, lambda: CrossEncoder(model_name))
            
            # Update status on success
            download_status_cache[model_name] = {
                "downloading": False,
                "completed": True,
                "message": f"Successfully downloaded {model_name}",
                "status": "completed"
            }
            
            logger.info(f"✅ Successfully downloaded HuggingFace reranker model: {model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to download HuggingFace model {model_name}: {e}")
            download_status_cache[model_name] = {
                "downloading": False,
                "completed": False,
                "message": f"Download failed: {str(e)}",
                "status": "failed"
            }
    
    @staticmethod
    def _get_huggingface_models() -> List[Dict[str, Any]]:
        """Get list of HuggingFace reranker models"""
        return [
            {
                "name": "BAAI/bge-reranker-v2-m3",
                "display_name": "BGE Reranker V2 M3",
                "description": "🤗 Multilingual BGE reranking model with excellent cross-lingual performance",
                "provider": "huggingface",
                "size": "1.2B",
                "is_local": False
            },
            {
                "name": "BAAI/bge-reranker-large",
                "display_name": "BGE Reranker Large",
                "description": "🤗 Large BGE reranking model for high-accuracy document ranking",
                "provider": "huggingface",
                "size": "560M",
                "is_local": False
            },
            {
                "name": "BAAI/bge-reranker-base",
                "display_name": "BGE Reranker Base",
                "description": "🤗 Base BGE reranking model, balanced performance and speed",
                "provider": "huggingface",
                "size": "278M",
                "is_local": False
            },
            {
                "name": "jinaai/jina-reranker-v1-base-en",
                "display_name": "Jina Reranker V1 Base (English)",
                "description": "🤗 Jina's dedicated reranking model optimized for English",
                "provider": "huggingface",
                "size": "278M",
                "is_local": False
            },
            {
                "name": "jinaai/jina-reranker-v1-tiny-en",
                "display_name": "Jina Reranker V1 Tiny (English)",
                "description": "🤗 Lightweight Jina reranker for fast processing",
                "provider": "huggingface",
                "size": "33M",
                "is_local": False
            },
            {
                "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "display_name": "MS Marco MiniLM L6 V2",
                "description": "🤗 Microsoft's cross-encoder reranker trained on MS MARCO",
                "provider": "huggingface",
                "size": "90M",
                "is_local": False
            },
            {
                "name": "cross-encoder/ms-marco-MiniLM-L-12-v2",
                "display_name": "MS Marco MiniLM L12 V2",
                "description": "🤗 Larger Microsoft cross-encoder with better accuracy",
                "provider": "huggingface",
                "size": "134M",
                "is_local": False
            },
            {
                "name": "mixedbread-ai/mxbai-rerank-large-v1",
                "display_name": "MixedBread AI Rerank Large V1",
                "description": "🤗 High-performance multilingual reranking model",
                "provider": "huggingface",
                "size": "560M",
                "is_local": False
            }
        ]
    
    @staticmethod
    def _get_fallback_ollama_models() -> List[Dict[str, Any]]:
        """Get fallback Ollama models"""
        return [
            {
                "name": "linux6200/bge-reranker-v2-m3",
                "display_name": "BGE Reranker V2 M3",
                "description": "High-performance BGE reranking model (available to download)",
                "is_local": False,
                "provider": "ollama"
            },
            {
                "name": "dengcao/Qwen3-Reranker-8B",
                "display_name": "Qwen3 Reranker 8B",
                "description": "Alibaba's multilingual reranking model (available to download)",
                "is_local": False,
                "provider": "ollama"
            },
            {
                "name": "qllama/bge-reranker-large",
                "display_name": "BGE Reranker Large (Quantized)",
                "description": "Quantized BGE reranking model (available to download)",
                "is_local": False,
                "provider": "ollama"
            }
        ]
    
    @staticmethod
    def _get_fallback_models() -> List[Dict[str, Any]]:
        """Get basic fallback models on error"""
        return [
            {
                "name": "",
                "display_name": "None (Vector + Keyword)",
                "description": "Use weighted combination of vector similarity and keyword matching",
                "provider": "none"
            },
            {
                "name": "BAAI/bge-reranker-v2-m3",
                "display_name": "BGE Reranker V2 M3",
                "description": "🤗 Multilingual BGE reranking model (error occurred)",
                "provider": "huggingface",
                "is_local": False
            },
            {
                "name": "linux6200/bge-reranker-v2-m3",
                "display_name": "BGE Reranker V2 M3",
                "description": "High-performance BGE reranking model (error occurred)",
                "provider": "ollama",
                "is_local": False
            }
        ]
