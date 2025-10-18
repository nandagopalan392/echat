"""
Model Service - Business Logic for Model Management

This service handles all model-related business logic including:
- Model status and availability checks
- Model downloads and updates
- GPU compatibility verification
- Provider management (Ollama, HuggingFace)
- Model parameter validation
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
import httpx
import requests

# Import dependencies
from rag import (
    get_chatpdf_instance,
    check_model_compatibility,
    detect_model_provider,
    HuggingFaceProvider,
    create_embedding_model
)
from ollama_scraper import get_available_ollama_models
from chat_db import ChatDB
from gpu_utils import get_gpu_memory_info, check_model_compatibility_detailed

# Import utility functions
from app.utils.model_utils import (
    format_model_size,
    categorize_model,
    normalize_model_name,
    validate_model_parameters,
    generate_compatibility_recommendations,
    get_model_variants
)

logger = logging.getLogger(__name__)


class ModelService:
    """
    Service class for managing AI models (LLM and embedding models).
    
    Handles all business logic for model operations including downloading,
    configuration, GPU compatibility checks, and provider management.
    """
    
    def __init__(self):
        """Initialize the model service with database and configuration."""
        self.chat_db = ChatDB()
        self.ollama_host = os.getenv('OLLAMA_HOST', 'http://ollama:11434')
        self.logger = logging.getLogger(__name__)
    
    async def get_model_status(self) -> Dict[str, Any]:
        """
        Get status of models in Ollama.
        
        Returns:
            Dict containing current model status and availability
            
        Raises:
            Exception: If unable to connect to Ollama or retrieve status
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.ollama_host}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = data.get('models', [])
                    
                    # Get current settings
                    rag = get_chatpdf_instance()
                    current_llm = rag.llm_model
                    current_embedding = rag.embedding_model
                    
                    # Check if current models are available
                    model_names = [model['name'] for model in models]
                    llm_available = current_llm in model_names
                    embedding_available = current_embedding in model_names
                    
                    return {
                        "success": True,
                        "current_llm": current_llm,
                        "current_embedding": current_embedding,
                        "llm_available": llm_available,
                        "embedding_available": embedding_available,
                        "available_models": model_names,
                        "total_models": len(models)
                    }
                else:
                    raise Exception("Could not connect to Ollama")
                    
        except Exception as e:
            self.logger.error(f"Error getting model status: {str(e)}")
            raise
    
    async def get_available_models(self) -> Dict[str, Any]:
        """
        Get list of available models from both local Ollama and Ollama library.
        
        Returns:
            Dict containing all available models categorized by type
            
        Raises:
            Exception: If unable to fetch models
        """
        try:
            # Get locally installed models
            local_model_names = set()
            local_models = []
            
            try:
                local_models = await self._fetch_local_ollama_models()
                local_model_names = {model['name'] for model in local_models}
            except Exception as e:
                self.logger.warning(f"Could not fetch local models: {e}")
                local_model_names = set()
            
            # Get available models from Ollama library
            try:
                available_models = get_available_ollama_models(use_cache=True)
            except Exception as e:
                self.logger.warning(f"Could not fetch Ollama library models: {e}")
                available_models = []
            
            # Combine local and available models
            all_models = self._combine_models(local_models, available_models, local_model_names)
            
            # Convert to list and separate by category
            models_list = list(all_models.values())
            llm_models = [m for m in models_list if m.get('category', 'llm') == 'llm']
            embedding_models = [m for m in models_list if m.get('category', 'embedding') == 'embedding']
            
            return {
                "success": True,
                "models": models_list,
                "llm_models": llm_models,
                "embedding_models": embedding_models,
                "total_models": len(models_list)
            }
                    
        except Exception as e:
            self.logger.error(f"Error getting available models: {str(e)}")
            raise
    
    async def get_current_models(self) -> Dict[str, Any]:
        """
        Get current model settings including parameters and provider.
        
        Returns:
            Dict containing current LLM, embedding model, provider, and parameters
            
        Raises:
            Exception: If unable to retrieve current settings
        """
        try:
            rag = get_chatpdf_instance()
            
            # Load parameters from database, fallback to defaults
            parameters = {
                'temperature': 0.7,
                'max_tokens': 2048,
                'top_p': 0.9,
                'frequency_penalty': 0.0,
                'presence_penalty': 0.0
            }
            provider = "ollama"
            
            try:
                db_settings = self.chat_db.get_latest_model_settings()
                if db_settings:
                    if 'parameters' in db_settings:
                        parameters.update(db_settings['parameters'])
                    if 'provider' in db_settings:
                        provider = db_settings['provider']
                else:
                    self.logger.warning("No model settings found in database, using defaults")
            except Exception as e:
                self.logger.warning(f"Could not load parameters from database: {e}")
            
            return {
                "success": True,
                "llm": rag.llm_model,
                "embedding": rag.embedding_model,
                "provider": provider,
                "parameters": parameters
            }
        except Exception as e:
            self.logger.error(f"Error getting current models: {str(e)}")
            raise
    
    async def get_model_providers(self) -> Dict[str, Any]:
        """
        Get available model providers and their models.
        
        Returns:
            Dict containing Ollama and HuggingFace providers with their models
            
        Raises:
            Exception: If unable to fetch provider information
        """
        try:
            providers = {
                "ollama": {
                    "name": "Ollama",
                    "icon": "ollama",
                    "models": []
                },
                "huggingface": {
                    "name": "Hugging Face",
                    "icon": "huggingface", 
                    "models": await self._fetch_huggingface_models()
                }
            }
            
            # Get Ollama models
            try:
                ollama_models = await self._fetch_all_ollama_models()
                providers["ollama"]["models"] = sorted(ollama_models, key=lambda x: x['name'])
                self.logger.info(f"Found {len(ollama_models)} total Ollama models (local + library)")
            except Exception as e:
                self.logger.warning(f"Could not fetch Ollama models: {e}")
            
            # Log summary
            hf_models = providers["huggingface"]["models"]
            hf_llm_count = len([m for m in hf_models if m.get('type') == 'llm'])
            hf_emb_count = len([m for m in hf_models if m.get('type') == 'embedding'])
            self.logger.info(f"Total HuggingFace models: {len(hf_models)} (LLM: {hf_llm_count}, Embedding: {hf_emb_count})")
            
            return {
                "success": True,
                "providers": providers
            }
        except Exception as e:
            self.logger.error(f"Error getting model providers: {str(e)}")
            raise
    
    async def check_gpu_compatibility(
        self,
        llm_model: str,
        embedding_model: str
    ) -> Dict[str, Any]:
        """
        Check GPU compatibility for selected models.
        
        Args:
            llm_model: Name of the LLM model
            embedding_model: Name of the embedding model
            
        Returns:
            Dict containing compatibility results and recommendations
            
        Raises:
            Exception: If compatibility check fails
        """
        try:
            if not llm_model or not embedding_model:
                return {
                    "success": False,
                    "compatible": False,
                    "message": "Both LLM and embedding models must be specified"
                }
            
            self.logger.info(f"Checking GPU compatibility for LLM: {llm_model}, Embedding: {embedding_model}")
            
            # Get model information for size estimation
            available_models_data = await self.get_available_models()
            
            llm_info = None
            embedding_info = None
            
            # Find model information
            for model in available_models_data.get('models', []):
                if model['name'] == llm_model:
                    llm_info = model
                elif model['name'] == embedding_model:
                    embedding_info = model
            
            # Check individual model compatibility
            llm_compatible, llm_message, llm_details = check_model_compatibility_detailed(
                llm_model, 
                llm_info.get('size') if llm_info else None
            )
            
            embedding_compatible, embedding_message, embedding_details = check_model_compatibility_detailed(
                embedding_model, 
                embedding_info.get('size') if embedding_info else None
            )
            
            self.logger.info(f"LLM compatibility check: {llm_compatible} - {llm_message}")
            self.logger.info(f"Embedding compatibility check: {embedding_compatible} - {embedding_message}")
            
            # Combined compatibility check
            total_required_mb = llm_details['required_memory_mb'] + embedding_details['required_memory_mb']
            gpu_info = get_gpu_memory_info()
            buffer_memory = max(1024, int(gpu_info['total'] * 0.2))
            usable_memory = gpu_info['total'] - buffer_memory
            
            combined_compatible = total_required_mb <= usable_memory
            
            return {
                "success": True,
                "compatible": llm_compatible and embedding_compatible and combined_compatible,
                "llm_check": {
                    "model": llm_model,
                    "compatible": llm_compatible,
                    "estimated_memory_mb": llm_details['required_memory_mb'],
                    "message": llm_message,
                    "details": llm_details
                },
                "embedding_check": {
                    "model": embedding_model,
                    "compatible": embedding_compatible,
                    "estimated_memory_mb": embedding_details['required_memory_mb'],
                    "message": embedding_message,
                    "details": embedding_details
                },
                "combined_check": {
                    "required_mb": total_required_mb,
                    "usable_mb": usable_memory,
                    "compatible": combined_compatible,
                    "message": f"Combined models require {total_required_mb}MB, {usable_memory}MB usable from {gpu_info['total']}MB total (after {buffer_memory}MB buffer)",
                    "shortage_mb": max(0, total_required_mb - usable_memory)
                },
                "gpu_info": gpu_info,
                "recommendation": (
                    "Models should fit in available GPU memory" 
                    if combined_compatible 
                    else "Consider using smaller models or upgrading GPU memory"
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error checking GPU compatibility: {str(e)}")
            return {
                "success": False,
                "compatible": False,
                "message": f"Failed to check GPU compatibility: {str(e)}"
            }
    
    # ==================== Private Helper Methods ====================
    
    async def _fetch_local_ollama_models(self) -> List[Dict]:
        """Fetch locally installed Ollama models."""
        local_models = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                data = response.json()
                
                for model in data.get('models', []):
                    model_name = model.get('name', '')
                    model_size = model.get('size', 0)
                    model_modified = model.get('modified_at', '')
                    
                    category = self._categorize_model(model_name)
                    
                    local_models.append({
                        'name': model_name,
                        'category': category,
                        'size': self._format_model_size(model_size),
                        'modified_at': model_modified,
                        'source': 'local',
                        'description': f"Locally installed {category} model"
                    })
        
        return local_models
    
    async def _fetch_all_ollama_models(self) -> List[Dict]:
        """Fetch all Ollama models (local + library)."""
        # Get locally installed models
        local_models = []
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.ollama_host}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    
                    for model in data.get('models', []):
                        model_name = model.get('name', '')
                        model_size = model.get('size', 0)
                        model_modified = model.get('modified_at', '')
                        
                        model_type = self._categorize_model(model_name)
                        
                        local_models.append({
                            'name': model_name,
                            'type': model_type,
                            'category': model_type,
                            'size': self._format_model_size(model_size),
                            'modified_at': model_modified,
                            'source': 'local',
                            'provider': 'ollama',
                            'description': f"Locally installed {model_type} model"
                        })
                else:
                    self.logger.warning(f"Could not fetch local Ollama models: {response.status_code}")
        except Exception as e:
            self.logger.warning(f"Could not fetch local Ollama models: {e}")
        
        # Get available models from Ollama library
        try:
            available_models = get_available_ollama_models(use_cache=True)
            self.logger.info(f"Found {len(available_models)} models from Ollama library")
        except Exception as e:
            self.logger.warning(f"Could not fetch Ollama library models: {e}")
            available_models = []
        
        # Combine local and available models
        all_ollama_models = {}
        
        # Add local models first (priority)
        for model in local_models:
            all_ollama_models[model['name']] = model
        
        # Add available models that aren't already local
        for model in available_models:
            model_name = model['name']
            if model_name not in all_ollama_models:
                model_type = self._categorize_model(model_name)
                
                all_ollama_models[model_name] = {
                    'name': model_name,
                    'type': model_type,
                    'category': model_type,
                    'description': model.get('description', ''),
                    'size': self._format_model_size(model.get('size', 'Unknown')),
                    'source': 'library',
                    'provider': 'ollama',
                    'tags': model.get('tags', [])
                }
            else:
                # Update local model with library info
                all_ollama_models[model_name].update({
                    'description': model.get('description', all_ollama_models[model_name].get('description', '')),
                    'tags': model.get('tags', [])
                })
        
        return list(all_ollama_models.values())
    
    async def _fetch_huggingface_models(self) -> List[Dict]:
        """Fetch available HuggingFace models."""
        try:
            from huggingface_hub import list_models
            self.logger.info("Fetching models from Hugging Face Hub using official SDK...")
            
            def fetch_models_sdk(task: str, max_models: int = 100):
                """Fetch models from HuggingFace Hub using the SDK."""
                models = []
                try:
                    for model in list_models(filter=task, sort="downloads", direction=-1, limit=max_models * 2):
                        if model.id.startswith("private/") or "/" not in model.id:
                            continue
                        
                        model_type = "llm" if task == "text-generation" else "embedding"
                        
                        # For embedding models, apply stricter filtering
                        if task in ["sentence-similarity", "feature-extraction"]:
                            model_name_lower = model.id.lower()
                            if any(keyword in model_name_lower for keyword in [
                                "sentence-transformers/", "embedding", "embed", "bge-", "e5-", 
                                "gte-", "multilingual", "mpnet", "minilm", "distilbert", 
                                "roberta", "bert-base", "msmarco", "paraphrase"
                            ]):
                                if not any(exclude in model_name_lower for exclude in [
                                    "bart", "gpt", "t5-", "flan", "bloom", "llama", "mistral", 
                                    "gemma", "qwen", "falcon", "vicuna", "generation", "chat"
                                ]):
                                    models.append({
                                        "name": model.id,
                                        "downloads": getattr(model, 'downloads', 0) or 0,
                                        "likes": getattr(model, 'likes', 0) or 0,
                                        "type": "embedding",
                                        "provider": "huggingface",
                                        "task": task,
                                        "last_modified": getattr(model, 'lastModified', None)
                                    })
                        else:
                            models.append({
                                "name": model.id,
                                "downloads": getattr(model, 'downloads', 0) or 0,
                                "likes": getattr(model, 'likes', 0) or 0,
                                "type": model_type,
                                "provider": "huggingface",
                                "task": task,
                                "last_modified": getattr(model, 'lastModified', None)
                            })
                        
                        if len(models) >= max_models:
                            break
                            
                except Exception as e:
                    self.logger.warning(f"Failed to fetch {task} models from HF Hub SDK: {e}")
                    
                return models
            
            # Fetch different model types
            llm_models = fetch_models_sdk("text-generation", max_models=150)
            embedding_models = fetch_models_sdk("sentence-similarity", max_models=100)
            feature_extraction_models = fetch_models_sdk("feature-extraction", max_models=50)
            
            all_embedding_models = embedding_models + [
                {**model, "type": "embedding"} for model in feature_extraction_models
            ]
            
            # Fallback to curated list if too few models
            if len(llm_models) < 10 and len(all_embedding_models) < 5:
                self.logger.info("SDK returned too few models, using fallback curated list")
                return self._get_fallback_huggingface_models()
            
            # Sort by downloads
            llm_models.sort(key=lambda x: x.get('downloads', 0), reverse=True)
            all_embedding_models.sort(key=lambda x: x.get('downloads', 0), reverse=True)
            
            all_models = llm_models + all_embedding_models
            self.logger.info(f"Total HuggingFace models fetched: {len(all_models)} (LLM: {len(llm_models)}, Embedding: {len(all_embedding_models)})")
            
            return all_models
            
        except ImportError:
            self.logger.warning("huggingface_hub not available, using fallback")
            return self._get_fallback_huggingface_models()
        except Exception as e:
            self.logger.error(f"Error fetching Hugging Face models: {e}")
            return self._get_fallback_huggingface_models()
    
    def _get_fallback_huggingface_models(self) -> List[Dict]:
        """Get fallback list of HuggingFace models."""
        return [
            {"name": "microsoft/DialoGPT-medium", "type": "llm", "provider": "huggingface", "downloads": 50000},
            {"name": "google/flan-t5-base", "type": "llm", "provider": "huggingface", "downloads": 75000},
            {"name": "sentence-transformers/all-MiniLM-L6-v2", "type": "embedding", "provider": "huggingface", "downloads": 200000}
        ]
    
    def _combine_models(
        self,
        local_models: List[Dict],
        available_models: List[Dict],
        local_model_names: set
    ) -> Dict[str, Dict]:
        """Combine local and library models, prioritizing local."""
        all_models = {}
        
        # Add local models first
        for model in local_models:
            all_models[model['name']] = model
        
        # Add available models that aren't local
        for model in available_models:
            model_name = model['name']
            if model_name not in all_models:
                model_category = categorize_model(model_name)
                
                all_models[model_name] = {
                    'name': model_name,
                    'category': model_category,
                    'description': model.get('description', ''),
                    'size': format_model_size(model.get('size', 'Unknown')),
                    'source': 'library',
                    'tags': model.get('tags', [])
                }
            else:
                # Update local model with library info
                all_models[model_name].update({
                    'description': model.get('description', all_models[model_name].get('description', '')),
                    'tags': model.get('tags', [])
                })
        
        return all_models
    
    async def update_model_settings(
        self, 
        llm_model: str, 
        embedding_model: str,
        llm_size: Optional[str] = None,
        embedding_size: Optional[str] = None,
        force_update: bool = False,
        model_parameters: Optional[Dict[str, Any]] = None,
        provider: str = 'ollama',
        embedding_provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update model settings with full validation and GPU compatibility checks.
        
        Args:
            llm_model: LLM model name
            embedding_model: Embedding model name
            llm_size: Optional LLM model size
            embedding_size: Optional embedding model size
            force_update: Bypass GPU compatibility check
            model_parameters: Optional model parameters
            provider: LLM provider (ollama/huggingface)
            embedding_provider: Embedding provider (defaults to LLM provider)
            
        Returns:
            Dict with update status and details
        """
        try:
            # Use LLM provider as default for embedding if not specified
            if not embedding_provider:
                embedding_provider = provider
            
            # Validate and set model parameters
            valid_parameters = validate_model_parameters(model_parameters)
            
            # Check GPU compatibility before proceeding (unless forced)
            if not force_update:
                try:
                    # Check LLM model compatibility
                    llm_compatible, llm_message, llm_details = check_model_compatibility(llm_model, llm_size)
                    
                    # Check embedding model compatibility
                    embedding_compatible, embedding_message, embedding_details = check_model_compatibility(
                        embedding_model, embedding_size
                    )
                    
                    # Calculate combined memory requirement
                    combined_memory = llm_details['required_memory_mb'] + embedding_details['required_memory_mb']
                    available_memory = llm_details['available_memory_mb']
                    combined_compatible = combined_memory <= available_memory
                    
                    if not (llm_compatible and embedding_compatible and combined_compatible):
                        # Models are not compatible with current GPU
                        error_details = {
                            "error": "GPU_MEMORY_INSUFFICIENT",
                            "message": "Selected models require more GPU memory than available",
                            "llm_check": {
                                "compatible": llm_compatible,
                                "message": llm_message,
                                "required_mb": llm_details['required_memory_mb']
                            },
                            "embedding_check": {
                                "compatible": embedding_compatible,
                                "message": embedding_message,
                                "required_mb": embedding_details['required_memory_mb']
                            },
                            "combined_check": {
                                "compatible": combined_compatible,
                                "required_mb": combined_memory,
                                "available_mb": available_memory,
                                "shortage_mb": max(0, combined_memory - available_memory)
                            },
                            "recommendations": generate_compatibility_recommendations(
                                llm_details, embedding_details, combined_compatible
                            )
                        }
                        
                        raise ValueError(json.dumps(error_details))
                    
                    logger.info(f"✅ GPU compatibility check passed for models {llm_model} + {embedding_model}")
                    
                except ValueError:
                    raise
                except Exception as e:
                    logger.warning(f"Could not check GPU compatibility: {str(e)}, proceeding anyway")
            
            # Get current models
            rag = get_chatpdf_instance()
            current_llm = rag.llm_model
            current_embedding = rag.embedding_model
            
            logger.info(f"Current models - LLM: '{current_llm}', Embedding: '{current_embedding}'")
            logger.info(f"Requested models - LLM: '{llm_model}', Embedding: '{embedding_model}'")
            
            models_unchanged = (current_llm == llm_model and current_embedding == embedding_model)
            
            # If models unchanged, only update parameters
            if models_unchanged:
                logger.info("Models unchanged, only updating parameters - skipping model downloads")
                
                try:
                    rag.update_models(llm_model, embedding_model, provider=provider)
                except Exception as e:
                    logger.error(f"Error updating parameters: {str(e)}")
                    raise Exception(f"Failed to update parameters: {str(e)}")
                
                # Save settings to database
                try:
                    self.chat_db.save_model_settings(
                        llm_model, embedding_model, valid_parameters, provider, embedding_provider
                    )
                    logger.info(f"Model settings saved to database")
                except Exception as e:
                    logger.error(f"Could not save to database: {e}")
                    raise Exception(f"Failed to save model settings: {e}")
                
                return {
                    "success": True,
                    "message": "Model parameters updated successfully (no downloads needed)",
                    "llm": llm_model,
                    "embedding": embedding_model,
                    "embedding_changed": False,
                    "downloaded_models": [],
                    "parameters_only": True
                }
            
            # Models changed - download if needed
            models_to_download = []
            
            # Detect providers from model names
            llm_provider = provider if provider != 'ollama' else detect_model_provider(llm_model)
            embedding_provider_final = embedding_provider if embedding_provider != 'ollama' else detect_model_provider(
                embedding_model
            )
            
            logger.info(
                f"Using providers - LLM: {llm_provider} for {llm_model}, "
                f"Embedding: {embedding_provider_final} for {embedding_model}"
            )
            
            # Download Ollama models if needed
            ollama_host = os.getenv('OLLAMA_HOST', 'http://ollama:11434')
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get currently installed Ollama models
                try:
                    response = await client.get(f"{ollama_host}/api/tags")
                    if response.status_code == 200:
                        data = response.json()
                        installed_models = {model.get('name', '') for model in data.get('models', [])}
                    else:
                        installed_models = set()
                except Exception as e:
                    logger.warning(f"Could not fetch installed models: {e}")
                    installed_models = set()
                
                # Check and download LLM model (if Ollama)
                if llm_provider.lower() == 'ollama':
                    if llm_model not in installed_models:
                        llm_variants = [llm_model, f"{llm_model}:latest", llm_model.replace(":latest", "")]
                        if not any(variant in installed_models for variant in llm_variants):
                            models_to_download.append(llm_model)
                            logger.info(f"Ollama LLM model {llm_model} needs to be downloaded")
                
                # Check and download embedding model (if Ollama)
                if embedding_provider_final.lower() == 'ollama':
                    embedding_model_normalized = normalize_model_name(embedding_model)
                    if embedding_model_normalized not in installed_models:
                        embedding_variants = [
                            embedding_model_normalized,
                            f"{embedding_model_normalized}:latest",
                            embedding_model_normalized.replace(":latest", "")
                        ]
                        if not any(variant in installed_models for variant in embedding_variants):
                            models_to_download.append(embedding_model_normalized)
                            logger.info(f"Ollama embedding model {embedding_model_normalized} needs to be downloaded")
                    
                    # Use normalized name
                    embedding_model = embedding_model_normalized
                
                # Download missing Ollama models
                for model_name in models_to_download:
                    logger.info(f"Downloading model: {model_name}")
                    try:
                        download_response = await client.post(
                            f"{ollama_host}/api/pull",
                            json={"name": model_name, "stream": True},
                            timeout=600.0
                        )
                        
                        if download_response.status_code != 200:
                            logger.error(f"Failed to download model {model_name}: {download_response.status_code}")
                            raise Exception(f"Failed to download model {model_name}: HTTP {download_response.status_code}")
                        
                        # Process streaming response
                        async for line in download_response.aiter_lines():
                            if line:
                                try:
                                    data = json.loads(line)
                                    status = data.get("status", "")
                                    if "success" in status.lower() or "pull complete" in status.lower():
                                        logger.info(f"Model {model_name}: Download completed successfully")
                                        break
                                except json.JSONDecodeError:
                                    continue
                        
                    except httpx.TimeoutException:
                        logger.error(f"Timeout downloading model {model_name}")
                        raise Exception(f"Timeout downloading model {model_name}. Please try again.")
                    except Exception as e:
                        logger.error(f"Error downloading model {model_name}: {e}")
                        raise Exception(f"Failed to download model {model_name}: {str(e)}")
            
            # Update models in RAG system
            embedding_changed = rag.embedding_model != embedding_model
            
            try:
                rag.update_models(llm_model, embedding_model, provider=provider, embedding_provider=embedding_provider_final)
            except Exception as e:
                logger.error(f"Error updating models: {str(e)}")
                raise Exception(f"Failed to update models: {str(e)}")
            
            # Save settings to database
            try:
                self.chat_db.save_model_settings(
                    llm_model, embedding_model, valid_parameters, provider, embedding_provider_final
                )
                logger.info(f"Model settings saved to database")
            except Exception as e:
                logger.error(f"Could not save to database: {e}")
                raise Exception(f"Failed to save model settings: {e}")
            
            response_data = {
                "success": True,
                "message": "Models updated successfully",
                "llm": llm_model,
                "embedding": embedding_model,
                "embedding_changed": embedding_changed,
                "downloaded_models": models_to_download
            }
            
            # Add download info to message
            if models_to_download:
                downloaded_list = ", ".join(models_to_download)
                response_data["message"] += f". Downloaded models: {downloaded_list}"
            
            # Handle embedding model change
            if embedding_changed:
                response_data["message"] += ". Embedding model changed - starting automatic reingestion of all documents."
                response_data["reingestion_started"] = True
                
                # Start reingestion in background
                try:
                    reingestion_result = rag.reingest_all_documents()
                    if reingestion_result:
                        response_data["message"] += " Reingestion completed successfully."
                        response_data["reingestion_success"] = True
                    else:
                        response_data["message"] += " Reingestion failed. Please reingest documents manually."
                        response_data["reingestion_success"] = False
                except Exception as reingest_error:
                    logger.warning(f"Automatic reingestion failed: {reingest_error}")
                    response_data["message"] += " Automatic reingestion failed. Please reingest documents manually."
                    response_data["reingestion_success"] = False
                    response_data["reingestion_error"] = str(reingest_error)
                response_data["reingest_suggested"] = True
            
            return response_data
            
        except ValueError as e:
            # Re-raise validation errors with their structured data
            raise
        except Exception as e:
            logger.error(f"Error updating model settings: {str(e)}")
            raise Exception(f"Failed to update model settings: {str(e)}")
    
    async def download_huggingface_model(
        self, 
        model_name: str, 
        model_type: str = 'llm'
    ) -> Dict[str, Any]:
        """
        Download and validate HuggingFace model.
        
        Args:
            model_name: HuggingFace model name
            model_type: Model type ('llm' or 'embedding')
            
        Returns:
            Dict with download status and details
        """
        try:
            logger.info(f"Attempting to download HuggingFace model: {model_name} (type: {model_type})")
            
            if model_type == 'llm':
                # Create HuggingFace provider and attempt to load model
                provider = HuggingFaceProvider(model_name, {})
                
                try:
                    # This will check if model is downloaded and download if needed
                    model = provider.create_model()
                    
                    # If we get here, download was successful
                    return {
                        "success": True,
                        "message": f"Successfully downloaded and validated LLM model: {model_name}",
                        "model_name": model_name,
                        "model_type": model_type,
                        "ready_to_use": True
                    }
                    
                except ValueError as e:
                    # Check if this is a gated model error
                    error_str = str(e)
                    if error_str.startswith("GATED_MODEL_ERROR:"):
                        # Parse the structured error data
                        try:
                            error_data = json.loads(error_str.replace("GATED_MODEL_ERROR:", ""))
                            return {
                                "success": False,
                                "error_type": "gated_model",
                                "model_name": model_name,
                                "model_type": model_type,
                                "message": error_data["message"],
                                "model_url": error_data["model_url"],
                                "steps": error_data["steps"],
                                "alternatives": error_data.get("alternatives", [])
                            }
                        except json.JSONDecodeError:
                            pass
                    raise Exception(str(e))
                    
                except Exception as e:
                    logger.error(f"Failed to download LLM model {model_name}: {e}")
                    raise Exception(f"Failed to download model: {str(e)}")
            
            elif model_type == 'embedding':
                try:
                    # Create embedding model and attempt to load
                    embedding_model = create_embedding_model(model_name, 'huggingface')
                    
                    # Test the embedding model
                    test_embedding = embedding_model.embed_query("test")
                    
                    if test_embedding and len(test_embedding) > 0:
                        return {
                            "success": True,
                            "message": f"Successfully downloaded and validated embedding model: {model_name}",
                            "model_name": model_name,
                            "model_type": model_type,
                            "embedding_dimension": len(test_embedding),
                            "ready_to_use": True
                        }
                    else:
                        raise Exception("Model downloaded but failed validation test")
                        
                except Exception as e:
                    logger.error(f"Failed to download embedding model {model_name}: {e}")
                    
                    # Check for gated model error
                    if "gated" in str(e).lower() or "access" in str(e).lower():
                        model_url = f"https://huggingface.co/{model_name}"
                        return {
                            "success": False,
                            "error_type": "gated_model",
                            "model_name": model_name,
                            "model_type": model_type,
                            "message": f"Model '{model_name}' requires special access",
                            "model_url": model_url,
                            "steps": [
                                f"Visit the model page: {model_url}",
                                "Click 'Request access' button",
                                "Wait for approval",
                                "Ensure you have a valid HuggingFace token in your .env file",
                                "Try again after approval is granted"
                            ]
                        }
                    
                    raise Exception(f"Failed to download embedding model: {str(e)}")
            
            else:
                raise ValueError("model_type must be 'llm' or 'embedding'")
                
        except Exception as e:
            logger.error(f"Error in download_huggingface_model: {str(e)}")
            raise Exception(f"Failed to download HuggingFace model: {str(e)}")
    
    async def update_simple_settings(
        self, 
        llm_model: str, 
        embedding_model: str,
        model_parameters: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None,
        embedding_provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update model settings with simplified validation (for basic UI).
        More permissive than update_model_settings, with warnings instead of blocking errors.
        
        Args:
            llm_model: LLM model name
            embedding_model: Embedding model name
            model_parameters: Optional model parameters
            provider: Optional provider (auto-detected if not provided)
            embedding_provider: Optional embedding provider
            
        Returns:
            Dict with update status and details
        """
        try:
            logger.info(f"Simple settings update - LLM: {llm_model}, Embedding: {embedding_model}")
            
            # Normalize model names - fix common issues
            embedding_model = normalize_model_name(embedding_model)
            
            # Detect providers
            if provider:
                detected_provider = provider
                logger.info(f"Using requested provider '{detected_provider}' for LLM model '{llm_model}'")
            else:
                detected_provider = detect_model_provider(llm_model)
                logger.info(f"Auto-detected provider '{detected_provider}' for LLM model '{llm_model}'")
            
            if embedding_provider:
                detected_embedding_provider = embedding_provider
                logger.info(
                    f"Using requested embedding provider '{detected_embedding_provider}' "
                    f"for embedding model '{embedding_model}'"
                )
            else:
                detected_embedding_provider = detect_model_provider(embedding_model)
                logger.info(
                    f"Auto-detected embedding provider '{detected_embedding_provider}' "
                    f"for embedding model '{embedding_model}'"
                )
            
            # Validate model parameters
            valid_parameters = validate_model_parameters(model_parameters)
            
            # Check GPU compatibility (warnings only, don't block)
            compatibility_warnings = []
            try:
                logger.info(f"Checking GPU compatibility for LLM: {llm_model}, Embedding: {embedding_model}")
                
                llm_compatible, llm_message, llm_details = check_model_compatibility(llm_model)
                embedding_compatible, embedding_message, embedding_details = check_model_compatibility(embedding_model)
                
                combined_memory = llm_details['required_memory_mb'] + embedding_details['required_memory_mb']
                available_memory = llm_details['available_memory_mb']
                combined_compatible = combined_memory <= available_memory
                
                if not llm_compatible:
                    compatibility_warnings.append(f"LLM model '{llm_model}' may not fit in GPU memory: {llm_message}")
                
                if not embedding_compatible:
                    compatibility_warnings.append(
                        f"Embedding model '{embedding_model}' may not fit in GPU memory: {embedding_message}"
                    )
                
                if not combined_compatible:
                    shortage = combined_memory - available_memory
                    compatibility_warnings.append(
                        f"Combined models may require ~{combined_memory}MB but only {available_memory}MB available "
                        f"(potential shortage: {shortage}MB)"
                    )
                
                if compatibility_warnings:
                    logger.warning("GPU compatibility warnings (proceeding): " + "; ".join(compatibility_warnings))
                
            except Exception as e:
                logger.warning(f"Could not check GPU compatibility: {str(e)}, proceeding anyway")
                compatibility_warnings = [f"Could not verify GPU compatibility: {str(e)}"]
            
            # Get current models
            rag = get_chatpdf_instance()
            current_llm = rag.llm_model
            current_embedding = rag.embedding_model
            
            models_unchanged = (current_llm == llm_model and current_embedding == embedding_model)
            
            # If models unchanged, only update parameters
            if models_unchanged:
                logger.info("Models unchanged, only updating parameters - skipping model downloads")
                
                try:
                    rag.update_models(
                        llm_model, embedding_model, 
                        provider=detected_provider, embedding_provider=detected_embedding_provider
                    )
                except Exception as e:
                    logger.error(f"Error updating parameters: {str(e)}")
                    raise Exception(f"Failed to update parameters: {str(e)}")
                
                # Save settings to database
                try:
                    self.chat_db.save_model_settings(
                        llm_model, embedding_model, valid_parameters, 
                        detected_provider, detected_embedding_provider
                    )
                    logger.info(f"Model settings saved to database")
                except Exception as e:
                    logger.error(f"Could not save to database: {e}")
                    raise Exception(f"Failed to save model settings: {e}")
                
                response_data = {
                    "success": True,
                    "message": "Model parameters updated successfully (no downloads needed)",
                    "llm": llm_model,
                    "embedding": embedding_model,
                    "embedding_changed": False,
                    "downloaded_models": [],
                    "parameters_only": True
                }
                
                if compatibility_warnings:
                    response_data["gpu_warnings"] = compatibility_warnings
                
                return response_data
            
            # Models changed - need to download and update
            models_to_download = []
            embedding_changed = rag.embedding_model != embedding_model
            
            # Try to initialize models with provider-specific logic
            if detected_provider == 'huggingface':
                # For HuggingFace, attempt direct initialization
                logger.info(f"Testing HuggingFace model download: {llm_model}")
                hf_provider = HuggingFaceProvider(llm_model)
                
                download_result = hf_provider._download_model_if_needed(llm_model)
                
                if download_result.get('success') == False:
                    logger.error(f"HuggingFace model download failed: {download_result}")
                    
                    if download_result.get('error') in ['gated_no_token', 'gated_no_access']:
                        gated_error_json = {
                            'error_type': download_result['error'],
                            'model_name': llm_model,
                            'model_url': download_result['model_url'],
                            'message': download_result['message'],
                            'steps': download_result['steps']
                        }
                        
                        if 'alternatives' in download_result:
                            gated_error_json['alternatives'] = download_result['alternatives']
                        
                        return {
                            "success": True,
                            "message": f"GATED_MODEL_ERROR:{json.dumps(gated_error_json)}",
                            "llm": llm_model,
                            "embedding": embedding_model,
                            "embedding_changed": False,
                            "downloaded_models": [],
                            "gated_model_error": True
                        }
                    else:
                        raise Exception(
                            f"Failed to download HuggingFace model {llm_model}: "
                            f"{download_result.get('message', 'Unknown error')}"
                        )
                
                models_to_download.append(llm_model)
                
            else:
                # For Ollama, use the download logic
                ollama_host = os.getenv('OLLAMA_HOST', 'http://ollama:11434')
                async with httpx.AsyncClient(timeout=30.0) as client:
                    try:
                        response = await client.get(f"{ollama_host}/api/tags")
                        if response.status_code == 200:
                            data = response.json()
                            installed_models = {model.get('name', '') for model in data.get('models', [])}
                        else:
                            installed_models = set()
                    except Exception as e:
                        logger.warning(f"Could not fetch installed models: {e}")
                        installed_models = set()
                    
                    # Check if models need downloading
                    if llm_model not in installed_models:
                        models_to_download.append(llm_model)
                        
                    if embedding_model not in installed_models:
                        models_to_download.append(embedding_model)
                    
                    # Download missing models
                    for model_name in models_to_download:
                        logger.info(f"Downloading model: {model_name}")
                        try:
                            download_response = await client.post(
                                f"{ollama_host}/api/pull",
                                json={"name": model_name},
                                timeout=600.0
                            )
                            
                            if download_response.status_code != 200:
                                logger.error(f"Failed to download model {model_name}: {download_response.status_code}")
                                raise Exception(f"Failed to download model {model_name}")
                            else:
                                logger.info(f"Successfully downloaded model: {model_name}")
                                
                        except httpx.TimeoutException:
                            logger.error(f"Timeout downloading model {model_name}")
                            raise Exception(f"Timeout downloading model {model_name}. Please try again.")
                        except Exception as e:
                            logger.error(f"Error downloading model {model_name}: {e}")
                            raise Exception(f"Failed to download model {model_name}: {str(e)}")
            
            # Update the models with provider-specific logic
            rag.update_models(
                llm_model, embedding_model, 
                provider=detected_provider, embedding_provider=detected_embedding_provider
            )
            
            # Save settings to database
            try:
                self.chat_db.save_model_settings(
                    llm_model, embedding_model, valid_parameters, 
                    detected_provider, detected_embedding_provider
                )
                logger.info(f"Model settings saved to database after successful download")
            except Exception as e:
                logger.error(f"Could not save to database: {e}")
                raise Exception(f"Failed to save model settings: {e}")
            
            response_data = {
                "success": True,
                "message": "Models updated successfully",
                "llm": llm_model,
                "embedding": embedding_model,
                "embedding_changed": embedding_changed,
                "downloaded_models": models_to_download
            }
            
            if compatibility_warnings:
                response_data["gpu_warnings"] = compatibility_warnings
                response_data["message"] += f". GPU compatibility warnings: {'; '.join(compatibility_warnings)}"
            
            if models_to_download:
                downloaded_list = ", ".join(models_to_download)
                response_data["message"] += f". Downloaded models: {downloaded_list}"
            
            if embedding_changed:
                response_data["message"] += ". Embedding model changed - you may want to re-ingest documents."
                response_data["reingest_suggested"] = True
            
            return response_data
            
        except Exception as e:
            # Check if the error message contains gated model information
            error_str = str(e)
            if 'GATED_MODEL_ERROR:' in error_str:
                return {
                    "success": True,
                    "message": error_str,
                    "llm": llm_model,
                    "embedding": embedding_model,
                    "embedding_changed": False,
                    "downloaded_models": [],
                    "gated_model_error": True
                }
            else:
                logger.error(f"Error in update_simple_settings: {str(e)}")
                raise Exception(f"Failed to update models: {str(e)}")

