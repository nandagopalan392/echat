"""
HuggingFace Provider - Handles HuggingFace-specific model operations

This provider manages HuggingFace models including downloading, validation,
and handling gated models that require special access.
"""

import json
import logging
from typing import List, Dict, Any, Optional
import requests

from .base import BaseModelProvider
from app.utils.rag_utils import create_embedding_model


class HuggingFaceProvider(BaseModelProvider):
    """
    Provider for HuggingFace models.
    
    Handles all HuggingFace-specific operations including:
    - Fetching available LLM and embedding models
    - Downloading and validating models
    - Handling gated models (models requiring access approval)
    - Model metadata and capabilities
    """
    
    def _initialize(self):
        """Initialize HuggingFace-specific configuration."""
        self.default_llm_limit = 150
        self.default_embedding_limit = 100
        self.request_timeout = 30.0
        
        # Curated list of popular HuggingFace models
        self.curated_llm_models = [
            'meta-llama/Llama-2-7b-hf',
            'meta-llama/Llama-2-13b-hf',
            'mistralai/Mistral-7B-v0.1',
            'mistralai/Mixtral-8x7B-v0.1',
            'tiiuae/falcon-7b',
            'tiiuae/falcon-40b',
            'google/flan-t5-base',
            'google/flan-t5-large',
            'google/flan-t5-xl',
            'EleutherAI/gpt-neo-2.7B',
            'EleutherAI/gpt-j-6B',
            'bigscience/bloom-7b1',
            'stabilityai/stablelm-base-alpha-7b'
        ]
        
        self.curated_embedding_models = [
            'sentence-transformers/all-MiniLM-L6-v2',
            'sentence-transformers/all-mpnet-base-v2',
            'sentence-transformers/multi-qa-mpnet-base-dot-v1',
            'BAAI/bge-small-en-v1.5',
            'BAAI/bge-base-en-v1.5',
            'BAAI/bge-large-en-v1.5',
            'BAAI/bge-m3',
            'thenlper/gte-small',
            'thenlper/gte-base',
            'thenlper/gte-large',
            'intfloat/e5-small-v2',
            'intfloat/e5-base-v2',
            'intfloat/e5-large-v2',
            'nomic-ai/nomic-embed-text-v1',
            'Snowflake/snowflake-arctic-embed-m',
            'mixedbread-ai/mxbai-embed-large-v1'
        ]
        
        self.logger.info("Initialized HuggingFaceProvider")
    
    async def get_installed_models(self) -> List[Dict[str, Any]]:
        """
        Get list of locally cached HuggingFace models.
        
        Note: HuggingFace models are downloaded to cache on first use.
        This method returns an empty list as we don't track cached models.
        
        Returns:
            Empty list (HF models are managed by transformers library)
        """
        # HuggingFace models are cached by the transformers library
        # We don't have a direct way to query what's cached
        self.logger.info("HuggingFace installed models query - returning empty (models managed by transformers cache)")
        return []
    
    async def get_available_models(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get list of available HuggingFace models (both LLM and embedding).
        
        Args:
            limit: Optional limit on number of models to return
            
        Returns:
            Combined list of LLM and embedding models
        """
        try:
            llm_limit = limit if limit else self.default_llm_limit
            emb_limit = limit if limit else self.default_embedding_limit
            
            llm_models = await self.get_available_llm_models(llm_limit)
            embedding_models = await self.get_available_embedding_models(emb_limit)
            
            all_models = llm_models + embedding_models
            
            if limit and len(all_models) > limit:
                all_models = all_models[:limit]
            
            self.logger.info(f"Found {len(all_models)} available HuggingFace models")
            return all_models
            
        except Exception as e:
            self.logger.error(f"Error fetching available HuggingFace models: {e}")
            return self._get_fallback_models()
    
    async def get_available_llm_models(self, limit: int = 150) -> List[Dict[str, Any]]:
        """
        Get available LLM models from HuggingFace.
        
        Args:
            limit: Maximum number of models to return
            
        Returns:
            List of LLM model dictionaries
        """
        try:
            # Try to fetch from HuggingFace Hub API
            models = []
            
            # For now, return curated list
            # In production, you could use huggingface_hub.list_models()
            for model_name in self.curated_llm_models[:limit]:
                models.append({
                    'name': model_name,
                    'type': 'llm',
                    'source': 'huggingface',
                    'description': f'HuggingFace LLM model: {model_name}',
                    'installed': False,
                    'tags': ['llm', 'huggingface', 'text-generation']
                })
            
            self.logger.info(f"Returning {len(models)} curated HuggingFace LLM models")
            return models
            
        except Exception as e:
            self.logger.error(f"Error fetching HuggingFace LLM models: {e}")
            return []
    
    async def get_available_embedding_models(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get available embedding models from HuggingFace.
        
        Args:
            limit: Maximum number of models to return
            
        Returns:
            List of embedding model dictionaries
        """
        try:
            models = []
            
            for model_name in self.curated_embedding_models[:limit]:
                models.append({
                    'name': model_name,
                    'type': 'embedding',
                    'source': 'huggingface',
                    'description': f'HuggingFace embedding model: {model_name}',
                    'installed': False,
                    'tags': ['embedding', 'huggingface', 'sentence-transformers']
                })
            
            self.logger.info(f"Returning {len(models)} curated HuggingFace embedding models")
            return models
            
        except Exception as e:
            self.logger.error(f"Error fetching HuggingFace embedding models: {e}")
            return []
    
    async def download_model(
        self, 
        model_name: str,
        model_type: str = 'llm',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Download and validate a HuggingFace model.
        
        Args:
            model_name: Name of the HuggingFace model (e.g., 'org/model-name')
            model_type: Type of model ('llm' or 'embedding')
            **kwargs: Additional options
            
        Returns:
            Dictionary with download status and details
        """
        try:
            self.logger.info(f"Attempting to download HuggingFace {model_type} model: {model_name}")
            
            if model_type == 'llm':
                return await self._download_llm_model(model_name)
            elif model_type == 'embedding':
                return await self._download_embedding_model(model_name)
            else:
                return {
                    'success': False,
                    'message': f"Invalid model type: {model_type}. Must be 'llm' or 'embedding'",
                    'model_name': model_name,
                    'model_type': model_type
                }
                
        except Exception as e:
            error_msg = f"Error downloading HuggingFace model {model_name}: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'model_name': model_name,
                'error': str(e)
            }
    
    async def _download_llm_model(self, model_name: str) -> Dict[str, Any]:
        """
        Download and validate an LLM model from HuggingFace with progress tracking.
        
        Args:
            model_name: HuggingFace model identifier
            
        Returns:
            Download status dictionary
        """
        try:
            # Download the model using HuggingFace Hub with progress tracking
            from transformers import AutoTokenizer, AutoConfig
            from huggingface_hub import snapshot_download, HfApi, try_to_load_from_cache
            import threading
            import time
            import os
            
            start_time = time.time()
            
            self.logger.info(f"📦 Checking HuggingFace LLM model: {model_name}")
            
            # Check if model is already cached - just verify files exist, don't load the model
            # Loading the model here would consume GPU memory and potentially crash
            try:
                cached_path = try_to_load_from_cache(model_name, "config.json")
                if cached_path:
                    cache_dir = os.path.dirname(cached_path)
                    self.logger.info(f"✅ Model already cached at: {cache_dir}")
                    
                    # Just verify the tokenizer and config can be loaded (lightweight check)
                    self.logger.info(f"📥 Verifying cached model files...")
                    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                    
                    # Check if model weights exist (without loading them)
                    weight_files = [f for f in os.listdir(cache_dir) 
                                   if f.endswith('.bin') or f.endswith('.safetensors')]
                    
                    if weight_files or os.path.exists(os.path.join(cache_dir, 'model.safetensors.index.json')):
                        elapsed = int(time.time() - start_time)
                        self.logger.info(f"✅ Model cache verified in {elapsed}s (weights: {weight_files[:3]}...)")
                        
                        return {
                            'success': True,
                            'message': f"LLM model verified from cache: {model_name}",
                            'model_name': model_name,
                            'model_type': 'llm',
                            'ready_to_use': True,
                            'from_cache': True,
                            'download_time_seconds': elapsed
                        }
                    else:
                        self.logger.warning(f"Cache found but missing weight files, will re-download")
            except Exception as cache_check_error:
                self.logger.info(f"Model not in cache or incomplete, will download: {cache_check_error}")
            
            self.logger.info(f"📦 Starting download of HuggingFace LLM model: {model_name}")
            self.logger.info(f"💾 Cache location: /root/.cache/huggingface/hub/")
            
            # Get model info to calculate total size
            total_size_bytes = 0
            files_to_download = []
            try:
                api = HfApi()
                model_info = api.model_info(model_name, files_metadata=True)
                
                # Calculate total size from siblings (files in the repo)
                if hasattr(model_info, 'siblings') and model_info.siblings:
                    for sibling in model_info.siblings:
                        if hasattr(sibling, 'size') and sibling.size:
                            total_size_bytes += sibling.size
                            files_to_download.append({
                                'name': sibling.rfilename,
                                'size': sibling.size
                            })
                
                total_size_mb = total_size_bytes / (1024 * 1024)
                total_size_gb = total_size_bytes / (1024 * 1024 * 1024)
                
                if total_size_gb >= 1:
                    self.logger.info(f"📊 Total model size: {total_size_gb:.2f} GB ({len(files_to_download)} files)")
                else:
                    self.logger.info(f"📊 Total model size: {total_size_mb:.2f} MB ({len(files_to_download)} files)")
                    
                # Log large files
                large_files = [f for f in files_to_download if f['size'] > 100 * 1024 * 1024]  # > 100MB
                for f in large_files:
                    f_size_mb = f['size'] / (1024 * 1024)
                    self.logger.info(f"   📁 {f['name']}: {f_size_mb:.1f} MB")
                    
            except Exception as e:
                self.logger.warning(f"Could not get model size info: {e}")
            
            # Progress tracking state
            download_complete = threading.Event()
            current_file = ['']
            downloaded_bytes = [0]
            current_file_downloaded = [0]
            current_file_total = [0]
            
            def log_progress():
                """Background thread to log progress every 10 seconds"""
                while not download_complete.is_set():
                    elapsed = int(time.time() - start_time)
                    elapsed_min = elapsed // 60
                    elapsed_sec = elapsed % 60
                    
                    if total_size_bytes > 0 and downloaded_bytes[0] > 0:
                        overall_percent = min(100, (downloaded_bytes[0] / total_size_bytes) * 100)
                        downloaded_mb = downloaded_bytes[0] / (1024 * 1024)
                        total_mb = total_size_bytes / (1024 * 1024)
                        
                        # Calculate speed and ETA
                        if elapsed > 0:
                            speed_mb_s = downloaded_mb / elapsed
                            remaining_mb = total_mb - downloaded_mb
                            eta_seconds = int(remaining_mb / speed_mb_s) if speed_mb_s > 0 else 0
                            eta_min = eta_seconds // 60
                            eta_sec = eta_seconds % 60
                            
                            self.logger.info(
                                f"⬇️ [{model_name}] {overall_percent:.1f}% "
                                f"({downloaded_mb:.0f}/{total_mb:.0f} MB) | "
                                f"Speed: {speed_mb_s:.1f} MB/s | "
                                f"Elapsed: {elapsed_min}m {elapsed_sec}s | ETA: {eta_min}m {eta_sec}s"
                            )
                    elif current_file[0]:
                        # Show current file being downloaded
                        if current_file_total[0] > 0:
                            file_percent = (current_file_downloaded[0] / current_file_total[0]) * 100
                            self.logger.info(
                                f"⬇️ [{model_name}] Downloading {current_file[0]}: {file_percent:.1f}% "
                                f"(Elapsed: {elapsed_min}m {elapsed_sec}s)"
                            )
                        else:
                            self.logger.info(
                                f"⬇️ [{model_name}] Downloading {current_file[0]}... "
                                f"(Elapsed: {elapsed_min}m {elapsed_sec}s)"
                            )
                    else:
                        self.logger.info(f"⬇️ [{model_name}] Downloading... (Elapsed: {elapsed_min}m {elapsed_sec}s)")
                    
                    if download_complete.wait(10):
                        break
            
            progress_thread = threading.Thread(target=log_progress, daemon=True)
            progress_thread.start()
            
            try:
                # Use snapshot_download for better progress tracking
                # This downloads all model files with progress
                self.logger.info(f"📥 Downloading model files for {model_name}...")
                
                local_dir = snapshot_download(
                    repo_id=model_name,
                    repo_type="model",
                    local_dir=None,  # Use default cache
                    resume_download=True  # Enable resume for interrupted downloads
                )
                
                self.logger.info(f"✅ Model files downloaded to: {local_dir}")
                
                # Verify the download by loading tokenizer and config (lightweight)
                # Don't load the full model here - it will be loaded when actually used
                self.logger.info(f"📥 Verifying tokenizer for {model_name}...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                self.logger.info(f"✅ Tokenizer verified for {model_name}")
                
                # Verify config loads correctly
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                self.logger.info(f"✅ Model config verified: {config.model_type}")
                
                # Verify weight files exist (without loading them - saves GPU memory)
                weight_files = [f for f in os.listdir(local_dir) 
                               if f.endswith('.bin') or f.endswith('.safetensors')]
                has_sharded = os.path.exists(os.path.join(local_dir, 'model.safetensors.index.json'))
                
                if weight_files or has_sharded:
                    self.logger.info(f"✅ Model weights verified: {len(weight_files)} files found")
                else:
                    self.logger.warning(f"⚠️ No weight files found in {local_dir}")
                
                download_complete.set()
                elapsed = int(time.time() - start_time)
                elapsed_min = elapsed // 60
                elapsed_sec = elapsed % 60
                
                # If we get here, download was successful
                self.logger.info(f"✅ Successfully downloaded LLM model: {model_name} ({elapsed_min}m {elapsed_sec}s)")
                return {
                    'success': True,
                    'message': f"Successfully downloaded and verified LLM model: {model_name}",
                    'model_name': model_name,
                    'model_type': 'llm',
                    'ready_to_use': True,
                    'download_time_seconds': elapsed
                }
            finally:
                download_complete.set()
            
        except ValueError as e:
            # Check if this is a gated model error
            error_str = str(e)
            if error_str.startswith("GATED_MODEL_ERROR:"):
                return self._handle_gated_model_error(model_name, error_str, 'llm')
            
            return {
                'success': False,
                'message': str(e),
                'model_name': model_name,
                'model_type': 'llm',
                'error': 'validation_error'
            }
            
        except Exception as e:
            error_msg = f"Failed to download LLM model {model_name}: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'model_name': model_name,
                'model_type': 'llm',
                'error': str(e)
            }
    
    async def _download_embedding_model(self, model_name: str) -> Dict[str, Any]:
        """
        Download and validate an embedding model from HuggingFace with progress tracking.
        
        Args:
            model_name: HuggingFace model identifier
            
        Returns:
            Download status dictionary
        """
        try:
            import threading
            import time
            
            start_time = time.time()
            download_complete = threading.Event()
            
            self.logger.info(f"📦 Starting download of HuggingFace embedding model: {model_name}")
            self.logger.info(f"💾 Cache location: ~/.cache/huggingface/hub/")
            
            # Get model info to calculate total size
            total_size_bytes = 0
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                model_info = api.model_info(model_name)
                
                if hasattr(model_info, 'siblings') and model_info.siblings:
                    for sibling in model_info.siblings:
                        if hasattr(sibling, 'size') and sibling.size:
                            total_size_bytes += sibling.size
                
                total_size_mb = total_size_bytes / (1024 * 1024)
                self.logger.info(f"📊 Model size: {total_size_mb:.2f} MB")
                    
            except Exception as e:
                self.logger.warning(f"Could not get model size info: {e}")
            
            def progress_logger():
                while not download_complete.is_set():
                    elapsed = int(time.time() - start_time)
                    self.logger.info(f"⏳ Downloading embedding '{model_name}'... ({elapsed}s elapsed)")
                    if download_complete.wait(10):
                        break
            
            progress_thread = threading.Thread(target=progress_logger, daemon=True)
            progress_thread.start()
            
            try:
                # Create embedding model (this will download if needed)
                embedding_model = create_embedding_model(model_name, 'huggingface')
                
                # Test the embedding model with a simple query
                test_embedding = embedding_model.embed_query("test")
                
                download_complete.set()
                elapsed = int(time.time() - start_time)
                
                if test_embedding and len(test_embedding) > 0:
                    self.logger.info(f"✅ Successfully downloaded embedding model: {model_name} ({elapsed}s)")
                    return {
                        'success': True,
                        'message': f"Successfully downloaded and validated embedding model: {model_name}",
                        'model_name': model_name,
                        'model_type': 'embedding',
                        'embedding_dimension': len(test_embedding),
                        'ready_to_use': True,
                        'download_time_seconds': elapsed
                    }
                else:
                    return {
                        'success': False,
                        'message': "Model downloaded but failed validation test",
                        'model_name': model_name,
                        'model_type': 'embedding',
                        'error': 'validation_failed'
                    }
            finally:
                download_complete.set()
                
        except Exception as e:
            error_msg = f"Failed to download embedding model {model_name}: {str(e)}"
            self.logger.error(error_msg)
            
            # Check for gated model error
            if "gated" in str(e).lower() or "access" in str(e).lower():
                return self._handle_gated_model_error(model_name, str(e), 'embedding')
            
            return {
                'success': False,
                'message': error_msg,
                'model_name': model_name,
                'model_type': 'embedding',
                'error': str(e)
            }
    
    def _handle_gated_model_error(
        self, 
        model_name: str, 
        error_str: str, 
        model_type: str
    ) -> Dict[str, Any]:
        """
        Handle gated model access errors with structured response.
        
        Args:
            model_name: Name of the gated model
            error_str: Error string from the exception
            model_type: Type of model ('llm' or 'embedding')
            
        Returns:
            Structured error response with access instructions
        """
        model_url = f"https://huggingface.co/{model_name}"
        
        # Try to parse structured error from error string
        if "GATED_MODEL_ERROR:" in error_str:
            try:
                json_str = error_str.split("GATED_MODEL_ERROR:", 1)[1].strip()
                error_data = json.loads(json_str)
                
                return {
                    'success': False,
                    'error_type': 'gated_model',
                    'model_name': model_name,
                    'model_type': model_type,
                    'message': error_data.get('message', f"Model '{model_name}' requires special access"),
                    'model_url': error_data.get('model_url', model_url),
                    'steps': error_data.get('steps', self._get_gated_model_steps(model_url)),
                    'alternatives': error_data.get('alternatives', [])
                }
            except (json.JSONDecodeError, IndexError):
                pass
        
        # Return default gated model error
        return {
            'success': False,
            'error_type': 'gated_model',
            'model_name': model_name,
            'model_type': model_type,
            'message': f"Model '{model_name}' requires special access approval",
            'model_url': model_url,
            'steps': self._get_gated_model_steps(model_url),
            'alternatives': []
        }
    
    def _get_gated_model_steps(self, model_url: str) -> List[str]:
        """
        Get steps to request access to a gated model.
        
        Args:
            model_url: URL to the model page
            
        Returns:
            List of steps to follow
        """
        return [
            f"Visit the model page: {model_url}",
            "Click the 'Request access' button",
            "Wait for approval from the model owner",
            "Ensure you have a valid HuggingFace token in your .env file (HUGGINGFACE_TOKEN)",
            "Try downloading the model again after approval is granted"
        ]
    
    def _get_fallback_models(self) -> List[Dict[str, Any]]:
        """
        Get fallback model list when API fetch fails.
        
        Returns:
            Curated list of LLM and embedding models
        """
        models = []
        
        # Add curated LLM models
        for model_name in self.curated_llm_models[:10]:
            models.append({
                'name': model_name,
                'type': 'llm',
                'source': 'huggingface',
                'installed': False
            })
        
        # Add curated embedding models
        for model_name in self.curated_embedding_models[:10]:
            models.append({
                'name': model_name,
                'type': 'embedding',
                'source': 'huggingface',
                'installed': False
            })
        
        return models
    
    async def check_model_exists(self, model_name: str) -> bool:
        """
        Check if a HuggingFace model exists and is accessible.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is accessible, False otherwise
        """
        try:
            # Try to get model info from HuggingFace Hub
            url = f"https://huggingface.co/api/models/{model_name}"
            response = requests.get(url, timeout=self.request_timeout)
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Error checking if HuggingFace model exists: {e}")
            return False
    
    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a HuggingFace model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary or None if not found
        """
        try:
            url = f"https://huggingface.co/api/models/{model_name}"
            response = requests.get(url, timeout=self.request_timeout)
            
            if response.status_code == 200:
                data = response.json()
                
                return {
                    'name': model_name,
                    'model_id': data.get('modelId', model_name),
                    'author': data.get('author', ''),
                    'downloads': data.get('downloads', 0),
                    'likes': data.get('likes', 0),
                    'tags': data.get('tags', []),
                    'pipeline_tag': data.get('pipeline_tag', ''),
                    'library_name': data.get('library_name', ''),
                    'gated': data.get('gated', False),
                    'source': 'huggingface',
                    'url': f"https://huggingface.co/{model_name}"
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting HuggingFace model info: {e}")
            return None
    
    async def validate_model(self, model_name: str) -> Dict[str, Any]:
        """
        Validate that a HuggingFace model is accessible.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            Validation result dictionary
        """
        try:
            model_info = await self.get_model_info(model_name)
            
            if model_info:
                is_gated = model_info.get('gated', False)
                
                return {
                    'valid': True,
                    'message': f"Model {model_name} is accessible",
                    'details': {
                        'gated': is_gated,
                        'downloads': model_info.get('downloads', 0),
                        'tags': model_info.get('tags', [])
                    },
                    'warning': "Model is gated and requires approval" if is_gated else None
                }
            else:
                return {
                    'valid': False,
                    'message': f"Model {model_name} not found or not accessible",
                    'details': {}
                }
                
        except Exception as e:
            return {
                'valid': False,
                'message': f"Error validating model: {str(e)}",
                'details': {'error': str(e)}
            }
    
    async def get_all_models(self) -> Dict[str, Any]:
        """
        Get all available HuggingFace models (LLM and embedding).
        
        Returns:
            Dictionary with categorized models
        """
        try:
            llm_models = await self.get_available_llm_models()
            embedding_models = await self.get_available_embedding_models()
            
            return {
                'llm': llm_models,
                'embedding': embedding_models,
                'total_llm': len(llm_models),
                'total_embedding': len(embedding_models),
                'total': len(llm_models) + len(embedding_models)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting all HuggingFace models: {e}")
            return {
                'llm': [],
                'embedding': [],
                'total_llm': 0,
                'total_embedding': 0,
                'total': 0,
                'error': str(e)
            }
