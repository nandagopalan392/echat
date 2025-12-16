"""
Model Cache - Production-grade caching for model lists

This module provides a centralized cache for model lists from all providers.
Models are fetched at application startup and cached for fast access.

Features:
- Background refresh at startup
- TTL-based cache expiration
- Fallback to hardcoded models
- Thread-safe operations
- Shared across all endpoints
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class CacheStatus(Enum):
    """Cache status states"""
    NOT_INITIALIZED = "not_initialized"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    STALE = "stale"


@dataclass
class CachedModelList:
    """Cached model list with metadata"""
    models: List[Dict[str, Any]]
    fetched_at: datetime
    ttl_seconds: int = 3600  # 1 hour default
    source: str = "unknown"  # "live", "fallback", "cache"
    
    @property
    def is_expired(self) -> bool:
        """Check if cache has expired"""
        age = (datetime.now() - self.fetched_at).total_seconds()
        return age > self.ttl_seconds
    
    @property
    def age_seconds(self) -> int:
        """Get cache age in seconds"""
        return int((datetime.now() - self.fetched_at).total_seconds())


class ModelCache:
    """
    Centralized model cache for all providers.
    
    Thread-safe singleton that caches model lists from Ollama and HuggingFace.
    Designed for production use with background refresh and fallback support.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._cache: Dict[str, CachedModelList] = {}
        self._status: Dict[str, CacheStatus] = {}
        self._cache_lock = threading.RLock()
        self._refresh_task = None
        
        # Configuration
        self.ollama_ttl = 1800  # 30 minutes
        self.huggingface_ttl = 3600  # 1 hour
        self.finetuning_ttl = 86400  # 24 hours (rarely changes)
        
        # Initialize fallback models
        self._init_fallback_models()
        
        logger.info("ModelCache initialized")
    
    def _init_fallback_models(self):
        """Initialize hardcoded fallback models"""
        
        # Ollama fallback models
        self._ollama_fallback = [
            {"name": "llama3.2:3b", "size": "3B", "type": "llm", "description": "Llama 3.2 3B"},
            {"name": "llama3.2:1b", "size": "1B", "type": "llm", "description": "Llama 3.2 1B"},
            {"name": "llama3.1:8b", "size": "8B", "type": "llm", "description": "Llama 3.1 8B"},
            {"name": "gemma2:2b", "size": "2B", "type": "llm", "description": "Gemma 2 2B"},
            {"name": "gemma2:9b", "size": "9B", "type": "llm", "description": "Gemma 2 9B"},
            {"name": "qwen2.5:0.5b", "size": "0.5B", "type": "llm", "description": "Qwen 2.5 0.5B"},
            {"name": "qwen2.5:1.5b", "size": "1.5B", "type": "llm", "description": "Qwen 2.5 1.5B"},
            {"name": "qwen2.5:3b", "size": "3B", "type": "llm", "description": "Qwen 2.5 3B"},
            {"name": "qwen2.5:7b", "size": "7B", "type": "llm", "description": "Qwen 2.5 7B"},
            {"name": "mistral:7b", "size": "7B", "type": "llm", "description": "Mistral 7B"},
            {"name": "phi3:mini", "size": "3.8B", "type": "llm", "description": "Phi-3 Mini"},
            {"name": "phi3:medium", "size": "14B", "type": "llm", "description": "Phi-3 Medium"},
            {"name": "codellama:7b", "size": "7B", "type": "llm", "description": "Code Llama 7B"},
            {"name": "deepseek-coder:6.7b", "size": "6.7B", "type": "llm", "description": "DeepSeek Coder"},
            # Embedding models
            {"name": "nomic-embed-text", "size": "137M", "type": "embedding", "description": "Nomic Embed Text"},
            {"name": "mxbai-embed-large", "size": "335M", "type": "embedding", "description": "MixedBread Embed Large"},
            {"name": "all-minilm", "size": "23M", "type": "embedding", "description": "All-MiniLM-L6-v2"},
            {"name": "bge-m3", "size": "567M", "type": "embedding", "description": "BGE-M3 Multilingual"},
        ]
        
        # HuggingFace fallback models (LLM) - Comprehensive list for all GPU sizes
        self._huggingface_llm_fallback = [
            # ============ TINY MODELS (< 500M params, ~1-2GB VRAM) ============
            {"name": "distilgpt2", "size": "82M", "type": "llm", "description": "DistilGPT-2 - Lightweight GPT-2"},
            {"name": "gpt2", "size": "124M", "type": "llm", "description": "GPT-2 Base"},
            {"name": "facebook/opt-125m", "size": "125M", "type": "llm", "description": "OPT 125M"},
            {"name": "EleutherAI/gpt-neo-125M", "size": "125M", "type": "llm", "description": "GPT-Neo 125M"},
            {"name": "facebook/opt-350m", "size": "350M", "type": "llm", "description": "OPT 350M"},
            {"name": "gpt2-medium", "size": "355M", "type": "llm", "description": "GPT-2 Medium"},
            
            # ============ SMALL MODELS (500M - 1.5B params, ~2-4GB VRAM) ============
            {"name": "Qwen/Qwen2.5-0.5B", "size": "0.5B", "type": "llm", "description": "Qwen 2.5 0.5B Base"},
            {"name": "Qwen/Qwen2.5-0.5B-Instruct", "size": "0.5B", "type": "llm", "description": "Qwen 2.5 0.5B Instruct"},
            {"name": "gpt2-large", "size": "774M", "type": "llm", "description": "GPT-2 Large"},
            {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "size": "1.1B", "type": "llm", "description": "TinyLlama 1.1B Chat"},
            {"name": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", "size": "1.1B", "type": "llm", "description": "TinyLlama 1.1B Base"},
            {"name": "microsoft/phi-1", "size": "1.3B", "type": "llm", "description": "Phi-1 1.3B"},
            {"name": "microsoft/phi-1_5", "size": "1.3B", "type": "llm", "description": "Phi-1.5 1.3B"},
            {"name": "facebook/opt-1.3b", "size": "1.3B", "type": "llm", "description": "OPT 1.3B"},
            {"name": "EleutherAI/gpt-neo-1.3B", "size": "1.3B", "type": "llm", "description": "GPT-Neo 1.3B"},
            {"name": "EleutherAI/pythia-1.4b", "size": "1.4B", "type": "llm", "description": "Pythia 1.4B"},
            {"name": "EleutherAI/pythia-1.4b-deduped", "size": "1.4B", "type": "llm", "description": "Pythia 1.4B Deduped"},
            {"name": "gpt2-xl", "size": "1.5B", "type": "llm", "description": "GPT-2 XL"},
            {"name": "Qwen/Qwen2.5-1.5B", "size": "1.5B", "type": "llm", "description": "Qwen 2.5 1.5B Base"},
            {"name": "Qwen/Qwen2.5-1.5B-Instruct", "size": "1.5B", "type": "llm", "description": "Qwen 2.5 1.5B Instruct"},
            {"name": "bigscience/bloomz-1b1", "size": "1.1B", "type": "llm", "description": "BLOOMZ 1.1B"},
            {"name": "bigscience/bloomz-1b7", "size": "1.7B", "type": "llm", "description": "BLOOMZ 1.7B"},
            
            # ============ MEDIUM MODELS (2B - 4B params, ~4-8GB VRAM) ============
            {"name": "microsoft/phi-2", "size": "2.7B", "type": "llm", "description": "Phi-2 2.7B"},
            {"name": "facebook/opt-2.7b", "size": "2.7B", "type": "llm", "description": "OPT 2.7B"},
            {"name": "EleutherAI/gpt-neo-2.7B", "size": "2.7B", "type": "llm", "description": "GPT-Neo 2.7B"},
            {"name": "EleutherAI/pythia-2.8b", "size": "2.8B", "type": "llm", "description": "Pythia 2.8B"},
            {"name": "EleutherAI/pythia-2.8b-deduped", "size": "2.8B", "type": "llm", "description": "Pythia 2.8B Deduped"},
            {"name": "google/gemma-2b", "size": "2B", "type": "llm", "description": "Gemma 2B Base"},
            {"name": "google/gemma-2b-it", "size": "2B", "type": "llm", "description": "Gemma 2B Instruct"},
            {"name": "google/gemma-2-2b", "size": "2B", "type": "llm", "description": "Gemma 2 2B Base"},
            {"name": "google/gemma-2-2b-it", "size": "2B", "type": "llm", "description": "Gemma 2 2B Instruct"},
            {"name": "Qwen/Qwen2.5-3B", "size": "3B", "type": "llm", "description": "Qwen 2.5 3B Base"},
            {"name": "Qwen/Qwen2.5-3B-Instruct", "size": "3B", "type": "llm", "description": "Qwen 2.5 3B Instruct"},
            {"name": "stabilityai/stablelm-3b-4e1t", "size": "3B", "type": "llm", "description": "StableLM 3B"},
            {"name": "stabilityai/stablelm-zephyr-3b", "size": "3B", "type": "llm", "description": "StableLM Zephyr 3B"},
            {"name": "microsoft/phi-3-mini-4k-instruct", "size": "3.8B", "type": "llm", "description": "Phi-3 Mini 4K Instruct"},
            {"name": "microsoft/phi-3-mini-128k-instruct", "size": "3.8B", "type": "llm", "description": "Phi-3 Mini 128K Instruct"},
            {"name": "microsoft/Phi-3.5-mini-instruct", "size": "3.8B", "type": "llm", "description": "Phi-3.5 Mini Instruct"},
            {"name": "bigscience/bloomz-3b", "size": "3B", "type": "llm", "description": "BLOOMZ 3B"},
            
            # ============ LARGE MODELS (6B - 8B params, ~12-16GB VRAM) ============
            {"name": "EleutherAI/gpt-j-6B", "size": "6B", "type": "llm", "description": "GPT-J 6B"},
            {"name": "EleutherAI/pythia-6.9b", "size": "6.9B", "type": "llm", "description": "Pythia 6.9B"},
            {"name": "EleutherAI/pythia-6.9b-deduped", "size": "6.9B", "type": "llm", "description": "Pythia 6.9B Deduped"},
            {"name": "facebook/opt-6.7b", "size": "6.7B", "type": "llm", "description": "OPT 6.7B"},
            {"name": "google/gemma-7b", "size": "7B", "type": "llm", "description": "Gemma 7B Base"},
            {"name": "google/gemma-7b-it", "size": "7B", "type": "llm", "description": "Gemma 7B Instruct"},
            {"name": "google/gemma-2-9b", "size": "9B", "type": "llm", "description": "Gemma 2 9B Base"},
            {"name": "google/gemma-2-9b-it", "size": "9B", "type": "llm", "description": "Gemma 2 9B Instruct"},
            {"name": "Qwen/Qwen2.5-7B", "size": "7B", "type": "llm", "description": "Qwen 2.5 7B Base"},
            {"name": "Qwen/Qwen2.5-7B-Instruct", "size": "7B", "type": "llm", "description": "Qwen 2.5 7B Instruct"},
            {"name": "mistralai/Mistral-7B-v0.1", "size": "7B", "type": "llm", "description": "Mistral 7B v0.1 Base"},
            {"name": "mistralai/Mistral-7B-v0.3", "size": "7B", "type": "llm", "description": "Mistral 7B v0.3 Base"},
            {"name": "mistralai/Mistral-7B-Instruct-v0.1", "size": "7B", "type": "llm", "description": "Mistral 7B Instruct v0.1"},
            {"name": "mistralai/Mistral-7B-Instruct-v0.2", "size": "7B", "type": "llm", "description": "Mistral 7B Instruct v0.2"},
            {"name": "mistralai/Mistral-7B-Instruct-v0.3", "size": "7B", "type": "llm", "description": "Mistral 7B Instruct v0.3"},
            {"name": "meta-llama/Llama-2-7b-hf", "size": "7B", "type": "llm", "description": "Llama 2 7B Base"},
            {"name": "meta-llama/Llama-2-7b-chat-hf", "size": "7B", "type": "llm", "description": "Llama 2 7B Chat"},
            {"name": "meta-llama/Meta-Llama-3-8B", "size": "8B", "type": "llm", "description": "Llama 3 8B Base"},
            {"name": "meta-llama/Meta-Llama-3-8B-Instruct", "size": "8B", "type": "llm", "description": "Llama 3 8B Instruct"},
            {"name": "meta-llama/Llama-3.1-8B", "size": "8B", "type": "llm", "description": "Llama 3.1 8B Base"},
            {"name": "meta-llama/Llama-3.1-8B-Instruct", "size": "8B", "type": "llm", "description": "Llama 3.1 8B Instruct"},
            {"name": "meta-llama/Llama-3.2-3B", "size": "3B", "type": "llm", "description": "Llama 3.2 3B Base"},
            {"name": "meta-llama/Llama-3.2-3B-Instruct", "size": "3B", "type": "llm", "description": "Llama 3.2 3B Instruct"},
            {"name": "meta-llama/Llama-3.2-1B", "size": "1B", "type": "llm", "description": "Llama 3.2 1B Base"},
            {"name": "meta-llama/Llama-3.2-1B-Instruct", "size": "1B", "type": "llm", "description": "Llama 3.2 1B Instruct"},
            {"name": "tiiuae/falcon-7b", "size": "7B", "type": "llm", "description": "Falcon 7B Base"},
            {"name": "tiiuae/falcon-7b-instruct", "size": "7B", "type": "llm", "description": "Falcon 7B Instruct"},
            {"name": "bigscience/bloom-7b1", "size": "7.1B", "type": "llm", "description": "BLOOM 7.1B"},
            {"name": "bigscience/bloomz-7b1", "size": "7.1B", "type": "llm", "description": "BLOOMZ 7.1B"},
            {"name": "mosaicml/mpt-7b", "size": "7B", "type": "llm", "description": "MPT 7B Base"},
            {"name": "mosaicml/mpt-7b-instruct", "size": "7B", "type": "llm", "description": "MPT 7B Instruct"},
            {"name": "mosaicml/mpt-7b-chat", "size": "7B", "type": "llm", "description": "MPT 7B Chat"},
            {"name": "togethercomputer/RedPajama-INCITE-7B-Base", "size": "7B", "type": "llm", "description": "RedPajama 7B Base"},
            {"name": "togethercomputer/RedPajama-INCITE-7B-Instruct", "size": "7B", "type": "llm", "description": "RedPajama 7B Instruct"},
            {"name": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO", "size": "7B", "type": "llm", "description": "Nous Hermes 2 Mistral 7B"},
            {"name": "teknium/OpenHermes-2.5-Mistral-7B", "size": "7B", "type": "llm", "description": "OpenHermes 2.5 Mistral 7B"},
            {"name": "HuggingFaceH4/zephyr-7b-beta", "size": "7B", "type": "llm", "description": "Zephyr 7B Beta"},
            {"name": "openchat/openchat-3.5-0106", "size": "7B", "type": "llm", "description": "OpenChat 3.5"},
            {"name": "lmsys/vicuna-7b-v1.5", "size": "7B", "type": "llm", "description": "Vicuna 7B v1.5"},
            
            # ============ XL MODELS (10B - 14B params, ~20-28GB VRAM) ============
            {"name": "EleutherAI/pythia-12b", "size": "12B", "type": "llm", "description": "Pythia 12B"},
            {"name": "EleutherAI/pythia-12b-deduped", "size": "12B", "type": "llm", "description": "Pythia 12B Deduped"},
            {"name": "facebook/opt-13b", "size": "13B", "type": "llm", "description": "OPT 13B"},
            {"name": "meta-llama/Llama-2-13b-hf", "size": "13B", "type": "llm", "description": "Llama 2 13B Base"},
            {"name": "meta-llama/Llama-2-13b-chat-hf", "size": "13B", "type": "llm", "description": "Llama 2 13B Chat"},
            {"name": "lmsys/vicuna-13b-v1.5", "size": "13B", "type": "llm", "description": "Vicuna 13B v1.5"},
            {"name": "NousResearch/Nous-Hermes-Llama2-13b", "size": "13B", "type": "llm", "description": "Nous Hermes Llama 2 13B"},
            {"name": "Qwen/Qwen2.5-14B", "size": "14B", "type": "llm", "description": "Qwen 2.5 14B Base"},
            {"name": "Qwen/Qwen2.5-14B-Instruct", "size": "14B", "type": "llm", "description": "Qwen 2.5 14B Instruct"},
            {"name": "microsoft/phi-3-medium-4k-instruct", "size": "14B", "type": "llm", "description": "Phi-3 Medium 4K Instruct"},
            {"name": "microsoft/phi-3-medium-128k-instruct", "size": "14B", "type": "llm", "description": "Phi-3 Medium 128K Instruct"},
            
            # ============ XXL MODELS (20B+ params, ~40GB+ VRAM) ============
            {"name": "EleutherAI/gpt-neox-20b", "size": "20B", "type": "llm", "description": "GPT-NeoX 20B"},
            {"name": "google/gemma-2-27b", "size": "27B", "type": "llm", "description": "Gemma 2 27B Base"},
            {"name": "google/gemma-2-27b-it", "size": "27B", "type": "llm", "description": "Gemma 2 27B Instruct"},
            {"name": "Qwen/Qwen2.5-32B", "size": "32B", "type": "llm", "description": "Qwen 2.5 32B Base"},
            {"name": "Qwen/Qwen2.5-32B-Instruct", "size": "32B", "type": "llm", "description": "Qwen 2.5 32B Instruct"},
            {"name": "mistralai/Mixtral-8x7B-v0.1", "size": "47B", "type": "llm", "description": "Mixtral 8x7B MoE Base"},
            {"name": "mistralai/Mixtral-8x7B-Instruct-v0.1", "size": "47B", "type": "llm", "description": "Mixtral 8x7B MoE Instruct"},
            {"name": "facebook/opt-30b", "size": "30B", "type": "llm", "description": "OPT 30B"},
            {"name": "tiiuae/falcon-40b", "size": "40B", "type": "llm", "description": "Falcon 40B Base"},
            {"name": "tiiuae/falcon-40b-instruct", "size": "40B", "type": "llm", "description": "Falcon 40B Instruct"},
            {"name": "meta-llama/Llama-2-70b-hf", "size": "70B", "type": "llm", "description": "Llama 2 70B Base"},
            {"name": "meta-llama/Llama-2-70b-chat-hf", "size": "70B", "type": "llm", "description": "Llama 2 70B Chat"},
            {"name": "meta-llama/Meta-Llama-3-70B", "size": "70B", "type": "llm", "description": "Llama 3 70B Base"},
            {"name": "meta-llama/Meta-Llama-3-70B-Instruct", "size": "70B", "type": "llm", "description": "Llama 3 70B Instruct"},
            {"name": "meta-llama/Llama-3.1-70B", "size": "70B", "type": "llm", "description": "Llama 3.1 70B Base"},
            {"name": "meta-llama/Llama-3.1-70B-Instruct", "size": "70B", "type": "llm", "description": "Llama 3.1 70B Instruct"},
            {"name": "Qwen/Qwen2.5-72B", "size": "72B", "type": "llm", "description": "Qwen 2.5 72B Base"},
            {"name": "Qwen/Qwen2.5-72B-Instruct", "size": "72B", "type": "llm", "description": "Qwen 2.5 72B Instruct"},
            {"name": "facebook/opt-66b", "size": "66B", "type": "llm", "description": "OPT 66B"},
            {"name": "bigscience/bloom", "size": "176B", "type": "llm", "description": "BLOOM 176B"},
            {"name": "bigscience/bloomz", "size": "176B", "type": "llm", "description": "BLOOMZ 176B"},
            
            # ============ CODE MODELS ============
            {"name": "bigcode/starcoder", "size": "15.5B", "type": "llm", "description": "StarCoder 15.5B"},
            {"name": "bigcode/starcoder2-3b", "size": "3B", "type": "llm", "description": "StarCoder2 3B"},
            {"name": "bigcode/starcoder2-7b", "size": "7B", "type": "llm", "description": "StarCoder2 7B"},
            {"name": "bigcode/starcoder2-15b", "size": "15B", "type": "llm", "description": "StarCoder2 15B"},
            {"name": "codellama/CodeLlama-7b-hf", "size": "7B", "type": "llm", "description": "CodeLlama 7B Base"},
            {"name": "codellama/CodeLlama-7b-Instruct-hf", "size": "7B", "type": "llm", "description": "CodeLlama 7B Instruct"},
            {"name": "codellama/CodeLlama-13b-hf", "size": "13B", "type": "llm", "description": "CodeLlama 13B Base"},
            {"name": "codellama/CodeLlama-13b-Instruct-hf", "size": "13B", "type": "llm", "description": "CodeLlama 13B Instruct"},
            {"name": "codellama/CodeLlama-34b-hf", "size": "34B", "type": "llm", "description": "CodeLlama 34B Base"},
            {"name": "codellama/CodeLlama-34b-Instruct-hf", "size": "34B", "type": "llm", "description": "CodeLlama 34B Instruct"},
            {"name": "deepseek-ai/deepseek-coder-1.3b-base", "size": "1.3B", "type": "llm", "description": "DeepSeek Coder 1.3B Base"},
            {"name": "deepseek-ai/deepseek-coder-1.3b-instruct", "size": "1.3B", "type": "llm", "description": "DeepSeek Coder 1.3B Instruct"},
            {"name": "deepseek-ai/deepseek-coder-6.7b-base", "size": "6.7B", "type": "llm", "description": "DeepSeek Coder 6.7B Base"},
            {"name": "deepseek-ai/deepseek-coder-6.7b-instruct", "size": "6.7B", "type": "llm", "description": "DeepSeek Coder 6.7B Instruct"},
            {"name": "deepseek-ai/deepseek-coder-33b-base", "size": "33B", "type": "llm", "description": "DeepSeek Coder 33B Base"},
            {"name": "deepseek-ai/deepseek-coder-33b-instruct", "size": "33B", "type": "llm", "description": "DeepSeek Coder 33B Instruct"},
            {"name": "Qwen/Qwen2.5-Coder-0.5B", "size": "0.5B", "type": "llm", "description": "Qwen 2.5 Coder 0.5B"},
            {"name": "Qwen/Qwen2.5-Coder-1.5B", "size": "1.5B", "type": "llm", "description": "Qwen 2.5 Coder 1.5B"},
            {"name": "Qwen/Qwen2.5-Coder-3B", "size": "3B", "type": "llm", "description": "Qwen 2.5 Coder 3B"},
            {"name": "Qwen/Qwen2.5-Coder-7B", "size": "7B", "type": "llm", "description": "Qwen 2.5 Coder 7B"},
            {"name": "Qwen/Qwen2.5-Coder-14B", "size": "14B", "type": "llm", "description": "Qwen 2.5 Coder 14B"},
            {"name": "Qwen/Qwen2.5-Coder-32B", "size": "32B", "type": "llm", "description": "Qwen 2.5 Coder 32B"},
            
            # ============ MATH/REASONING MODELS ============
            {"name": "Qwen/Qwen2.5-Math-1.5B", "size": "1.5B", "type": "llm", "description": "Qwen 2.5 Math 1.5B"},
            {"name": "Qwen/Qwen2.5-Math-7B", "size": "7B", "type": "llm", "description": "Qwen 2.5 Math 7B"},
            {"name": "Qwen/Qwen2.5-Math-72B", "size": "72B", "type": "llm", "description": "Qwen 2.5 Math 72B"},
            {"name": "deepseek-ai/deepseek-math-7b-base", "size": "7B", "type": "llm", "description": "DeepSeek Math 7B Base"},
            {"name": "deepseek-ai/deepseek-math-7b-instruct", "size": "7B", "type": "llm", "description": "DeepSeek Math 7B Instruct"},
            
            # ============ SEQUENCE-TO-SEQUENCE MODELS (T5, BART, etc.) ============
            {"name": "google/flan-t5-small", "size": "80M", "type": "llm", "description": "Flan-T5 Small"},
            {"name": "google/flan-t5-base", "size": "250M", "type": "llm", "description": "Flan-T5 Base"},
            {"name": "google/flan-t5-large", "size": "780M", "type": "llm", "description": "Flan-T5 Large"},
            {"name": "google/flan-t5-xl", "size": "3B", "type": "llm", "description": "Flan-T5 XL"},
            {"name": "google/flan-t5-xxl", "size": "11B", "type": "llm", "description": "Flan-T5 XXL"},
            {"name": "google/flan-ul2", "size": "20B", "type": "llm", "description": "Flan-UL2 20B"},
            {"name": "google/t5-v1_1-small", "size": "77M", "type": "llm", "description": "T5 v1.1 Small"},
            {"name": "google/t5-v1_1-base", "size": "250M", "type": "llm", "description": "T5 v1.1 Base"},
            {"name": "google/t5-v1_1-large", "size": "780M", "type": "llm", "description": "T5 v1.1 Large"},
            {"name": "google/t5-v1_1-xl", "size": "3B", "type": "llm", "description": "T5 v1.1 XL"},
            {"name": "facebook/bart-base", "size": "140M", "type": "llm", "description": "BART Base"},
            {"name": "facebook/bart-large", "size": "400M", "type": "llm", "description": "BART Large"},
        ]
        
        # HuggingFace fallback embedding models
        self._huggingface_embedding_fallback = [
            {"name": "sentence-transformers/all-MiniLM-L6-v2", "size": "23M", "type": "embedding", "description": "All-MiniLM-L6-v2"},
            {"name": "sentence-transformers/all-mpnet-base-v2", "size": "110M", "type": "embedding", "description": "All-MPNet-Base-v2"},
            {"name": "BAAI/bge-small-en-v1.5", "size": "33M", "type": "embedding", "description": "BGE Small EN"},
            {"name": "BAAI/bge-base-en-v1.5", "size": "110M", "type": "embedding", "description": "BGE Base EN"},
            {"name": "BAAI/bge-large-en-v1.5", "size": "335M", "type": "embedding", "description": "BGE Large EN"},
            {"name": "BAAI/bge-m3", "size": "567M", "type": "embedding", "description": "BGE-M3 Multilingual"},
            {"name": "mixedbread-ai/mxbai-embed-large-v1", "size": "335M", "type": "embedding", "description": "MixedBread Embed Large"},
            {"name": "thenlper/gte-small", "size": "33M", "type": "embedding", "description": "GTE Small"},
            {"name": "thenlper/gte-base", "size": "110M", "type": "embedding", "description": "GTE Base"},
            {"name": "thenlper/gte-large", "size": "335M", "type": "embedding", "description": "GTE Large"},
            {"name": "intfloat/e5-small-v2", "size": "33M", "type": "embedding", "description": "E5 Small v2"},
            {"name": "intfloat/e5-base-v2", "size": "110M", "type": "embedding", "description": "E5 Base v2"},
            {"name": "intfloat/e5-large-v2", "size": "335M", "type": "embedding", "description": "E5 Large v2"},
            {"name": "nomic-ai/nomic-embed-text-v1", "size": "137M", "type": "embedding", "description": "Nomic Embed Text"},
        ]
        
        # Finetuning models (comprehensive list)
        self._finetuning_fallback = [
            # Small models (good for limited GPU)
            {"name": "gpt2", "description": "GPT-2 base model", "size": "small", "type": "text-generation"},
            {"name": "distilgpt2", "description": "Lightweight GPT-2", "size": "small", "type": "text-generation"},
            {"name": "facebook/opt-125m", "description": "OPT 125M", "size": "small", "type": "text-generation"},
            {"name": "facebook/opt-350m", "description": "OPT 350M", "size": "small", "type": "text-generation"},
            {"name": "EleutherAI/gpt-neo-125M", "description": "GPT-Neo 125M", "size": "small", "type": "text-generation"},
            {"name": "microsoft/phi-1_5", "description": "Phi-1.5 1.3B", "size": "small", "type": "text-generation"},
            
            # Qwen models (recommended)
            {"name": "Qwen/Qwen2.5-0.5B", "description": "Qwen 2.5 0.5B base", "size": "small", "type": "text-generation"},
            {"name": "Qwen/Qwen2.5-0.5B-Instruct", "description": "Qwen 2.5 0.5B instruct", "size": "small", "type": "instruction-following"},
            {"name": "Qwen/Qwen2.5-1.5B", "description": "Qwen 2.5 1.5B base", "size": "medium", "type": "text-generation"},
            {"name": "Qwen/Qwen2.5-1.5B-Instruct", "description": "Qwen 2.5 1.5B instruct", "size": "medium", "type": "instruction-following"},
            {"name": "Qwen/Qwen2.5-3B", "description": "Qwen 2.5 3B base", "size": "medium", "type": "text-generation"},
            {"name": "Qwen/Qwen2.5-3B-Instruct", "description": "Qwen 2.5 3B instruct", "size": "medium", "type": "instruction-following"},
            {"name": "Qwen/Qwen2.5-7B", "description": "Qwen 2.5 7B base", "size": "large", "type": "text-generation"},
            {"name": "Qwen/Qwen2.5-7B-Instruct", "description": "Qwen 2.5 7B instruct", "size": "large", "type": "instruction-following"},
            
            # TinyLlama
            {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "description": "TinyLlama 1.1B Chat", "size": "small", "type": "conversational"},
            {"name": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", "description": "TinyLlama 1.1B base", "size": "small", "type": "text-generation"},
            
            # Phi models (Microsoft)
            {"name": "microsoft/phi-2", "description": "Phi-2 2.7B", "size": "medium", "type": "text-generation"},
            
            # Gemma models (Google)
            {"name": "google/gemma-2b", "description": "Gemma 2B base", "size": "medium", "type": "text-generation"},
            {"name": "google/gemma-2b-it", "description": "Gemma 2B instruct", "size": "medium", "type": "instruction-following"},
            {"name": "google/gemma-7b", "description": "Gemma 7B base", "size": "large", "type": "text-generation"},
            {"name": "google/gemma-7b-it", "description": "Gemma 7B instruct", "size": "large", "type": "instruction-following"},
            
            # Medium models
            {"name": "gpt2-medium", "description": "GPT-2 Medium", "size": "medium", "type": "text-generation"},
            {"name": "gpt2-large", "description": "GPT-2 Large", "size": "large", "type": "text-generation"},
            {"name": "facebook/opt-1.3b", "description": "OPT 1.3B", "size": "medium", "type": "text-generation"},
            {"name": "facebook/opt-2.7b", "description": "OPT 2.7B", "size": "large", "type": "text-generation"},
            {"name": "EleutherAI/gpt-neo-1.3B", "description": "GPT-Neo 1.3B", "size": "medium", "type": "text-generation"},
            {"name": "EleutherAI/gpt-neo-2.7B", "description": "GPT-Neo 2.7B", "size": "large", "type": "text-generation"},
            
            # T5 models
            {"name": "t5-small", "description": "T5 Small", "size": "small", "type": "text-to-text"},
            {"name": "t5-base", "description": "T5 Base", "size": "medium", "type": "text-to-text"},
            {"name": "google/flan-t5-small", "description": "Flan-T5 Small", "size": "small", "type": "instruction-following"},
            {"name": "google/flan-t5-base", "description": "Flan-T5 Base", "size": "medium", "type": "instruction-following"},
            
            # Mistral models
            {"name": "mistralai/Mistral-7B-v0.1", "description": "Mistral 7B base", "size": "large", "type": "text-generation"},
            {"name": "mistralai/Mistral-7B-Instruct-v0.1", "description": "Mistral 7B instruct", "size": "large", "type": "instruction-following"},
            
            # Falcon models
            {"name": "tiiuae/falcon-7b", "description": "Falcon 7B base", "size": "large", "type": "text-generation"},
            {"name": "tiiuae/falcon-7b-instruct", "description": "Falcon 7B instruct", "size": "large", "type": "instruction-following"},
            
            # BLOOM models
            {"name": "bigscience/bloom-560m", "description": "BLOOM 560M", "size": "small", "type": "text-generation"},
            {"name": "bigscience/bloom-1b1", "description": "BLOOM 1.1B", "size": "medium", "type": "text-generation"},
            
            # Code models
            {"name": "Salesforce/codegen-350M-mono", "description": "CodeGen 350M", "size": "small", "type": "code-generation"},
            {"name": "microsoft/CodeGPT-small-py", "description": "CodeGPT Small Python", "size": "small", "type": "code-generation"},
            
            # Conversational models
            {"name": "microsoft/DialoGPT-small", "description": "DialoGPT Small", "size": "small", "type": "conversational"},
            {"name": "microsoft/DialoGPT-medium", "description": "DialoGPT Medium", "size": "medium", "type": "conversational"},
        ]
        
        # Reranker models (for retrieval)
        self._reranker_fallback = [
            # None option
            {"name": "", "display_name": "None (Vector + Keyword)", "description": "Use weighted combination of vector similarity and keyword matching", "provider": "none", "is_local": True},
            
            # Ollama reranker models
            {"name": "bge-reranker-v2-m3", "display_name": "BGE Reranker V2 M3", "description": "🎯 BGE dedicated reranking model (multilingual)", "provider": "ollama", "is_local": False},
            {"name": "qwen3-reranker:0.6b", "display_name": "Qwen3 Reranker 0.6B", "description": "🎯 Qwen dedicated reranking model (fast)", "provider": "ollama", "is_local": False},
            {"name": "qwen3-reranker:4b", "display_name": "Qwen3 Reranker 4B", "description": "🎯 Qwen dedicated reranking model (accurate)", "provider": "ollama", "is_local": False},
            
            # HuggingFace reranker models
            {"name": "BAAI/bge-reranker-v2-m3", "display_name": "BGE Reranker V2 M3", "description": "🤗 Multilingual BGE reranking model", "provider": "huggingface", "is_local": False},
            {"name": "BAAI/bge-reranker-large", "display_name": "BGE Reranker Large", "description": "🤗 BGE large reranking model (high accuracy)", "provider": "huggingface", "is_local": False},
            {"name": "BAAI/bge-reranker-base", "display_name": "BGE Reranker Base", "description": "🤗 BGE base reranking model", "provider": "huggingface", "is_local": False},
            {"name": "cross-encoder/ms-marco-MiniLM-L-6-v2", "display_name": "MS MARCO MiniLM L6", "description": "🤗 Fast cross-encoder reranker", "provider": "huggingface", "is_local": False},
            {"name": "cross-encoder/ms-marco-MiniLM-L-12-v2", "display_name": "MS MARCO MiniLM L12", "description": "🤗 Accurate cross-encoder reranker", "provider": "huggingface", "is_local": False},
            {"name": "mixedbread-ai/mxbai-rerank-large-v1", "display_name": "MixedBread Rerank Large", "description": "🤗 MixedBread large reranking model", "provider": "huggingface", "is_local": False},
            {"name": "mixedbread-ai/mxbai-rerank-base-v1", "display_name": "MixedBread Rerank Base", "description": "🤗 MixedBread base reranking model", "provider": "huggingface", "is_local": False},
            {"name": "Alibaba-NLP/gte-rerank-base", "display_name": "GTE Rerank Base", "description": "🤗 GTE base reranking model", "provider": "huggingface", "is_local": False},
            {"name": "Alibaba-NLP/gte-rerank-large", "display_name": "GTE Rerank Large", "description": "🤗 GTE large reranking model", "provider": "huggingface", "is_local": False},
        ]
        
        self.reranker_ttl = 1800  # 30 minutes for reranker models
        
        logger.info(f"Initialized fallback models: Ollama={len(self._ollama_fallback)}, "
                   f"HF-LLM={len(self._huggingface_llm_fallback)}, "
                   f"HF-Embed={len(self._huggingface_embedding_fallback)}, "
                   f"Finetuning={len(self._finetuning_fallback)}, "
                   f"Reranker={len(self._reranker_fallback)}")
    
    def _merge_models_with_fallback(
        self, 
        live_models: List[Dict[str, Any]], 
        fallback_models: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge live models with fallback models, ensuring essential models are always included.
        
        Live models take priority if the same model appears in both lists.
        Fallback models that don't appear in live data are appended.
        """
        # Create a set of live model names for quick lookup
        live_names = {m.get('name', '').lower() for m in live_models}
        
        # Start with live models
        merged = list(live_models)
        
        # Add fallback models that aren't in live data
        for fallback_model in fallback_models:
            fallback_name = fallback_model.get('name', '').lower()
            if fallback_name and fallback_name not in live_names:
                merged.append(fallback_model)
                logger.debug(f"Added fallback model: {fallback_model.get('name')}")
        
        return merged
    
    def get_status(self) -> Dict[str, Any]:
        """Get cache status for all providers"""
        with self._cache_lock:
            status = {}
            for key, cached in self._cache.items():
                status[key] = {
                    "status": self._status.get(key, CacheStatus.NOT_INITIALIZED).value,
                    "model_count": len(cached.models),
                    "source": cached.source,
                    "age_seconds": cached.age_seconds,
                    "is_expired": cached.is_expired,
                    "fetched_at": cached.fetched_at.isoformat()
                }
            return status
    
    def get_ollama_models(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Get Ollama models from cache or fallback"""
        return self._get_cached_models("ollama", self._ollama_fallback, force_refresh)
    
    def get_huggingface_llm_models(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Get HuggingFace LLM models from cache or fallback"""
        return self._get_cached_models("huggingface_llm", self._huggingface_llm_fallback, force_refresh)
    
    def get_huggingface_embedding_models(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Get HuggingFace embedding models from cache or fallback"""
        return self._get_cached_models("huggingface_embedding", self._huggingface_embedding_fallback, force_refresh)
    
    def get_finetuning_models(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Get finetuning models - combines HuggingFace LLM models with fallback list.
        
        For finetuning, we use HuggingFace LLM models since they can be finetuned.
        Falls back to curated list if HF models aren't available.
        """
        # Get HuggingFace LLM models from cache
        hf_llm_models = self._get_cached_models("huggingface_llm", self._huggingface_llm_fallback, force_refresh)
        
        if hf_llm_models and len(hf_llm_models) > 0:
            # Use HuggingFace LLM models for finetuning
            # Ensure they have the required fields
            result = []
            for model in hf_llm_models:
                result.append({
                    "name": model.get("name", ""),
                    "description": model.get("description", ""),
                    "size": model.get("size", "unknown"),
                    "type": model.get("type", "text-generation")
                })
            return result
        
        # Fallback to curated finetuning list
        return self._finetuning_fallback
    
    def get_reranker_models(self, provider: Optional[str] = None, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Get reranker models from cache or fallback.
        
        Args:
            provider: Optional filter by provider ('ollama', 'huggingface', or None for all)
            force_refresh: Force refresh from source
            
        Returns:
            List of reranker model dicts
        """
        models = self._get_cached_models("reranker", self._reranker_fallback, force_refresh)
        
        # Apply provider filter if specified
        if provider:
            if provider == 'ollama':
                models = [m for m in models if m.get('provider') in ['ollama', 'none']]
            elif provider == 'huggingface':
                models = [m for m in models if m.get('provider') in ['huggingface', 'none']]
        
        return models
    
    def _get_cached_models(
        self, 
        cache_key: str, 
        fallback: List[Dict[str, Any]], 
        force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """Get models from cache with fallback support"""
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            
            # Return cached if valid and not forcing refresh
            if cached and not cached.is_expired and not force_refresh:
                return cached.models
            
            # Return fallback if no cache or expired
            if not cached:
                logger.debug(f"Cache miss for {cache_key}, returning fallback")
                return fallback
            
            # Return stale cache if available (better than nothing)
            if cached.is_expired:
                logger.debug(f"Cache expired for {cache_key}, returning stale data")
                self._status[cache_key] = CacheStatus.STALE
                return cached.models
            
            return fallback
    
    def update_cache(
        self, 
        cache_key: str, 
        models: List[Dict[str, Any]], 
        ttl_seconds: Optional[int] = None,
        source: str = "live"
    ):
        """Update cache with new model list"""
        with self._cache_lock:
            ttl = ttl_seconds or self._get_default_ttl(cache_key)
            self._cache[cache_key] = CachedModelList(
                models=models,
                fetched_at=datetime.now(),
                ttl_seconds=ttl,
                source=source
            )
            self._status[cache_key] = CacheStatus.READY
            logger.info(f"Updated cache for {cache_key}: {len(models)} models (TTL: {ttl}s)")
    
    def _get_default_ttl(self, cache_key: str) -> int:
        """Get default TTL for cache key"""
        if cache_key == "ollama":
            return self.ollama_ttl
        elif cache_key.startswith("huggingface"):
            return self.huggingface_ttl
        elif cache_key == "finetuning":
            return self.finetuning_ttl
        return 3600  # 1 hour default
    
    async def refresh_all_caches(self):
        """Refresh all model caches from live sources"""
        logger.info("🔄 Starting cache refresh for all providers...")
        start_time = time.time()
        
        tasks = [
            self._refresh_ollama_cache(),
            self._refresh_huggingface_cache(),
        ]
        
        # Also populate finetuning cache from fallback (it's comprehensive)
        self.update_cache("finetuning", self._finetuning_fallback, source="fallback")
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Cache refresh completed in {elapsed:.2f}s")
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Cache refresh task {i} failed: {result}")
    
    async def _refresh_ollama_cache(self):
        """Refresh Ollama models cache"""
        try:
            self._status["ollama"] = CacheStatus.LOADING
            logger.info("Fetching Ollama models...")
            
            from app.core.providers import OllamaProvider
            provider = OllamaProvider()
            
            # Get installed models
            installed = await provider.get_installed_models()
            
            # Get available models from library
            available = await provider.get_available_models()
            
            # Combine and deduplicate
            all_models = []
            seen = set()
            
            # Add installed models first
            for model in installed:
                name = model.get("name", "")
                if name and name not in seen:
                    model["installed"] = True
                    all_models.append(model)
                    seen.add(name)
            
            # Add available models
            for model in available:
                name = model.get("name", "")
                if name and name not in seen:
                    model["installed"] = False
                    all_models.append(model)
                    seen.add(name)
            
            if all_models:
                self.update_cache("ollama", all_models, source="live")
                logger.info(f"✅ Ollama cache updated: {len(all_models)} models")
            else:
                logger.warning("No Ollama models fetched, keeping fallback")
                self.update_cache("ollama", self._ollama_fallback, source="fallback")
                
        except Exception as e:
            logger.error(f"Failed to refresh Ollama cache: {e}")
            self._status["ollama"] = CacheStatus.ERROR
            # Use fallback
            self.update_cache("ollama", self._ollama_fallback, source="fallback")
    
    async def _refresh_huggingface_cache(self):
        """Refresh HuggingFace models cache"""
        try:
            self._status["huggingface_llm"] = CacheStatus.LOADING
            self._status["huggingface_embedding"] = CacheStatus.LOADING
            logger.info("Fetching HuggingFace models...")
            
            from app.core.providers import HuggingFaceProvider
            provider = HuggingFaceProvider()
            
            # Get LLM models
            llm_models = await provider.get_available_llm_models(limit=200)
            
            # Get embedding models
            embedding_models = await provider.get_available_embedding_models(limit=100)
            
            if llm_models:
                # IMPORTANT: Merge live models with fallback to ensure essential models are always available
                merged_llm = self._merge_models_with_fallback(llm_models, self._huggingface_llm_fallback)
                self.update_cache("huggingface_llm", merged_llm, source="live")
                logger.info(f"✅ HuggingFace LLM cache updated: {len(merged_llm)} models (live: {len(llm_models)}, fallback: {len(self._huggingface_llm_fallback)})")
            else:
                self.update_cache("huggingface_llm", self._huggingface_llm_fallback, source="fallback")
            
            if embedding_models:
                # IMPORTANT: Merge live models with fallback to ensure essential models are always available
                merged_embedding = self._merge_models_with_fallback(embedding_models, self._huggingface_embedding_fallback)
                self.update_cache("huggingface_embedding", merged_embedding, source="live")
                logger.info(f"✅ HuggingFace embedding cache updated: {len(merged_embedding)} models (live: {len(embedding_models)}, fallback: {len(self._huggingface_embedding_fallback)})")
            else:
                self.update_cache("huggingface_embedding", self._huggingface_embedding_fallback, source="fallback")
                
        except Exception as e:
            logger.error(f"Failed to refresh HuggingFace cache: {e}")
            self._status["huggingface_llm"] = CacheStatus.ERROR
            self._status["huggingface_embedding"] = CacheStatus.ERROR
            # Use fallback
            self.update_cache("huggingface_llm", self._huggingface_llm_fallback, source="fallback")
            self.update_cache("huggingface_embedding", self._huggingface_embedding_fallback, source="fallback")
    
    def get_huggingface_models(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Get all HuggingFace models (LLM + embedding) from cache or fallback"""
        llm_models = self.get_huggingface_llm_models(force_refresh)
        embedding_models = self.get_huggingface_embedding_models(force_refresh)
        
        # Ensure proper provider field
        for model in llm_models:
            model["provider"] = "huggingface"
        for model in embedding_models:
            model["provider"] = "huggingface"
        
        return llm_models + embedding_models
    
    def get_all_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all models from all providers"""
        return {
            "ollama": self.get_ollama_models(),
            "huggingface": self.get_huggingface_models(),
            "finetuning": self.get_finetuning_models(),
            "reranker": self.get_reranker_models()
        }
    
    def warm_up(self):
        """Warm up the cache synchronously using fallback data.
        
        This method immediately populates the cache with fallback data
        so the application can start serving requests immediately.
        Live data will be fetched in the background.
        """
        logger.info("🔥 Warming up model cache with fallback data...")
        
        # Populate with fallback data immediately
        self.update_cache("ollama", self._ollama_fallback, source="fallback")
        self.update_cache("huggingface_llm", self._huggingface_llm_fallback, source="fallback")
        self.update_cache("huggingface_embedding", self._huggingface_embedding_fallback, source="fallback")
        self.update_cache("finetuning", self._finetuning_fallback, source="fallback")
        self.update_cache("reranker", self._reranker_fallback, source="fallback")
        
        logger.info(f"✅ Cache warmed up with fallback data - "
                   f"Ollama: {len(self._ollama_fallback)}, "
                   f"HF-LLM: {len(self._huggingface_llm_fallback)}, "
                   f"HF-Embed: {len(self._huggingface_embedding_fallback)}, "
                   f"Finetuning: {len(self._finetuning_fallback)}, "
                   f"Reranker: {len(self._reranker_fallback)}")
        
        # Start background refresh
        import threading
        def background_refresh():
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.refresh_all_caches())
                loop.close()
            except Exception as e:
                logger.warning(f"Background cache refresh failed: {e}")
        
        thread = threading.Thread(target=background_refresh, daemon=True)
        thread.start()
        logger.info("🔄 Background cache refresh started")


# Singleton instance
_model_cache: Optional[ModelCache] = None


def get_model_cache() -> ModelCache:
    """Get the singleton model cache instance"""
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelCache()
    return _model_cache


async def initialize_model_cache():
    """Initialize and populate the model cache at startup"""
    cache = get_model_cache()
    await cache.refresh_all_caches()
    return cache
