"""
RAG Utility Functions

Helper functions for RAG operations including model compatibility checks,
provider detection, and embedding model creation.
"""
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


def detect_model_provider(model_name: str) -> str:
    """
    Detect the provider for a given model based on its name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Provider name ('ollama', 'huggingface', or 'openai')
        
    Examples:
        >>> detect_model_provider("llama2:latest")
        'ollama'
        >>> detect_model_provider("sentence-transformers/all-MiniLM-L6-v2")
        'huggingface'
    """
    if not model_name:
        return 'ollama'  # Default to ollama
    
    model_lower = model_name.lower()
    
    # HuggingFace models typically have / in the name (org/model format)
    if '/' in model_name and not model_name.startswith('http'):
        return 'huggingface'
    
    # OpenAI models
    if model_lower.startswith(('gpt-', 'text-embedding-', 'text-davinci', 'text-curie')):
        return 'openai'
    
    # Common HuggingFace patterns
    hf_indicators = [
        'sentence-transformers',
        'BAAI',
        'microsoft',
        'facebook',
        'google',
        'bert-',
        'roberta-',
        't5-'
    ]
    
    if any(indicator in model_name for indicator in hf_indicators):
        return 'huggingface'
    
    # Default to Ollama (most common for local models)
    return 'ollama'


def check_model_compatibility(
    model_name: str,
    model_size: Optional[str] = None
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Check if a model is compatible with the current system.
    
    This is a simplified compatibility check. For GPU-specific checks,
    use check_model_compatibility_detailed from gpu_utils.
    
    Args:
        model_name: Name of the model to check
        model_size: Size of the model (e.g., "7B", "13B")
        
    Returns:
        Tuple of (is_compatible, message, details)
        
    Examples:
        >>> compatible, msg, details = check_model_compatibility("llama2", "7B")
        >>> if compatible:
        ...     print(f"Model is compatible: {msg}")
    """
    if not model_name:
        return False, "No model specified", {"error": "Model name required"}
    
    try:
        # Import GPU utilities for detailed check
        from app.utils.gpu_utils import check_model_compatibility_detailed
        
        # Use the detailed GPU compatibility check
        return check_model_compatibility_detailed(model_name, model_size)
        
    except ImportError:
        logger.warning("GPU utils not available, using basic compatibility check")
        
        # Basic compatibility check - assume compatible if model name is valid
        return True, f"Model {model_name} is compatible (basic check)", {
            "model": model_name,
            "size": model_size or "Unknown",
            "status": "compatible_basic",
            "check_type": "fallback"
        }
    except Exception as e:
        logger.error(f"Error checking model compatibility for {model_name}: {e}")
        return False, f"Error checking compatibility: {str(e)}", {
            "error": str(e),
            "model": model_name
        }


def create_embedding_model(
    model_name: str,
    provider: Optional[str] = None,
    **kwargs
):
    """
    Create an embedding model instance based on provider.
    
    Args:
        model_name: Name of the embedding model
        provider: Provider name ('ollama', 'huggingface', 'openai')
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        Embedding model instance
        
    Examples:
        >>> embeddings = create_embedding_model("mxbai-embed-large", provider="ollama")
        >>> embeddings = create_embedding_model("BAAI/bge-large-en-v1.5", provider="huggingface")
    """
    if provider is None:
        provider = detect_model_provider(model_name)
    
    logger.info(f"Creating embedding model: {model_name} (provider: {provider})")
    
    try:
        if provider == 'ollama':
            from langchain_ollama import OllamaEmbeddings
            import os
            
            ollama_host = os.getenv('OLLAMA_HOST', 'http://ollama:11434')
            return OllamaEmbeddings(
                model=model_name,
                base_url=ollama_host,
                **kwargs
            )
            
        elif provider == 'huggingface':
            from langchain_huggingface import HuggingFaceEmbeddings
            
            return HuggingFaceEmbeddings(
                model_name=model_name,
                **kwargs
            )
            
        elif provider == 'openai':
            from langchain_openai import OpenAIEmbeddings
            
            return OpenAIEmbeddings(
                model=model_name,
                **kwargs
            )
            
        else:
            raise ValueError(f"Unknown provider: {provider}")
            
    except ImportError as e:
        logger.error(f"Failed to import embedding model for provider {provider}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to create embedding model {model_name}: {e}")
        raise


def get_model_provider_info(model_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a model's provider.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with provider information
        
    Example:
        >>> info = get_model_provider_info("llama2:7b")
        >>> print(info['provider'])  # 'ollama'
        >>> print(info['is_local'])  # True
    """
    provider = detect_model_provider(model_name)
    
    provider_info = {
        'provider': provider,
        'model_name': model_name,
        'is_local': provider in ['ollama'],
        'is_cloud': provider in ['openai', 'anthropic', 'cohere'],
        'requires_api_key': provider in ['openai', 'anthropic', 'cohere'],
        'supports_gpu': provider in ['ollama', 'huggingface']
    }
    
    return provider_info
