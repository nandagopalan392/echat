"""
Model Utilities - Helper functions for model management

This module provides utility functions for working with AI models including:
- Model name normalization
- Model categorization
- Size formatting
- Parameter validation
- GPU compatibility recommendations
"""

import json
from typing import Dict, Any, List, Optional, Tuple


def format_model_size(size: Any) -> str:
    """
    Format model size in human-readable format.
    
    Args:
        size: Size as string, int, or float (in bytes)
        
    Returns:
        Human-readable size string (e.g., "7.2GB", "335MB")
        
    Examples:
        >>> format_model_size(7200000000)
        '6.7GB'
        >>> format_model_size("7.2GB")
        '7.2GB'
        >>> format_model_size(1024)
        '1.0KB'
    """
    if isinstance(size, str):
        return size
    elif isinstance(size, (int, float)):
        if size < 1024:
            return f"{size}B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f}KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024):.1f}MB"
        else:
            return f"{size / (1024 * 1024 * 1024):.1f}GB"
    else:
        return "Unknown"


def categorize_model(model_name: str) -> str:
    """
    Determine if a model is an LLM or embedding model based on its name.
    
    Args:
        model_name: Model name to categorize
        
    Returns:
        'llm' or 'embedding'
        
    Examples:
        >>> categorize_model('llama2')
        'llm'
        >>> categorize_model('bge-m3')
        'embedding'
        >>> categorize_model('nomic-embed-text')
        'embedding'
    """
    model_name_lower = model_name.lower()
    
    # Embedding model indicators
    embedding_indicators = [
        'embed', 'bge', 'minilm', 'all-minilm', 'nomic', 'e5-',
        'sentence', 'text-embedding', 'instructor', 'gte-',
        'multilingual-e5', 'arctic-embed', 'mxbai-embed',
        'snowflake-arctic-embed', 'paraphrase-', 'distiluse',
        'voyage-', 'cohere-embed', 'jina-'
    ]
    
    # Common embedding model prefixes
    embedding_prefixes = [
        'bge-', 'all-minilm-', 'e5-', 'gte-', 'nomic-',
        'mxbai-', 'snowflake-', 'voyage-', 'jina-'
    ]
    
    is_embedding = (
        any(indicator in model_name_lower for indicator in embedding_indicators) or
        any(model_name_lower.startswith(prefix) for prefix in embedding_prefixes)
    )
    
    return 'embedding' if is_embedding else 'llm'


def is_embedding_model(model_name: str) -> bool:
    """
    Check if a model is an embedding model.
    
    Args:
        model_name: Model name to check
        
    Returns:
        True if model is an embedding model, False otherwise
        
    Examples:
        >>> is_embedding_model('bge-large')
        True
        >>> is_embedding_model('llama2')
        False
    """
    return categorize_model(model_name) == 'embedding'


def normalize_bge_model_name(model_name: str) -> str:
    """
    Normalize BGE (BAAI General Embedding) model names.
    
    BGE models often have size suffixes that need to be removed for Ollama.
    
    Args:
        model_name: BGE model name to normalize
        
    Returns:
        Normalized model name
        
    Examples:
        >>> normalize_bge_model_name('bge-m3:1.7M')
        'bge-m3'
        >>> normalize_bge_model_name('bge-large:335m')
        'bge-large'
        >>> normalize_bge_model_name('bge-small:33.4m')
        'bge-small'
    """
    # Common BGE model variants that need normalization
    bge_mappings = {
        'bge-m3:1.7M': 'bge-m3',
        'bge-m3:1.7m': 'bge-m3',
        'bge-large:335m': 'bge-large',
        'bge-large:335M': 'bge-large',
        'bge-base:109m': 'bge-base',
        'bge-base:109M': 'bge-base',
        'bge-small:33.4m': 'bge-small',
        'bge-small:33.4M': 'bge-small',
        'bge-m3:567m': 'bge-m3',
        'bge-m3:567M': 'bge-m3'
    }
    
    # Check exact match first
    if model_name in bge_mappings:
        return bge_mappings[model_name]
    
    # Check case-insensitive match
    model_name_lower = model_name.lower()
    for key, value in bge_mappings.items():
        if model_name_lower == key.lower():
            return value
    
    # If it's a BGE model but not in mappings, try to strip size suffix
    if 'bge-' in model_name_lower:
        # Remove common size patterns like :1.7M, :335m, etc.
        import re
        normalized = re.sub(r':\d+\.?\d*[mMbBkK]', '', model_name)
        return normalized
    
    return model_name


def normalize_nomic_model_name(model_name: str) -> str:
    """
    Normalize Nomic embedding model names.
    
    Args:
        model_name: Nomic model name to normalize
        
    Returns:
        Normalized model name
        
    Examples:
        >>> normalize_nomic_model_name('nomic-embed-text:33.3M')
        'nomic-embed-text'
        >>> normalize_nomic_model_name('nomic-embed-text:latest')
        'nomic-embed-text'
    """
    nomic_mappings = {
        'nomic-embed-text:33.3M': 'nomic-embed-text',
        'nomic-embed-text:33.3m': 'nomic-embed-text',
        'nomic-embed-text:v1': 'nomic-embed-text',
        'nomic-embed-text:v1.5': 'nomic-embed-text'
    }
    
    # Check exact match
    if model_name in nomic_mappings:
        return nomic_mappings[model_name]
    
    # Remove :latest suffix
    if model_name.endswith(':latest'):
        return model_name.replace(':latest', '')
    
    # Remove version/size suffixes for nomic models
    if 'nomic-' in model_name.lower():
        import re
        normalized = re.sub(r':(v?\d+\.?\d*[mMbBkK]?|latest)', '', model_name)
        return normalized
    
    return model_name


def normalize_model_name(model_name: str) -> str:
    """
    Normalize model names by fixing common naming issues.
    
    This function handles:
    - BGE model size suffixes
    - Nomic model version suffixes
    - Other common model naming patterns
    
    Args:
        model_name: Model name to normalize
        
    Returns:
        Normalized model name
        
    Examples:
        >>> normalize_model_name('bge-m3:1.7M')
        'bge-m3'
        >>> normalize_model_name('nomic-embed-text:33.3M')
        'nomic-embed-text'
        >>> normalize_model_name('llama2:latest')
        'llama2'
    """
    if not model_name:
        return model_name
    
    # Try BGE normalization first
    if 'bge' in model_name.lower():
        return normalize_bge_model_name(model_name)
    
    # Try Nomic normalization
    if 'nomic' in model_name.lower():
        return normalize_nomic_model_name(model_name)
    
    # Remove :latest suffix for any model
    if model_name.endswith(':latest'):
        return model_name.replace(':latest', '')
    
    return model_name


def get_model_variants(model_name: str) -> List[str]:
    """
    Generate common variants of a model name for matching.
    
    Args:
        model_name: Base model name
        
    Returns:
        List of possible model name variants
        
    Examples:
        >>> get_model_variants('llama2')
        ['llama2', 'llama2:latest', 'llama2']
        >>> get_model_variants('bge-m3:1.7M')
        ['bge-m3', 'bge-m3:latest', 'bge-m3:1.7M']
    """
    normalized = normalize_model_name(model_name)
    variants = [
        model_name,
        f"{normalized}:latest",
        normalized.replace(":latest", ""),
        normalized
    ]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_variants = []
    for variant in variants:
        if variant not in seen:
            seen.add(variant)
            unique_variants.append(variant)
    
    return unique_variants


def validate_model_parameters(parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Validate and normalize model parameters.
    
    Args:
        parameters: Dictionary of model parameters to validate
        
    Returns:
        Dictionary of validated parameters with defaults
        
    Raises:
        ValueError: If parameters are invalid
        
    Examples:
        >>> validate_model_parameters({'temperature': 0.7})
        {'temperature': 0.7, 'max_tokens': 2048, 'top_p': 0.9, ...}
        >>> validate_model_parameters({'temperature': 5.0})  # Invalid
        {'temperature': 0.7, 'max_tokens': 2048, ...}  # Uses default
    """
    # Default parameters
    defaults = {
        'temperature': 0.7,
        'max_tokens': 2048,
        'top_p': 0.9,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0
    }
    
    if not parameters:
        return defaults.copy()
    
    valid_parameters = {}
    
    # Validate temperature (0.0 to 2.0)
    temp = parameters.get('temperature', 0.7)
    if isinstance(temp, (int, float)) and 0.0 <= temp <= 2.0:
        valid_parameters['temperature'] = float(temp)
    else:
        valid_parameters['temperature'] = defaults['temperature']
    
    # Validate max_tokens (1 to 32768)
    max_tokens = parameters.get('max_tokens', 2048)
    if isinstance(max_tokens, int) and 1 <= max_tokens <= 32768:
        valid_parameters['max_tokens'] = max_tokens
    else:
        valid_parameters['max_tokens'] = defaults['max_tokens']
    
    # Validate top_p (0.0 to 1.0)
    top_p = parameters.get('top_p', 0.9)
    if isinstance(top_p, (int, float)) and 0.0 <= top_p <= 1.0:
        valid_parameters['top_p'] = float(top_p)
    else:
        valid_parameters['top_p'] = defaults['top_p']
    
    # Validate frequency_penalty (-2.0 to 2.0)
    freq_penalty = parameters.get('frequency_penalty', 0.0)
    if isinstance(freq_penalty, (int, float)) and -2.0 <= freq_penalty <= 2.0:
        valid_parameters['frequency_penalty'] = float(freq_penalty)
    else:
        valid_parameters['frequency_penalty'] = defaults['frequency_penalty']
    
    # Validate presence_penalty (-2.0 to 2.0)
    presence_penalty = parameters.get('presence_penalty', 0.0)
    if isinstance(presence_penalty, (int, float)) and -2.0 <= presence_penalty <= 2.0:
        valid_parameters['presence_penalty'] = float(presence_penalty)
    else:
        valid_parameters['presence_penalty'] = defaults['presence_penalty']
    
    return valid_parameters


def generate_compatibility_recommendations(
    llm_details: Dict[str, Any],
    embedding_details: Dict[str, Any],
    combined_compatible: bool
) -> List[str]:
    """
    Generate GPU compatibility recommendations based on model requirements.
    
    Args:
        llm_details: Dictionary containing LLM memory requirements
        embedding_details: Dictionary containing embedding model memory requirements
        combined_compatible: Whether models fit together in GPU memory
        
    Returns:
        List of recommendation strings
        
    Examples:
        >>> llm_details = {'required_memory_mb': 8000, 'available_memory_mb': 6000}
        >>> emb_details = {'required_memory_mb': 500, 'available_memory_mb': 6000}
        >>> generate_compatibility_recommendations(llm_details, emb_details, False)
        ['Consider using smaller models...', 'LLM model requires 8000MB...']
    """
    recommendations = []
    
    if not combined_compatible:
        recommendations.append("Consider using smaller models that require less GPU memory")
        recommendations.append("Upgrade your GPU to one with more VRAM")
        recommendations.append("Use quantized models if available (e.g., 4-bit or 8-bit versions)")
        recommendations.append("Consider using CPU-only mode for less demanding tasks")
    
    # LLM-specific recommendations
    if llm_details.get('required_memory_mb', 0) > llm_details.get('available_memory_mb', 0):
        required_mb = llm_details['required_memory_mb']
        recommendations.append(
            f"LLM model requires {required_mb}MB - consider a smaller variant or quantized version"
        )
        
        # Suggest specific alternatives based on size
        if required_mb > 16000:
            recommendations.append("Try 7B or 13B parameter models instead of 70B+ models")
        elif required_mb > 8000:
            recommendations.append("Try 3B or 7B parameter models for lower memory usage")
    
    # Embedding-specific recommendations
    if embedding_details.get('required_memory_mb', 0) > embedding_details.get('available_memory_mb', 0):
        required_mb = embedding_details['required_memory_mb']
        recommendations.append(
            f"Embedding model requires {required_mb}MB - consider a smaller variant"
        )
        recommendations.append("Try 'all-minilm-l6-v2' or 'bge-small' for lower memory usage")
    
    return recommendations


def is_gated_model_error(error_message: str) -> bool:
    """
    Check if an error message indicates a gated model access issue.
    
    Gated models on HuggingFace require special access permissions.
    
    Args:
        error_message: Error message to check
        
    Returns:
        True if error indicates gated model issue, False otherwise
        
    Examples:
        >>> is_gated_model_error("GATED_MODEL_ERROR: Access denied")
        True
        >>> is_gated_model_error("Model requires access approval")
        True
        >>> is_gated_model_error("Network timeout")
        False
    """
    if not error_message:
        return False
    
    error_lower = error_message.lower()
    
    # Check for explicit gated model error marker
    if "GATED_MODEL_ERROR:" in error_message:
        return True
    
    # Check for gated model indicators
    gated_indicators = [
        'gated',
        'access',
        'requires approval',
        'request access',
        'permission denied',
        'authentication required',
        'not authorized',
        'access token'
    ]
    
    return any(indicator in error_lower for indicator in gated_indicators)


def parse_gated_model_error(error_message: str) -> Optional[Dict[str, Any]]:
    """
    Parse a gated model error message and extract structured data.
    
    Args:
        error_message: Error message containing gated model information
        
    Returns:
        Dictionary with error details or None if not a gated model error
        
    Examples:
        >>> error = "GATED_MODEL_ERROR:{'model_name': 'meta-llama/Llama-2-70b'}"
        >>> parse_gated_model_error(error)
        {'model_name': 'meta-llama/Llama-2-70b'}
    """
    if not is_gated_model_error(error_message):
        return None
    
    # Try to extract JSON from error message
    if "GATED_MODEL_ERROR:" in error_message:
        try:
            json_str = error_message.split("GATED_MODEL_ERROR:", 1)[1].strip()
            return json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            pass
    
    # Return basic structure if can't parse
    return {
        "error_type": "gated_model",
        "message": error_message,
        "requires_access": True
    }


# Additional utility functions for model management

def extract_model_size_from_name(model_name: str) -> Optional[str]:
    """
    Extract size information from model name.
    
    Args:
        model_name: Model name that may contain size info
        
    Returns:
        Size string or None if not found
        
    Examples:
        >>> extract_model_size_from_name('llama2:7b')
        '7b'
        >>> extract_model_size_from_name('mistral:8x7b')
        '8x7b'
    """
    import re
    
    # Pattern for size like :7b, :13b, :70b, :8x7b
    size_pattern = r':(\d+x?\d*[bBmMkK])'
    match = re.search(size_pattern, model_name)
    
    if match:
        return match.group(1).lower()
    
    return None


def compare_model_sizes(size1: str, size2: str) -> int:
    """
    Compare two model sizes.
    
    Args:
        size1: First size string (e.g., '7b', '13b')
        size2: Second size string
        
    Returns:
        -1 if size1 < size2, 0 if equal, 1 if size1 > size2
        
    Examples:
        >>> compare_model_sizes('7b', '13b')
        -1
        >>> compare_model_sizes('70b', '13b')
        1
    """
    import re
    
    def parse_size(size_str: str) -> float:
        """Convert size string to numeric value for comparison."""
        if not size_str:
            return 0
        
        size_lower = size_str.lower()
        
        # Extract numeric part
        match = re.search(r'(\d+\.?\d*)', size_lower)
        if not match:
            return 0
        
        value = float(match.group(1))
        
        # Apply multiplier based on unit
        if 'b' in size_lower:  # Billion parameters
            value *= 1_000_000_000
        elif 'm' in size_lower:  # Million parameters
            value *= 1_000_000
        elif 'k' in size_lower:  # Thousand parameters
            value *= 1_000
        
        return value
    
    val1 = parse_size(size1)
    val2 = parse_size(size2)
    
    if val1 < val2:
        return -1
    elif val1 > val2:
        return 1
    else:
        return 0
