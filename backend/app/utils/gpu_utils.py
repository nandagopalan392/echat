"""
GPU Memory Management Utilities

Handles dynamic GPU/CPU switching based on memory availability.
Used for model loading, document processing, and other GPU-intensive tasks.
"""
import os
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


def check_gpu_memory_available(min_memory_gb: float = 2.0) -> Tuple[bool, str]:
    """
    Check if GPU has sufficient memory available for processing.
    
    Args:
        min_memory_gb: Minimum required GPU memory in GB
        
    Returns:
        Tuple of (is_available, status_message)
        
    Examples:
        >>> available, msg = check_gpu_memory_available(2.0)
        >>> if available:
        ...     print("GPU ready for use")
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        
        # Check each GPU device
        for device_id in range(torch.cuda.device_count()):
            # Get GPU memory info
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            allocated_memory = torch.cuda.memory_allocated(device_id)
            reserved_memory = torch.cuda.memory_reserved(device_id)
            
            # Calculate available memory
            available_memory = total_memory - max(allocated_memory, reserved_memory)
            available_gb = available_memory / (1024**3)
            total_gb = total_memory / (1024**3)
            
            logger.info(f"🔍 GPU {device_id}: {available_gb:.2f}GB available / {total_gb:.2f}GB total")
            
            if available_gb >= min_memory_gb:
                return True, f"GPU {device_id} has {available_gb:.2f}GB available (>= {min_memory_gb}GB required)"
        
        return False, f"No GPU has sufficient memory (need {min_memory_gb}GB, best available: {available_gb:.2f}GB)"
        
    except ImportError:
        return False, "PyTorch not available for GPU memory check"
    except Exception as e:
        return False, f"GPU memory check failed: {e}"


def get_gpu_memory_info() -> Dict[str, Any]:
    """
    Get detailed GPU memory information for all available devices.
    
    Returns:
        Dictionary with GPU memory stats for each device
        
    Example:
        >>> info = get_gpu_memory_info()
        >>> print(f"GPU 0: {info['devices'][0]['available_gb']:.2f}GB available")
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {
                "available": False,
                "message": "CUDA not available",
                "devices": []
            }
        
        devices = []
        for device_id in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(device_id)
            total_memory = props.total_memory
            allocated_memory = torch.cuda.memory_allocated(device_id)
            reserved_memory = torch.cuda.memory_reserved(device_id)
            
            available_memory = total_memory - max(allocated_memory, reserved_memory)
            
            devices.append({
                "id": device_id,
                "name": props.name,
                "total_gb": total_memory / (1024**3),
                "allocated_gb": allocated_memory / (1024**3),
                "reserved_gb": reserved_memory / (1024**3),
                "available_gb": available_memory / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}"
            })
        
        return {
            "available": True,
            "message": f"Found {len(devices)} GPU device(s)",
            "devices": devices
        }
        
    except ImportError:
        return {
            "available": False,
            "message": "PyTorch not available",
            "devices": []
        }
    except Exception as e:
        return {
            "available": False,
            "message": f"Error: {e}",
            "devices": []
        }


def configure_docling_device(force_cpu: bool = False) -> Tuple[bool, str]:
    """
    Configure Docling to use GPU or CPU based on memory availability.
    
    Args:
        force_cpu: Force CPU usage regardless of GPU availability
        
    Returns:
        Tuple of (use_gpu, configuration_message)
        
    Example:
        >>> use_gpu, msg = configure_docling_device()
        >>> print(f"Using {'GPU' if use_gpu else 'CPU'}: {msg}")
    """
    if force_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return False, "CPU usage forced by parameter"
    
    # Check GPU memory availability
    gpu_available, gpu_status = check_gpu_memory_available(min_memory_gb=1.5)  # Need at least 1.5GB for Docling
    
    if gpu_available:
        # Allow GPU usage
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        return True, f"GPU enabled: {gpu_status}"
    else:
        # Force CPU usage
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        return False, f"CPU forced: {gpu_status}"


def check_model_compatibility_detailed(
    model_name: str,
    model_size: Optional[str] = None,
    min_vram_gb: float = 2.0
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Check if a model is compatible with available GPU resources.
    
    Args:
        model_name: Name of the model to check
        model_size: Size of the model (e.g., "7B", "13B")
        min_vram_gb: Minimum VRAM required in GB
        
    Returns:
        Tuple of (is_compatible, message, details_dict)
        
    Example:
        >>> compatible, msg, details = check_model_compatibility_detailed("llama2", "7B")
        >>> if compatible:
        ...     print(f"Model {model_name} is compatible")
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return False, "CUDA not available", {
                "cuda_available": False,
                "message": "No GPU detected"
            }
        
        gpu_info = get_gpu_memory_info()
        
        if not gpu_info["available"] or not gpu_info["devices"]:
            return False, "No GPU devices found", {
                "cuda_available": True,
                "devices": 0,
                "message": "CUDA available but no devices detected"
            }
        
        # Get the best GPU (most available memory)
        best_device = max(gpu_info["devices"], key=lambda d: d["available_gb"])
        available_gb = best_device["available_gb"]
        
        if available_gb >= min_vram_gb:
            return True, f"Compatible - {available_gb:.2f}GB available", {
                "cuda_available": True,
                "device_count": len(gpu_info["devices"]),
                "best_device": best_device,
                "available_gb": available_gb,
                "required_gb": min_vram_gb,
                "compatible": True
            }
        else:
            return False, f"Insufficient VRAM - {available_gb:.2f}GB available, need {min_vram_gb}GB", {
                "cuda_available": True,
                "device_count": len(gpu_info["devices"]),
                "best_device": best_device,
                "available_gb": available_gb,
                "required_gb": min_vram_gb,
                "compatible": False
            }
            
    except ImportError:
        return False, "PyTorch not available", {
            "cuda_available": False,
            "message": "PyTorch not installed"
        }
    except Exception as e:
        return False, f"Error checking compatibility: {e}", {
            "cuda_available": False,
            "message": str(e)
        }


def is_cuda_memory_error(error: Exception) -> bool:
    """
    Check if an exception is related to CUDA memory issues.
    
    Args:
        error: The exception to check
        
    Returns:
        True if it's a CUDA memory error
        
    Example:
        >>> try:
        ...     # Some CUDA operation
        ...     pass
        ... except Exception as e:
        ...     if is_cuda_memory_error(e):
        ...         clear_gpu_memory()
    """
    error_str = str(error).lower()
    cuda_memory_indicators = [
        'cuda out of memory',
        'out of memory',
        'cuda runtime error',
        'gpu memory',
        'cuda error',
        'torch.cuda.outofmemoryerror'
    ]
    
    return any(indicator in error_str for indicator in cuda_memory_indicators)


def clear_gpu_memory() -> None:
    """
    Clear GPU memory cache if possible.
    
    Useful after encountering CUDA out of memory errors or
    when switching between models.
    
    Example:
        >>> clear_gpu_memory()  # Frees up GPU memory
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 GPU memory cache cleared")
    except ImportError:
        logger.debug("PyTorch not available for GPU memory clearing")
    except Exception as e:
        logger.warning(f"Failed to clear GPU memory: {e}")


def get_optimal_device() -> str:
    """
    Get the optimal device (cuda:0, cuda:1, or cpu) based on availability.
    
    Returns:
        Device string (e.g., "cuda:0" or "cpu")
        
    Example:
        >>> device = get_optimal_device()
        >>> model.to(device)
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return "cpu"
        
        # Find GPU with most available memory
        gpu_info = get_gpu_memory_info()
        if gpu_info["available"] and gpu_info["devices"]:
            best_device = max(gpu_info["devices"], key=lambda d: d["available_gb"])
            return f"cuda:{best_device['id']}"
        
        return "cpu"
        
    except Exception:
        return "cpu"
