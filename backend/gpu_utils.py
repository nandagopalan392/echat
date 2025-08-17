"""
GPU Memory Management Utilities for Docling
Handles dynamic GPU/CPU switching based on memory availability
"""
import os
import logging

logger = logging.getLogger(__name__)

def check_gpu_memory_available(min_memory_gb: float = 2.0) -> tuple[bool, str]:
    """
    Check if GPU has sufficient memory available for Docling processing.
    
    Args:
        min_memory_gb: Minimum required GPU memory in GB
        
    Returns:
        Tuple of (is_available, status_message)
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

def configure_docling_device(force_cpu: bool = False) -> tuple[bool, str]:
    """
    Configure Docling to use GPU or CPU based on memory availability.
    
    Args:
        force_cpu: Force CPU usage regardless of GPU availability
        
    Returns:
        Tuple of (use_gpu, configuration_message)
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

def is_cuda_memory_error(error: Exception) -> bool:
    """
    Check if an exception is related to CUDA memory issues.
    
    Args:
        error: The exception to check
        
    Returns:
        True if it's a CUDA memory error
    """
    error_str = str(error).lower()
    cuda_memory_indicators = [
        'cuda out of memory',
        'out of memory',
        'cuda runtime error',
        'gpu memory',
        'cuda error',
        'torch.cuda.OutOfMemoryError'
    ]
    
    return any(indicator in error_str for indicator in cuda_memory_indicators)

def clear_gpu_memory():
    """
    Clear GPU memory cache if possible.
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
