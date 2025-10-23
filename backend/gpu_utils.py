"""
GPU Memory Management Utilities for Docling
Handles dynamic GPU/CPU switching based on memory availability
"""
import os
import re
import logging
import subprocess
from typing import Dict

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


def get_gpu_memory_info() -> Dict[str, int]:
    """Get GPU memory information in MB"""
    try:
        # Try nvidia-smi first
        logger.info("Attempting to get GPU info via nvidia-smi...")
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines:
                # Take the first GPU
                memory_info = lines[0].split(', ')
                if len(memory_info) >= 3:
                    total_mb = int(memory_info[0])
                    used_mb = int(memory_info[1])
                    free_mb = int(memory_info[2])
                    logger.info(f"GPU detected via nvidia-smi: Total={total_mb}MB, Used={used_mb}MB, Free={free_mb}MB")
                    return {
                        'total': total_mb,
                        'used': used_mb,
                        'free': free_mb,
                        'available': free_mb
                    }
        logger.warning(f"nvidia-smi failed with return code {result.returncode}, stderr: {result.stderr}")
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        logger.warning(f"nvidia-smi command failed: {e}")
    
    try:
        # Try PyTorch GPU detection
        logger.info("Attempting to get GPU info via PyTorch...")
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"PyTorch detected {gpu_count} GPU(s)")
            if gpu_count > 0:
                # Get info for the first GPU
                total_memory = torch.cuda.get_device_properties(0).total_memory
                total_mb = int(total_memory / (1024 * 1024))
                
                # Get current memory usage
                torch.cuda.empty_cache()  # Clear cache to get accurate reading
                allocated = torch.cuda.memory_allocated(0)
                cached = torch.cuda.memory_reserved(0)
                used_mb = int((allocated + cached) / (1024 * 1024))
                free_mb = total_mb - used_mb
                
                logger.info(f"GPU detected via PyTorch: Total={total_mb}MB, Used={used_mb}MB, Free={free_mb}MB")
                return {
                    'total': total_mb,
                    'used': used_mb,
                    'free': free_mb,
                    'available': free_mb
                }
        else:
            logger.info("PyTorch says CUDA is not available")
    except Exception as e:
        logger.warning(f"PyTorch GPU detection failed: {e}")
    
    try:
        # Fallback: Try to get info from /proc/driver/nvidia/gpus/
        logger.info("Attempting to get GPU info via /proc/driver/nvidia/gpus/...")
        gpu_dirs = [d for d in os.listdir('/proc/driver/nvidia/gpus/') if os.path.isdir(f'/proc/driver/nvidia/gpus/{d}')]
        if gpu_dirs:
            # Read memory info from the first GPU
            gpu_dir = gpu_dirs[0]
            with open(f'/proc/driver/nvidia/gpus/{gpu_dir}/information', 'r') as f:
                content = f.read()
                # Extract memory info
                memory_match = re.search(r'Video Memory:\s+(\d+)\s+MB', content)
                if memory_match:
                    total_mb = int(memory_match.group(1))
                    # Estimate available as 80% of total (conservative)
                    available_mb = int(total_mb * 0.8)
                    logger.info(f"GPU detected via /proc: Total={total_mb}MB, Estimated available={available_mb}MB")
                    return {
                        'total': total_mb,
                        'used': total_mb - available_mb,
                        'free': available_mb,
                        'available': available_mb
                    }
    except (FileNotFoundError, PermissionError, ValueError) as e:
        logger.warning(f"/proc/driver/nvidia detection failed: {e}")
    
    # If no GPU info available, return default values
    logger.warning("Could not determine GPU memory, using default estimates. Running in CPU-only mode.")
    return {
        'total': 8192,  # 8GB default
        'used': 2048,   # 2GB used
        'free': 6144,   # 6GB free
        'available': 6144
    }


def estimate_model_memory_requirement(model_name: str, model_size: str = None) -> int:
    """Estimate memory requirement for a model in MB"""
    name_lower = model_name.lower()
    
    # If size is provided, try to parse it
    if model_size and isinstance(model_size, str):
        size_lower = model_size.lower()
        # Extract numeric value from size string
        size_match = re.search(r'(\d+\.?\d*)', size_lower)
        if size_match:
            size_value = float(size_match.group(1))
            
            # Convert based on unit
            if 'gb' in size_lower:
                return int(size_value * 1024)  # Convert GB to MB
            elif 'mb' in size_lower:
                return int(size_value)
            elif 'b' in size_lower and 'gb' not in size_lower and 'mb' not in size_lower:
                # Assume it's parameters (e.g., "7b", "13b")
                # Rule of thumb: 1B parameters ≈ 2GB in FP16, ≈ 1GB in Q4
                return int(size_value * 1500)  # Conservative estimate for Q4 quantization
    
    # Fallback: estimate based on model name patterns
    if any(size in name_lower for size in ['0.5b', '500m']):
        return 1024   # ~1GB
    elif any(size in name_lower for size in ['1b', '1.5b']):
        return 2048   # ~2GB
    elif any(size in name_lower for size in ['3b', '2.8b']):
        return 4096   # ~4GB
    elif any(size in name_lower for size in ['7b', '6.7b', '8b']):
        return 8192   # ~8GB
    elif any(size in name_lower for size in ['13b', '14b', '15b']):
        return 16384  # ~16GB
    elif any(size in name_lower for size in ['30b', '32b', '34b']):
        return 32768  # ~32GB
    elif any(size in name_lower for size in ['70b', '72b']):
        return 65536  # ~64GB
    elif any(size in name_lower for size in ['175b', '180b']):
        return 131072 # ~128GB
    
    # Embedding models are typically smaller
    if any(keyword in name_lower for keyword in ['embed', 'bge', 'minilm', 'e5', 'sentence']):
        if 'large' in name_lower:
            return 1024   # ~1GB for large embedding models
        else:
            return 512    # ~512MB for smaller embedding models
    
    # Default estimate for unknown models
    return 4096  # ~4GB default


def check_model_compatibility_detailed(model_name: str, model_size: str = None) -> tuple:
    """Check if a model is compatible with current GPU memory"""
    gpu_info = get_gpu_memory_info()
    required_memory = estimate_model_memory_requirement(model_name, model_size)
    
    # Leave some buffer for system and other processes (20% of total or min 1GB)
    buffer_memory = max(1024, int(gpu_info['total'] * 0.2))
    usable_memory = gpu_info['total'] - buffer_memory  # Use total memory, not available
    
    is_compatible = required_memory <= usable_memory
    
    if is_compatible:
        message = f"✅ Model {model_name} is compatible (requires ~{required_memory}MB, {usable_memory}MB usable from {gpu_info['total']}MB total)"
    else:
        shortage = required_memory - usable_memory
        message = f"❌ Model {model_name} requires ~{required_memory}MB but only {usable_memory}MB usable from {gpu_info['total']}MB total (shortage: {shortage}MB)"
    
    details = {
        'required_memory_mb': required_memory,
        'usable_memory_mb': usable_memory,  # Usable memory (total - buffer)
        'gpu_total_mb': gpu_info['total'],
        'gpu_used_mb': gpu_info['used'],
        'gpu_free_mb': gpu_info['free'],
        'buffer_memory_mb': buffer_memory,
        'compatible': is_compatible,
        'shortage_mb': max(0, required_memory - usable_memory)
    }
    
    return is_compatible, message, details


def format_model_size(size):
    """Format model size from bytes to human readable format"""
    if isinstance(size, str):
        # If it's already a string, try to parse it or return as-is
        if size.lower() in ['unknown', 'n/a', '', 'none']:
            return 'Unknown'
        # If it's already formatted (contains B, KB, MB, GB), return as-is
        if any(unit in size.upper() for unit in ['B', 'KB', 'MB', 'GB', 'TB']):
            return size
        # Try to convert string to int
        try:
            size = int(size)
        except (ValueError, TypeError):
            return 'Unknown'
    
    if not isinstance(size, (int, float)) or size <= 0:
        return 'Unknown'
    
    # Convert bytes to human readable format
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            if unit == 'B':
                return f"{int(size)} {unit}"
            else:
                return f"{size:.1f} {unit}"
        size /= 1024.0
    
    return f"{size:.1f} PB"
