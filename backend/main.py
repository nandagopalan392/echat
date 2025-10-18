import os
# Disable DeepSpeed and problematic integrations before any transformers imports
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"
os.environ["WANDB_DISABLED"] = "true"
os.environ["DEEPSPEED_DISABLE"] = "true"
os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
os.environ["ACCELERATE_USE_FSDP"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import sys
import json
import uuid
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Request, status, Form, BackgroundTasks, Header, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import jwt
import datetime
# Import centralized configuration
from app.config import settings

# Authentication imports
from app.dependencies import get_current_user, get_current_admin_user
from app.api.v1.endpoints.auth import router as auth_router
from app.api.v1.endpoints.users import router as users_router
from app.api.v1.endpoints.admin import router as admin_router
from app.api.v1.endpoints.documents import router as documents_router
from chat_db import ChatDB
from rag import ChatPDF, get_chatpdf_instance
from rlhf import RLHF
import logging
from chunking_config import ChunkingMethod, ChunkingConfig, get_chunking_config_manager, FileFormatSupport
from app.core.rag.chunking.enhanced_document_processor import get_document_processor
import pandas as pd
from docx import Document
import sqlite3
from sse_starlette.sse import EventSourceResponse
import asyncio
import time
from contextlib import contextmanager
import shutil
from pathlib import Path
import random
import requests
import subprocess
import re

# Global download status tracking
download_status_cache = {}

# WebSocket connection manager for download progress
class DownloadProgressManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_progress(self, model_name: str, message: dict):
        disconnected_clients = []
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json({
                    "type": "download_progress",
                    "model_name": model_name,
                    **message
                })
            except:
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)

download_progress_manager = DownloadProgressManager()

async def download_huggingface_model_background(model_name: str):
    """Download HuggingFace model in background"""
    try:
        logger.info(f"🔄 Starting background download of HuggingFace reranker model: {model_name}")
        
        # Update status and notify WebSocket clients
        status_update = {"downloading": True, "completed": False, "message": "Starting download...", "status": "downloading"}
        download_status_cache[model_name] = status_update
        await download_progress_manager.send_progress(model_name, status_update)
        
        # Update progress
        status_update = {"downloading": True, "completed": False, "message": "Downloading model files...", "status": "downloading"}
        download_status_cache[model_name] = status_update
        await download_progress_manager.send_progress(model_name, status_update)
        
        from sentence_transformers import CrossEncoder
        
        # This will download the model to cache if not already present
        # Run in thread pool to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(None, lambda: CrossEncoder(model_name))
        
        # Update status on success
        status_update = {"downloading": False, "completed": True, "message": f"Successfully downloaded {model_name}", "status": "completed"}
        download_status_cache[model_name] = status_update
        await download_progress_manager.send_progress(model_name, status_update)
        
        logger.info(f"✅ Successfully downloaded HuggingFace reranker model: {model_name}")
        
    except Exception as e:
        logger.error(f"❌ Failed to download HuggingFace model {model_name}: {e}")
        status_update = {"downloading": False, "completed": False, "message": f"Download failed: {str(e)}", "status": "failed"}
        download_status_cache[model_name] = status_update
        await download_progress_manager.send_progress(model_name, status_update)

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

app = FastAPI(
    title="Chat API",
    description="API for chat application with PDF processing capabilities",
    version="1.0.0"
)

# Dataset generation progress tracking
dataset_generation_progress = {}

chat_db = ChatDB()
rlhf_db = RLHF()

# Add lazy loading for RAG instance
# RAG instance is now managed by the rag.py module singleton

def get_rag():
    """Get RAG instance using the singleton from rag.py"""
    return get_chatpdf_instance()

# Setup logging - Enable DEBUG for key components to track performance
logging.basicConfig(
    level=logging.DEBUG,  # Enable DEBUG to see timing information
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set specific loggers to WARNING to suppress their output, except key components
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('rag').setLevel(logging.WARNING)
logging.getLogger('chat_db').setLevel(logging.WARNING)
logging.getLogger('ollama_scraper').setLevel(logging.WARNING)
logging.getLogger('document_storage').setLevel(logging.WARNING)
# Keep DEBUG for document processor to see timing
logging.getLogger('enhanced_document_processor').setLevel(logging.DEBUG)
logging.getLogger('table_extraction').setLevel(logging.DEBUG)
logging.getLogger('chunking_config').setLevel(logging.WARNING)
logging.getLogger('reranker').setLevel(logging.WARNING)
logging.getLogger('fastapi').setLevel(logging.WARNING)
logging.getLogger('uvicorn').setLevel(logging.WARNING)
logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
logging.getLogger('uvicorn.error').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)
# Suppress LangChain chain tracing logs
logging.getLogger('langchain').setLevel(logging.WARNING)
logging.getLogger('langchain_core').setLevel(logging.WARNING)
logging.getLogger('langchain_ollama').setLevel(logging.WARNING)
logging.getLogger('langchain_community').setLevel(logging.WARNING)
logging.getLogger('langchain_chroma').setLevel(logging.WARNING)

# Main logger for general app logs - enable INFO to see startup messages
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Only allow evaluation-related logs
evaluation_logger = logging.getLogger('evaluation')
evaluation_logger.setLevel(logging.INFO)

# Create evaluation logger for use in evaluation endpoints
def get_evaluation_logger():
    return evaluation_logger

# CORS configuration - must be before any routes
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    f"http://{os.getenv('HOST_IP', '0.0.0.0')}:3000",  # Add IP-based access
    "http://192.168.8.205:3000",  # Explicitly add this origin
    "*"  # Allow all origins in development
]

# Replace the existing CORS middleware with this more explicit configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=".*",  # Allow all origins with regex
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*", "Authorization", "Content-Type", "X-Requested-With"],
    expose_headers=["*"],
    max_age=86400,  # Cache preflight requests for 24 hours
)

# Add a custom middleware to handle CORS issues
@app.middleware("http")
async def cors_middleware(request: Request, call_next):
    # For OPTIONS requests, return an early response with CORS headers
    if request.method == "OPTIONS":
        logger.info(f"Handling OPTIONS request for {request.url.path}")
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "Authorization, Content-Type, Accept, X-Requested-With",
            "Access-Control-Max-Age": "86400",
            "Access-Control-Allow-Credentials": "true",
        }
        return JSONResponse(
            content={},
            status_code=200,
            headers=headers
        )
    
    # For non-OPTIONS requests, process normally
    response = await call_next(request)
    
    # Make sure CORS headers are present in the response
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    
    return response

# Add request logging middleware - DISABLED for cleaner logs
# @app.middleware("http")
# async def log_requests(request: Request, call_next):
#     logger.debug(f"Incoming request: {request.method} {request.url}")
#     try:
#         body = await request.body()
#         logger.debug(f"Request body: {body.decode()}")
#     except:
#         pass
#     response = await call_next(request)
#     return response

# Startup event
@app.on_event("startup")
async def startup_event():
    import time
    total_startup_time = time.time()
    logger.info("🚀 DEBUG: Starting up the application...")
    
    try:
        # Initialize database
        db_start = time.time()
        chat_db.init_db()
        db_time = time.time() - db_start
        logger.info(f"🚀 DEBUG: Database initialized successfully in {db_time:.2f} seconds")
        
        # Check vector store status
        vs_start = time.time()
        chroma_path = os.getenv('CHROMA_DB_PATH', '/app/data/chroma_db')
        if os.path.exists(chroma_path):
            logger.info("🚀 DEBUG: Found existing vector store")
            # Force reload of vector store
            rag = get_rag()
            rag.ensure_models_loaded()
            if rag.vector_store:
                vs_time = time.time() - vs_start
                logger.info(f"🚀 DEBUG: Vector store loaded successfully in {vs_time:.2f} seconds")
            else:
                logger.warning("Vector store exists but couldn't be loaded")
        else:
            logger.info("No existing vector store found")
        
        # Initialize Docling processors to trigger model downloads at startup
        logger.info("🚀 DEBUG: Initializing Docling processors to preload models...")
        docling_start = time.time()
        
        try:
            from app.core.rag.chunking.enhanced_document_processor import get_document_processor
            doc_processor = get_document_processor()
            docling_init_time = time.time() - docling_start
            logger.info(f"🚀 DEBUG: Docling document processor initialized in {docling_init_time:.2f} seconds")
            
            # Also initialize table extractor
            table_start = time.time()
            from table_extraction import get_table_extractor
            table_extractor = get_table_extractor()
            table_init_time = time.time() - table_start
            logger.info(f"🚀 DEBUG: Docling table extractor initialized in {table_init_time:.2f} seconds")
            
        except Exception as e:
            logger.warning(f"🚀 DEBUG: Error initializing Docling processors: {e}")
        
        total_time = time.time() - total_startup_time
        logger.info(f"🚀 DEBUG: Application startup completed in {total_time:.2f} seconds")
            
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down the application...")
    try:
        # Cleanup code here
        rag = get_rag()
        if rag.vector_store:
            rag.clear()
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")




class Message(BaseModel):
    content: str
    session_id: Optional[int] = None

  
class RLHFFeedback(BaseModel):
    session_id: int
    chosen_index: int  # 0 for first response, 1 for second response

# Include TruLens evaluation routes (after defining get_current_user to avoid circular imports)
from routes.evaluation import router as evaluation_router
app.include_router(evaluation_router, prefix="/api/evaluation", tags=["evaluation"])

# Include API routers
app.include_router(auth_router, prefix="/api/auth", tags=["authentication"])
app.include_router(users_router, prefix="/api/users", tags=["users"])
app.include_router(admin_router, prefix="/api/admin", tags=["admin"])
app.include_router(documents_router, prefix="/api", tags=["documents"])

# Chat endpoints
@contextmanager
def timeout(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds} seconds")

    # Set the timeout handler and start the timer
    try:
        signal_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        yield
    finally:
        # Restore the original handler and cancel the timer
        signal.alarm(0)
        signal.signal(signal.SIGALRM, signal_handler)

# Fix the send_message function to return properly formatted JSON instead of SSE events
@app.post("/api/chat/send")
async def send_message(message: Message, token: str = Depends(oauth2_scheme)):
    try:
        user = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = user["sub"]
        logger.info(f"🔧 CHAT DEBUG: Decoded user from JWT - username='{username}'")
        
        if not message.session_id:
            message.session_id = chat_db.create_session(username, message.content)
        
        chat_db.save_message(message.session_id, message.content, True)

        # Store message data for RLHF
        user_prompt = message.content
        session_id = message.session_id

        # First get the responses from RAG
        try:
            logger.info(f"Getting responses for prompt: {user_prompt[:50]}...")
            
            # Get the first response with conversational style (friendly, detailed explanations)
            response_a_chunks = []
            rag = get_rag()
            logger.info(f"🔧 CHAT DEBUG: About to call stream_response with user_id='{username}'")
            async for chunk in rag.stream_response(user_prompt, style="conversational", user_id=username):
                response_a_chunks.append(chunk)
            
            # Parse the final response from chunks
            response_a_content = ""
            response_a_thinking = ""
            for chunk in response_a_chunks:
                # Handle different chunk types
                if isinstance(chunk, str):
                    try:
                        chunk_data = json.loads(chunk)
                        if isinstance(chunk_data, dict):
                            if chunk_data.get("thinking"):
                                response_a_thinking = chunk_data["thinking"]
                            if chunk_data.get("content"):
                                response_a_content = chunk_data["content"]
                        else:
                            # If parsed JSON is not a dict, treat as plain text
                            response_a_content += str(chunk)
                    except (json.JSONDecodeError, TypeError):
                        # If it's not JSON, treat as plain text
                        response_a_content += str(chunk)
                else:
                    # Handle non-string chunks
                    response_a_content += str(chunk)
            
            # Fallback if no content was extracted
            if not response_a_content:
                response_a_content = "".join(response_a_chunks)
            
            logger.info(f"Response A (conversational) - Thinking: {len(response_a_thinking)} chars, Content: {len(response_a_content)} chars")
            
            # Add a small delay between requests
            await asyncio.sleep(2)
            
            # Get second response with detailed/analytical style
            response_b_chunks = []
            async for chunk in rag.stream_response(user_prompt, style="detailed", user_id=username):
                response_b_chunks.append(chunk)
            
            # Parse the final response from chunks
            response_b_content = ""
            response_b_thinking = ""
            for chunk in response_b_chunks:
                # Handle different chunk types
                if isinstance(chunk, str):
                    try:
                        chunk_data = json.loads(chunk)
                        if isinstance(chunk_data, dict):
                            if chunk_data.get("thinking"):
                                response_b_thinking = chunk_data["thinking"]
                            if chunk_data.get("content"):
                                response_b_content = chunk_data["content"]
                        else:
                            # If parsed JSON is not a dict, treat as plain text
                            response_b_content += str(chunk)
                    except (json.JSONDecodeError, TypeError):
                        # If it's not JSON, treat as plain text
                        response_b_content += str(chunk)
                else:
                    # Handle non-string chunks
                    response_b_content += str(chunk)
            
            # Fallback if no content was extracted
            if not response_b_content:
                response_b_content = "".join(response_b_chunks)
            
            logger.info(f"Response B (detailed) - Thinking: {len(response_b_thinking)} chars, Content: {len(response_b_content)} chars")
            
            # Prepare the response options for RLHF with structured data
            response_options = [
                {
                    "thinking": response_a_thinking,
                    "content": response_a_content,
                    "style": "conversational"
                },
                {
                    "thinking": response_b_thinking,
                    "content": response_b_content,
                    "style": "detailed"
                }
            ]
            
            # Save the response options to the RLHF database for later retrieval
            try:
                rlhf_db.save_response_options(
                    session_id=session_id,
                    question=user_prompt,
                    response_option_0=response_a_content,
                    response_option_1=response_b_content,
                    username=username
                )
                logger.info(f"Saved RLHF response options for session {session_id}")
            except Exception as e:
                logger.error(f"Failed to save RLHF response options: {str(e)}")
            
            # Return a proper JSON response with structured data
            return JSONResponse(content={
                "content": "I've generated two different responses for you to choose from: one conversational and friendly, the other detailed and analytical. Please select your preferred approach:",
                "full_response": "I've generated two different responses for you to choose from: one conversational and friendly, the other detailed and analytical. Please select your preferred approach:",
                "is_final": True,
                "session_id": session_id,
                "response_options": response_options,
                "rlhf_enabled": True,  # Signal to frontend this is for RLHF
                "message": "Choose between conversational (friendly explanations) and detailed (comprehensive analysis) responses:",
                "thinking_included": bool(response_a_thinking or response_b_thinking)
            })
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )
        
    except Exception as e:
        logger.error(f"Error in send_message: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

def generate_mock_response(user_prompt, style_idx):
    """
    Generate mock responses with different styles for RLHF
    
    Args:
        user_prompt: The user's input message
        style_idx: Index of the style to use (0 = formal, 1 = conversational)
        
    Returns:
        A mock response string
    """
    # List of response templates for different styles
    formal_templates = [
        "Based on your query regarding {topic}, I would recommend considering the following points: {points}",
        "In response to your question about {topic}, here is a detailed analysis: {points}",
        "Regarding {topic}, the following information may be helpful: {points}",
        "After analyzing your question about {topic}, I can provide these insights: {points}"
    ]
    
    conversational_templates = [
        "I've been thinking about your question on {topic}. Here's what I found: {points}",
        "Great question about {topic}! Here's what I think: {points}",
        "Let me share some thoughts on {topic}: {points}",
        "That's an interesting question about {topic}! Here's my take: {points}"
    ]
    
    # Extract a topic from the user prompt (simplified)
    topic_words = user_prompt.split()
    if len(topic_words) > 3:
        topic = " ".join(topic_words[1:4])
    else:
        topic = user_prompt
        
    # Generate some mock points
    points_templates = [
        "First, {p1}. Second, {p2}. Finally, {p3}.",
        "{p1}. Additionally, {p2}. In conclusion, {p3}.",
        "The primary consideration is {p1}. We should also note that {p2}. Lastly, {p3}."
    ]
    
    # Simple point generators
    p1_options = [
        "this approach offers significant advantages",
        "we should consider the environmental impact",
        "the financial implications are substantial",
        "there are technical challenges to overcome"
    ]
    
    p2_options = [
        "research supports multiple perspectives on this issue",
        "stakeholders have expressed various concerns",
        "the timeline might need adjustment",
        "alternative solutions exist that might be more efficient"
    ]
    
    p3_options = [
        "careful planning will be essential for success",
        "further analysis may reveal additional insights",
        "a balanced approach seems most appropriate",
        "consultation with experts is recommended"
    ]
    
    # Randomly select components for the response
    points = random.choice(points_templates).format(
        p1=random.choice(p1_options),
        p2=random.choice(p2_options),
        p3=random.choice(p3_options)
    )
    
    # Generate the response based on the style
    if style_idx == 0:
        response = random.choice(formal_templates).format(topic=topic, points=points)
    else:
        response = random.choice(conversational_templates).format(topic=topic, points=points)
        
    return response

@app.get("/api/chat/sessions")
async def get_sessions(token: str = Depends(oauth2_scheme)):
    user = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    sessions = chat_db.get_user_sessions(user["sub"])
    return {"sessions": sessions}

@app.get("/api/chat/sessions/{session_id}/messages")
async def get_session_messages(session_id: int, token: str = Depends(oauth2_scheme)):
    try:
        user = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        logger.info(f"Fetching messages for session {session_id} by user {user['sub']}")
        
        messages = chat_db.get_session_messages(session_id)
        
        # Ensure messages are properly ordered and formatted
        formatted_messages = []
        for i, msg in enumerate(messages):
            formatted_msg = {
                "id": f"{session_id}-{i}",  # Create a unique ID
                "content": msg[0], 
                "isUser": bool(msg[1]),
                "timestamp": msg[2] if len(msg) > 2 else None
            }
            formatted_messages.append(formatted_msg)
            logger.debug(f"Message {i+1}: isUser={formatted_msg['isUser']}, content='{formatted_msg['content'][:50]}...'")
        
        logger.info(f"Returning {len(formatted_messages)} messages for session {session_id}")
        
        return {
            "messages": formatted_messages,
            "session_id": session_id
        }
    except Exception as e:
        logger.error(f"Error getting session messages: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/rlhf-feedback")
async def submit_rlhf_feedback(feedback: RLHFFeedback, token: str = Depends(oauth2_scheme)):
    """
    Endpoint to receive user feedback on which AI response was preferred
    """
    try:
        user = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = user["sub"]
        
        logger.info(f"Processing RLHF feedback for session {feedback.session_id}, chosen_index: {feedback.chosen_index}")
        
        # First, get the response options for this session to find the chosen response
        chosen_response_content = None
        try:
            with sqlite3.connect(rlhf_db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT response_option_0, response_option_1 
                    FROM rlhf_response_options 
                    WHERE session_id = ? AND username = ?
                    ORDER BY created_at DESC LIMIT 1
                    ''',
                    (feedback.session_id, username)
                )
                
                result = cursor.fetchone()
                if result:
                    response_options = [result[0], result[1]]
                    if 0 <= feedback.chosen_index < len(response_options):
                        chosen_response_content = response_options[feedback.chosen_index]
                        logger.info(f"Found chosen response: {chosen_response_content[:100]}...")
                    else:
                        logger.error(f"Invalid chosen_index {feedback.chosen_index}, defaulting to 0")
                        chosen_response_content = response_options[0]
                        feedback.chosen_index = 0
                else:
                    logger.error(f"No response options found for session {feedback.session_id}")
        
        except Exception as e:
            logger.error(f"Error retrieving response options: {str(e)}")
        
        # Save the user's preference
        success = rlhf_db.save_selected_response(
            session_id=feedback.session_id,
            chosen_index=feedback.chosen_index,
            user_id=username
        )
        
        if not success:
            logger.error(f"Failed to save RLHF preference for session {feedback.session_id}")
            raise HTTPException(status_code=500, detail="Failed to save RLHF feedback")
        
        # CRITICAL: Save the chosen response as a regular chat message for chat history
        if chosen_response_content:
            try:
                # Check if this response is already saved to avoid duplicates
                with sqlite3.connect(chat_db.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """SELECT COUNT(*) FROM messages 
                           WHERE session_id = ? AND content = ? AND is_user = 0""",
                        (feedback.session_id, chosen_response_content)
                    )
                    
                    if cursor.fetchone()[0] == 0:  # Response not found, save it
                        chat_db.save_message(feedback.session_id, chosen_response_content, False)  # False = AI message
                        logger.info(f"✅ Successfully saved chosen RLHF response as chat message for session {feedback.session_id}")
                    else:
                        logger.info(f"✅ Chosen RLHF response already exists in chat history for session {feedback.session_id}")
                        
            except Exception as e:
                logger.error(f"❌ Error saving chosen response as chat message: {str(e)}")
                # This is critical - if we can't save the response, the user won't see it in history
                raise HTTPException(status_code=500, detail="Failed to save response to chat history")
        else:
            logger.error(f"❌ No chosen response content to save for session {feedback.session_id}")
            raise HTTPException(status_code=500, detail="No response content found to save")
        
        logger.info(f"✅ RLHF feedback processing completed successfully for session {feedback.session_id}")
        
        return {"status": "success", "message": "Feedback received and processed"}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ Error processing RLHF feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class MessageUpdate(BaseModel):
    session_id: int
    content: str

@app.put("/api/chat/message/update")
async def update_message(message_update: MessageUpdate, token: str = Depends(oauth2_scheme)):
    """
    Update the latest AI message in a session (used after RLHF response selection)
    """
    try:
        user = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Get the latest AI message ID for this session
        message_id = chat_db.get_latest_ai_message_id(message_update.session_id)
        
        if not message_id:
            raise HTTPException(status_code=404, detail="No AI message found in session")
        
        # Update the message content
        success = chat_db.update_message(message_id, message_update.content)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update message")
        
        logger.info(f"Message {message_id} updated for session {message_update.session_id}")
        return {"status": "success", "message_id": message_id}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error updating message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add upload progress tracking
upload_progress = {}

async def track_progress(file_id: str, total_size: int):
    progress = 0
    while progress < 100:
        if file_id in upload_progress:
            progress = upload_progress[file_id]
            yield json.dumps({"progress": progress})
        await asyncio.sleep(0.5)
    yield json.dumps({"progress": 100})

@app.get("/api/upload-progress/{file_id}")
async def get_upload_progress(file_id: str):
    return EventSourceResponse(track_progress(file_id, 100))

@app.post("/api/chat/upload")
async def upload_file(file: UploadFile = File(...), token: str = Depends(oauth2_scheme)):
    try:
        file_id = f"upload_{datetime.datetime.now().timestamp()}"
        upload_progress[file_id] = 0
        
        contents = await file.read()
        total_size = len(contents)
        
        with open(f"temp_{file.filename}", "wb") as f:
            f.write(contents)
            upload_progress[file_id] = 50  # File written
        
        # Process file using new storage system
        success = get_rag().ingest_with_storage(f"temp_{file.filename}", file.filename)
        upload_progress[file_id] = 100  # Processing complete
        
        # Cleanup
        if os.path.exists(f"temp_{file.filename}"):
            os.remove(f"temp_{file.filename}")
        if file_id in upload_progress:
            del upload_progress[file_id]
            
        if success:
            return {"message": "File processed successfully", "file_id": file_id}
        else:
            raise HTTPException(status_code=400, detail="Failed to process file")
    except Exception as e:
        if file_id in upload_progress:
            del upload_progress[file_id]
        raise HTTPException(status_code=400, detail=str(e))

# Debug collection info endpoint kept for diagnostics
@app.get("/api/debug/collection-info")
async def get_collection_debug_info(current_user: dict = Depends(get_current_user)):
    """Debug endpoint to inspect collection structure"""
    try:
        from rag import get_chatpdf_instance
        chatpdf = get_chatpdf_instance()
        
        if not chatpdf or not chatpdf.vector_store:
            raise HTTPException(status_code=503, detail="Vector store not available")
        
        chroma_client = chatpdf.vector_store._client
        collection_name = chatpdf._get_collection_name()
        
        try:
            collection = chroma_client.get_collection(collection_name)
            
            # Get basic collection info
            count = collection.count()
            
            # Get sample documents
            sample_results = collection.get(limit=10, include=["metadatas", "documents"])
            
            return {
                "collection_name": collection_name,
                "total_documents": count,
                "sample_metadata": sample_results.get('metadatas', []),
                "sample_document_previews": [doc[:100] + "..." if len(doc) > 100 else doc for doc in sample_results.get('documents', [])]
            }
            
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Collection error: {e}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to Chat API"}

@app.get("/api/models/status")
async def get_model_status(current_user: dict = Depends(get_current_user)):
    """Get status of models in Ollama"""
    try:
        import httpx
        ollama_host = os.getenv('OLLAMA_HOST', 'http://ollama:11434')
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{ollama_host}/api/tags")
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
                raise HTTPException(status_code=500, detail="Could not connect to Ollama")
                
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")

@app.get("/api/vector-store/stats")
async def get_vector_store_stats(current_user: dict = Depends(get_current_user)):
    """Get detailed statistics about the vector store and collections"""
    try:
        rag = get_rag()
        stats = rag.get_vector_store_stats()
        
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting vector store stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get vector store stats: {str(e)}")

@app.delete("/api/vector-store/clear")
async def clear_vector_store(admin: dict = Depends(get_current_admin_user)):
    """Clear the entire vector store - admin only"""
    try:
        rag = get_rag()
        success = rag.clear_vectorstore()
        
        if success:
            return {"success": True, "message": "Vector store cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear vector store")
        
    except Exception as e:
        logger.error(f"Error clearing vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear vector store: {str(e)}")


# Chunking configuration endpoints
@app.get("/api/chunking/methods")
async def get_chunking_methods(token: str = Depends(oauth2_scheme)):
    """Get available chunking methods and their supported file formats"""
    try:
        from chunking_config import ChunkingMethod, FileFormatSupport
        
        methods = {}
        for method in ChunkingMethod:
            methods[method.value] = {
                'name': method.value,
                'description': _get_method_description(method),
                'supported_formats': FileFormatSupport.get_supported_formats(method)
            }
        
        return {"methods": methods}
    except Exception as e:
        logger.error(f"Error getting chunking methods: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chunking/config/{method}")
async def get_chunking_config(
    method: str, 
    current_user: dict = Depends(get_current_user)
):
    """Get chunking configuration for a specific method"""
    try:
        from chunking_config import ChunkingMethod, get_chunking_config_manager
        
        try:
            chunking_method = ChunkingMethod(method)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid chunking method: {method}")
        
        config_manager = get_chunking_config_manager()
        
        # Get user-specific config if user is available
        user_id = current_user.get('sub', 'default') if current_user else None
        config = config_manager.get_config(chunking_method, user_id)
        
        return {"config": config.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chunking config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chunking/config/{method}")
async def update_chunking_config(
    method: str,
    config_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Update chunking configuration for a specific method"""
    try:
        from chunking_config import ChunkingMethod, ChunkingConfig, get_chunking_config_manager
        
        try:
            chunking_method = ChunkingMethod(method)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid chunking method: {method}")
        
        # Create config from provided data
        config = ChunkingConfig.from_dict(config_data)
        
        # Validate configuration
        config_manager = get_chunking_config_manager()
        warnings = config_manager.validate_config(config)
        
        # Save configuration (user-specific based on token)
        user_id = current_user.get('sub', 'default')
        config_manager.save_config(chunking_method, config, user_id)
        
        return {
            "message": "Configuration updated successfully",
            "warnings": warnings,
            "config": config.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating chunking config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chunking/optimal/{file_extension}")
async def get_optimal_chunking_method(file_extension: str, token: str = Depends(oauth2_scheme)):
    """Get optimal chunking method for a file extension"""
    try:
        from chunking_config import FileFormatSupport
        
        # Remove dot if present
        ext = file_extension.lstrip('.')
        
        optimal_method = FileFormatSupport.get_optimal_method(ext)
        available_methods = FileFormatSupport.get_available_methods(ext)
        
        return {
            "file_extension": ext,
            "optimal_method": optimal_method.value,
            "available_methods": [method.value for method in available_methods]
        }
    except Exception as e:
        logger.error(f"Error getting optimal chunking method: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Retrieval Configuration API Endpoints

@app.get("/api/retrieval/config")
async def get_retrieval_config(token: str = Depends(oauth2_scheme)):
    """Get current retrieval configuration"""
    try:
        current_user = await get_current_user(token)
        user_id = current_user.get('sub') if current_user else None
        
        from retrieval_config import get_retrieval_config_manager
        config_manager = get_retrieval_config_manager()
        config = config_manager.get_config(user_id)
        
        return {
            "config": config.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting retrieval config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def check_and_download_reranker_model(model_name: str, provider: str = "ollama") -> Dict[str, Any]:
    """Check if a reranker model is available locally, download if not"""
    if not model_name or model_name.lower() == "none":
        return {"success": True, "downloaded": False, "message": "No reranker model specified"}
    
    # For HuggingFace models, handle download with status tracking
    if provider == "huggingface":
        # Check if already downloading
        if model_name in download_status_cache:
            status = download_status_cache[model_name]
            return {"success": True, "downloaded": status.get("completed", False), "message": status.get("message", "Download in progress"), "downloading": status.get("downloading", False)}
        
        # Start download process
        try:
            logger.info(f"🔄 Downloading HuggingFace reranker model: {model_name}")
            download_status_cache[model_name] = {"downloading": True, "completed": False, "message": "Starting download..."}
            
            from sentence_transformers import CrossEncoder
            
            # This will download the model to cache if not already present
            model = CrossEncoder(model_name)
            
            # Update status on success
            download_status_cache[model_name] = {"downloading": False, "completed": True, "message": f"Successfully downloaded {model_name}"}
            
            return {"success": True, "downloaded": True, "message": f"HuggingFace model {model_name} downloaded successfully"}
        except Exception as e:
            logger.error(f"Failed to download HuggingFace model {model_name}: {e}")
            download_status_cache[model_name] = {"downloading": False, "completed": False, "message": f"Download failed: {str(e)}"}
            return {"success": False, "downloaded": False, "message": f"Failed to download HuggingFace model: {str(e)}"}
    
    try:
        import httpx
        ollama_host = os.getenv('OLLAMA_HOST', 'http://ollama:11434')
        
        # Check if model is available locally
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
                
                # Try download with enhanced error handling and progress tracking
                download_response = await client.post(
                    f"{ollama_host}/api/pull",
                    json={"name": model_name, "stream": True},
                    timeout=600.0  # 10 minute timeout
                )
                
                if download_response.status_code != 200:
                    error_text = await download_response.text()
                    error_msg = f"Failed to download reranker model {model_name}: HTTP {download_response.status_code} - {error_text}"
                    logger.error(error_msg)
                    return {"success": False, "downloaded": False, "message": error_msg}
                
                # Process download stream with better status tracking
                download_success = False
                last_status = ""
                try:
                    async for line in download_response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                status = data.get("status", "")
                                
                                # Log status changes
                                if status != last_status and status:
                                    logger.info(f"🎯 Reranker model {model_name}: {status}")
                                    last_status = status
                                
                                # Check for completion indicators
                                if any(indicator in status.lower() for indicator in [
                                    "success", "pull complete", "already exists"
                                ]):
                                    download_success = True
                                    logger.info(f"✅ Reranker model {model_name}: Download completed - {status}")
                                    break
                                    
                                # Check for error indicators
                                if any(error_word in status.lower() for error_word in [
                                    "error", "failed", "not found"
                                ]):
                                    logger.error(f"❌ Reranker model {model_name}: Download failed - {status}")
                                    return {"success": False, "downloaded": False, "message": f"Download failed: {status}"}
                                    
                            except json.JSONDecodeError:
                                # Skip malformed JSON lines
                                continue
                                
                except Exception as stream_error:
                    logger.error(f"Error processing download stream for {model_name}: {stream_error}")
                    # Don't fail immediately, try to verify if model was downloaded
                
                # Verify download by checking if model is now available
                verification_response = await client.get(f"{ollama_host}/api/tags")
                if verification_response.status_code == 200:
                    updated_models_data = verification_response.json()
                    updated_available_models = {model.get('name', '') for model in updated_models_data.get('models', [])}
                    
                    # Check if any variant of the model is now available
                    verification_found = any(variant in updated_available_models for variant in model_variants)
                    
                    if verification_found:
                        download_success = True
                        logger.info(f"✅ Reranker model {model_name}: Verified successful download")
                    else:
                        logger.warning(f"⚠️ Reranker model {model_name}: Download status unclear, model not found in verification")
                
                if download_success:
                    return {"success": True, "downloaded": True, "message": f"Successfully downloaded reranker model {model_name}"}
                else:
                    error_msg = f"Download verification failed for reranker model {model_name}. Model may not be available in Ollama registry."
                    logger.error(error_msg)
                    return {"success": False, "downloaded": False, "message": error_msg}
            else:
                error_msg = f"Failed to check available models: HTTP {response.status_code}"
                logger.error(error_msg)
                return {"success": False, "downloaded": False, "message": error_msg}
                
    except httpx.TimeoutException:
        error_msg = f"Timeout downloading reranker model {model_name}. Large models may take longer to download."
        logger.error(error_msg)
        return {"success": False, "downloaded": False, "message": error_msg}
    except Exception as e:
        error_msg = f"Error with reranker model {model_name}: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "downloaded": False, "message": error_msg}

@app.put("/api/retrieval/config")
async def update_retrieval_config(
    config_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    token: str = Depends(oauth2_scheme)
):
    """Update retrieval configuration with auto-download for reranker models"""
    try:
        current_user = await get_current_user(token)
        user_id = current_user.get('sub') if current_user else None
        
        from retrieval_config import get_retrieval_config_manager, RetrievalConfig
        config_manager = get_retrieval_config_manager()
        
        # Create config from provided data
        config = RetrievalConfig.from_dict(config_data)
        
        # Check and handle reranker model download
        download_result = {"success": True, "downloaded": False, "message": "No reranker model specified"}
        if config.reranker_enabled and config.reranker_model and config.reranker_model.lower() != "none":
            logger.info(f"🎯 Checking reranker model availability: {config.reranker_model}")
            
            if config.reranker_provider == "huggingface":
                # For HuggingFace models, start download in background
                if config.reranker_model not in download_status_cache:
                    download_status_cache[config.reranker_model] = {"downloading": True, "completed": False, "message": "Download starting..."}
                    # Use asyncio.create_task instead of BackgroundTasks
                    asyncio.create_task(download_huggingface_model_background(config.reranker_model))
                download_result = {"success": True, "downloaded": False, "message": f"HuggingFace model {config.reranker_model} download started in background"}
            else:
                # For Ollama models, download synchronously
                download_result = await check_and_download_reranker_model(config.reranker_model, config.reranker_provider)
                
                if not download_result["success"]:
                    # If download failed, still save config but include warning
                    logger.warning(f"⚠️ Reranker model download failed but continuing with config save: {download_result['message']}")
        
        # Validate configuration
        warnings = config_manager.validate_config(config)
        
        # Add download-related warning if download failed
        if not download_result["success"]:
            warnings.append(f"Reranker model download failed: {download_result['message']}")
        
        # Save configuration
        success = config_manager.save_config(config, user_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save retrieval configuration")
        
        response_data = {
            "success": True,
            "config": config.to_dict(),
            "warnings": warnings,
            "reranker_download": download_result
        }
        
        # Add success message if model was downloaded
        if download_result["downloaded"]:
            response_data["message"] = f"Configuration saved and reranker model '{config.reranker_model}' downloaded successfully"
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating retrieval config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/retrieval/reranker-models")
async def get_available_reranker_models(provider: Optional[str] = None, token: str = Depends(oauth2_scheme)):
    """Get list of available reranker models from Ollama and HuggingFace"""
    try:
        # Get models from Ollama API (locally installed) and Ollama scraper (library)
        ollama_url = os.getenv('OLLAMA_HOST', 'http://ollama:11434')
        
        import httpx
        from ollama_scraper import get_available_ollama_models
        
        # Get locally installed models
        local_models = []
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ollama_url}/api/tags")
            
            if response.status_code == 200:
                ollama_data = response.json()
                local_models = ollama_data.get('models', [])
            else:
                logger.error(f"Failed to fetch models from Ollama API: {response.status_code}")
                # Continue without local models, will use library models only
        
        # Get models from Ollama library
        try:
            library_models = get_available_ollama_models(use_cache=True)
            # logger.info(f"Found {len(library_models)} models from Ollama library")
        except Exception as e:
            logger.warning(f"Could not fetch Ollama library models: {e}")
            library_models = []
        
        # Combine local and library models
        all_ollama_models = list(local_models)  # Start with local models
        
        # Add library models that aren't already installed locally
        local_model_names = {model.get('name', '') for model in local_models}
        for lib_model in library_models:
            if lib_model.get('name', '') not in local_model_names:
                # Convert library model format to match local model format
                all_ollama_models.append({
                    'name': lib_model.get('name', ''),
                    'category': lib_model.get('category', 'llm')
                })
        
        # Filter for reranker/embedding models that can be used for reranking
        reranker_models = []
        
        # Add "None" option first
        reranker_models.append({
            "name": "",
            "display_name": "None (Vector + Keyword)",
            "description": "Use weighted combination of vector similarity and keyword matching",
            "provider": "none"
        })
        
        # Check each model (both local and library)
        total_models_checked = 0
        reranker_models_found = 0
        
        for model in all_ollama_models:
            total_models_checked += 1
            model_name = model.get('name', '').lower()
            original_name = model.get('name', '')
            model_category = model.get('category', 'llm')
            
            # Prioritize dedicated reranker models ONLY - exclude general embedding models
            is_dedicated_reranker = (
                model_category == 'reranker' or
                any(keyword in model_name for keyword in [
                    'rerank', 'cross-encoder', 'qwen3-reranker', 'qwen-reranker', 
                    'bce-reranker', 'bge-reranker', 'reranker', 'ranking'
                ])
            )
            
            # Only include BGE models that are specifically rerankers (not general embedding models)
            is_bge_reranker = (
                'bge' in model_name and 
                ('reranker' in model_name or 'rerank' in model_name)
            )
            
            # Include ONLY dedicated reranker models, exclude general embedding models
            if is_dedicated_reranker or is_bge_reranker:
                reranker_models_found += 1
                
                # Create display name by cleaning up the model name
                display_name = original_name.replace(':', ' ').replace('-', ' ').replace('_', ' ')
                display_name = ' '.join(word.capitalize() for word in display_name.split())
                
                # Add description - all should be reranker-specific
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
                
                # Check if it's locally installed
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
        huggingface_reranker_models = [
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
        
        # Add HuggingFace models to the list
        reranker_models.extend(huggingface_reranker_models)
        
        # logger.info(f"Reranker model filtering: Checked {total_models_checked} models, found {reranker_models_found} dedicated reranker models")
        
        # If no ollama reranker models found, add some fallback ollama models
        if reranker_models_found == 0:
            logger.warning("No reranker models found in Ollama, adding fallback Ollama models")
            fallback_ollama_models = [
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
                },
                {
                    "name": "BAAI/bge-reranker-large",
                    "display_name": "BGE Reranker Large (Original)",
                    "description": "Original BGE reranking model (may need to be pulled)",
                    "is_local": False,
                    "provider": "ollama"
                }
            ]
            reranker_models.extend(fallback_ollama_models)
        
        # Apply server-side filtering if provider is specified
        if provider:
            if provider == 'ollama':
                # Filter for Ollama models and the "None" option
                reranker_models = [model for model in reranker_models if model.get('provider') in ['ollama', 'none']]
            elif provider == 'huggingface':
                # Filter for HuggingFace models and the "None" option
                reranker_models = [model for model in reranker_models if model.get('provider') in ['huggingface', 'none']]
        
        return {
            "models": reranker_models
        }
        
    except Exception as e:
        logger.error(f"Error getting reranker models: {e}")
        # Return basic fallback on error
        fallback_models = [
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
        return {
            "models": fallback_models
        }

@app.get("/api/retrieval/reranker-download-status")
async def get_reranker_download_status(model_name: str, token: str = Depends(oauth2_scheme)):
    """Get the download status of a reranker model"""
    try:
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
    except Exception as e:
        logger.error(f"Error getting download status for {model_name}: {e}")
        return {
            "model_name": model_name,
            "downloading": False,
            "completed": False,
            "message": f"Error getting status: {str(e)}"
        }

@app.websocket("/api/ws/download-progress")
async def websocket_download_progress(websocket: WebSocket, token: str = None):
    """WebSocket endpoint for real-time download progress updates"""
    import uuid
    client_id = str(uuid.uuid4())
    
    try:
        # Add to manager with client ID (this will accept the connection)
        await download_progress_manager.connect(websocket, client_id)
        
        try:
            # Keep connection alive and listen for disconnection
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        download_progress_manager.disconnect(client_id)

def _get_method_description(method: ChunkingMethod) -> str:
    """Get description for chunking method"""
    descriptions = {
        ChunkingMethod.GENERAL: "General document chunking for PDF, DOCX, MD, TXT files",
        ChunkingMethod.QA: "For question-answer formatted documents",
        ChunkingMethod.RESUME: "Enterprise edition for resume documents", 
        ChunkingMethod.TABLE: "For spreadsheet/tabular data",
        ChunkingMethod.PRESENTATION: "For PPT/presentation files",
        ChunkingMethod.PICTURE: "Image/visual content processing",
        ChunkingMethod.EMAIL: "Email content chunking"
    }
    return descriptions.get(method, "Custom chunking method")

async def flexible_oauth2_scheme(
    request: Request,
    authorization: str = Header(None),
    token: str = Query(None)
):
    """OAuth2 scheme that accepts token from either Authorization header or query parameter"""
    auth_token = None
    
    # Try to get token from Authorization header first
    if authorization:
        try:
            scheme, _, param = authorization.partition(" ")
            if scheme.lower() == "bearer":
                auth_token = param
        except Exception:
            pass
    
    # If no header token, try query parameter
    if not auth_token and token:
        auth_token = token
        
    if not auth_token:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return auth_token

# ================================
# FINETUNING API ENDPOINTS
# ================================

from experiment_db import experiment_db, ExperimentStatus
from hf_finetuner import hf_finetuner

@app.get("/api/finetuning/models")
async def get_available_models(token: str = Depends(oauth2_scheme)):
    """Get list of available models for finetuning"""
    try:
        models = hf_finetuner.get_available_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/finetuning/experiments")
async def create_experiment(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    description: str = Form(""),
    model_name: str = Form(...),
    dataset_id: str = Form(...),
    learning_rate: float = Form(0.0001),
    num_epochs: int = Form(3),
    batch_size: int = Form(4),
    lora_r: int = Form(16),
    lora_alpha: int = Form(32),
    lora_dropout: float = Form(0.1),
    target_modules: str = Form("q_proj,v_proj"),
    max_seq_length: int = Form(512),
    warmup_ratio: float = Form(0.03),
    weight_decay: float = Form(0.01),
    gradient_accumulation_steps: int = Form(1),
    logging_steps: int = Form(10),
    save_steps: int = Form(500),
    eval_steps: int = Form(100),
    save_total_limit: int = Form(2),
    load_best_model_at_end: bool = Form(True),
    metric_for_best_model: str = Form("eval_loss"),
    greater_is_better: bool = Form(False),
    evaluation_strategy: str = Form("steps"),
    save_strategy: str = Form("steps"),
    current_user: dict = Depends(get_current_user)
):
    """Create a new finetuning experiment"""
    try:
        # Get user ID from current user
        user_id = current_user['sub']
        
        # Get dataset info from database
        dataset_info = experiment_db.get_dataset(dataset_id)
        if not dataset_info:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Create experiment configuration
        config = {
            'base_model': model_name,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': num_epochs,
            'use_lora': True,  # Always use LoRA for now
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'lora_dropout': lora_dropout,
            'target_modules': target_modules,
            'max_length': max_seq_length,
            'warmup_ratio': warmup_ratio,
            'weight_decay': weight_decay,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'logging_steps': logging_steps,
            'save_steps': save_steps,
            'eval_steps': eval_steps,
            'save_total_limit': save_total_limit,
            'load_best_model_at_end': load_best_model_at_end,
            'metric_for_best_model': metric_for_best_model,
            'greater_is_better': greater_is_better,
            'evaluation_strategy': evaluation_strategy,
            'save_strategy': save_strategy,
            'dataset_path': dataset_info['file_path'],
            'dataset_id': dataset_id
        }
        
        # Create experiment
        experiment_data = {
            'name': name,
            'description': description,
            'user_id': user_id,
            'base_model': model_name,
            'model_provider': 'huggingface',
            'config': config
        }
        
        experiment_id = experiment_db.create_experiment(experiment_data)
        
        logger.info(f"Created experiment {experiment_id} for user {user_id}")
        
        # Automatically start training after creation
        try:
            background_tasks.add_task(hf_finetuner.start_training, experiment_id, config)
            logger.info(f"Auto-started training for experiment {experiment_id}")
            training_status = "training_started"
        except Exception as e:
            logger.error(f"Failed to start training for experiment {experiment_id}: {e}")
            training_status = "draft"
        
        return {
            "experiment": {
                "id": experiment_id,
                "name": name,
                "description": description,
                "base_model": model_name,
                "dataset_id": dataset_id
            },
            "status": "created",
            "training_status": training_status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/finetuning/experiments/{experiment_id}/start")
async def start_experiment(
    experiment_id: str,
    background_tasks: BackgroundTasks,
    token: str = Depends(oauth2_scheme)
):
    """Start training for an experiment"""
    try:
        user_id = get_user_from_token(token)
        
        # Get experiment
        experiment = experiment_db.get_experiment(experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        # Check ownership
        if experiment['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Not authorized")
        
        # Check status
        if experiment['status'] not in [ExperimentStatus.DRAFT.value, ExperimentStatus.FAILED.value]:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot start experiment in {experiment['status']} status"
            )
        
        # Start training in background
        asyncio.create_task(hf_finetuner.start_training(experiment_id, experiment['config']))
        
        return {"status": "training_started", "experiment_id": experiment_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/finetuning/experiments")
async def get_user_experiments(
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Get experiments for the current user"""
    try:
        user_id = current_user['sub']
        experiments = experiment_db.get_user_experiments(user_id, limit)
        return {"experiments": experiments}
        
    except Exception as e:
        logger.error(f"Error getting experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/finetuning/experiments/{experiment_id}")
async def get_experiment_details(
    experiment_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed information about an experiment"""
    try:
        user_id = current_user['sub']
        
        experiment = experiment_db.get_experiment(experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        # Check ownership
        if experiment['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Not authorized")
        
        # Get training logs
        training_logs = experiment_db.get_training_logs(experiment_id)
        experiment['training_logs'] = training_logs
        
        return experiment
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/finetuning/experiments/{experiment_id}/logs")
async def get_experiment_logs(
    experiment_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get training logs for an experiment"""
    try:
        user_id = current_user['sub']
        
        experiment = experiment_db.get_experiment(experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        if experiment['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Not authorized")
        
        logs = experiment_db.get_training_logs(experiment_id)
        return {"logs": logs}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/finetuning/experiments/{experiment_id}")
async def delete_experiment(
    experiment_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete an experiment"""
    try:
        user_id = current_user['sub']
        
        success = experiment_db.delete_experiment(experiment_id, user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Experiment not found or not authorized")
        
        return {"status": "deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/finetuning/experiments/{experiment_id}/metrics")
async def get_experiment_metrics(
    experiment_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get live training metrics for an experiment"""
    try:
        from training_metrics import get_metrics_collector
        
        user_id = current_user['sub']
        
        # Check if experiment belongs to user
        experiment = experiment_db.get_experiment(experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        if experiment.get('user_id') != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get metrics collector if training is active
        metrics_collector = get_metrics_collector(experiment_id)
        if metrics_collector:
            return metrics_collector.get_metrics_summary()
        else:
            # Return historical data from database if available
            if experiment.get('metrics') and experiment['metrics'].get('training_history'):
                training_history = experiment['metrics']['training_history']
                final_progress = experiment['metrics'].get('final_progress', {})
                
                return {
                    "experiment_id": experiment_id,
                    "progress": final_progress,
                    "metrics": training_history,
                    "system": {"current": {}, "history": []},
                    "training_logs": experiment_db.get_training_logs(experiment_id)[-20:],
                    "training_completed": True
                }
            else:
                # Return empty structure if no training data available
                logs = experiment_db.get_training_logs(experiment_id)
                return {
                    "experiment_id": experiment_id,
                    "progress": {"current_epoch": 0, "total_epochs": 0, "current_step": 0, "total_steps": 0},
                    "metrics": {"train_losses": [], "eval_losses": [], "learning_rates": [], "accuracies": []},
                    "system": {"current": {}, "history": []},
                    "training_logs": logs[-20:] if logs else [],
                    "training_completed": False
                }
    except Exception as e:
        logger.error(f"Error getting experiment metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/finetuning/datasets")
async def get_user_datasets(current_user: dict = Depends(get_current_user)):
    """Get datasets for the current user"""
    try:
        user_id = current_user['sub']
        datasets = experiment_db.get_user_datasets(user_id)
        return {"datasets": datasets}
        
    except Exception as e:
        logger.error(f"Error getting datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/finetuning/validate-dataset")
async def validate_dataset_file(
    dataset: UploadFile = File(...),
    token: str = Depends(oauth2_scheme)
):
    """Validate a dataset file without creating an experiment"""
    try:
        if not dataset.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not dataset.filename.endswith(('.jsonl', '.csv')):
            raise HTTPException(
                status_code=400, 
                detail="Dataset must be in JSONL or CSV format"
            )
        
        # Save temporary file
        temp_id = str(uuid.uuid4())
        temp_path = f"/tmp/{temp_id}_{dataset.filename}"
        
        with open(temp_path, "wb") as buffer:
            content = await dataset.read()
            buffer.write(content)
        
        try:
            # Validate
            result = hf_finetuner.validate_dataset(temp_path)
            return result
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/api/ws/finetuning/{experiment_id}")
async def websocket_experiment_progress(websocket: WebSocket, experiment_id: str):
    """WebSocket endpoint for real-time experiment progress updates"""
    await websocket.accept()
    
    try:
        while True:
            # Check experiment status
            experiment = experiment_db.get_experiment(experiment_id)
            if experiment:
                # Get latest logs
                logs = experiment_db.get_training_logs(experiment_id)
                latest_logs = logs[-10:] if logs else []  # Last 10 entries
                
                # Get live training metrics if available
                from training_metrics import get_metrics_collector
                metrics_collector = get_metrics_collector(experiment_id)
                training_metrics = None
                if metrics_collector:
                    training_metrics = metrics_collector.get_metrics_summary()
                
                status = experiment.get('status')
                status_lower = (status or "").lower()
                message_data = {
                    "type": "experiment_update",
                    "experiment_id": experiment_id,
                    "status": status_lower,
                    "metrics": experiment.get('metrics', {}),
                    "latest_logs": latest_logs,
                    "error_message": experiment.get('error_message'),
                    "training_metrics": training_metrics  # Include live training metrics
                }
                
                await websocket.send_json(message_data)
                logger.info(f"WebSocket sent update for experiment {experiment_id}: status={status_lower}")
                
                # Close connection if experiment is completed or failed (check both original and lowercase)
                if status_lower in ["completed", "failed", "cancelled"] or status in [ExperimentStatus.COMPLETED.value, ExperimentStatus.FAILED.value, ExperimentStatus.CANCELLED.value]:
                    logger.info(f"Experiment {experiment_id} finished with status {status_lower}, closing WebSocket")
                    await websocket.close()
                    break
            else:
                # Experiment not found
                await websocket.send_json({
                    "type": "error",
                    "message": f"Experiment {experiment_id} not found"
                })
                break
            
            await asyncio.sleep(2)  # Update every 2 seconds for more responsive UI
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for experiment {experiment_id}")
        pass
    except Exception as e:
        logger.error(f"WebSocket error for experiment {experiment_id}: {e}")
        await websocket.close()
    except Exception as e:
        logger.error(f"WebSocket error for experiment {experiment_id}: {e}")
        await websocket.close()

# Additional Dataset Management Endpoints

@app.post("/api/finetuning/datasets/create")
async def create_finetuning_dataset(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Create a new finetuning dataset from documents using Q-C-A format"""
    try:
        data = await request.json()
        name = data.get('name')
        description = data.get('description', '')
        document_ids = data.get('document_ids', [])
        questions_per_doc = data.get('questions_per_doc', 5)
        
        if not name:
            raise HTTPException(status_code=400, detail="Dataset name is required")
        
        if not document_ids:
            raise HTTPException(status_code=400, detail="At least one document must be selected")
        
        # Get documents from document storage
        from document_storage import get_document_storage
        doc_storage = get_document_storage()
        
        documents = []
        for doc_id in document_ids:
            try:
                doc = doc_storage.get_document(doc_id)
                if doc:
                    documents.append(doc)
                else:
                    logger.warning(f"Document {doc_id} not found")
            except Exception as e:
                logger.error(f"Error loading document {doc_id}: {e}")
                continue
        
        if not documents:
            raise HTTPException(status_code=404, detail="No valid documents found")
        
        # Start background task for dataset generation
        from qca_dataset_generator import QCADatasetGenerator
        
        # Create unique dataset ID for tracking
        import time
        dataset_id = f"qca_{int(time.time())}"
        
        # Store generation progress
        global dataset_generation_progress
        if 'dataset_generation_progress' not in globals():
            dataset_generation_progress = {}
        
        dataset_generation_progress[dataset_id] = {
            "status": "starting",
            "progress": 0,
            "current_document": "",
            "completed_documents": 0,
            "total_documents": len(documents),
            "created_at": datetime.datetime.now().isoformat()
        }
        
        # Start background generation using Celery
        from qca_tasks import create_qca_dataset_background
        
        task = create_qca_dataset_background.delay(
            name=name,
            description=description,
            document_ids=document_ids,
            questions_per_doc=questions_per_doc,
            model_name="gemma2:2b",
            user_id=current_user['sub']
        )
        
        logger.info(f"Started Q-C-A dataset generation task: {task.id}")
        
        return {
            "success": True,
            "task_id": task.id,
            "message": f"Dataset creation started. Processing {len(documents)} documents with {questions_per_doc} questions per document.",
            "total_documents": len(documents)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating finetuning dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/api/ws/qca-dataset/{task_id}")
async def qca_dataset_websocket(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for Q-C-A dataset generation progress"""
    await websocket.accept()
    logger.info(f"Q-C-A dataset WebSocket connected for task {task_id}")
    
    try:
        # Check for progress updates from Celery backend
        import redis
        redis_client = redis.Redis(host=os.getenv('REDIS_HOST', 'redis'), port=6379, db=0)
        
        last_update = None
        while True:
            try:
                # Get progress from Redis
                progress_key = f"qca_progress_{task_id}"
                progress_data = redis_client.get(progress_key)
                
                if progress_data:
                    import json
                    progress_info = json.loads(progress_data)
                    
                    # Only send if there's a new update
                    if progress_info != last_update:
                        await websocket.send_json(progress_info)
                        last_update = progress_info
                        
                        # Break if task is complete or failed
                        if progress_info.get("status") in ["SUCCESS", "FAILURE"]:
                            break
                
                await asyncio.sleep(1)  # Check every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {e}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info(f"Q-C-A dataset WebSocket disconnected for task {task_id}")

@app.get("/api/finetuning/datasets/create/{dataset_id}/progress")
async def get_dataset_creation_progress(
    dataset_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get progress of dataset creation"""
    global dataset_generation_progress
    
    if dataset_id not in dataset_generation_progress:
        raise HTTPException(status_code=404, detail="Dataset creation not found")
    
    return dataset_generation_progress[dataset_id]

@app.get("/api/finetuning/datasets/{dataset_id}")
async def get_dataset_details(
    dataset_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed information about a finetuning dataset"""
    try:
        dataset = experiment_db.get_dataset(dataset_id)
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Get sample data (first 10 samples for preview)
        samples = experiment_db.get_dataset_samples(dataset_id, limit=10)
        
        return {
            "id": dataset['id'],
            "name": dataset['name'],
            "description": dataset['description'],
            "num_samples": dataset['num_samples'],
            "file_size": dataset.get('file_size', 0),
            "format": dataset.get('format', 'jsonl'),
            "created_at": dataset['created_at'],
            "samples": samples
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/finetuning/datasets/{dataset_id}/download")
async def download_dataset(
    dataset_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Download a finetuning dataset in JSONL format"""
    try:
        dataset = experiment_db.get_dataset(dataset_id)
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Get all samples
        samples = experiment_db.get_dataset_samples(dataset_id)
        
        # Create JSONL content
        jsonl_lines = []
        for sample in samples:
            jsonl_lines.append(json.dumps(sample, ensure_ascii=False))
        
        jsonl_content = '\n'.join(jsonl_lines)
        
        # Create filename
        safe_name = "".join(c for c in dataset['name'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_name}.jsonl"
        
        # Return as file download
        return Response(
            content=jsonl_content.encode('utf-8'),
            media_type='application/jsonl',
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/finetuning/datasets/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a finetuning dataset"""
    try:
        success = experiment_db.delete_dataset(dataset_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        return {"success": True, "message": "Dataset deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))
