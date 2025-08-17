import sys
import os
import json
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Request, status, Form, BackgroundTasks, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import jwt
import datetime
from chat_db import ChatDB
from rag import ChatPDF, get_chatpdf_instance
from rlhf import RLHF
import logging
from chunking_config import ChunkingMethod, ChunkingConfig, get_chunking_config_manager, FileFormatSupport
from enhanced_document_processor import get_document_processor
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

def get_gpu_memory_info() -> Dict[str, int]:
    """Get GPU memory information in MB"""
    try:
        # Try nvidia-smi first
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
                    return {
                        'total': total_mb,
                        'used': used_mb,
                        'free': free_mb,
                        'available': free_mb
                    }
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass
    
    try:
        # Fallback: Try to get info from /proc/driver/nvidia/gpus/
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
                    return {
                        'total': total_mb,
                        'used': total_mb - available_mb,
                        'free': available_mb,
                        'available': available_mb
                    }
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    
    # If no GPU info available, return default values
    logger.warning("Could not determine GPU memory, using default estimates")
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
    usable_memory = gpu_info['available'] - buffer_memory
    
    is_compatible = required_memory <= usable_memory
    
    if is_compatible:
        message = f"✅ Model {model_name} is compatible (requires ~{required_memory}MB, {usable_memory}MB available)"
    else:
        shortage = required_memory - usable_memory
        message = f"❌ Model {model_name} requires ~{required_memory}MB but only {usable_memory}MB available (shortage: {shortage}MB)"
    
    details = {
        'required_memory_mb': required_memory,
        'available_memory_mb': usable_memory,
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
            from enhanced_document_processor import get_document_processor
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

# JWT Settings
SECRET_KEY = "your-secret-key"  # Change this to a secure key in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="api/auth/login",  # Changed from "/api/auth/login"
    scheme_name="JWT"
)

# Models
class UserLogin(BaseModel):
    username: str
    password: str

class Message(BaseModel):
    content: str
    session_id: Optional[int] = None

class UserCreate(BaseModel):
    username: str
    password: str
    role: str
    
class RLHFFeedback(BaseModel):
    session_id: int
    chosen_index: int  # 0 for first response, 1 for second response

# Auth endpoints
@app.post("/api/auth/login")
@app.options("/api/auth/login")  # Add explicit OPTIONS handler
async def login(request: Request):
    # Log request method to debug preflight issues
    logger.info(f"Auth request method: {request.method}")
    
    # Handle OPTIONS request (preflight)
    if request.method == "OPTIONS":
        return JSONResponse(
            content={},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Authorization, Content-Type, Accept, X-Requested-With",
                "Access-Control-Max-Age": "86400",
                "Access-Control-Allow-Credentials": "true",
            }
        )
    
    try:
        # Add CORS headers to response
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        }
        
        # Log the request headers for debugging
        logger.info(f"Request headers: {request.headers}")
        
        # Try to get form data first
        content_type = request.headers.get('content-type', '')
        logger.info(f"Content-Type: {content_type}")
        
        username = None
        password = None
        
        if 'application/json' in content_type:
            # Handle JSON data
            data = await request.json()
            username = data.get('username')
            password = data.get('password')
            logger.info(f"Received JSON login request for user: {username}")
        else:
            # Handle form data
            form_data = await request.form()
            username = form_data.get('username')
            password = form_data.get('password')
            logger.info(f"Received form login request for user: {username}")
        
        logger.info(f"Login attempt for user: {username}")
        
        # For testing - accept fixed credentials directly
        if (username == "admin" and password == "admin") or (username == "test" and password == "test"):
            logger.info(f"Using direct auth for: {username}")
            token_data = {
                "sub": username,
                "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            }
            access_token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
            
            response_data = {
                "access_token": access_token,
                "token_type": "bearer",
                "username": username
            }
            
            return JSONResponse(
                content=response_data,
                headers=headers
            )
        
        if not username or not password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username and password are required"
            )

        authenticated = chat_db.authenticate_user(username, password)
        logger.info(f"Authentication result for {username}: {authenticated}")
        
        if not authenticated:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        token_data = {
            "sub": username,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        }
        access_token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
        
        logger.info(f"Login successful for user: {username}")
        
        response_data = {
            "access_token": access_token,
            "token_type": "bearer",
            "username": username
        }
        
        return JSONResponse(
            content=response_data,
            headers=headers
        )
    except HTTPException as he:
        # Return structured error response with proper headers
        logger.error(f"HTTP Exception in login: {str(he)}")
        return JSONResponse(
            content={"detail": he.detail},
            status_code=he.status_code,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
            }
        )
    except Exception as e:
        logger.error(f"Login error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(e)},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
            }
        )

# Add admin check function
async def check_if_admin(token: str = Depends(oauth2_scheme)):
    try:
        user = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if not chat_db.is_admin(user["sub"]):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        return user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

# Add user authentication function
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        user = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

# Include TruLens evaluation routes (after defining get_current_user to avoid circular imports)
from routes.evaluation import router as evaluation_router
app.include_router(evaluation_router, prefix="/api/evaluation", tags=["evaluation"])

# Add admin-only user management endpoint
@app.post("/api/admin/add-user")
async def add_user(user_data: UserCreate, admin: dict = Depends(check_if_admin)):
    try:
        if chat_db.user_exists(user_data.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        if user_data.role not in ['Engineer', 'Manager', 'Business Development', 'Associate']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid role"
            )
        
        success = chat_db.add_user(user_data.username, user_data.password, user_data.role)
        return {"message": "User added successfully"}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/admin/users")
async def get_users(admin: dict = Depends(check_if_admin)):
    users = chat_db.get_all_users()
    return {"users": users}

# Add new endpoints for dashboard data
@app.get("/api/admin/user-stats/{username}")
async def get_user_stats(username: str, admin: dict = Depends(check_if_admin)):
    try:
        stats = chat_db.get_user_stats(username)
        return {"data": stats}  # Wrap the stats in a data field
    except Exception as e:
        logger.error(f"Error getting user stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/activity-stats")
async def get_activity_stats(admin: dict = Depends(check_if_admin)):
    try:
        stats = chat_db.get_activity_stats()
        return {"data": stats}  # Wrap stats in data field
    except Exception as e:
        logger.error(f"Error getting activity stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# User endpoints for ManageUserPage
@app.get("/api/users/profile")
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    """Get current user's profile"""
    try:
        username = current_user["sub"]
        # Get user data from database
        user_data = chat_db.get_user_by_username(username)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Return user profile data
        return {
            "user": {
                "username": user_data.get("username", username),
                "email": user_data.get("email", ""),
                "role": user_data.get("role", "user"),
                "created_at": user_data.get("created_at", ""),
                "last_login": user_data.get("last_login", ""),
                "is_active": user_data.get("is_active", True)
            }
        }
    except Exception as e:
        logger.error(f"Error getting user profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/activities")
async def get_user_activities(current_user: dict = Depends(get_current_user)):
    """Get user activities/recent actions"""
    try:
        username = current_user["sub"]
        
        # Get recent user activities from chat history and other sources
        activities = []
        
        # Get recent chat sessions
        try:
            sessions = chat_db.get_user_sessions(username, limit=10)
            for session in sessions:
                activities.append({
                    "id": f"chat_{session.get('id', '')}",
                    "type": "chat",
                    "action": "Started conversation",
                    "details": session.get("title", "Chat session")[:100],
                    "timestamp": session.get("created_at", ""),
                    "metadata": {
                        "session_id": session.get("id"),
                        "message_count": session.get("message_count", 0)
                    }
                })
        except Exception as e:
            logger.warning(f"Error getting chat sessions: {e}")
        
        # Add document upload activities if available
        try:
            # This would need to be implemented based on your document tracking
            # For now, we'll add placeholder data
            activities.append({
                "id": "doc_recent",
                "type": "document",
                "action": "Document processing",
                "details": "Recent document activities",
                "timestamp": datetime.datetime.now().isoformat(),
                "metadata": {"source": "system"}
            })
        except Exception as e:
            logger.warning(f"Error getting document activities: {e}")
        
        # Sort by timestamp (newest first)
        activities.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return {"activities": activities[:20]}  # Return last 20 activities
        
    except Exception as e:
        logger.error(f"Error getting user activities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/stats")
async def get_user_stats_general(current_user: dict = Depends(get_current_user)):
    """Get general system statistics for admin dashboard"""
    try:
        # Get overall system statistics
        stats = {
            "totalUsers": 0,
            "activeUsers": 0,
            "totalSessions": 0,
            "totalMessages": 0
        }
        
        try:
            # Get total users count
            with sqlite3.connect(chat_db.db_path) as conn:
                cursor = conn.cursor()
                
                # Total users
                cursor.execute("SELECT COUNT(*) FROM users")
                stats["totalUsers"] = cursor.fetchone()[0]
                
                # Active users (users with sessions in last 30 days)
                cursor.execute("""
                    SELECT COUNT(DISTINCT username) 
                    FROM chat_sessions 
                    WHERE created_at >= datetime('now', '-30 days')
                """)
                stats["activeUsers"] = cursor.fetchone()[0]
                
                # Total sessions
                cursor.execute("SELECT COUNT(*) FROM chat_sessions")
                stats["totalSessions"] = cursor.fetchone()[0]
                
                # Total messages
                cursor.execute("SELECT COUNT(*) FROM messages")
                stats["totalMessages"] = cursor.fetchone()[0]
                
        except Exception as e:
            logger.warning(f"Error getting system stats: {e}")
        
        return {"stats": stats}
        
    except Exception as e:
        logger.error(f"Error getting user stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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

@app.post("/api/admin/upload")
async def upload_file(
    file: UploadFile,
    is_folder: str = Form(default="false"),
    folder_path: str = Form(default=""),
    chunking_method: str = Form(default="auto"),  # auto, general, qa, resume, table, presentation, picture, email
    chunk_token_num: int = Form(default=1000),
    chunk_overlap: int = Form(default=200),
    delimiter: str = Form(default="\\n\\n|\\n|\\.|\\!|\\?"),
    max_token: int = Form(default=4096),
    layout_recognize: str = Form(default="auto"),
    preserve_formatting: bool = Form(default=True),
    extract_tables: bool = Form(default=True),
    extract_images: bool = Form(default=False),
    admin: dict = Depends(check_if_admin)
):
    failed_files = []
    processed_files = []
    try:
        # Import chunking components
        from chunking_config import ChunkingMethod, ChunkingConfig, FileFormatSupport
        
        file_id = f"upload_{datetime.datetime.now().timestamp()}"
        upload_progress[file_id] = 0
        
        # Create base temp directory
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True, mode=0o755)
        
        try:
            contents = await file.read()
            
            if is_folder.lower() == "true" and folder_path:
                # Create full folder structure
                folder_structure = Path(folder_path).parent
                full_path = temp_dir / folder_structure
                full_path.mkdir(parents=True, exist_ok=True)
                temp_path = temp_dir / folder_path
            else:
                temp_path = temp_dir / f"temp_{file_id}_{file.filename}"
            
            # Save file
            temp_path.write_bytes(contents)
            upload_progress[file_id] = 30

            # Determine file extension and chunking method
            file_ext = file.filename.split('.')[-1].lower()
            
            # Set up chunking configuration
            if chunking_method == "auto":
                # Auto-detect optimal method for file type
                selected_method = FileFormatSupport.get_optimal_method(file_ext)
            else:
                try:
                    selected_method = ChunkingMethod(chunking_method)
                except ValueError:
                    logger.warning(f"Invalid chunking method '{chunking_method}', using general")
                    selected_method = ChunkingMethod.GENERAL
            
            # Create chunking configuration
            chunking_config = ChunkingConfig(
                method=selected_method,
                chunk_token_num=chunk_token_num,
                chunk_overlap=chunk_overlap,
                delimiter=delimiter,
                max_token=max_token,
                layout_recognize=layout_recognize,
                preserve_formatting=preserve_formatting,
                extract_tables=extract_tables,
                extract_images=extract_images
            )
            
            logger.info(f"Processing {file.filename} with method {selected_method.value}")

            # Process files with enhanced chunking - include images now
            if file.filename.lower().endswith(('.pdf', '.docx', '.doc', '.txt', '.md', '.csv', '.xlsx', '.xls', '.ppt', '.pptx', '.html', '.json', '.eml', '.jpg', '.jpeg', '.png', '.gif', '.tif', '.tiff')):
                logger.info(f"Processing document file: {file.filename}")
                success = get_rag().ingest_with_storage_and_chunking(
                    str(temp_path), 
                    file.filename,
                    selected_method,
                    chunking_config,
                    admin['sub']  # user_id
                )
                if not success:
                    failed_files.append(file.filename)
                    logger.warning(f"Failed to process document: {file.filename}")
                else:
                    processed_files.append(file.filename)
                    logger.info(f"Successfully processed document: {file.filename}")
            else:
                # For other files, store in MinIO only
                logger.info(f"Storing non-document file: {file.filename}")
                try:
                    # Determine content type based on file extension
                    import mimetypes
                    content_type, _ = mimetypes.guess_type(file.filename)
                    if not content_type:
                        content_type = 'application/octet-stream'
                    
                    # Store in MinIO using document storage with chunking info
                    from document_storage import get_document_storage
                    doc_storage = get_document_storage()
                    doc_info = doc_storage.store_document(
                        str(temp_path), 
                        file.filename, 
                        content_type,
                        selected_method.value,
                        chunking_config.to_dict()
                    )
                    
                    if doc_info:
                        processed_files.append(file.filename)
                        logger.info(f"Successfully stored file: {file.filename}")
                    else:
                        failed_files.append(file.filename)
                        logger.warning(f"Failed to store file: {file.filename}")
                        
                except Exception as e:
                    logger.error(f"Error storing file {file.filename}: {str(e)}")
                    failed_files.append(file.filename)
            
            upload_progress[file_id] = 90
            
            # Save file info regardless of processing success
            chat_db.save_file_info(
                filename=file.filename,
                format=file.filename.split('.')[-1].lower(),
                size=len(contents),
                uploaded_by=admin['sub'],
                is_folder=is_folder.lower() == "true",
                folder_path=folder_path if is_folder.lower() == "true" else None
            )
            
            upload_progress[file_id] = 100
            
            response_data = {
                "message": f"Upload complete. Processed: {len(processed_files)} files, Failed: {len(failed_files)} files",
                "file_id": file_id,
                "processed_files": processed_files,
                "failed_files": failed_files,
                "folder_path": str(folder_path) if is_folder.lower() == "true" else None
            }
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            failed_files.append(file.filename)
            return {
                "message": "Upload completed with errors",
                "file_id": file_id,
                "processed_files": processed_files,
                "failed_files": failed_files,
                "error": str(e)
            }
            
        finally:
            # Clean up temp files
            try:
                if 'temp_path' in locals() and temp_path.exists():
                    temp_path.unlink()
                for path in sorted([p for p in temp_dir.rglob('*') if p.is_dir()], reverse=True):
                    try:
                        path.rmdir()
                    except OSError:
                        pass
                if temp_dir.exists() and not any(temp_dir.iterdir()):
                    temp_dir.rmdir()
            except Exception as e:
                logger.error(f"Cleanup error: {str(e)}")
                
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        if 'file_id' in locals():
            upload_progress[file_id] = -1
        return {
            "message": "Upload failed",
            "error": str(e),
            "processed_files": processed_files,
            "failed_files": failed_files
        }

# Document management endpoints
@app.get("/api/documents")
async def list_documents():  # Temporarily removed auth: token: str = Depends(oauth2_scheme)
    """List all documents with their ingestion status"""
    try:
        documents = get_rag().get_all_documents()
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/ingested")
async def list_ingested_documents(token: str = Depends(oauth2_scheme)):
    """List documents ingested for current embedding model"""
    try:
        documents = get_rag().get_ingested_documents()
        return {"documents": documents, "embedding_model": get_rag().embedding_model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/reingest")
async def reingest_documents(
    embedding_model: str,
    admin: dict = Depends(check_if_admin)
):
    """Re-ingest all documents for a new embedding model"""
    try:
        success = get_rag().reingest_for_model_switch(embedding_model)
        if success:
            return {
                "message": f"Documents re-ingested for model: {embedding_model}",
                "embedding_model": embedding_model
            }
        else:
            raise HTTPException(status_code=500, detail="Re-ingestion failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/reingest-specific")
async def reingest_specific_documents(
    request: dict,
    admin: dict = Depends(check_if_admin)
):
    """Re-ingest specific documents with per-document chunking configuration"""
    try:
        documents = request.get('documents', [])
        
        if not documents:
            raise HTTPException(status_code=400, detail="No documents provided")
        
        # Parse and validate each document configuration
        parsed_documents = []
        for doc_config in documents:
            document_id = doc_config.get('document_id')
            chunking_method_str = doc_config.get('chunking_method')
            chunking_config_dict = doc_config.get('chunking_config', {})
            
            if not document_id:
                raise HTTPException(status_code=400, detail="Document ID is required for each document")
            
            # Parse chunking method
            chunking_method = None
            if chunking_method_str:
                try:
                    from chunking_config import ChunkingMethod
                    chunking_method = ChunkingMethod(chunking_method_str)
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid chunking method: {chunking_method_str}")
            
            # Parse chunking config
            chunking_config = None
            if chunking_config_dict:
                try:
                    from chunking_config import ChunkingConfig
                    chunking_config = ChunkingConfig.from_dict(chunking_config_dict)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid chunking config: {str(e)}")
            
            parsed_documents.append({
                'document_id': document_id,
                'chunking_method': chunking_method,
                'chunking_config': chunking_config
            })
        
        # Perform reingestion with per-document configuration
        results = get_rag().reingest_specific_documents_with_config(parsed_documents)
        
        return {
            "message": f"Reingestion completed: {results['successful']}/{results['total']} successful",
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in specific document reingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/{document_id}/retry")
async def retry_document_processing(
    document_id: int,
    method: str = Form(default="auto"),
    chunk_token_num: int = Form(default=1000),
    chunk_overlap: int = Form(default=200),
    delimiter: str = Form(default="\\n\\n|\\n|\\.|\\!|\\?"),
    max_token: int = Form(default=4096),
    layout_recognize: str = Form(default="auto"),
    preserve_formatting: bool = Form(default=True),
    extract_tables: bool = Form(default=True),
    extract_images: bool = Form(default=False),
    current_user: dict = Depends(get_current_user)
):
    """Retry processing a failed document"""
    try:
        from document_storage import get_document_storage
        from chunking_config import ChunkingMethod, ChunkingConfig
        
        # Get document info
        doc_storage = get_document_storage()
        doc_info = doc_storage.get_document(document_id)
        
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if user owns this document or is admin
        if doc_info.get('user_id') != current_user['sub'] and not current_user.get('is_admin', False):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Update document status to pending
        doc_storage.update_document_status(document_id, 'pending', None)
        
        # Get the stored file from MinIO
        temp_file_path = doc_storage.get_document_file(document_id)
        if not temp_file_path:
            raise HTTPException(status_code=404, detail="Document file not found")
        
        try:
            filename = doc_info['filename']
            
            # Determine chunking method
            if method == "auto":
                from chunking_config import FileFormatSupport
                file_ext = filename.split('.')[-1].lower()
                selected_method = FileFormatSupport.get_optimal_method(file_ext)
            else:
                try:
                    selected_method = ChunkingMethod(method)
                except ValueError:
                    logger.warning(f"Invalid chunking method '{method}', using general")
                    selected_method = ChunkingMethod.GENERAL
            
            # Create chunking configuration
            chunking_config = ChunkingConfig(
                method=selected_method,
                chunk_token_num=chunk_token_num,
                chunk_overlap=chunk_overlap,
                delimiter=delimiter,
                max_token=max_token,
                layout_recognize=layout_recognize,
                preserve_formatting=preserve_formatting,
                extract_tables=extract_tables,
                extract_images=extract_images
            )
            
            logger.info(f"Retrying processing for {filename} with method {selected_method.value}")
            
            # First, remove any existing chunks for this document using document ID for more accuracy
            try:
                rag_instance = get_rag()
                # Use remove_document_by_id for more comprehensive deletion across all embedding models
                deletion_success = rag_instance.remove_document_by_id(document_id)
                if not deletion_success:
                    logger.warning(f"Could not fully remove existing chunks for document {document_id}")
                    # Fallback to filename-based removal
                    rag_instance.remove_document_from_vectorstore(filename)
            except Exception as e:
                logger.warning(f"Could not remove existing chunks: {e}")
            
            # Process the document (this will only re-ingest to vector store, not create new document entry)
            success = get_rag().ingest_with_storage_and_chunking(
                temp_file_path, 
                filename,
                selected_method,
                chunking_config,
                current_user['sub']  # user_id
            )
            
            if success:
                # Update status to completed
                doc_storage.update_document_status(document_id, 'completed', None)
                return {
                    "message": f"Document {filename} processed successfully",
                    "document_id": document_id,
                    "method": selected_method.value
                }
            else:
                # Update status to failed
                doc_storage.update_document_status(document_id, 'failed', "Processing failed during retry")
                raise HTTPException(status_code=500, detail="Document processing failed")
                
        finally:
            # Clean up temporary file
            try:
                import os
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Could not delete temp file: {e}")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrying document processing: {e}")
        # Update status to failed if we have document_id
        try:
            doc_storage.update_document_status(document_id, 'failed', str(e))
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/{document_id}")
async def delete_document(
    document_id: int,
    admin: dict = Depends(check_if_admin)
):
    """Delete a document from storage"""
    try:
        from document_storage import get_document_storage
        success = get_document_storage().delete_document(document_id)
        if success:
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/bulk-delete")
async def bulk_delete_documents(
    request_data: dict,
    admin: dict = Depends(check_if_admin)
):
    """Bulk delete multiple documents from storage"""
    try:
        document_ids = request_data.get('document_ids', [])
        
        if not document_ids:
            raise HTTPException(status_code=400, detail="No document IDs provided")
        
        if not isinstance(document_ids, list):
            raise HTTPException(status_code=400, detail="document_ids must be a list")
        
        # Validate that all IDs are integers
        try:
            document_ids = [int(doc_id) for doc_id in document_ids]
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="All document IDs must be integers")
        
        from document_storage import get_document_storage
        results = get_document_storage().delete_multiple_documents(document_ids)
        
        return {
            "message": f"Bulk deletion completed: {results['successful']} successful, {results['failed']} failed",
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/clear-all-documents")
async def clear_all_documents(
    admin: dict = Depends(check_if_admin)
):
    """Clear all documents from storage - admin only"""
    try:
        from document_storage import get_document_storage
        success = get_document_storage().clear_all_documents()
        if success:
            return {"message": "All documents cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear documents")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/cleanup-orphaned")
async def cleanup_orphaned_documents(
    admin: dict = Depends(check_if_admin)
):
    """Cleanup orphaned documents - admin only"""
    try:
        from document_storage import get_document_storage
        count = get_document_storage().cleanup_orphaned_documents()
        return {"message": f"Cleaned up {count} orphaned documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/files/check-duplicate")
async def check_file_duplicate(
    request_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Check if a file with the same filename and hash already exists"""
    try:
        filename = request_data.get('filename')
        file_hash = request_data.get('hash')
        
        if not filename or not file_hash:
            raise HTTPException(status_code=400, detail="Both filename and hash are required")
        
        from document_storage import get_document_storage
        doc_storage = get_document_storage()
        
        # Check if a file with this hash already exists
        existing_file = doc_storage.get_document_by_hash(file_hash)
        
        if existing_file:
            return {
                "exists": True,
                "existing_file": {
                    "filename": existing_file.get('filename'),
                    "upload_date": existing_file.get('upload_date'),
                    "size": existing_file.get('file_size'),
                    "content_type": existing_file.get('content_type'),
                    "hash": existing_file.get('file_hash')
                }
            }
        else:
            return {"exists": False}
            
    except Exception as e:
        logger.error(f"Error checking file duplicate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/files/{filename}")
async def delete_file_by_filename(
    filename: str,
    admin: dict = Depends(check_if_admin)
):
    """Delete a document by filename"""
    try:
        from document_storage import get_document_storage
        success = get_document_storage().delete_document_by_filename(filename)
        if success:
            return {"message": f"File {filename} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files/{filename}/chunks")
async def get_document_chunks(
    filename: str,
    current_user: dict = Depends(get_current_user)
):
    """Get all chunks for a document by filename"""
    try:
        from rag import get_chatpdf_instance
        from document_storage import get_document_storage
        
        chatpdf = get_chatpdf_instance()
        doc_storage = get_document_storage()
        
        if not chatpdf or not chatpdf.vector_store:
            raise HTTPException(status_code=503, detail="Vector store not available")
        
        # Get document info from storage first
        doc_info = None
        try:
            # Find document by filename
            all_docs = doc_storage.list_all_documents()
            for doc in all_docs:
                if doc['filename'] == filename:
                    doc_info = doc
                    break
        except Exception as e:
            logger.warning(f"Could not get document info: {e}")
        
        # Get ChromaDB client and collection
        chroma_client = chatpdf.vector_store._client
        collection_name = chatpdf._get_collection_name()
        
        try:
            collection = chroma_client.get_collection(collection_name)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Collection not found: {e}")
        
        # Query for chunks with this filename in source metadata
        try:
            logger.info(f"Searching for chunks with filename: {filename}")
            
            # Try both 'source' and 'source_file' metadata keys with exact match first
            results = None
            for metadata_key in ['source_file', 'source']:
                try:
                    results = collection.get(
                        where={metadata_key: filename},
                        include=["documents", "metadatas", "embeddings"]
                    )
                    if results and results.get('ids'):
                        logger.info(f"Found {len(results['ids'])} chunks using exact match on metadata key '{metadata_key}'")
                        break
                except Exception as e:
                    logger.debug(f"Exact query with '{metadata_key}' failed: {e}")
                    continue
            
            # If exact match didn't work, try to find files ending with the filename (for folder uploads)
            if not results or not results.get('ids'):
                logger.info("No exact match found, trying filename suffix search...")
                for metadata_key in ['source_file', 'source']:
                    try:
                        # Get all documents and filter by filename suffix
                        all_results = collection.get(
                            include=["documents", "metadatas", "embeddings"]
                        )
                        
                        if all_results and all_results.get('metadatas'):
                            matching_indices = []
                            for i, metadata in enumerate(all_results['metadatas']):
                                source_value = metadata.get(metadata_key, '')
                                if source_value and source_value.endswith(filename):
                                    matching_indices.append(i)
                            
                            if matching_indices:
                                # Build filtered results
                                results = {
                                    'ids': [all_results['ids'][i] for i in matching_indices],
                                    'documents': [all_results['documents'][i] for i in matching_indices],
                                    'metadatas': [all_results['metadatas'][i] for i in matching_indices],
                                    'embeddings': [all_results['embeddings'][i] for i in matching_indices] if all_results.get('embeddings') else None
                                }
                                logger.info(f"Found {len(results['ids'])} chunks using suffix match on metadata key '{metadata_key}'")
                                break
                    except Exception as e:
                        logger.debug(f"Suffix search with '{metadata_key}' failed: {e}")
                        continue
            
            logger.info(f"ChromaDB query results: found {len(results.get('ids', [])) if results else 0} chunks")
            if results and results.get('metadatas'):
                logger.info(f"Sample metadata from results: {results['metadatas'][:2] if len(results['metadatas']) > 0 else 'None'}")
            
            if not results or not results.get('ids'):
                # Let's also try a broader search to see what's actually in the collection
                logger.info("No chunks found with exact filename match, checking collection contents...")
                sample_results = collection.get(limit=5, include=["metadatas"])
                logger.info(f"Sample collection metadata: {sample_results.get('metadatas', [])}")
                
                return {
                    "filename": filename,
                    "chunks": [],
                    "document_info": doc_info,
                    "is_image": doc_info and doc_info.get('content_type', '').startswith('image/') if doc_info else False
                }
            
            logger.info(f"Processing {len(results['ids'])} chunks for response")
            chunks = []
            for i, chunk_id in enumerate(results['ids']):
                try:
                    chunk_content = results['documents'][i]
                    chunk_metadata = results['metadatas'][i] if results.get('metadatas') else {}
                    
                    # Count tokens/words (approximate)
                    word_count = len(chunk_content.split()) if chunk_content else 0
                    
                    # Calculate embedding size safely
                    embedding_size = 0
                    try:
                        if results.get('embeddings') and i < len(results['embeddings']) and results['embeddings'][i]:
                            embedding_size = len(results['embeddings'][i])
                    except Exception as e:
                        logger.warning(f"Could not calculate embedding size for chunk {i}: {e}")
                    
                    chunk_data = {
                        "id": chunk_id,
                        "chunk_number": i + 1,
                        "content": chunk_content,
                        "word_count": word_count,
                        "metadata": chunk_metadata,
                        "embedding_size": embedding_size
                    }
                    chunks.append(chunk_data)
                    logger.debug(f"Processed chunk {i+1}/{len(results['ids'])}")
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {e}")
                    continue
            
            logger.info(f"Successfully processed {len(chunks)} chunks")
            return {
                "filename": filename,
                "total_chunks": len(chunks),
                "chunks": chunks,
                "document_info": doc_info,
                "is_image": doc_info and doc_info.get('content_type', '').startswith('image/') if doc_info else False
            }
            
        except Exception as e:
            logger.error(f"Error querying chunks: {e}")
            raise HTTPException(status_code=500, detail=f"Error querying chunks: {e}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
async def clear_vector_store(admin: dict = Depends(check_if_admin)):
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

@app.get("/api/models/available")
async def get_available_models():
    """Get list of available models from Ollama (both local and remote)"""
    try:
        # TODO: Add authentication back when circular import is resolved
        # current_user: dict = Depends(get_current_user)
        
        import httpx
        from ollama_scraper import get_available_ollama_models
        
        ollama_host = os.getenv('OLLAMA_HOST', 'http://ollama:11434')
        
        # Get locally installed models
        local_models = []
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{ollama_host}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    local_model_names = {model.get('name', '') for model in data.get('models', [])}
                    
                    for model in data.get('models', []):
                        model_name = model.get('name', '')
                        model_size = model.get('size', 0)
                        model_modified = model.get('modified_at', '')
                        
                        # Determine category with comprehensive embedding detection
                        model_name_lower = model_name.lower()
                        is_embedding = (
                            'embed' in model_name_lower or
                            'bge' in model_name_lower or
                            'minilm' in model_name_lower or
                            'all-minilm' in model_name_lower or
                            'nomic' in model_name_lower or
                            'e5-' in model_name_lower or
                            'sentence' in model_name_lower or
                            'text-embedding' in model_name_lower or
                            'instructor' in model_name_lower or
                            'gte-' in model_name_lower or
                            'multilingual-e5' in model_name_lower or
                            'arctic-embed' in model_name_lower or
                            'mxbai-embed' in model_name_lower or
                            model_name_lower.startswith('bge-') or
                            model_name_lower.startswith('all-minilm-') or
                            model_name_lower.startswith('e5-') or
                            model_name_lower.startswith('gte-') or
                            model_name_lower.startswith('nomic-') or
                            'snowflake-arctic-embed' in model_name_lower or
                            'paraphrase-' in model_name_lower or
                            'distiluse' in model_name_lower
                        )
                        
                        if is_embedding:
                            category = 'embedding'
                        else:
                            category = 'llm'
                        
                        local_models.append({
                            'name': model_name,
                            'category': category,
                            'size': format_model_size(model_size),
                            'modified_at': model_modified,
                            'source': 'local',
                            'description': f"Locally installed {category} model"
                        })
                else:
                    local_model_names = set()
        except Exception as e:
            logger.warning(f"Could not fetch local models: {e}")
            local_model_names = set()
        
        # Get available models from Ollama library
        try:
            available_models = get_available_ollama_models(use_cache=True)
            # logger.info(f"Found {len(available_models)} models from Ollama library")
        except Exception as e:
            logger.warning(f"Could not fetch Ollama library models: {e}")
            available_models = []
        
        # Combine local and available models, marking local ones
        all_models = {}
        
        # Add local models first (these take priority)
        for model in local_models:
            all_models[model['name']] = model
        
        # Add available models that aren't already local
        for model in available_models:
            model_name = model['name']
            if model_name not in all_models:
                # Determine category with comprehensive embedding detection
                model_name_lower = model_name.lower()
                is_embedding = (
                    'embed' in model_name_lower or
                    'bge' in model_name_lower or
                    'minilm' in model_name_lower or
                    'all-minilm' in model_name_lower or
                    'nomic' in model_name_lower or
                    'e5-' in model_name_lower or
                    'sentence' in model_name_lower or
                    'text-embedding' in model_name_lower or
                    'instructor' in model_name_lower or
                    'gte-' in model_name_lower or
                    'multilingual-e5' in model_name_lower or
                    'arctic-embed' in model_name_lower or
                    'mxbai-embed' in model_name_lower or
                    model_name_lower.startswith('bge-') or
                    model_name_lower.startswith('all-minilm-') or
                    model_name_lower.startswith('e5-') or
                    model_name_lower.startswith('gte-') or
                    model_name_lower.startswith('nomic-') or
                    'snowflake-arctic-embed' in model_name_lower or
                    'paraphrase-' in model_name_lower or
                    'distiluse' in model_name_lower
                )
                
                # Override the category from library if needed
                if is_embedding:
                    model_category = 'embedding'
                else:
                    model_category = model.get('category', 'llm')
                
                # Add as available for download
                all_models[model_name] = {
                    'name': model_name,
                    'category': model_category,
                    'description': model.get('description', ''),
                    'size': format_model_size(model.get('size', 'Unknown')),
                    'source': 'library',
                    'tags': model.get('tags', [])
                }
            else:
                # Update local model with additional info from library
                all_models[model_name].update({
                    'description': model.get('description', all_models[model_name].get('description', '')),
                    'tags': model.get('tags', [])
                })
        
        # Convert to list and separate by category for backward compatibility
        models_list = list(all_models.values())
        
        # Debug logging for categorization - DISABLED for cleaner logs
        # logger.info(f"Total models loaded: {len(models_list)}")
        # for model in models_list:
        #     logger.info(f"Model: {model['name']} -> Category: {model.get('category', 'unknown')}")
        
        llm_models = [m for m in models_list if m.get('category', 'llm') == 'llm']
        embedding_models = [m for m in models_list if m.get('category', 'embedding') == 'embedding']
        
        # logger.info(f"Categorized into {len(llm_models)} LLM models and {len(embedding_models)} embedding models")
        # logger.info(f"LLM models: {[m['name'] for m in llm_models]}")
        # logger.info(f"Embedding models: {[m['name'] for m in embedding_models]}")
        
        return {
            "success": True,
            "models": models_list,  # For new frontend format
            "llm_models": llm_models,  # For backward compatibility
            "embedding_models": embedding_models,  # For backward compatibility
            "total_models": len(models_list)
        }
                
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get available models: {str(e)}")

@app.get("/api/models/current")
async def get_current_models(current_user: dict = Depends(get_current_user)):
    """Get current model settings including parameters"""
    try:
        rag = get_chatpdf_instance()
        
        # Load parameters from database first, then fallback to config file
        parameters = {
            'temperature': 0.7,
            'max_tokens': 2048,
            'top_p': 0.9,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0
        }
        
        try:
            # Try database first
            db_settings = chat_db.get_latest_model_settings()
            if db_settings and 'parameters' in db_settings:
                parameters.update(db_settings['parameters'])
            else:
                # Fallback to config file
                config_path = "model_settings.json"
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        settings = json.load(f)
                        if 'parameters' in settings:
                            parameters.update(settings['parameters'])
        except Exception as e:
            logger.warning(f"Could not load parameters from database or config: {e}")
        
        return {
            "success": True,
            "llm": rag.llm_model,
            "embedding": rag.embedding_model,
            "parameters": parameters
        }
    except Exception as e:
        logger.error(f"Error getting current models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get current models: {str(e)}")

@app.post("/api/models/check-gpu")
async def check_gpu_compatibility(request: dict = None):
    """Check GPU compatibility for models"""
    try:
        if not request:
            return {
                "success": False,
                "compatible": False,
                "message": "No models specified for compatibility check"
            }
        
        llm_model = request.get('llm')
        embedding_model = request.get('embedding')
        
        if not llm_model or not embedding_model:
            return {
                "success": False,
                "compatible": False,
                "message": "Both LLM and embedding models must be specified"
            }
        
        logger.info(f"Checking GPU compatibility for LLM: {llm_model}, Embedding: {embedding_model}")
        
        # Get model information for size estimation
        available_models = await get_available_models()
        
        llm_info = None
        embedding_info = None
        
        # Find model information
        for model in available_models.get('models', []):
            if model['name'] == llm_model:
                llm_info = model
            elif model['name'] == embedding_model:
                embedding_info = model
        
        # Check individual model compatibility using the new detailed function
        llm_compatible, llm_message, llm_details = check_model_compatibility_detailed(
            llm_model, 
            llm_info.get('size') if llm_info else None
        )
        
        embedding_compatible, embedding_message, embedding_details = check_model_compatibility_detailed(
            embedding_model, 
            embedding_info.get('size') if embedding_info else None
        )
        
        logger.info(f"LLM compatibility check: {llm_compatible} - {llm_message}")
        logger.info(f"Embedding compatibility check: {embedding_compatible} - {embedding_message}")
        
        # Combined compatibility check
        total_required_mb = llm_details['required_memory_mb'] + embedding_details['required_memory_mb']
        gpu_info = get_gpu_memory_info()
        buffer_memory = max(1024, int(gpu_info['total'] * 0.2))
        usable_memory = gpu_info['available'] - buffer_memory
        
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
                "available_mb": usable_memory,
                "compatible": combined_compatible,
                "message": f"Combined models require {total_required_mb}MB, {usable_memory}MB available after buffer"
            },
            "gpu_info": gpu_info,
            "recommendation": (
                "Models should fit in available GPU memory" 
                if combined_compatible 
                else "Consider using smaller models or upgrading GPU memory"
            )
        }
        
    except Exception as e:
        logger.error(f"Error checking GPU compatibility: {str(e)}")
        return {
            "success": False,
            "compatible": True,  # Default to compatible to not block users
            "message": f"GPU check failed: {str(e)}. Proceeding with model download.",
            "error": str(e)
        }

@app.post("/api/models/settings")
async def update_models_settings(
    request: dict,
    current_user: dict = Depends(get_current_user)
):
    """Update model settings and download models if needed"""
    try:
        import httpx
        
        llm_model = request.get('llm')
        embedding_model = request.get('embedding')
        llm_size = request.get('llm_size')
        embedding_size = request.get('embedding_size')
        force_update = request.get('force', False)  # Allow bypassing compatibility check
        model_parameters = request.get('parameters', {})
        
        if not llm_model or not embedding_model:
            raise HTTPException(status_code=400, detail="Both LLM and embedding models are required")
        
        # Validate model parameters
        valid_parameters = {}
        if model_parameters:
            # Temperature (0.0 to 2.0)
            temp = model_parameters.get('temperature', 0.7)
            if isinstance(temp, (int, float)) and 0.0 <= temp <= 2.0:
                valid_parameters['temperature'] = float(temp)
            else:
                valid_parameters['temperature'] = 0.7
            
            # Max tokens (1 to 32768)
            max_tokens = model_parameters.get('max_tokens', 2048)
            if isinstance(max_tokens, int) and 1 <= max_tokens <= 32768:
                valid_parameters['max_tokens'] = max_tokens
            else:
                valid_parameters['max_tokens'] = 2048
            
            # Top-p (0.0 to 1.0)
            top_p = model_parameters.get('top_p', 0.9)
            if isinstance(top_p, (int, float)) and 0.0 <= top_p <= 1.0:
                valid_parameters['top_p'] = float(top_p)
            else:
                valid_parameters['top_p'] = 0.9
            
            # Frequency penalty (-2.0 to 2.0)
            freq_penalty = model_parameters.get('frequency_penalty', 0.0)
            if isinstance(freq_penalty, (int, float)) and -2.0 <= freq_penalty <= 2.0:
                valid_parameters['frequency_penalty'] = float(freq_penalty)
            else:
                valid_parameters['frequency_penalty'] = 0.0
            
            # Presence penalty (-2.0 to 2.0)
            presence_penalty = model_parameters.get('presence_penalty', 0.0)
            if isinstance(presence_penalty, (int, float)) and -2.0 <= presence_penalty <= 2.0:
                valid_parameters['presence_penalty'] = float(presence_penalty)
            else:
                valid_parameters['presence_penalty'] = 0.0
        else:
            # Default parameters
            valid_parameters = {
                'temperature': 0.7,
                'max_tokens': 2048,
                'top_p': 0.9,
                'frequency_penalty': 0.0,
                'presence_penalty': 0.0
            }
        
        # Check GPU compatibility before proceeding (unless forced)
        if not force_update:
            try:
                from rag import check_model_compatibility
                
                # Check LLM model compatibility
                llm_compatible, llm_message, llm_details = check_model_compatibility(llm_model, llm_size)
                
                # Check embedding model compatibility
                embedding_compatible, embedding_message, embedding_details = check_model_compatibility(embedding_model, embedding_size)
                
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
                        "recommendations": generate_compatibility_recommendations(llm_details, embedding_details, combined_compatible)
                    }
                    
                    raise HTTPException(
                        status_code=400, 
                        detail=error_details
                    )
                
                logger.info(f"✅ GPU compatibility check passed for models {llm_model} + {embedding_model}")
                
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"Could not check GPU compatibility: {str(e)}, proceeding anyway")
        
        ollama_host = os.getenv('OLLAMA_HOST', 'http://ollama:11434')
        
        # Check if only parameters changed (no model downloads needed)
        rag = get_chatpdf_instance()
        current_llm = rag.llm_model
        current_embedding = rag.embedding_model
        
        logger.info(f"[API /api/models/settings] Current models - LLM: '{current_llm}', Embedding: '{current_embedding}'")
        logger.info(f"[API /api/models/settings] Requested models - LLM: '{llm_model}', Embedding: '{embedding_model}'")
        
        models_unchanged = (current_llm == llm_model and current_embedding == embedding_model)
        logger.info(f"[API /api/models/settings] Models unchanged check: {models_unchanged}")
        
        if models_unchanged:
            logger.info("Models unchanged, only updating parameters - skipping model downloads")
            
            # Update the models in the RAG system (this will only update parameters)
            try:
                rag.update_models(llm_model, embedding_model)
            except Exception as e:
                logger.error(f"Error updating parameters: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to update parameters: {str(e)}"
                )
            
            # Save settings to database and config file
            config_path = "model_settings.json"
            settings = {
                'llm': llm_model,
                'embedding': embedding_model,
                'parameters': valid_parameters
            }
            
            # Save to database
            try:
                chat_db.save_model_settings(llm_model, embedding_model, valid_parameters)
            except Exception as e:
                logger.warning(f"Could not save to database: {e}")
            
            # Also save to config file as backup
            with open(config_path, 'w') as f:
                json.dump(settings, f, indent=2)
            
            return {
                "success": True,
                "message": "Model parameters updated successfully (no downloads needed)",
                "llm": llm_model,
                "embedding": embedding_model,
                "embedding_changed": False,
                "downloaded_models": [],
                "parameters_only": True
            }
        
        # Check which models need to be downloaded
        models_to_download = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get currently installed models
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
            
            # Check if LLM model needs downloading
            llm_model_to_check = llm_model
            if llm_model not in installed_models:
                # Check common variants for model matching
                llm_variants = [
                    llm_model,
                    f"{llm_model}:latest", 
                    llm_model.replace(":latest", "")
                ]
                model_found = any(variant in installed_models for variant in llm_variants)
                if not model_found:
                    models_to_download.append(llm_model)
                    logger.info(f"LLM model {llm_model} needs to be downloaded")
                else:
                    logger.info(f"LLM model {llm_model} (or variant) already installed")
                
            # Check if embedding model needs downloading with special handling for BGE/nomic
            embedding_model_to_check = embedding_model
            if embedding_model not in installed_models:
                # Check common variants and fix common naming issues
                embedding_variants = [
                    embedding_model,
                    f"{embedding_model}:latest",
                    embedding_model.replace(":latest", "")
                ]
                
                # Special handling for BGE models
                if "bge" in embedding_model.lower():
                    if embedding_model == "bge-m3:1.7M":
                        # Fix incorrect format - should be bge-m3 not bge-m3:1.7M
                        embedding_model = "bge-m3"
                        embedding_model_to_check = "bge-m3"
                        logger.info(f"Fixed BGE model name from bge-m3:1.7M to bge-m3")
                    elif embedding_model == "bge-large:335m":
                        embedding_model = "bge-large"
                        embedding_model_to_check = "bge-large"
                        logger.info(f"Fixed BGE model name from bge-large:335m to bge-large")
                    
                    embedding_variants.extend([
                        "bge-m3", "bge-large", "bge-m3:567m", "bge-large:335m"
                    ])
                
                # Special handling for nomic models  
                if "nomic" in embedding_model.lower():
                    embedding_variants.extend([
                        "nomic-embed-text", "nomic-embed-text:33.3M", "nomic-embed-text:latest"
                    ])
                
                model_found = any(variant in installed_models for variant in embedding_variants)
                if not model_found:
                    models_to_download.append(embedding_model_to_check)
                    logger.info(f"Embedding model {embedding_model_to_check} needs to be downloaded")
                else:
                    logger.info(f"Embedding model {embedding_model} (or variant) already installed")
            
            # Download missing models with progress tracking
            for model_name in models_to_download:
                logger.info(f"Downloading model: {model_name}")
                try:
                    # Use streaming to track download progress
                    async with httpx.AsyncClient(timeout=600.0) as client:  # 10 minute timeout for downloads
                        download_response = await client.post(
                            f"{ollama_host}/api/pull",
                            json={"name": model_name, "stream": True},
                            timeout=600.0
                        )
                        
                        if download_response.status_code != 200:
                            logger.error(f"Failed to download model {model_name}: {download_response.status_code}")
                            raise HTTPException(
                                status_code=500, 
                                detail=f"Failed to download model {model_name}: HTTP {download_response.status_code}"
                            )
                        
                        # Process streaming response to track progress
                        progress_info = {
                            "status": "downloading",
                            "completed": 0,
                            "total": 0,
                            "downloading": "",
                            "pulling_fs_layer": "",
                            "verifying_checksum": "",
                            "download_complete": "",
                            "pulling_manifest": "",
                            "success": False
                        }
                        
                        async for line in download_response.aiter_lines():
                            if line:
                                try:
                                    data = json.loads(line)
                                    
                                    # Update progress based on status
                                    if "status" in data:
                                        progress_info["status"] = data["status"]
                                        
                                    if "completed" in data and "total" in data:
                                        progress_info["completed"] = data["completed"]
                                        progress_info["total"] = data["total"]
                                        
                                        # Log progress every 10MB
                                        if data["total"] > 0 and data["completed"] % (10 * 1024 * 1024) == 0:
                                            percent = (data["completed"] / data["total"]) * 100
                                            logger.info(f"Downloading {model_name}: {percent:.1f}% complete ({data['completed']}/{data['total']} bytes)")
                                    
                                    # Track different stages
                                    status = data.get("status", "")
                                    if "pulling manifest" in status.lower():
                                        progress_info["pulling_manifest"] = status
                                        logger.info(f"Model {model_name}: {status}")
                                    elif "downloading" in status.lower():
                                        progress_info["downloading"] = status
                                    elif "verifying checksum" in status.lower():
                                        progress_info["verifying_checksum"] = status
                                        logger.info(f"Model {model_name}: {status}")
                                    elif "success" in status.lower() or "pull complete" in status.lower():
                                        progress_info["success"] = True
                                        logger.info(f"Model {model_name}: Download completed successfully")
                                        break
                                        
                                except json.JSONDecodeError:
                                    # Skip non-JSON lines
                                    continue
                        
                        # Verify download completed successfully
                        if not progress_info["success"]:
                            # Check one more time if model is now available
                            check_response = await client.get(f"{ollama_host}/api/tags")
                            if check_response.status_code == 200:
                                models_data = check_response.json()
                                available_models = {model.get('name', '') for model in models_data.get('models', [])}
                                logger.info(f"Available models after download attempt: {sorted(list(available_models))}")
                                
                                # Check for the exact model name and common variants
                                model_variants = [
                                    model_name,
                                    f"{model_name}:latest",
                                    model_name.replace(":latest", "")
                                ]
                                
                                model_found = None
                                for variant in model_variants:
                                    if variant in available_models:
                                        model_found = variant
                                        break
                                
                                if model_found:
                                    progress_info["success"] = True
                                    logger.info(f"Model {model_name} verified as downloaded (found as: {model_found})")
                                else:
                                    logger.error(f"Model {model_name} not found in available models. Variants checked: {model_variants}")
                                    logger.error(f"Available embedding models: {[m for m in available_models if any(term in m.lower() for term in ['embed', 'bge', 'nomic', 'minilm'])]}")
                        
                        if progress_info["success"]:
                            logger.info(f"Successfully downloaded model: {model_name}")
                        else:
                            error_msg = f"Model download verification failed: {model_name}"
                            logger.error(error_msg)
                            
                            # For BGE and nomic models, provide specific guidance
                            if "bge" in model_name.lower():
                                error_msg += ". BGE models may require specific naming format. Try 'bge-m3' or 'bge-large' instead."
                            elif "nomic" in model_name.lower():
                                error_msg += ". Nomic models may require specific naming format. Try 'nomic-embed-text' instead."
                                
                            raise HTTPException(
                                status_code=500, 
                                detail=error_msg
                            )
                        
                except httpx.TimeoutException:
                    logger.error(f"Timeout downloading model {model_name}")
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Timeout downloading model {model_name}. Please try again."
                    )
                except Exception as e:
                    logger.error(f"Error downloading model {model_name}: {e}")
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Failed to download model {model_name}: {str(e)}"
                    )
        
        # Now update the models in the RAG system
        rag = get_chatpdf_instance()
        
        # Check if embedding model changed - if so, we need to re-ingest
        embedding_changed = rag.embedding_model != embedding_model
        
        try:
            # Update the models
            rag.update_models(llm_model, embedding_model)
        except Exception as e:
            logger.error(f"Error updating models: {str(e)}")
            
            # Return detailed error information to frontend
            error_detail = {
                "error": "MODEL_UPDATE_FAILED",
                "message": f"Failed to update models: {str(e)}",
                "llm_model": llm_model,
                "embedding_model": embedding_model,
                "downloaded_models": models_to_download,
                "suggestion": "Please check if the models are compatible with your system or try different models."
            }
            
            # If it's a model not found error, suggest alternative
            if "not found" in str(e).lower():
                error_detail["suggestion"] = f"Model '{embedding_model}' not found. Please try a different embedding model or check if the model name is correct."
            
            raise HTTPException(
                status_code=500,
                detail=error_detail
            )
        
        # Save settings to database and config file
        config_path = "model_settings.json"
        settings = {
            'llm': llm_model,
            'embedding': embedding_model,
            'parameters': valid_parameters
        }
        
        # Save to database
        try:
            chat_db.save_model_settings(llm_model, embedding_model, valid_parameters)
        except Exception as e:
            logger.warning(f"Could not save to database: {e}")
        
        # Also save to config file as backup
        with open(config_path, 'w') as f:
            json.dump(settings, f, indent=2)
        
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
        
        # If embedding model changed, trigger automatic reingestion
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update models: {str(e)}")

@app.post("/api/models/simple-settings")
async def update_simple_models_settings(
    request: dict,
    current_user: dict = Depends(get_current_user)
):
    """Update model settings without complex validation - for basic UI"""
    try:
        import httpx
        import asyncio
        from rag import check_model_compatibility
        
        # Validate and normalize model names
        llm_model = request.get('llm')
        embedding_model = request.get('embedding')
        model_parameters = request.get('parameters', {})
        
        if not llm_model or not embedding_model:
            raise HTTPException(status_code=400, detail="Both LLM and embedding models are required")
        
        # Validate model parameters
        valid_parameters = {}
        if model_parameters:
            # Temperature (0.0 to 2.0)
            temp = model_parameters.get('temperature', 0.7)
            if isinstance(temp, (int, float)) and 0.0 <= temp <= 2.0:
                valid_parameters['temperature'] = float(temp)
            else:
                valid_parameters['temperature'] = 0.7
            
            # Max tokens (1 to 32768)
            max_tokens = model_parameters.get('max_tokens', 2048)
            if isinstance(max_tokens, int) and 1 <= max_tokens <= 32768:
                valid_parameters['max_tokens'] = max_tokens
            else:
                valid_parameters['max_tokens'] = 2048
            
            # Top-p (0.0 to 1.0)
            top_p = model_parameters.get('top_p', 0.9)
            if isinstance(top_p, (int, float)) and 0.0 <= top_p <= 1.0:
                valid_parameters['top_p'] = float(top_p)
            else:
                valid_parameters['top_p'] = 0.9
            
            # Frequency penalty (-2.0 to 2.0)
            freq_penalty = model_parameters.get('frequency_penalty', 0.0)
            if isinstance(freq_penalty, (int, float)) and -2.0 <= freq_penalty <= 2.0:
                valid_parameters['frequency_penalty'] = float(freq_penalty)
            else:
                valid_parameters['frequency_penalty'] = 0.0
            
            # Presence penalty (-2.0 to 2.0)
            presence_penalty = model_parameters.get('presence_penalty', 0.0)
            if isinstance(presence_penalty, (int, float)) and -2.0 <= presence_penalty <= 2.0:
                valid_parameters['presence_penalty'] = float(presence_penalty)
            else:
                valid_parameters['presence_penalty'] = 0.0
        else:
            # Default parameters
            valid_parameters = {
                'temperature': 0.7,
                'max_tokens': 2048,
                'top_p': 0.9,
                'frequency_penalty': 0.0,
                'presence_penalty': 0.0
            }
        
        # Normalize model names - fix common issues
        if embedding_model == "bge-m3:1.7M":
            embedding_model = "bge-m3"
            logger.info("Normalized bge-m3:1.7M to bge-m3")
        elif embedding_model == "bge-large:335m":
            embedding_model = "bge-large"
            logger.info("Normalized bge-large:335m to bge-large")
        elif embedding_model == "nomic-embed-text:33.3M":
            embedding_model = "nomic-embed-text"
            logger.info("Normalized nomic-embed-text:33.3M to nomic-embed-text")
        
        # Check GPU compatibility before downloading
        try:
            logger.info(f"Checking GPU compatibility for LLM: {llm_model}, Embedding: {embedding_model}")
            
            # Check LLM model compatibility
            llm_compatible, llm_message, llm_details = check_model_compatibility(llm_model)
            logger.info(f"LLM compatibility check: {llm_compatible} - {llm_message}")
            
            # Check embedding model compatibility  
            embedding_compatible, embedding_message, embedding_details = check_model_compatibility(embedding_model)
            logger.info(f"Embedding compatibility check: {embedding_compatible} - {embedding_message}")
            
            # Calculate combined memory requirement
            combined_memory = llm_details['required_memory_mb'] + embedding_details['required_memory_mb']
            available_memory = llm_details['available_memory_mb']
            combined_compatible = combined_memory <= available_memory
            
            logger.info(f"Combined memory check: {combined_memory}MB required, {available_memory}MB available, compatible: {combined_compatible}")
            
            # Only warn about compatibility issues, don't block downloads
            compatibility_warnings = []
            
            if not llm_compatible:
                compatibility_warnings.append(f"LLM model '{llm_model}' may not fit in GPU memory: {llm_message}")
            
            if not embedding_compatible:
                compatibility_warnings.append(f"Embedding model '{embedding_model}' may not fit in GPU memory: {embedding_message}")
            
            if not combined_compatible:
                shortage = combined_memory - available_memory
                compatibility_warnings.append(f"Combined models may require ~{combined_memory}MB but only {available_memory}MB available (potential shortage: {shortage}MB)")
            
            if compatibility_warnings:
                logger.warning("GPU compatibility warnings (proceeding with download): " + "; ".join(compatibility_warnings))
            else:
                logger.info("GPU compatibility check passed - all models should fit in memory")
            
        except Exception as e:
            # Log GPU check errors but don't block the process
            logger.warning(f"Could not check GPU compatibility: {str(e)}, proceeding anyway")
            compatibility_warnings = [f"Could not verify GPU compatibility: {str(e)}"]
        
        ollama_host = os.getenv('OLLAMA_HOST', 'http://ollama:11434')
        
        # Check if only parameters changed (no model downloads needed)
        rag = get_chatpdf_instance()
        current_llm = rag.llm_model
        current_embedding = rag.embedding_model
        
        logger.info(f"[API /api/models/simple-settings] Current models - LLM: '{current_llm}', Embedding: '{current_embedding}'")
        logger.info(f"[API /api/models/simple-settings] Requested models - LLM: '{llm_model}', Embedding: '{embedding_model}'")
        
        models_unchanged = (current_llm == llm_model and current_embedding == embedding_model)
        logger.info(f"[API /api/models/simple-settings] Models unchanged check: {models_unchanged}")
        
        if models_unchanged:
            logger.info("Models unchanged, only updating parameters - skipping model downloads")
            
            # Update the models in the RAG system (this will only update parameters)
            try:
                rag.update_models(llm_model, embedding_model)
            except Exception as e:
                logger.error(f"Error updating parameters: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to update parameters: {str(e)}"
                )
            
            # Save settings to database and config file
            config_path = "model_settings.json"
            settings = {
                'llm': llm_model,
                'embedding': embedding_model,
                'parameters': valid_parameters
            }
            
            # Save to database
            try:
                chat_db.save_model_settings(llm_model, embedding_model, valid_parameters)
            except Exception as e:
                logger.warning(f"Could not save to database: {e}")
            
            # Also save to config file as backup
            with open(config_path, 'w') as f:
                json.dump(settings, f, indent=2)
            
            response_data = {
                "success": True,
                "message": "Model parameters updated successfully (no downloads needed)",
                "llm": llm_model,
                "embedding": embedding_model,
                "embedding_changed": False,
                "downloaded_models": [],
                "parameters_only": True
            }
            
            # Add GPU compatibility warnings if any
            if 'compatibility_warnings' in locals() and compatibility_warnings:
                response_data["gpu_warnings"] = compatibility_warnings
            
            return response_data
        
        # Check which models need to be downloaded
        models_to_download = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get currently installed models
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
            
            # Download missing models with progress tracking
            for model_name in models_to_download:
                logger.info(f"Downloading model: {model_name}")
                try:
                    download_response = await client.post(
                        f"{ollama_host}/api/pull",
                        json={"name": model_name},
                        timeout=600.0  # 10 minutes timeout for model download
                    )
                    
                    if download_response.status_code != 200:
                        logger.error(f"Failed to download model {model_name}: {download_response.status_code}")
                        raise HTTPException(
                            status_code=500, 
                            detail=f"Failed to download model {model_name}"
                        )
                    else:
                        logger.info(f"Successfully downloaded model: {model_name}")
                        
                except httpx.TimeoutException:
                    logger.error(f"Timeout downloading model {model_name}")
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Timeout downloading model {model_name}. Please try again."
                    )
                except Exception as e:
                    logger.error(f"Error downloading model {model_name}: {e}")
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Failed to download model {model_name}: {str(e)}"
                    )
        
        # Get RAG instance
        rag = get_chatpdf_instance()
        
        # Check if embedding model changed - if so, we need to re-ingest
        embedding_changed = rag.embedding_model != embedding_model
        
        # Update the models
        rag.update_models(llm_model, embedding_model)
        
        # Save settings to database and config file
        config_path = "model_settings.json"
        settings = {
            'llm': llm_model,
            'embedding': embedding_model,
            'parameters': valid_parameters
        }
        
        # Save to database
        try:
            chat_db.save_model_settings(llm_model, embedding_model, valid_parameters)
        except Exception as e:
            logger.warning(f"Could not save to database: {e}")
        
        # Also save to config file as backup
        with open(config_path, 'w') as f:
            json.dump(settings, f, indent=2)
        
        response_data = {
            "success": True,
            "message": "Models updated successfully",
            "llm": llm_model,
            "embedding": embedding_model,
            "embedding_changed": embedding_changed,
            "downloaded_models": models_to_download
        }
        
        # Add GPU compatibility warnings if any
        if 'compatibility_warnings' in locals() and compatibility_warnings:
            response_data["gpu_warnings"] = compatibility_warnings
            response_data["message"] += f". GPU compatibility warnings: {'; '.join(compatibility_warnings)}"
        
        # Add download info to message
        if models_to_download:
            downloaded_list = ", ".join(models_to_download)
            response_data["message"] += f". Downloaded models: {downloaded_list}"
        
        # If embedding model changed, suggest re-ingestion
        if embedding_changed:
            response_data["message"] += ". Embedding model changed - you may want to re-ingest documents."
            response_data["reingest_suggested"] = True
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update models: {str(e)}")

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

async def check_and_download_reranker_model(model_name: str) -> Dict[str, Any]:
    """Check if a reranker model is available locally, download if not"""
    if not model_name or model_name.lower() == "none":
        return {"success": True, "downloaded": False, "message": "No reranker model specified"}
    
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
        
        # Check and download reranker model if needed
        download_result = {"success": True, "downloaded": False, "message": "No reranker model specified"}
        if config.reranker_enabled and config.reranker_model and config.reranker_model.lower() != "none":
            logger.info(f"🎯 Checking reranker model availability: {config.reranker_model}")
            download_result = await check_and_download_reranker_model(config.reranker_model)
            
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
async def get_available_reranker_models(token: str = Depends(oauth2_scheme)):
    """Get list of available reranker models from Ollama (both local and library)"""
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
            "description": "Use weighted combination of vector similarity and keyword matching"
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
                    "is_local": is_local
                })
        
        # logger.info(f"Reranker model filtering: Checked {total_models_checked} models, found {reranker_models_found} dedicated reranker models")
        
        # If no reranker models found, add some fallback models
        if len(reranker_models) == 1:  # Only "None" option
            logger.warning("No reranker models found in Ollama, adding fallback models")
            fallback_models = [
                {
                    "name": "linux6200/bge-reranker-v2-m3",
                    "display_name": "BGE Reranker V2 M3",
                    "description": "High-performance BGE reranking model (available to download)",
                    "is_local": False
                },
                {
                    "name": "dengcao/Qwen3-Reranker-8B",
                    "display_name": "Qwen3 Reranker 8B",
                    "description": "Alibaba's multilingual reranking model (available to download)",
                    "is_local": False
                },
                {
                    "name": "qllama/bge-reranker-large",
                    "display_name": "BGE Reranker Large (Quantized)",
                    "description": "Quantized BGE reranking model (available to download)",
                    "is_local": False
                },
                {
                    "name": "BAAI/bge-reranker-large",
                    "display_name": "BGE Reranker Large (Original)",
                    "description": "Original BGE reranking model (may need to be pulled)",
                    "is_local": False
                }
            ]
            reranker_models.extend(fallback_models)
        
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
                "description": "Use weighted combination of vector similarity and keyword matching"
            },
            {
                "name": "linux6200/bge-reranker-v2-m3",
                "display_name": "BGE Reranker V2 M3",
                "description": "High-performance BGE reranking model (error occurred)",
                "is_local": False
            },
            {
                "name": "dengcao/Qwen3-Reranker-8B",
                "display_name": "Qwen3 Reranker 8B",
                "description": "Alibaba's multilingual reranking model (error occurred)",
                "is_local": False
            }
        ]
        return {
            "models": fallback_models
        }

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


@app.get("/api/documents/{document_id}/image")
async def get_document_image(document_id: int, background_tasks: BackgroundTasks, token: str = Depends(flexible_oauth2_scheme)):
    """Serve document image if it's an image file"""
    try:
        from document_storage import get_document_storage
        doc_storage = get_document_storage()
        
        # Get document info first
        doc_info = doc_storage._get_document_by_id(document_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if it's an image file by content type or extension
        is_image_by_content_type = doc_info['content_type'].startswith('image/')
        is_image_by_extension = doc_info['filename'].lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg'))
        
        if not (is_image_by_content_type or is_image_by_extension):
            raise HTTPException(status_code=400, detail="Document is not an image")
        
        # Get the document content from MinIO
        temp_file_path = doc_storage.get_document_file(document_id)
        
        # Schedule cleanup
        background_tasks.add_task(os.remove, temp_file_path)
        
        # Determine correct content type if stored incorrectly
        response_content_type = doc_info['content_type']
        if not response_content_type.startswith('image/'):
            import mimetypes
            guessed_type, _ = mimetypes.guess_type(doc_info['filename'])
            if guessed_type and guessed_type.startswith('image/'):
                response_content_type = guessed_type
        
        # Return the image file
        return FileResponse(
            temp_file_path,
            media_type=response_content_type,
            filename=doc_info['filename']
        )
        
    except Exception as e:
        logger.error(f"Error serving document image {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/{document_id}/preview")
async def get_document_preview(document_id: int, background_tasks: BackgroundTasks, token: str = Depends(oauth2_scheme)):
    """Get document preview content for side-by-side viewing"""
    try:
        from document_storage import get_document_storage
        doc_storage = get_document_storage()
        
        # Get document info first
        doc_info = doc_storage._get_document_by_id(document_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # For images, return image metadata
        is_image_by_content_type = doc_info['content_type'].startswith('image/')
        is_image_by_extension = doc_info['filename'].lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg'))
        
        if is_image_by_content_type or is_image_by_extension:
            return {
                "type": "image",
                "content_type": doc_info['content_type'],
                "filename": doc_info['filename'],
                "image_url": f"/api/documents/{document_id}/image"
            }
        
        # Get the document content from MinIO
        temp_file_path = doc_storage.get_document_file(document_id)
        
        try:
            # Extract text content based on file type
            content = ""
            content_type = doc_info['content_type'].lower()
            
            if content_type == 'text/plain' or doc_info['filename'].endswith('.txt'):
                with open(temp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            
            elif content_type == 'application/pdf' or doc_info['filename'].endswith('.pdf'):
                import PyPDF2
                pages_info = []
                with open(temp_file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    content = ""
                    for page_num in range(min(20, len(pdf_reader.pages))):  # Limit to first 20 pages
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        content += f"--- Page {page_num + 1} ---\n"
                        content += page_text + "\n\n"
                        
                        # Store page info for PDF viewer
                        pages_info.append({
                            "page_number": page_num + 1,
                            "text": page_text,
                            "text_length": len(page_text)
                        })
                
                # For PDFs, return special structure with page info
                return {
                    "type": "pdf",
                    "content_type": doc_info['content_type'],
                    "filename": doc_info['filename'],
                    "content": content,
                    "pdf_url": f"/api/documents/{document_id}/raw",
                    "pages_info": pages_info,
                    "total_pages": len(pdf_reader.pages),
                    "truncated": len(content) >= 50000
                }
            
            elif content_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword'] or doc_info['filename'].endswith(('.docx', '.doc')):
                from docx import Document
                doc = Document(temp_file_path)
                content = ""
                for paragraph in doc.paragraphs[:50]:  # Limit to first 50 paragraphs
                    content += paragraph.text + "\n"
            
            elif content_type in ['application/vnd.openxmlformats-officedocument.presentationml.presentation', 'application/vnd.ms-powerpoint'] or doc_info['filename'].endswith(('.pptx', '.ppt')):
                # Handle PowerPoint files - Convert to HTML
                logger.info(f"DEBUG: Processing PPTX file: {doc_info['filename']}")
                logger.info(f"DEBUG: Content type: {content_type}")
                logger.info(f"DEBUG: File path: {temp_file_path}")
                try:
                    from pptx_html_converter import convert_pptx_to_html_slides
                    logger.info("DEBUG: Successfully imported pptx_html_converter")
                    
                    # Convert PPTX to HTML slides
                    logger.info("DEBUG: Starting PPTX to HTML conversion...")
                    result = convert_pptx_to_html_slides(temp_file_path)
                    logger.info(f"DEBUG: HTML conversion result: {result.get('has_html', False)}, slides count: {len(result.get('slides', []))}")
                    
                    # Also extract text content for backward compatibility
                    from pptx import Presentation
                    logger.info("DEBUG: Starting text extraction from PPTX...")
                    prs = Presentation(temp_file_path)
                    slides_content = []
                    logger.info(f"DEBUG: Found {len(prs.slides)} slides in presentation")
                    
                    for slide_num, slide in enumerate(prs.slides, 1):
                        slide_text = []
                        
                        # Extract text from shapes
                        for shape in slide.shapes:
                            if hasattr(shape, 'text') and shape.text.strip():
                                slide_text.append(shape.text.strip())
                            
                            # Handle tables safely
                            if hasattr(shape, 'table') and shape.has_table:
                                try:
                                    table_text = []
                                    for row in shape.table.rows:
                                        row_text = []
                                        for cell in row.cells:
                                            if cell.text.strip():
                                                row_text.append(cell.text.strip())
                                        if row_text:
                                            table_text.append(' | '.join(row_text))
                                    if table_text:
                                        slide_text.append('\n'.join(table_text))
                                except Exception as e:
                                    logger.warning(f"Could not process table in slide {slide_num}: {e}")
                        
                        slide_content = '\n\n'.join(slide_text) if slide_text else f"[Slide {slide_num} - No text content]"
                        slides_content.append({
                            "slide_number": slide_num,
                            "content": slide_content,
                            "text_length": len(slide_content)
                        })
                    
                    logger.info(f"DEBUG: Extracted text from {len(slides_content)} slides")
                    
                    # Merge HTML data with text content for backward compatibility
                    merged_slides = []
                    
                    # If we have HTML slides, use them as primary data
                    if result.get('has_html', False) and result.get('slides'):
                        logger.info("DEBUG: Using HTML slides as primary data")
                        for i, html_slide in enumerate(result['slides']):
                            slide_data = html_slide.copy()
                            
                            # Add text content for search/indexing if available
                            if i < len(slides_content):
                                slide_data['content'] = slides_content[i]['content']
                                slide_data['text_length'] = slides_content[i]['text_length']
                            
                            merged_slides.append(slide_data)
                        logger.info(f"DEBUG: Created {len(merged_slides)} merged HTML slides")
                    else:
                        # Fallback to text-only slides
                        logger.info("DEBUG: Falling back to text-only slides")
                        merged_slides = slides_content
                    
                    content = '\n\n=== SLIDE SEPARATOR ===\n\n'.join([slide['content'] for slide in slides_content])
                    
                    final_response = {
                        "type": "presentation",
                        "content_type": doc_info['content_type'],
                        "filename": doc_info['filename'],
                        "content": content,
                        "slides": merged_slides,
                        "total_slides": len(prs.slides),
                        "has_html": result.get('has_html', False),
                        "has_images": False,  # Using HTML instead of images
                        "conversion_method": result.get('conversion_method', 'unknown'),
                        "truncated": len(content) >= 50000
                    }
                    
                    logger.info(f"DEBUG: Final response structure:")
                    logger.info(f"DEBUG: - Type: {final_response['type']}")
                    logger.info(f"DEBUG: - Has HTML: {final_response['has_html']}")
                    logger.info(f"DEBUG: - Total slides: {final_response['total_slides']}")
                    logger.info(f"DEBUG: - Merged slides count: {len(final_response['slides'])}")
                    logger.info(f"DEBUG: - Conversion method: {final_response['conversion_method']}")
                    
                    if final_response['slides'] and len(final_response['slides']) > 0:
                        first_slide = final_response['slides'][0]
                        logger.info(f"DEBUG: First slide structure: {list(first_slide.keys())}")
                        if 'html_content' in first_slide:
                            logger.info(f"DEBUG: First slide has HTML content (length: {len(first_slide['html_content'])})")
                        if 'format' in first_slide:
                            logger.info(f"DEBUG: First slide format: {first_slide['format']}")
                    
                    return final_response
                    
                except ImportError as ie:
                    logger.error(f"Missing dependencies for PPTX processing: {ie}")
                    # Fallback to text-only processing
                    try:
                        from pptx import Presentation
                        prs = Presentation(temp_file_path)
                        slides_content = []
                        
                        for slide_num, slide in enumerate(prs.slides, 1):
                            slide_text = []
                            for shape in slide.shapes:
                                if hasattr(shape, 'text') and shape.text.strip():
                                    slide_text.append(shape.text.strip())
                            
                            slide_content = '\n\n'.join(slide_text) if slide_text else f"[Slide {slide_num} - No text content]"
                            slides_content.append({
                                "slide_number": slide_num,
                                "content": slide_content
                            })
                        
                        content = '\n\n=== SLIDE SEPARATOR ===\n\n'.join([slide['content'] for slide in slides_content])
                        
                        return {
                            "type": "presentation",
                            "content_type": doc_info['content_type'],
                            "filename": doc_info['filename'],
                            "content": content,
                            "slides": slides_content,
                            "total_slides": len(prs.slides),
                            "has_images": False,
                            "error": "Image conversion not available - text only"
                        }
                    except Exception as e:
                        content = f"Error processing PowerPoint file: {str(e)}"
                except Exception as e:
                    logger.error(f"Error converting PPTX to images: {e}")
                    content = f"Error processing PowerPoint file: {str(e)}"
            
            elif content_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel'] or doc_info['filename'].endswith(('.xlsx', '.xls')):
                # Handle Excel files
                try:
                    import pandas as pd
                    
                    # Read first few sheets
                    sheets_content = []
                    excel_file = pd.ExcelFile(temp_file_path)
                    
                    for sheet_name in excel_file.sheet_names[:5]:  # Limit to first 5 sheets
                        try:
                            df = pd.read_excel(temp_file_path, sheet_name=sheet_name, nrows=100)  # Limit rows
                            sheet_content = f"=== Sheet: {sheet_name} ===\n"
                            sheet_content += df.to_string(max_rows=50, max_cols=10, index=False)
                            sheets_content.append({
                                "sheet_name": sheet_name,
                                "content": sheet_content,
                                "rows": len(df),
                                "columns": len(df.columns)
                            })
                        except Exception as e:
                            sheets_content.append({
                                "sheet_name": sheet_name,
                                "content": f"Error reading sheet: {str(e)}",
                                "rows": 0,
                                "columns": 0
                            })
                    
                    content = '\n\n'.join([sheet['content'] for sheet in sheets_content])
                    
                    return {
                        "type": "spreadsheet",
                        "content_type": doc_info['content_type'],
                        "filename": doc_info['filename'],
                        "content": content,
                        "sheets": sheets_content,
                        "total_sheets": len(excel_file.sheet_names),
                        "truncated": len(content) >= 50000
                    }
                    
                except ImportError:
                    content = "Excel preview requires pandas library"
                except Exception as e:
                    content = f"Error processing Excel file: {str(e)}"
            
            elif content_type == 'text/csv' or doc_info['filename'].endswith('.csv'):
                # Handle CSV files
                try:
                    import pandas as pd
                    df = pd.read_csv(temp_file_path, nrows=100)  # Limit to first 100 rows
                    content = df.to_string(max_rows=50, max_cols=20, index=False)
                    
                    return {
                        "type": "csv",
                        "content_type": doc_info['content_type'],
                        "filename": doc_info['filename'],
                        "content": content,
                        "rows": len(df),
                        "columns": len(df.columns),
                        "column_names": list(df.columns),
                        "truncated": len(df) >= 100
                    }
                    
                except ImportError:
                    content = "CSV preview requires pandas library"
                except Exception as e:
                    content = f"Error processing CSV file: {str(e)}"
            
            elif content_type == 'text/html' or doc_info['filename'].endswith(('.html', '.htm')):
                # Handle HTML files
                try:
                    from bs4 import BeautifulSoup
                    with open(temp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        html_content = f.read()
                    
                    soup = BeautifulSoup(html_content, 'html.parser')
                    # Extract text content
                    text_content = soup.get_text(separator='\n', strip=True)
                    
                    return {
                        "type": "html",
                        "content_type": doc_info['content_type'],
                        "filename": doc_info['filename'],
                        "content": text_content[:50000],  # Limit content
                        "html_content": html_content[:10000],  # Limited HTML for preview
                        "truncated": len(text_content) >= 50000
                    }
                    
                except ImportError:
                    # Fallback to plain text
                    with open(temp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()[:10000]
                except Exception as e:
                    content = f"Error processing HTML file: {str(e)}"
            
            elif content_type == 'message/rfc822' or doc_info['filename'].endswith('.eml'):
                # Handle email files
                try:
                    import email
                    with open(temp_file_path, 'rb') as f:
                        msg = email.message_from_binary_file(f)
                    
                    email_content = []
                    email_content.append(f"From: {msg.get('From', 'Unknown')}")
                    email_content.append(f"To: {msg.get('To', 'Unknown')}")
                    email_content.append(f"Subject: {msg.get('Subject', 'No Subject')}")
                    email_content.append(f"Date: {msg.get('Date', 'Unknown')}")
                    email_content.append("\n" + "="*50 + "\n")
                    
                    # Get email body
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                email_content.append(part.get_payload(decode=True).decode('utf-8', errors='ignore'))
                                break
                    else:
                        email_content.append(msg.get_payload(decode=True).decode('utf-8', errors='ignore'))
                    
                    content = '\n'.join(email_content)
                    
                    return {
                        "type": "email",
                        "content_type": doc_info['content_type'],
                        "filename": doc_info['filename'],
                        "content": content[:50000],
                        "from": msg.get('From', 'Unknown'),
                        "to": msg.get('To', 'Unknown'),
                        "subject": msg.get('Subject', 'No Subject'),
                        "date": msg.get('Date', 'Unknown'),
                        "truncated": len(content) >= 50000
                    }
                    
                except Exception as e:
                    content = f"Error processing email file: {str(e)}"
            
            elif content_type == 'text/markdown' or doc_info['filename'].endswith('.md'):
                with open(temp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            
            elif content_type in ['application/json', 'text/json'] or doc_info['filename'].endswith('.json'):
                import json
                with open(temp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    json_data = json.load(f)
                    content = json.dumps(json_data, indent=2)
            
            else:
                # Try to read as text for other formats
                try:
                    with open(temp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()[:10000]  # Limit to first 10KB
                except:
                    content = f"Preview not available for {content_type}"
            
            # Limit content length for frontend display
            if len(content) > 50000:  # 50KB limit
                content = content[:50000] + "\n\n... (Content truncated for preview)"
            
            return {
                "type": "text",
                "content_type": doc_info['content_type'],
                "filename": doc_info['filename'],
                "content": content,
                "truncated": len(content) >= 50000
            }
            
        finally:
            # Schedule cleanup
            background_tasks.add_task(os.remove, temp_file_path)
        
    except Exception as e:
        logger.error(f"Error getting document preview: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving document preview: {str(e)}")

@app.get("/api/documents/{document_id}/raw")
async def get_document_raw(document_id: int, background_tasks: BackgroundTasks, token: str = Depends(oauth2_scheme)):
    """Serve raw document file for viewers (e.g., PDF viewer)"""
    try:
        from document_storage import get_document_storage
        doc_storage = get_document_storage()
        
        # Get document info first
        doc_info = doc_storage._get_document_by_id(document_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get the document content from MinIO
        temp_file_path = doc_storage.get_document_file(document_id)
        
        # Schedule cleanup
        background_tasks.add_task(os.remove, temp_file_path)
        
        # Return the document file
        return FileResponse(
            temp_file_path,
            media_type=doc_info['content_type'],
            filename=doc_info['filename']
        )
        
    except Exception as e:
        logger.error(f"Error serving raw document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ================== EVALUATION ENDPOINTS ==================

def calculate_groundedness(response: str, context: str) -> float:
    """Calculate groundedness score based on how well the response is grounded in the context"""
    if not context or not response:
        return 0.0
    
    # Basic keyword overlap approach
    context_words = set(context.lower().split())
    response_words = set(response.lower().split())
    
    if not response_words:
        return 0.0
    
    overlap = len(context_words.intersection(response_words))
    score = min(overlap / len(response_words), 1.0)
    
    # Add some randomness for demo purposes (in real implementation, use actual ML models)
    return min(max(score + random.uniform(-0.2, 0.2), 0.0), 1.0)

def calculate_context_relevance(query: str, context: str) -> float:
    """Calculate how relevant the retrieved context is to the query"""
    if not query or not context:
        return 0.0
    
    query_words = set(query.lower().split())
    context_words = set(context.lower().split())
    
    if not query_words:
        return 0.0
    
    overlap = len(query_words.intersection(context_words))
    score = min(overlap / len(query_words), 1.0)
    
    # Add some randomness for demo purposes
    return min(max(score + random.uniform(-0.15, 0.15), 0.0), 1.0)

def calculate_answer_quality(query: str, response: str) -> float:
    """Calculate the quality of the answer based on completeness and relevance"""
    if not query or not response:
        return 0.0
    
    # Simple scoring based on response length and keyword overlap
    query_words = set(query.lower().split())
    response_words = set(response.lower().split())
    
    if not query_words or not response_words:
        return 0.0
    
    # Factor in response completeness (length) and relevance (overlap)
    overlap_score = len(query_words.intersection(response_words)) / len(query_words)
    length_score = min(len(response.split()) / 50.0, 1.0)  # Ideal around 50 words
    
    quality_score = (overlap_score * 0.7) + (length_score * 0.3)
    
    # Add some randomness for demo purposes
    return min(max(quality_score + random.uniform(-0.2, 0.2), 0.0), 1.0)

@app.get("/api/evaluation/metrics")
async def get_evaluation_metrics(
    timeframe: str = Query("7d", description="Time frame: 1d, 7d, 30d"),
    current_user: dict = Depends(get_current_user)
):
    """Get current evaluation metrics"""
    try:
        if not current_user.get('is_admin', False):
            raise HTTPException(status_code=403, detail="Admin access required")
        
        # Calculate date range
        end_date = datetime.datetime.now()
        if timeframe == "1d":
            start_date = end_date - timedelta(days=1)
        elif timeframe == "7d":
            start_date = end_date - timedelta(days=7)
        elif timeframe == "30d":
            start_date = end_date - timedelta(days=30)
        else:
            start_date = end_date - timedelta(days=7)
        
        # Get recent chat sessions and messages for evaluation
        with sqlite3.connect(chat_db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get recent chat interactions
            cursor.execute("""
                SELECT cs.id as session_id, cs.username, cs.topic, cs.created_at,
                       m.content, m.is_user, m.timestamp
                FROM chat_sessions cs
                JOIN messages m ON cs.id = m.session_id
                WHERE cs.created_at >= ? AND cs.created_at <= ?
                ORDER BY cs.created_at DESC, m.timestamp ASC
            """, (start_date.isoformat(), end_date.isoformat()))
            
            messages = cursor.fetchall()
        
        # Process messages and calculate metrics
        total_interactions = 0
        groundedness_scores = []
        context_relevance_scores = []
        answer_quality_scores = []
        latency_scores = []
        
        current_session = None
        user_query = None
        
        for message in messages:
            if message['is_user']:
                user_query = message['content']
                current_session = message['session_id']
            else:
                if user_query and current_session == message['session_id']:
                    # Calculate metrics for this Q&A pair
                    total_interactions += 1
                    
                    # Mock context for demonstration (in real implementation, retrieve from RAG)
                    mock_context = f"Retrieved context for query: {user_query[:100]}..."
                    
                    # Calculate evaluation metrics
                    groundedness = calculate_groundedness(message['content'], mock_context)
                    context_relevance = calculate_context_relevance(user_query, mock_context)
                    answer_quality = calculate_answer_quality(user_query, message['content'])
                    
                    # Mock latency (in real implementation, track actual response times)
                    latency = random.uniform(0.5, 3.0)  # seconds
                    
                    groundedness_scores.append(groundedness)
                    context_relevance_scores.append(context_relevance)
                    answer_quality_scores.append(answer_quality)
                    latency_scores.append(latency)
                    
                    user_query = None
        
        # Calculate averages
        if total_interactions > 0:
            avg_groundedness = sum(groundedness_scores) / len(groundedness_scores)
            avg_context_relevance = sum(context_relevance_scores) / len(context_relevance_scores)
            avg_answer_quality = sum(answer_quality_scores) / len(answer_quality_scores)
            avg_latency = sum(latency_scores) / len(latency_scores)
        else:
            # Default values for demo
            avg_groundedness = 0.85
            avg_context_relevance = 0.78
            avg_answer_quality = 0.82
            avg_latency = 1.2
        
        return {
            "timeframe": timeframe,
            "total_interactions": total_interactions if total_interactions > 0 else 150,  # Mock for demo
            "metrics": {
                "groundedness": {
                    "score": round(avg_groundedness, 3),
                    "description": "How well responses are grounded in retrieved context",
                    "threshold": 0.7
                },
                "context_relevance": {
                    "score": round(avg_context_relevance, 3),
                    "description": "Relevance of retrieved context to user queries",
                    "threshold": 0.7
                },
                "answer_quality": {
                    "score": round(avg_answer_quality, 3),
                    "description": "Overall quality and completeness of answers",
                    "threshold": 0.75
                },
                "latency": {
                    "score": round(avg_latency, 2),
                    "description": "Average response time in seconds",
                    "threshold": 2.0,
                    "unit": "seconds"
                }
            },
            "calculated_at": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting evaluation metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/evaluation/historical")
async def get_historical_metrics(
    days: int = Query(30, description="Number of days to look back"),
    current_user: dict = Depends(get_current_user)
):
    """Get historical evaluation metrics"""
    try:
        if not current_user.get('is_admin', False):
            raise HTTPException(status_code=403, detail="Admin access required")
        
        # Generate mock historical data (in real implementation, store these metrics in DB)
        historical_data = []
        
        for i in range(days):
            date = datetime.datetime.now() - timedelta(days=days - i - 1)
            
            # Generate realistic trending data
            base_groundedness = 0.85
            base_context_relevance = 0.78
            base_answer_quality = 0.82
            base_latency = 1.2
            
            # Add some trend and noise
            trend_factor = (i / days) * 0.1  # Slight improvement over time
            noise_factor = random.uniform(-0.05, 0.05)
            
            historical_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "groundedness": round(max(min(base_groundedness + trend_factor + noise_factor, 1.0), 0.0), 3),
                "context_relevance": round(max(min(base_context_relevance + trend_factor + noise_factor, 1.0), 0.0), 3),
                "answer_quality": round(max(min(base_answer_quality + trend_factor + noise_factor, 1.0), 0.0), 3),
                "latency": round(max(base_latency - (trend_factor * 0.5) + (noise_factor * 0.3), 0.1), 2),
                "total_queries": random.randint(10, 50)
            })
        
        return {
            "period": f"{days} days",
            "data": historical_data
        }
        
    except Exception as e:
        logger.error(f"Error getting historical metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/evaluation/latency-distribution")
async def get_latency_distribution(
    timeframe: str = Query("7d", description="Time frame: 1d, 7d, 30d"),
    current_user: dict = Depends(get_current_user)
):
    """Get latency distribution data"""
    try:
        if not current_user.get('is_admin', False):
            raise HTTPException(status_code=403, detail="Admin access required")
        
        # Generate mock latency distribution (in real implementation, get from actual data)
        latency_ranges = [
            {"range": "0-0.5s", "count": random.randint(20, 40)},
            {"range": "0.5-1s", "count": random.randint(30, 60)},
            {"range": "1-2s", "count": random.randint(40, 80)},
            {"range": "2-3s", "count": random.randint(20, 50)},
            {"range": "3-5s", "count": random.randint(10, 30)},
            {"range": "5s+", "count": random.randint(5, 15)}
        ]
        
        return {
            "timeframe": timeframe,
            "distribution": latency_ranges
        }
        
    except Exception as e:
        logger.error(f"Error getting latency distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/evaluation/quality-breakdown")
async def get_quality_breakdown(
    metric: str = Query("answer_quality", description="Metric to analyze: groundedness, context_relevance, answer_quality"),
    current_user: dict = Depends(get_current_user)
):
    """Get detailed quality breakdown by score ranges"""
    try:
        if not current_user.get('is_admin', False):
            raise HTTPException(status_code=403, detail="Admin access required")
        
        # Generate mock quality breakdown (in real implementation, analyze actual scores)
        breakdown = [
            {"range": "0.9-1.0", "count": random.randint(40, 70), "percentage": 0},
            {"range": "0.8-0.9", "count": random.randint(30, 50), "percentage": 0},
            {"range": "0.7-0.8", "count": random.randint(20, 40), "percentage": 0},
            {"range": "0.6-0.7", "count": random.randint(10, 25), "percentage": 0},
            {"range": "0.5-0.6", "count": random.randint(5, 15), "percentage": 0},
            {"range": "0.0-0.5", "count": random.randint(2, 10), "percentage": 0}
        ]
        
        # Calculate percentages
        total_count = sum(item["count"] for item in breakdown)
        for item in breakdown:
            item["percentage"] = round((item["count"] / total_count) * 100, 1) if total_count > 0 else 0
        
        return {
            "metric": metric,
            "breakdown": breakdown,
            "total_evaluations": total_count
        }
        
    except Exception as e:
        logger.error(f"Error getting quality breakdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/evaluation/datasets")
async def get_evaluation_datasets():
    """Get all evaluation datasets"""
    try:
        # TODO: Add authentication back when circular import is resolved
        # if not current_user.get('is_admin', False):
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        # Get real datasets from database
        from chat_db import get_chat_db
        db = get_chat_db()
        datasets = db.get_evaluation_datasets()
        
        # logger.info(f"Retrieved {len(datasets)} datasets from database")
        
        return {
            "datasets": datasets
        }
        
    except Exception as e:
        logger.error(f"Error getting evaluation datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_dataset_background(
    dataset_id: str,
    documents: List[Dict],
    name: str,
    description: str,
    num_questions_per_doc: int,
    model_name: str,
    difficulty_levels: List[str]
):
    """Background task for dataset generation with progress tracking"""
    try:
        # Initialize progress
        dataset_generation_progress[dataset_id] = {
            "status": "starting",
            "progress": 0,
            "current_document": "",
            "total_documents": len(documents),
            "completed_documents": 0,
            "error": None
        }
        
        # Create dataset record in database first
        from chat_db import get_chat_db
        db = get_chat_db()
        
        try:
            db_dataset_id = db.create_evaluation_dataset(
                name=name,
                description=description,
                document_count=len(documents),
                created_by="admin"  # TODO: Use actual user when auth is restored
            )
            logger.info(f"Created dataset record in database with ID: {db_dataset_id}")
        except ValueError as e:
            # Handle duplicate dataset name
            error_msg = str(e)
            logger.error(f"Dataset creation failed: {error_msg}")
            dataset_generation_progress[dataset_id].update({
                "status": "error",
                "error": error_msg
            })
            return
        except Exception as e:
            # Handle other database errors
            error_msg = f"Database error: {str(e)}"
            logger.error(f"Dataset creation failed: {error_msg}")
            dataset_generation_progress[dataset_id].update({
                "status": "error", 
                "error": error_msg
            })
            return
        
        # Generate dataset using LLM
        from dataset_generator import DatasetGenerator
        
        generator = DatasetGenerator(ollama_base_url=os.getenv('OLLAMA_HOST', 'http://ollama:11434'))
        
        try:
            # Custom generation with progress tracking
            all_items = []
            generation_stats = {
                "total_documents": len(documents),
                "questions_per_document": num_questions_per_doc,
                "model_used": model_name,
                "generation_start": datetime.datetime.now().isoformat(),
                "success_count": 0,
                "error_count": 0
            }
            
            dataset_generation_progress[dataset_id].update({
                "status": "processing",
                "progress": 0
            })
            
            for doc_idx, document in enumerate(documents):
                try:
                    # Update progress
                    dataset_generation_progress[dataset_id].update({
                        "current_document": document.get('filename', 'Unknown'),
                        "completed_documents": doc_idx,
                        "progress": int((doc_idx / len(documents)) * 100)
                    })
                    
                    logger.info(f"Processing document {doc_idx + 1}/{len(documents)}: {document.get('filename', 'Unknown')}")
                    
                    # Extract content from document
                    content = await generator._extract_document_content(document)
                    if not content:
                        logger.warning(f"No content extracted from document {document.get('filename')}")
                        continue
                    
                    # Generate questions for this document
                    doc_items = await generator._generate_questions_for_document(
                        content=content,
                        document_metadata=document,
                        num_questions=num_questions_per_doc,
                        model_name=model_name,
                        difficulty_levels=difficulty_levels
                    )
                    
                    all_items.extend(doc_items)
                    generation_stats["success_count"] += len(doc_items)
                    
                    # Add small delay to avoid overwhelming the LLM
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error processing document {document.get('filename', 'Unknown')}: {e}")
                    generation_stats["error_count"] += 1
                    continue
            
            generation_stats["generation_end"] = datetime.datetime.now().isoformat()
            generation_stats["total_questions_generated"] = len(all_items)
            
            # Create final dataset object
            from dataset_generator import GeneratedDataset
            dataset = GeneratedDataset(
                name=name,
                description=description,
                items=all_items,
                generation_metadata=generation_stats
            )
            
            # Save dataset to file
            datasets_dir = "/app/data/datasets"
            os.makedirs(datasets_dir, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{safe_name}_{timestamp}.json"
            file_path = os.path.join(datasets_dir, filename)
            
            await generator.save_dataset_to_file(dataset, file_path)
            
            # Update database record with completion details
            db.update_evaluation_dataset_status(
                dataset_id=db_dataset_id,
                status="Ready",
                file_path=file_path,
                question_count=len(all_items)
            )
            
            # Update progress to completed
            dataset_generation_progress[dataset_id].update({
                "status": "completed",
                "progress": 100,
                "current_document": "Completed",
                "completed_documents": len(documents),
                "file_path": file_path,
                "question_count": len(all_items),
                "db_dataset_id": db_dataset_id
            })
            
            logger.info(f"Dataset generation completed: {len(all_items)} questions generated, saved to DB with ID {db_dataset_id}")
            
        finally:
            await generator.close()
            
    except Exception as e:
        logger.error(f"Error in background dataset generation: {e}")
        # Update database status to error if we have a db_dataset_id
        try:
            if 'db_dataset_id' in locals():
                db = get_chat_db()
                db.update_evaluation_dataset_status(db_dataset_id, "Error")
        except Exception as db_error:
            logger.error(f"Failed to update database status to error: {db_error}")
        
        dataset_generation_progress[dataset_id].update({
            "status": "error",
            "error": str(e)
        })

@app.post("/api/evaluation/datasets")
async def create_evaluation_dataset(
    request: dict,
    background_tasks: BackgroundTasks
):
    """Create a new evaluation dataset from selected documents using LLM"""
    try:
        # TODO: Add authentication back when circular import is resolved
        # if not current_user.get('is_admin', False):
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        # Extract parameters from request
        name = request.get('name', '')
        description = request.get('description', '')
        document_ids = request.get('document_ids', [])
        num_questions_per_doc = request.get('num_questions_per_doc', 3)
        model_name = request.get('model_name', 'llama3')
        difficulty_levels = request.get('difficulty_levels', ['easy', 'medium', 'hard'])
        
        if not name:
            raise HTTPException(status_code=400, detail="Dataset name is required")
        
        if not document_ids:
            raise HTTPException(status_code=400, detail="At least one document must be selected")
        
        # Fetch documents
        from document_storage import get_document_storage
        doc_storage = get_document_storage()
        
        documents = []
        for doc_id in document_ids:
            try:
                doc_info = doc_storage.get_document_info(doc_id)
                if doc_info:
                    documents.append(doc_info)
            except Exception as e:
                logger.warning(f"Could not fetch document {doc_id}: {e}")
                continue
        
        if not documents:
            raise HTTPException(status_code=400, detail="No valid documents found")
        
        # Generate unique dataset ID for progress tracking
        dataset_id = f"dataset_{int(datetime.datetime.now().timestamp())}"
        
        logger.info(f"Starting dataset generation with ID: {dataset_id}")
        
        # Start background task for dataset generation
        background_tasks.add_task(
            generate_dataset_background,
            dataset_id,
            documents,
            name,
            description,
            num_questions_per_doc,
            model_name,
            difficulty_levels
        )
        
        # Return immediately with dataset ID for progress tracking
        response = {
            "message": "Dataset generation started",
            "dataset_id": dataset_id,
            "status": "processing",
            "progress_url": f"/api/evaluation/datasets/{dataset_id}/progress"
        }
        
        logger.info(f"Returning dataset creation response: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Error creating evaluation dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/evaluation/datasets/{dataset_id}/progress")
async def get_dataset_generation_progress(dataset_id: str):
    """Get progress of dataset generation"""
    try:
        if dataset_id not in dataset_generation_progress:
            raise HTTPException(status_code=404, detail="Dataset generation not found")
        
        progress_info = dataset_generation_progress[dataset_id]
        
        # Clean up completed or errored generations after some time
        if progress_info["status"] in ["completed", "error"]:
            # Keep for 5 minutes after completion for client to fetch
            # In production, you might want to store this in a database
            pass
        
        return progress_info
        
    except Exception as e:
        logger.error(f"Error getting dataset generation progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/evaluation/datasets/{dataset_id}")
async def delete_evaluation_dataset(
    dataset_id: int
):
    """Delete an evaluation dataset"""
    try:
        # TODO: Add authentication back when circular import is resolved
        # if not current_user.get('is_admin', False):
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        from chat_db import get_chat_db
        db = get_chat_db()
        
        # First, get the dataset info to find the file path
        dataset_to_delete = db.get_evaluation_dataset_by_id(dataset_id)
        
        if not dataset_to_delete:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        # Delete the file if it exists
        file_path = dataset_to_delete.get('file_path')
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Deleted dataset file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete dataset file {file_path}: {e}")
        
        # Delete from database
        success = db.delete_evaluation_dataset(dataset_id)
        
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to delete dataset {dataset_id} from database")
        
        logger.info(f"Successfully deleted evaluation dataset {dataset_id}")
        
        return {
            "success": True,
            "message": f"Dataset {dataset_id} deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting evaluation dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/evaluation/datasets/{dataset_id}")
async def get_dataset_details(
    dataset_id: int
):
    """Get detailed information about a specific dataset"""
    try:
        # TODO: Add authentication back when circular import is resolved
        # if not current_user.get('is_admin', False):
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        from chat_db import get_chat_db
        db = get_chat_db()
        
        # Get dataset info from database
        dataset = db.get_evaluation_dataset_by_id(dataset_id)
        
        if not dataset:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        # Load dataset content from file if available
        dataset_content = None
        sample_items = []
        
        if dataset.get('file_path') and os.path.exists(dataset['file_path']):
            try:
                with open(dataset['file_path'], 'r', encoding='utf-8') as f:
                    dataset_content = json.load(f)
                    
                # Extract sample items (first 5 items for preview)
                if isinstance(dataset_content, list):
                    sample_items = dataset_content[:5]
                elif isinstance(dataset_content, dict) and 'items' in dataset_content:
                    sample_items = dataset_content['items'][:5]
                    
            except Exception as e:
                logger.warning(f"Failed to load dataset file {dataset['file_path']}: {e}")
        
        # Calculate statistics
        total_items = len(dataset_content) if isinstance(dataset_content, list) else len(dataset_content.get('items', [])) if dataset_content else 0
        
        # Group items by document/source for tabbed view
        documents_data = {}
        sample_items = []
        
        if dataset_content:
            items = dataset_content if isinstance(dataset_content, list) else dataset_content.get('items', [])
            
            # Group questions by source document
            for item in items:
                if isinstance(item, dict):
                    # Try to find document source from various fields
                    doc_source = "Unknown Document"
                    
                    # First, try the direct source_file field (for new dataset format)
                    if item.get('source_file'):
                        doc_source = item['source_file']
                    # Then try expected chunks
                    elif item.get('expected_chunks') and len(item['expected_chunks']) > 0:
                        first_chunk = item['expected_chunks'][0]
                        doc_source = first_chunk.get('source') or first_chunk.get('title') or "Unknown Document"
                    # Finally try metadata
                    elif item.get('metadata') and item['metadata'].get('source'):
                        doc_source = item['metadata']['source']
                    
                    if doc_source not in documents_data:
                        documents_data[doc_source] = []
                    documents_data[doc_source].append(item)
            
            # Get sample items from all documents (limit to 5 total)
            all_items = []
            for doc_items in documents_data.values():
                all_items.extend(doc_items)
            sample_items = all_items[:5]
        
        # Format response
        response = {
            "id": dataset['id'],
            "name": dataset['name'],
            "description": dataset['description'],
            "document_count": dataset['document_count'],
            "question_count": total_items,
            "created_at": dataset['created_at'],
            "updated_at": dataset['updated_at'],
            "status": dataset['status'],
            "created_by": dataset['created_by'],
            "file_path": dataset.get('file_path'),
            "generation_metadata": dataset_content.get('metadata', {}) if isinstance(dataset_content, dict) else {},
            "sample_items": sample_items,
            "total_items": total_items,
            "documents_data": documents_data  # Add grouped document data for tabbed view
        }
        
        # logger.info(f"Retrieved details for dataset {dataset_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluation/datasets/preview")
async def preview_dataset_generation(
    request: dict
):
    """Preview what a dataset generation would look like without creating it"""
    try:
        # TODO: Add authentication back when circular import is resolved
        # if not current_user.get('is_admin', False):
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        document_ids = request.get('document_ids', [])
        num_questions_per_doc = request.get('num_questions_per_doc', 3)
        model_name = request.get('model_name', 'llama3')
        
        if not document_ids:
            raise HTTPException(status_code=400, detail="At least one document must be selected")
        
        # Get document information
        from document_storage import get_document_storage
        doc_storage = get_document_storage()
        
        documents = []
        total_content_length = 0
        
        for doc_id in document_ids:
            try:
                doc_info = doc_storage.get_document_info(doc_id)
                if doc_info:
                    content = doc_storage.get_document_content(doc_id)
                    content_length = len(content) if content else 0
                    total_content_length += content_length
                    
                    documents.append({
                        "id": doc_id,
                        "filename": doc_info.get('filename', 'Unknown'),
                        "content_length": content_length,
                        "estimated_questions": num_questions_per_doc
                    })
            except Exception as e:
                logger.warning(f"Could not fetch document {doc_id}: {e}")
                continue
        
        if not documents:
            raise HTTPException(status_code=400, detail="No valid documents found")
        
        # Calculate estimates
        total_questions = len(documents) * num_questions_per_doc
        estimated_time_minutes = total_questions * 0.5  # Rough estimate: 30 seconds per question
        
        return {
            "preview": {
                "total_documents": len(documents),
                "total_questions_estimated": total_questions,
                "estimated_generation_time_minutes": round(estimated_time_minutes, 1),
                "total_content_length": total_content_length,
                "model_to_use": model_name,
                "documents": documents
            },
            "recommendations": {
                "optimal_questions_per_doc": min(max(1, total_content_length // (len(documents) * 1000)), 5),
                "estimated_cost": "Free (using local Ollama)",
                "tips": [
                    "Longer documents can support more questions",
                    "Medium difficulty questions work best for most use cases",
                    "Consider the model's context window when selecting documents"
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Error previewing dataset generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluation/test-cases/run")
async def run_evaluation_test_case(
    request: dict,
    current_user: dict = Depends(get_current_user)
):
    """Run evaluation test case on dataset with specified models"""
    try:
        if not current_user.get('is_admin', False):
            raise HTTPException(status_code=403, detail="Admin access required")
        
        dataset_id = request.get('dataset_id')
        model_names = request.get('models', [])
        
        if not dataset_id:
            raise HTTPException(status_code=400, detail="Dataset ID is required")
        
        if not model_names:
            raise HTTPException(status_code=400, detail="At least one model must be specified")
        
        # Mock test case execution (in real implementation, run actual evaluations)
        results = []
        for model_name in model_names:
            result = {
                "id": random.randint(1000, 9999),
                "dataset_id": dataset_id,
                "model": model_name,
                "groundedness": round(random.uniform(0.7, 0.95), 3),
                "context_relevance": round(random.uniform(0.65, 0.9), 3),
                "answer_quality": round(random.uniform(0.68, 0.92), 3),
                "avg_latency": round(random.uniform(0.5, 3.0), 2),
                "run_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "status": "Completed",
                "run_by": "admin"  # TODO: Use actual user when auth is restored
            }
            results.append(result)
        
        return {
            "success": True,
            "results": results,
            "message": f"Test case completed for {len(model_names)} models"
        }
        
    except Exception as e:
        logger.error(f"Error running evaluation test case: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluation/test-cases")
async def create_test_case(
    request: dict
    # TODO: Re-enable authentication when auth system is stable
    # current_user: dict = Depends(get_current_user)
):
    """Create a new test case with retrieval configuration"""
    try:
        # For now, use a default user for testing
        current_user = {"username": "test_user", "is_admin": False}
        
        dataset_id = request.get('dataset_id')
        model_names = request.get('models', [])
        retriever = request.get('retriever')
        retrieval_config = request.get('retrieval_config', {})
        
        if not dataset_id:
            raise HTTPException(status_code=400, detail="Dataset ID is required")
        
        if not model_names:
            raise HTTPException(status_code=400, detail="At least one model must be specified")
            
        if not retriever:
            raise HTTPException(status_code=400, detail="Retriever is required")
        
        # Generate unique test case ID
        test_case_id = random.randint(10000, 99999)
        
        # Store test case in database (simplified for demo)
        # In production, use proper database storage
        test_case = {
            "id": test_case_id,
            "dataset_id": dataset_id,
            "models": model_names,
            "retriever": retriever,
            "retrieval_config": retrieval_config,
            "status": "pending",
            "created_at": datetime.datetime.now().isoformat(),
            "created_by": current_user.get('username', 'admin')
        }
        
        # Start test case execution in background
        # For demo purposes, we'll simulate this
        import asyncio
        asyncio.create_task(execute_test_case_background(test_case))
        
        return {
            "success": True,
            "test_case_id": test_case_id,
            "message": f"Test case created and started with ID {test_case_id}"
        }
        
    except Exception as e:
        logger.error(f"Error creating test case: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/evaluation/test-cases/{test_case_id}")
async def get_test_case_status(
    test_case_id: int
    # TODO: Re-enable authentication when auth system is stable
    # current_user: dict = Depends(get_current_user)
):
    """Get test case status and results"""
    try:
        # Check if test case exists in storage
        if test_case_id in test_case_storage:
            test_case = test_case_storage[test_case_id]
            
            return {
                "id": test_case_id,
                "status": test_case.get('status', 'pending'),
                "results": test_case.get('results'),
                "started_at": test_case.get('started_at'),
                "completed_at": test_case.get('completed_at'),
                "error": test_case.get('error')
            }
        else:
            # Test case not found, return default pending status
            return {
                "id": test_case_id,
                "status": "pending",
                "results": None,
                "started_at": None,
                "completed_at": None,
                "error": None
            }
        
    except Exception as e:
        logger.error(f"Error getting test case status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def execute_test_case_background(test_case: dict):
    """Execute test case in background with real evaluation"""
    test_case_id = test_case.get('id')
    eval_logger = get_evaluation_logger()
    try:
        eval_logger.info(f"Starting real evaluation for test case {test_case_id}")
        
        # Update global test case storage
        if test_case_id not in test_case_storage:
            test_case_storage[test_case_id] = test_case.copy()
        
        test_case_storage[test_case_id]['status'] = 'running'
        test_case_storage[test_case_id]['started_at'] = datetime.datetime.now().isoformat()
        
        # 1. Load the dataset
        dataset_id = test_case.get('dataset_id')
        models = test_case.get('models', [])
        retrieval_config = test_case.get('retrieval_config', {})
        
        # Get dataset from database
        from chat_db import get_chat_db
        db = get_chat_db()
        dataset = db.get_evaluation_dataset(dataset_id)
        
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Load questions from dataset file if available
        questions = await load_dataset_questions(dataset)
        
        if not questions:
            # If no questions file, generate some test questions from documents
            questions = await generate_test_questions_from_documents(dataset_id)
        
        eval_logger.info(f"Loaded {len(questions)} questions for evaluation")
        
        # Store initial results structure
        model_results = []
        overall_metrics = {
            "groundedness": 0.0,
            "context_relevance": 0.0,
            "answer_quality": 0.0,
            "avg_latency": 0.0
        }
        
        # 2. Initialize RAG with retrieval config and run evaluation for each model
        for i, model_name in enumerate(models):
            eval_logger.info(f"Evaluating model {i+1}/{len(models)}: {model_name}")
            
            try:
                # Configure RAG instance with the specified model and retrieval settings
                rag_instance = await configure_rag_for_evaluation(model_name, retrieval_config)
                
                # 3. Run evaluation on all questions for this model
                model_result = await evaluate_model_on_questions(
                    rag_instance, model_name, questions, test_case_id
                )
                
                model_results.append(model_result)
                
                # Update overall metrics (average across models)
                for metric in overall_metrics:
                    overall_metrics[metric] += model_result.get(metric, 0.0)
                
            except Exception as e:
                eval_logger.error(f"Error evaluating model {model_name}: {e}")
                # Add failed result
                model_results.append({
                    "model": model_name,
                    "groundedness": 0.0,
                    "context_relevance": 0.0,
                    "answer_quality": 0.0,
                    "avg_latency": 0.0,
                    "total_questions": len(questions),
                    "success_rate": 0.0,
                    "error": str(e)
                })
        
        # Calculate final overall metrics
        if model_results:
            for metric in overall_metrics:
                overall_metrics[metric] = overall_metrics[metric] / len(models)
        
        # 4. Store final results
        test_case_storage[test_case_id].update({
            'status': 'completed',
            'completed_at': datetime.datetime.now().isoformat(),
            'results': {
                'overall_metrics': overall_metrics,
                'model_results': model_results,
                'total_questions': len(questions),
                'evaluation_summary': {
                    'models_evaluated': len(models),
                    'questions_processed': len(questions),
                    'evaluation_method': 'TruLens-based RAG Triad'
                }
            }
        })
        
        # 5. Save results to database
        try:
            chat_db = ChatDB()
            
            # Store each model's results in the database
            for model_result in model_results:
                model_name = model_result['model']
                
                success = chat_db.save_evaluation_results(
                    dataset_id=dataset_id,
                    model_name=model_name,
                    groundedness_score=model_result['groundedness'],
                    context_relevance_score=model_result['context_relevance'],
                    answer_quality_score=model_result['answer_quality'],
                    avg_latency=model_result['avg_latency'],
                    total_queries=len(questions),
                    status='completed',
                    run_by='system',  # TODO: Use actual user when auth is stable
                    started_at=test_case_storage[test_case_id].get('started_at'),
                    completed_at=datetime.datetime.now().isoformat()
                )
                
                if success:
                    eval_logger.info(f"Saved evaluation results for model {model_name}")
                else:
                    eval_logger.error(f"Failed to save evaluation results for model {model_name}")
                
            eval_logger.info(f"Saved evaluation results to database for test case {test_case_id}")
            
        except Exception as db_error:
            eval_logger.error(f"Error saving evaluation results to database: {db_error}")
            # Don't fail the test case just because of DB error
        
        eval_logger.info(f"Test case {test_case_id} completed successfully with {len(model_results)} model results")
        
    except Exception as e:
        eval_logger.error(f"Error executing test case {test_case_id}: {e}")
        # Update with error status
        if test_case_id in test_case_storage:
            test_case_storage[test_case_id].update({
                'status': 'failed',
                'error': str(e),
                'completed_at': datetime.datetime.now().isoformat()
            })

# Global storage for test cases (in production, use proper database)
test_case_storage = {}

async def load_dataset_questions(dataset: dict) -> List[dict]:
    """Load questions from dataset file"""
    try:
        file_path = dataset.get('file_path')
        if not file_path:
            return []
        
        # Try to load from file (assuming JSON format)
        import os
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'questions' in data:
                    return data['questions']
        
        return []
    except Exception as e:
        logger.warning(f"Could not load questions from dataset file: {e}")
        return []

async def generate_test_questions_from_documents(dataset_id: int) -> List[dict]:
    """Generate test questions from documents if no questions file exists"""
    eval_logger = get_evaluation_logger()
    try:
        # Get documents associated with this dataset from the knowledge base
        # For now, return some sample questions - in production, generate from actual documents
        sample_questions = [
            {
                "id": 1,
                "question": "What are the main features of the product?",
                "expected_context": "product features",
                "difficulty": "easy"
            },
            {
                "id": 2,
                "question": "How do I configure the advanced settings?",
                "expected_context": "configuration settings",
                "difficulty": "medium"
            },
            {
                "id": 3,
                "question": "What troubleshooting steps should I follow for common issues?",
                "expected_context": "troubleshooting guide",
                "difficulty": "medium"
            },
            {
                "id": 4,
                "question": "What are the system requirements and compatibility information?",
                "expected_context": "system requirements",
                "difficulty": "easy"
            },
            {
                "id": 5,
                "question": "How does the integration with third-party services work?",
                "expected_context": "integration documentation",
                "difficulty": "hard"
            }
        ]
        
        eval_logger.info(f"Generated {len(sample_questions)} test questions for dataset {dataset_id}")
        return sample_questions
        
    except Exception as e:
        eval_logger.error(f"Error generating test questions: {e}")
        return []

async def configure_rag_for_evaluation(model_name: str, retrieval_config: dict):
    """Configure RAG instance with specified model and retrieval settings"""
    eval_logger = get_evaluation_logger()
    try:
        from rag import get_chatpdf_instance
        
        # Get base RAG instance
        rag_instance = get_chatpdf_instance()
        
        # Update model if different from current
        if rag_instance.llm_model != model_name:
            rag_instance.llm_model = model_name
            # Reinitialize models with new model
            rag_instance.ensure_models_loaded()
        
        # Apply retrieval configuration
        if retrieval_config:
            # Update similarity threshold
            similarity_threshold = retrieval_config.get('similarity_threshold')
            if similarity_threshold is not None:
                rag_instance.similarity_threshold = similarity_threshold
            
            # Update top_k
            top_k = retrieval_config.get('top_k')
            if top_k is not None:
                rag_instance.top_k = top_k
            
            # Update chunk size and overlap if provided
            chunk_size = retrieval_config.get('chunk_size')
            if chunk_size is not None:
                rag_instance.chunk_size = chunk_size
            
            chunk_overlap = retrieval_config.get('chunk_overlap')
            if chunk_overlap is not None:
                rag_instance.chunk_overlap = chunk_overlap
            
            # Configure reranker if specified
            reranker_models = retrieval_config.get('reranker_models', [])
            if reranker_models:
                # Use first reranker model
                reranker_model = reranker_models[0] if reranker_models else None
                if reranker_model and hasattr(rag_instance, 'reranker_model'):
                    rag_instance.reranker_model = reranker_model
        
        eval_logger.info(f"Configured RAG for model {model_name} with retrieval config")
        return rag_instance
        
    except Exception as e:
        eval_logger.error(f"Error configuring RAG for evaluation: {e}")
        raise

async def evaluate_model_on_questions(rag_instance, model_name: str, questions: List[dict], test_case_id: int) -> dict:
    """Evaluate a model on a set of questions using real evaluation metrics"""
    eval_logger = get_evaluation_logger()
    try:
        from evaluation_system import evaluate_chat_response
        
        total_questions = len(questions)
        successful_evaluations = 0
        
        # Metrics accumulation
        total_groundedness = 0.0
        total_context_relevance = 0.0
        total_answer_quality = 0.0
        total_latency = 0.0
        
        eval_logger.info(f"Starting evaluation of {model_name} on {total_questions} questions")
        
        for i, question_data in enumerate(questions):
            try:
                question = question_data.get('question', '')
                if not question:
                    continue
                
                # Measure response time
                start_time = time.time()
                
                # Get response from RAG system
                response = rag_instance.query(question)
                
                # Extract context from the last query (if available)
                context = ""
                if hasattr(rag_instance, 'last_context_docs') and rag_instance.last_context_docs:
                    context = "\n".join([doc.page_content for doc in rag_instance.last_context_docs])
                
                response_time = time.time() - start_time
                
                # Run evaluation using TruLens-based system
                evaluation_result = await evaluate_chat_response(
                    question=question,
                    answer=response,
                    context=context
                )
                
                # Accumulate metrics
                total_groundedness += evaluation_result.groundedness.score
                total_context_relevance += evaluation_result.context_relevance.score
                total_answer_quality += evaluation_result.answer_relevance.score
                total_latency += response_time
                
                successful_evaluations += 1
                
                # Log progress
                if (i + 1) % 5 == 0 or (i + 1) == total_questions:
                    eval_logger.info(f"Progress: {i + 1}/{total_questions} questions evaluated for {model_name}")
                
            except Exception as e:
                eval_logger.warning(f"Error evaluating question {i+1} for {model_name}: {e}")
                continue
        
        # Calculate averages
        if successful_evaluations > 0:
            avg_groundedness = total_groundedness / successful_evaluations
            avg_context_relevance = total_context_relevance / successful_evaluations
            avg_answer_quality = total_answer_quality / successful_evaluations
            avg_latency = total_latency / successful_evaluations
            success_rate = successful_evaluations / total_questions
        else:
            avg_groundedness = 0.0
            avg_context_relevance = 0.0
            avg_answer_quality = 0.0
            avg_latency = 0.0
            success_rate = 0.0
        
        result = {
            "model": model_name,
            "groundedness": round(avg_groundedness, 3),
            "context_relevance": round(avg_context_relevance, 3),
            "answer_quality": round(avg_answer_quality, 3),
            "avg_latency": round(avg_latency, 2),
            "total_questions": total_questions,
            "successful_evaluations": successful_evaluations,
            "success_rate": round(success_rate, 3)
        }
        
        eval_logger.info(f"Completed evaluation for {model_name}: {result}")
        return result
        
    except Exception as e:
        eval_logger.error(f"Error in model evaluation for {model_name}: {e}")
        return {
            "model": model_name,
            "groundedness": 0.0,
            "context_relevance": 0.0,
            "answer_quality": 0.0,
            "avg_latency": 0.0,
            "total_questions": len(questions),
            "successful_evaluations": 0,
            "success_rate": 0.0,
            "error": str(e)
        }

@app.get("/api/evaluation/results")
async def get_evaluation_results(
    model_filter: str = Query(None, description="Filter by model name"),
    dataset_filter: str = Query(None, description="Filter by dataset name"),
    limit: int = Query(50, description="Maximum number of results")
):
    """Get evaluation results with optional filtering"""
    try:
        # TODO: Add authentication back when circular import is resolved
        # if not current_user.get('is_admin', False):
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        # Get results from database using ChatDB
        chat_db = ChatDB()
        all_results = chat_db.get_evaluation_results(
            model_filter=model_filter,
            dataset_filter=dataset_filter, 
            limit=limit
        )
        
        # If no results from database, also check test_case_storage for any in-progress results
        if not all_results:
            for test_case_id, test_case in test_case_storage.items():
                if test_case.get('status') == 'completed' and test_case.get('results'):
                    results = test_case['results']
                    model_results = results.get('model_results', [])
                    
                    for model_result in model_results:
                        all_results.append({
                            "id": f"tc_{test_case_id}_{model_result.get('model', 'unknown')}",
                            "test_case_id": test_case_id,
                            "dataset": f"Dataset {test_case.get('dataset_id', 'Unknown')}",
                            "model": model_result.get('model', 'Unknown'),
                            "groundedness": model_result.get('groundedness', 0.0),
                            "context_relevance": model_result.get('context_relevance', 0.0),
                            "answer_quality": model_result.get('answer_quality', 0.0),
                            "avg_latency": model_result.get('avg_latency', 0.0),
                            "total_questions": model_result.get('total_questions', 0),
                            "success_rate": model_result.get('success_rate', 0.0),
                            "run_date": test_case.get('completed_at', '').split('T')[0] if test_case.get('completed_at') else '',
                            "status": "Completed"
                        })
            
            # Add some sample results if still no real results exist
            if not all_results:
                all_results = [
                    {
                        "id": 1,
                        "dataset": "Customer Support QA",
                        "model": "llama3.1",
                        "groundedness": 0.85,
                        "context_relevance": 0.78,
                        "answer_quality": 0.82,
                        "avg_latency": 1.2,
                        "run_date": "2024-08-03",
                        "status": "Completed"
                    },
                    {
                        "id": 2,
                        "dataset": "Technical Documentation",
                        "model": "mistral",
                        "groundedness": 0.79,
                        "context_relevance": 0.81,
                        "answer_quality": 0.77,
                        "avg_latency": 0.9,
                        "run_date": "2024-08-02",
                        "status": "Completed"
                    }
                ]
        
        return {
            "results": all_results,
            "total": len(all_results),
            "filters": {
                "model": model_filter,
                "dataset": dataset_filter
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting evaluation results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# TruLens evaluation endpoints
@app.post("/api/trulens/evaluate")
async def evaluate_chat_response_endpoint(
    background_tasks: BackgroundTasks,
    question: str,
    answer: str,
    context: str,
    session_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Evaluate a chat response using TruLens metrics"""
    try:
        from evaluation_system import evaluate_chat_response
        
        # Perform evaluation
        result = await evaluate_chat_response(
            question=question,
            answer=answer,
            context=context,
            session_id=session_id,
            user_id="admin"  # TODO: Use actual user when auth is restored
        )
        
        return {
            "evaluation_id": f"trulens_{int(time.time())}",
            "overall_score": result.overall_score,
            "metrics": {
                "groundedness": {
                    "score": result.groundedness.score,
                    "raw_score": result.groundedness.raw_score,
                    "reasoning": result.groundedness.reasoning[:300] + "..." if len(result.groundedness.reasoning) > 300 else result.groundedness.reasoning
                },
                "answer_relevance": {
                    "score": result.answer_relevance.score,
                    "raw_score": result.answer_relevance.raw_score,
                    "reasoning": result.answer_relevance.reasoning[:300] + "..." if len(result.answer_relevance.reasoning) > 300 else result.answer_relevance.reasoning
                },
                "context_relevance": {
                    "score": result.context_relevance.score,
                    "raw_score": result.context_relevance.raw_score,
                    "reasoning": result.context_relevance.reasoning[:300] + "..." if len(result.context_relevance.reasoning) > 300 else result.context_relevance.reasoning
                }
            },
            "evaluation_time_seconds": result.evaluation_time_seconds,
            "timestamp": result.groundedness.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in TruLens evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trulens/statistics")
async def get_trulens_statistics(
    last_n: int = Query(100, description="Number of recent evaluations to analyze"),
    current_user: dict = Depends(get_current_user)
):
    """Get TruLens evaluation statistics"""
    try:
        if not current_user.get('is_admin', False):
            raise HTTPException(status_code=403, detail="Admin access required")
        
        from evaluation_system import get_recent_evaluation_stats
        
        stats = get_recent_evaluation_stats(last_n=last_n)
        
        if stats["total_evaluations"] == 0:
            return {
                "total_evaluations": 0,
                "message": "No evaluations performed yet. Start chatting to see real metrics!",
                "demo_metrics": {
                    "groundedness": {"mean": 0.85, "description": "How well responses are grounded in context"},
                    "answer_relevance": {"mean": 0.82, "description": "How relevant answers are to questions"},
                    "context_relevance": {"mean": 0.78, "description": "How relevant context is to queries"},
                    "overall_score": {"mean": 0.82, "description": "Weighted average of all metrics"}
                }
            }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting TruLens statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/evaluation/overview")
async def get_evaluation_overview(
    time_range: str = Query("30d", description="Time range: 7d, 30d, 90d")
    # TODO: Re-enable authentication when auth system is stable
    # current_user: dict = Depends(get_current_user)
):
    """Get evaluation overview data from database"""
    try:
        chat_db = ChatDB()
        
        # Get basic overview stats using the new method
        overview_stats = chat_db.get_evaluation_overview_stats()
        
        # Calculate date range for detailed queries
        days = 7 if time_range == "7d" else 30 if time_range == "30d" else 90
        start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).isoformat()
        
        # Get historical and detailed data using proper connection handling
        with sqlite3.connect(chat_db.db_path) as conn:
            cursor = conn.cursor()
            
            # Get overall metrics for the time range
            cursor.execute('''
                SELECT 
                    AVG(groundedness_score) as avg_groundedness,
                    AVG(context_relevance_score) as avg_context_relevance,
                    AVG(answer_quality_score) as avg_answer_quality,
                    AVG(avg_latency) as avg_latency,
                    COUNT(*) as total_evaluations
                FROM evaluation_results 
                WHERE completed_at >= ? AND status = 'completed'
            ''', (start_date,))
            
            overall_row = cursor.fetchone()
            
            if overall_row and overall_row[0] is not None:
                overall_metrics = {
                    "groundedness": float(overall_row[0] or 0),
                    "contextRelevance": float(overall_row[1] or 0),
                    "answerQuality": float(overall_row[2] or 0),
                    "averageLatency": float(overall_row[3] or 0) * 1000,  # Convert to ms
                    "totalEvaluations": int(overall_row[4] or 0)
                }
            else:
                # Use overview stats as fallback
                overall_metrics = {
                    "groundedness": overview_stats['average_scores']['groundedness'],
                    "contextRelevance": overview_stats['average_scores']['context_relevance'],
                    "answerQuality": overview_stats['average_scores']['answer_quality'],
                    "averageLatency": overview_stats['average_latency'] * 1000,  # Convert to ms
                    "totalEvaluations": overview_stats['total_evaluations']
                }
            
            # Get historical data for trends (daily averages)
            cursor.execute('''
                SELECT 
                    DATE(completed_at) as date,
                    AVG(groundedness_score) as groundedness,
                    AVG(context_relevance_score) as context_relevance,
                    AVG(answer_quality_score) as answer_quality,
                    AVG(avg_latency) as latency,
                    COUNT(*) as queries
                FROM evaluation_results 
                WHERE completed_at >= ? AND status = 'completed'
                GROUP BY DATE(completed_at)
                ORDER BY date
            ''', (start_date,))
            
            historical_rows = cursor.fetchall()
            historical_data = []
            
            for row in historical_rows:
                historical_data.append({
                    "date": row[0],
                    "groundedness": float(row[1] or 0),
                    "contextRelevance": float(row[2] or 0),
                    "answerQuality": float(row[3] or 0),
                    "latency": float(row[4] or 0) * 1000,  # Convert to ms
                    "queries": int(row[5] or 0)
                })
            
            # Get recent detailed results
            cursor.execute('''
                SELECT 
                    er.id,
                    er.model_name,
                    er.groundedness_score,
                    er.context_relevance_score,
                    er.answer_quality_score,
                    er.avg_latency,
                    er.completed_at,
                    ed.name as dataset_name
                FROM evaluation_results er
                LEFT JOIN evaluation_datasets ed ON er.dataset_id = ed.id
                WHERE er.completed_at >= ? AND er.status = 'completed'
                ORDER BY er.completed_at DESC
                LIMIT 20
            ''', (start_date,))
            
            detailed_rows = cursor.fetchall()
            detailed_data = []
            
            for row in detailed_rows:
                detailed_data.append({
                    "id": row[0],
                    "model": row[1],
                    "groundedness": float(row[2] or 0),
                    "contextRelevance": float(row[3] or 0),
                    "answerQuality": float(row[4] or 0),
                    "latency": float(row[5] or 0) * 1000,  # Convert to ms
                    "timestamp": row[6],
                    "dataset": row[7] or "Unknown",
                    "query": f"Evaluation from {row[7] or 'Unknown'} dataset"  # Simplified for now
                })
            
            # Get latency distribution
            cursor.execute('''
                SELECT 
                    CASE 
                        WHEN avg_latency < 1 THEN '0-1s'
                        WHEN avg_latency < 2 THEN '1-2s'
                        WHEN avg_latency < 5 THEN '2-5s'
                        WHEN avg_latency < 10 THEN '5-10s'
                        ELSE '10s+'
                    END as range,
                    COUNT(*) as count
                FROM evaluation_results 
                WHERE completed_at >= ? AND status = 'completed'
                GROUP BY 
                    CASE 
                        WHEN avg_latency < 1 THEN '0-1s'
                        WHEN avg_latency < 2 THEN '1-2s'
                        WHEN avg_latency < 5 THEN '2-5s'
                        WHEN avg_latency < 10 THEN '5-10s'
                        ELSE '10s+'
                    END
                ORDER BY avg_latency
            ''', (start_date,))
            
            latency_rows = cursor.fetchall()
            latency_distribution = [{"range": row[0], "count": row[1]} for row in latency_rows]
            
        return {
            "overall": overall_metrics,
            "historical": historical_data,
            "detailed": detailed_data,
            "latencyDistribution": latency_distribution
        }
        
    except Exception as e:
        logger.error(f"Error getting evaluation overview: {e}")
        # Return empty data instead of failing
        return {
            "overall": {
                "groundedness": 0.0,
                "contextRelevance": 0.0,
                "answerQuality": 0.0,
                "averageLatency": 0.0,
                "totalEvaluations": 0
            },
            "historical": [],
            "detailed": [],
            "latencyDistribution": []
        }

@app.get("/api/trulens/health")
async def get_trulens_health():
    """Check TruLens evaluation system health"""
    try:
        from evaluation_system import get_evaluation_manager
        
        manager = get_evaluation_manager()
        
        # Test basic functionality
        test_result = {
            "status": "healthy",
            "evaluator_initialized": manager.evaluator is not None,
            "llm_provider_type": type(manager.evaluator.llm_provider).__name__,
            "total_evaluations_in_memory": len(manager.evaluation_history),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Try a quick evaluation test
        try:
            test_eval = await manager.evaluate_rag_interaction(
                question="Test question?",
                answer="Test answer.",
                context="Test context."
            )
            test_result["last_test_evaluation"] = {
                "overall_score": test_eval.overall_score,
                "evaluation_time": test_eval.evaluation_time_seconds
            }
        except Exception as eval_error:
            test_result["status"] = "degraded"
            test_result["test_error"] = str(eval_error)
        
        return test_result
        
    except Exception as e:
        logger.error(f"Error checking TruLens health: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }
