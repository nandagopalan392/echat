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

chat_db = ChatDB()
rlhf_db = RLHF()

# Add lazy loading for RAG instance
# RAG instance is now managed by the rag.py module singleton

def get_rag():
    """Get RAG instance using the singleton from rag.py"""
    return get_chatpdf_instance()

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.debug(f"Incoming request: {request.method} {request.url}")
    try:
        body = await request.body()
        logger.debug(f"Request body: {body.decode()}")
    except:
        pass
    response = await call_next(request)
    return response

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the application...")
    try:
        # Initialize database
        chat_db.init_db()
        logger.info("Database initialized successfully")
        
        # Check vector store status
        chroma_path = os.getenv('CHROMA_DB_PATH', '/app/data/chroma_db')
        if os.path.exists(chroma_path):
            logger.info("Found existing vector store")
            # Force reload of vector store
            rag = get_rag()
            rag.ensure_models_loaded()
            if rag.vector_store:
                logger.info("Vector store loaded successfully")
            else:
                logger.warning("Vector store exists but couldn't be loaded")
        else:
            logger.info("No existing vector store found")
            
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
            async for chunk in rag.stream_response(user_prompt, style="conversational"):
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
            async for chunk in rag.stream_response(user_prompt, style="detailed"):
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
    chunking_method: str = Form(default="auto"),  # auto, naive, qa, resume, manual, table, laws, presentation, picture, one, email
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
                    logger.warning(f"Invalid chunking method '{chunking_method}', using naive")
                    selected_method = ChunkingMethod.NAIVE
            
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
async def list_documents(token: str = Depends(oauth2_scheme)):
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
                    logger.warning(f"Invalid chunking method '{method}', using naive")
                    selected_method = ChunkingMethod.NAIVE
            
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
            
            # First, remove any existing chunks for this document
            try:
                get_rag().remove_document_from_vectorstore(filename)
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
            
            # Try both 'source' and 'source_file' metadata keys
            results = None
            for metadata_key in ['source_file', 'source']:
                try:
                    results = collection.get(
                        where={metadata_key: filename},
                        include=["documents", "metadatas", "embeddings"]
                    )
                    if results and results.get('ids'):
                        logger.info(f"Found {len(results['ids'])} chunks using metadata key '{metadata_key}'")
                        break
                except Exception as e:
                    logger.debug(f"Query with '{metadata_key}' failed: {e}")
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
async def get_available_models(current_user: dict = Depends(get_current_user)):
    """Get list of available models from Ollama (both local and remote)"""
    try:
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
            logger.info(f"Found {len(available_models)} models from Ollama library")
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
        
        # Debug logging for categorization
        logger.info(f"Total models loaded: {len(models_list)}")
        for model in models_list:
            logger.info(f"Model: {model['name']} -> Category: {model.get('category', 'unknown')}")
        
        llm_models = [m for m in models_list if m.get('category', 'llm') == 'llm']
        embedding_models = [m for m in models_list if m.get('category', 'embedding') == 'embedding']
        
        logger.info(f"Categorized into {len(llm_models)} LLM models and {len(embedding_models)} embedding models")
        logger.info(f"LLM models: {[m['name'] for m in llm_models]}")
        logger.info(f"Embedding models: {[m['name'] for m in embedding_models]}")
        
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
    """Get current model settings"""
    try:
        rag = get_chatpdf_instance()
        return {
            "success": True,
            "llm": rag.llm_model,
            "embedding": rag.embedding_model
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
        
        if not llm_model or not embedding_model:
            raise HTTPException(status_code=400, detail="Both LLM and embedding models are required")
        
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
            if llm_model not in installed_models:
                models_to_download.append(llm_model)
                
            # Check if embedding model needs downloading
            if embedding_model not in installed_models:
                models_to_download.append(embedding_model)
            
            # Download missing models
            for model_name in models_to_download:
                logger.info(f"Downloading model: {model_name}")
                try:
                    download_response = await client.post(
                        f"{ollama_host}/api/pull",
                        json={"name": model_name},
                        timeout=300.0  # 5 minutes timeout for model download
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
        
        # Now update the models in the RAG system
        rag = get_chatpdf_instance()
        
        # Check if embedding model changed - if so, we need to re-ingest
        embedding_changed = rag.embedding_model != embedding_model
        
        # Update the models
        rag.update_models(llm_model, embedding_model)
        
        # Save settings to config file
        config_path = "model_settings.json"
        settings = {
            'llm': llm_model,
            'embedding': embedding_model
        }
        
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
        
        llm_model = request.get('llm')
        embedding_model = request.get('embedding')
        
        if not llm_model or not embedding_model:
            raise HTTPException(status_code=400, detail="Both LLM and embedding models are required")
        
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
        
        # Save settings to config file
        config_path = "model_settings.json"
        settings = {
            'llm': llm_model,
            'embedding': embedding_model
        }
        
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

def _get_method_description(method: ChunkingMethod) -> str:
    """Get description for chunking method"""
    descriptions = {
        ChunkingMethod.NAIVE: "Default consecutive chunking based on token limits",
        ChunkingMethod.QA: "For question-answer formatted documents",
        ChunkingMethod.RESUME: "Enterprise edition for resume documents", 
        ChunkingMethod.MANUAL: "Manual chunking for PDFs",
        ChunkingMethod.TABLE: "For spreadsheet/tabular data",
        ChunkingMethod.LAWS: "Legal document chunking",
        ChunkingMethod.PRESENTATION: "For PPT/presentation files",
        ChunkingMethod.PICTURE: "Image/visual content processing",
        ChunkingMethod.ONE: "Treats entire document as single chunk",
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
