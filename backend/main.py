import sys
import os
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Request, status, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import jwt
import datetime
from chat_db import ChatDB
from rag import ChatPDF
from rlhf import RLHF
import logging
import os
import pandas as pd
from docx import Document
import sqlite3
from sse_starlette.sse import EventSourceResponse
import asyncio
import json
import time
from contextlib import contextmanager
import shutil
from pathlib import Path
import random

app = FastAPI(
    title="Chat API",
    description="API for chat application with PDF processing capabilities",
    version="1.0.0"
)

chat_db = ChatDB()
rlhf_db = RLHF()

# Add lazy loading for RAG instance
_rag_instance = None

def get_rag():
    """Lazy load RAG instance only when needed"""
    global _rag_instance
    if _rag_instance is None:
        try:
            logging.info("Initializing RAG models...")
            _rag_instance = ChatPDF()  # Initialize only once when needed
            logging.info("RAG models initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize RAG: {str(e)}")
            raise
    return _rag_instance

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
            
            # Get the first response with standard style
            response_a_chunks = []
            rag = get_rag()
            async for chunk in rag.stream_response(user_prompt, style="standard"):
                response_a_chunks.append(chunk)
            
            # Parse the final response from chunks
            response_a_content = ""
            response_a_thinking = ""
            for chunk in response_a_chunks:
                try:
                    chunk_data = json.loads(chunk)
                    if chunk_data.get("thinking"):
                        response_a_thinking = chunk_data["thinking"]
                    if chunk_data.get("content"):
                        response_a_content = chunk_data["content"]
                except json.JSONDecodeError:
                    # If it's not JSON, treat as plain text
                    response_a_content += chunk
            
            # Fallback if no content was extracted
            if not response_a_content:
                response_a_content = "".join(response_a_chunks)
            
            logger.info(f"Response A - Thinking: {len(response_a_thinking)} chars, Content: {len(response_a_content)} chars")
            
            # Add a small delay between requests
            await asyncio.sleep(1)
            
            # Get second response with conversational style
            response_b_chunks = []
            async for chunk in rag.stream_response(user_prompt, style="conversational"):
                response_b_chunks.append(chunk)
            
            # Parse the final response from chunks
            response_b_content = ""
            response_b_thinking = ""
            for chunk in response_b_chunks:
                try:
                    chunk_data = json.loads(chunk)
                    if chunk_data.get("thinking"):
                        response_b_thinking = chunk_data["thinking"]
                    if chunk_data.get("content"):
                        response_b_content = chunk_data["content"]
                except json.JSONDecodeError:
                    # If it's not JSON, treat as plain text
                    response_b_content += chunk
            
            # Fallback if no content was extracted
            if not response_b_content:
                response_b_content = "".join(response_b_chunks)
            
            logger.info(f"Response B - Thinking: {len(response_b_thinking)} chars, Content: {len(response_b_content)} chars")
            
            # Prepare the response options for RLHF with structured data
            response_options = [
                {
                    "thinking": response_a_thinking,
                    "content": response_a_content,
                    "style": "standard"
                },
                {
                    "thinking": response_b_thinking,
                    "content": response_b_content,
                    "style": "conversational"
                }
            ]
            
            # Return a proper JSON response with structured data
            return JSONResponse(content={
                "content": "I've generated two possible responses from different approaches. Please select the one you prefer:",
                "full_response": "I've generated two possible responses from different approaches. Please select the one you prefer:",
                "is_final": True,
                "session_id": session_id,
                "response_options": response_options,
                "rlhf_enabled": True,  # Signal to frontend this is for RLHF
                "message": "Please choose between the following responses:",
                "thinking_included": bool(response_a_thinking or response_b_thinking)
            })
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
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
        messages = chat_db.get_session_messages(session_id)
        
        # Ensure messages are properly ordered and formatted
        return {
            "messages": [
                {
                    "content": msg[0], 
                    "isUser": bool(msg[1]),
                    "timestamp": msg[2] if len(msg) > 2 else None
                } 
                for msg in messages
            ],
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
        
        # Simply save the user's preference directly without regenerating responses
        success = rlhf_db.save_preference(
            session_id=feedback.session_id,
            chosen_index=feedback.chosen_index,
            user_id=username
        )
        
        logger.info(f"RLHF preference saved for session {feedback.session_id}")
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save RLHF feedback")
        
        # Return immediately without calling Ollama
        return {"status": "success", "message": "Feedback received and processed"}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing RLHF feedback: {str(e)}")
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
        
        # Process file
        get_rag().ingest(f"temp_{file.filename}")
        upload_progress[file_id] = 100  # Processing complete
        
        # Cleanup
        if os.path.exists(f"temp_{file.filename}"):
            os.remove(f"temp_{file.filename}")
        if file_id in upload_progress:
            del upload_progress[file_id]
            
        return {"message": "File processed successfully", "file_id": file_id}
    except Exception as e:
        if file_id in upload_progress:
            del upload_progress[file_id]
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/admin/upload")
async def upload_file(
    file: UploadFile,
    is_folder: str = Form(default="false"),
    folder_path: str = Form(default=""),
    admin: dict = Depends(check_if_admin)
):
    failed_files = []
    processed_files = []
    try:
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

            # Process PDF files
            if file.filename.lower().endswith('.pdf'):
                logger.info(f"Processing PDF file: {file.filename}")
                success = get_rag().ingest(str(temp_path))
                if not success:
                    failed_files.append(file.filename)
                    logger.warning(f"Failed to process PDF: {file.filename}")
                else:
                    processed_files.append(file.filename)
                    logger.info(f"Successfully processed PDF: {file.filename}")
            
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

@app.get("/api/admin/files")
async def get_files(admin: dict = Depends(check_if_admin)):
    try:
        files = chat_db.get_all_files()
        stats = chat_db.get_file_stats()
        logger.debug(f"Retrieved files: {files}")
        return {
            "data": {
                "files": files,
                "stats": stats
            }
        }
    except Exception as e:
        logger.error(f"Error getting files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/api/admin/rlhf-stats")
async def get_rlhf_stats(admin: dict = Depends(check_if_admin)):
    """
    Admin endpoint to get statistics about collected RLHF data
    """
    try:
        stats = rlhf_db.get_rlhf_stats()
        return {"data": stats}
    except Exception as e:
        logger.error(f"Error getting RLHF stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/api/admin/rlhf-data")
async def get_rlhf_data(limit: int = 100, admin: dict = Depends(check_if_admin)):
    """
    Admin endpoint to get RLHF training data
    """
    try:
        data = rlhf_db.get_preference_data(limit=limit)
        return {"data": data}
    except Exception as e:
        logger.error(f"Error getting RLHF data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to Chat API"}
