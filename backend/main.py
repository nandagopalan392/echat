import sys
import os
import json
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Request, status, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import jwt
import datetime
from chat_db import ChatDB
from rag import ChatPDF, get_chatpdf_instance
from rlhf import RLHF
import logging
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
                success = get_rag().ingest_with_storage(str(temp_path), file.filename)
                if not success:
                    failed_files.append(file.filename)
                    logger.warning(f"Failed to process PDF: {file.filename}")
                else:
                    processed_files.append(file.filename)
                    logger.info(f"Successfully processed PDF: {file.filename}")
            else:
                # For non-PDF files, we still want to store them in MinIO even if we can't process them
                logger.info(f"Storing non-PDF file: {file.filename}")
                try:
                    # Determine content type
                    content_type = None
                    if file.filename.lower().endswith('.docx'):
                        content_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                    elif file.filename.lower().endswith('.xlsx'):
                        content_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    elif file.filename.lower().endswith('.csv'):
                        content_type = 'text/csv'
                    elif file.filename.lower().endswith('.txt'):
                        content_type = 'text/plain'
                    
                    # Store in MinIO using document storage
                    from document_storage import get_document_storage
                    doc_storage = get_document_storage()
                    doc_info = doc_storage.store_document(str(temp_path), file.filename, content_type)
                    
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
            logger.warning(f"Could not check GPU compatibility (proceeding anyway): {e}")
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

@app.get("/api/admin/files")
async def list_admin_files():
    """Get list of all files with statistics for admin dashboard"""
    try:
        rag = get_chatpdf_instance()
        doc_storage = rag.doc_storage
        
        # Get all documents
        all_documents = doc_storage.list_all_documents()
        
        # Calculate statistics
        total_files = len(all_documents)
        total_size = sum(doc.get('file_size', 0) for doc in all_documents)
        
        # Calculate format statistics
        format_stats = {}
        for doc in all_documents:
            # Get proper format extension
            content_type = doc.get('content_type', 'unknown')
            filename = doc.get('filename', '')
            
            if content_type == 'unknown' or not content_type:
                # Try to get extension from filename
                if '.' in filename:
                    format_ext = filename.split('.')[-1].lower()
                else:
                    format_ext = 'unknown'
            else:
                # Map content type to format
                format_mapping = {
                    'application/pdf': 'pdf',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
                    'text/csv': 'csv',
                    'text/plain': 'txt'
                }
                format_ext = format_mapping.get(content_type, content_type.split('/')[-1] if '/' in content_type else content_type)
            
            if format_ext not in format_stats:
                format_stats[format_ext] = {'count': 0, 'size': 0}
            format_stats[format_ext]['count'] += 1
            format_stats[format_ext]['size'] += doc.get('file_size', 0)
        
        # Convert format stats to list
        format_stats_list = [
            {
                'format': format_name,
                'count': stats['count'],
                'size': stats['size']
            }
            for format_name, stats in format_stats.items()
        ]
        
        # Prepare file list with additional metadata
        files_list = []
        for doc in all_documents:
            # Get ingestion status for current model
            ingestion_status = doc.get('model_status', {})
            current_model = rag.embedding_model
            is_ingested = current_model in ingestion_status
            
            # Extract file format from content type or filename
            content_type = doc.get('content_type', 'unknown')
            if content_type == 'unknown' or not content_type:
                # Try to get extension from filename
                filename = doc.get('filename', '')
                if '.' in filename:
                    format_ext = filename.split('.')[-1].lower()
                else:
                    format_ext = 'unknown'
            else:
                # Map content type to format
                format_mapping = {
                    'application/pdf': 'pdf',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
                    'text/csv': 'csv',
                    'text/plain': 'txt'
                }
                format_ext = format_mapping.get(content_type, content_type.split('/')[-1] if '/' in content_type else content_type)
            
            files_list.append({
                'id': doc['id'],
                'filename': doc['filename'],
                'format': format_ext,  # Match frontend expectation
                'size': doc.get('file_size', 0),
                'upload_date': doc.get('uploaded_at', doc.get('upload_timestamp', '')),  # Match frontend expectation
                'uploadDate': doc.get('uploaded_at', doc.get('upload_timestamp', '')),  # Keep for backward compatibility
                'contentType': doc.get('content_type', 'unknown'),
                'isIngested': is_ingested,
                'ingestedModels': list(ingestion_status.keys()) if ingestion_status else [],
                'status': 'ingested' if is_ingested else 'uploaded'
            })
        logger.info(f"Admin files listed: {len(files_list)} files")
        return {
            "files": files_list,
            "stats": {
                "totalFiles": total_files,
                "totalSize": total_size,
                "formatStats": format_stats_list
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing admin files: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")

@app.get("/api/admin/ingestion-status")
async def get_ingestion_status(current_user: dict = Depends(get_current_user)):
    """Get ingestion status for current embedding model"""
    try:
        rag = get_chatpdf_instance()
        doc_storage = rag.doc_storage
        
        # Get all documents
        all_documents = doc_storage.list_all_documents()
        current_model = rag.embedding_model
        
        # Categorize documents by ingestion status
        ingested_docs = []
        pending_docs = []
        
        for doc in all_documents:
            ingestion_status = doc.get('model_status', {})
            is_ingested = current_model in ingestion_status
            
            doc_info = {
                'id': doc['id'],
                'filename': doc['filename'],
                'size': doc.get('file_size', 0),
                'uploaded_at': doc.get('uploaded_at', ''),
                'content_type': doc.get('content_type', 'unknown'),
                'status': ingestion_status.get(current_model, 'pending')
            }
            
            if is_ingested:
                ingested_docs.append(doc_info)
            else:
                pending_docs.append(doc_info)
        
        # Get vector store stats
        vector_stats = rag.get_vector_store_stats()
        
        return {
            "success": True,
            "current_embedding_model": current_model,
            "total_documents": len(all_documents),
            "ingested_documents": {
                "count": len(ingested_docs),
                "files": ingested_docs
            },
            "pending_documents": {
                "count": len(pending_docs),
                "files": pending_docs
            },
            "vector_store_stats": vector_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting ingestion status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get ingestion status: {str(e)}")

@app.post("/api/admin/reingest-all")
async def reingest_all_documents(current_user: dict = Depends(get_current_user)):
    """Re-ingest all documents for current embedding model"""
    try:
        rag = get_chatpdf_instance()
        
        # Trigger re-ingestion
        success = rag.reingest_all_documents_for_current_model()
        
        if success:
            return {
                "success": True,
                "message": f"Documents re-ingestion started for embedding model: {rag.embedding_model}"
            }
        else:
            raise HTTPException(status_code=500, detail="Re-ingestion failed")
            
    except Exception as e:
        logger.error(f"Error during re-ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to re-ingest documents: {str(e)}")

@app.post("/api/admin/cleanup-orphaned-documents")
async def cleanup_orphaned_documents(current_user: dict = Depends(get_current_user)):
    """Clean up orphaned documents (in database but not in MinIO)"""
    try:
        doc_storage = get_document_storage()
        
        # Clean up orphaned documents
        orphaned_count = doc_storage.cleanup_orphaned_documents()
        
        return {
            "success": True,
            "message": f"Cleaned up {orphaned_count} orphaned documents",
            "orphaned_count": orphaned_count
        }
        
    except Exception as e:
        logger.error(f"Error during orphaned document cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup orphaned documents: {str(e)}")

@app.get("/api/gpu/memory-info")
async def get_gpu_memory_info_endpoint(current_user: dict = Depends(get_current_user)):
    """Get current GPU memory information"""
    try:
        from rag import get_gpu_memory_info
        gpu_info = get_gpu_memory_info()
        
        return {
            "success": True,
            "gpu_memory": gpu_info,
            "message": f"GPU Memory - Total: {gpu_info['total']}MB, Used: {gpu_info['used']}MB, Available: {gpu_info['available']}MB"
        }
    except Exception as e:
        logger.error(f"Error getting GPU memory info: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "gpu_memory": {
                "total": 8192,
                "used": 2048,
                "free": 6144,
                "available": 6144
            },
            "message": "Could not determine GPU memory, using default estimates"
        }

@app.post("/api/models/check-compatibility")
async def check_model_compatibility_endpoint(
    request: dict,
    current_user: dict = Depends(get_current_user)
):
    """Check if selected models are compatible with current GPU memory"""
    try:
        from rag import check_model_compatibility, get_gpu_memory_info
        
        llm_model = request.get('llm')
        embedding_model = request.get('embedding')
        llm_size = request.get('llm_size')
        embedding_size = request.get('embedding_size')
        
        if not llm_model or not embedding_model:
            raise HTTPException(status_code=400, detail="Both LLM and embedding models are required")
        
        # Check LLM model compatibility
        llm_compatible, llm_message, llm_details = check_model_compatibility(llm_model, llm_size)
        
        # Check embedding model compatibility
        embedding_compatible, embedding_message, embedding_details = check_model_compatibility(embedding_model, embedding_size)
        
        # Calculate combined memory requirement
        combined_memory = llm_details['required_memory_mb'] + embedding_details['required_memory_mb']
        available_memory = llm_details['available_memory_mb']
        
        combined_compatible = combined_memory <= available_memory
        
        if combined_compatible:
            combined_message = f"✅ Both models compatible together (combined ~{combined_memory}MB, {available_memory}MB available)"
        else:
            shortage = combined_memory - available_memory
            combined_message = f"❌ Combined models require ~{combined_memory}MB but only {available_memory}MB available (shortage: {shortage}MB)"
        
        # Get current GPU info
        gpu_info = get_gpu_memory_info()
        
        return {
            "success": True,
            "compatible": llm_compatible and embedding_compatible and combined_compatible,
            "llm_check": {
                "compatible": llm_compatible,
                "message": llm_message,
                "details": llm_details
            },
            "embedding_check": {
                "compatible": embedding_compatible,
                "message": embedding_message,
                "details": embedding_details
            },
            "combined_check": {
                "compatible": combined_compatible,
                "message": combined_message,
                "required_memory_mb": combined_memory,
                "available_memory_mb": available_memory,
                "shortage_mb": max(0, combined_memory - available_memory)
            },
            "gpu_info": gpu_info,
            "recommendations": generate_compatibility_recommendations(llm_details, embedding_details, combined_compatible)
        }
        
    except Exception as e:
        logger.error(f"Error checking model compatibility: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to check model compatibility: {str(e)}")

@app.post("/api/models/check-gpu")
async def check_gpu_compatibility(
    request: dict,
    current_user: dict = Depends(get_current_user)
):
    """Check GPU compatibility for model selection"""
    try:
        from rag import check_model_compatibility
        
        llm_model = request.get('llm')
        embedding_model = request.get('embedding')
        
        if not llm_model or not embedding_model:
            raise HTTPException(status_code=400, detail="Both LLM and embedding models are required")
        
        # Check compatibility for both models
        llm_compatible, llm_message, llm_details = check_model_compatibility(llm_model)
        embedding_compatible, embedding_message, embedding_details = check_model_compatibility(embedding_model)
        
        # Calculate combined memory requirement
        combined_memory = llm_details['required_memory_mb'] + embedding_details['required_memory_mb']
        available_memory = llm_details['available_memory_mb']
        combined_compatible = combined_memory <= available_memory
        
        return {
            "compatible": llm_compatible and embedding_compatible and combined_compatible,
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
            }
        }
        
    except Exception as e:
        logger.error(f"Error checking GPU compatibility: {str(e)}")
        # Return compatible=true as fallback to not block users
        return {
            "compatible": True,
            "message": f"Could not check compatibility: {str(e)}",
            "llm_check": {"compatible": True, "message": "Check skipped"},
            "embedding_check": {"compatible": True, "message": "Check skipped"},
            "combined_check": {"compatible": True, "message": "Check skipped"}
        }

@app.get("/api/files/list")
async def list_files(token: str = Depends(oauth2_scheme)):
    """Get list of uploaded files"""
    try:
        rag = get_chatpdf_instance()
        doc_storage = rag.doc_storage
        
        # Get all documents
        all_documents = doc_storage.list_all_documents()
        
        # Format for frontend
        files = []
        for doc in all_documents:
            # Determine if document is indexed (has successful ingestion)
            model_status = doc.get('model_status', {})
            indexed = any(status == 'completed' for status in model_status.values())
            
            # Get the embedding model used (first completed one, or first available)
            embedding_model = None
            for model, status in model_status.items():
                if status == 'completed':
                    embedding_model = model
                    break
            if not embedding_model and model_status:
                embedding_model = list(model_status.keys())[0]
            
            files.append({
                "filename": doc.get('filename', 'Unknown'),
                "size": doc.get('file_size', 0),  # Map file_size to size
                "upload_date": doc.get('uploaded_at', ''),  # Map uploaded_at to upload_date  
                "indexed": indexed,  # Determine from model_status
                "embedding_model": embedding_model or 'Unknown'  # Get from model_status
            })
        
        return {"files": files, "total": len(files)}
        
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/vectorstore/stats")
async def get_vectorstore_stats(token: str = Depends(oauth2_scheme)):
    """Get vector store statistics"""
    try:
        rag = get_chatpdf_instance()
        
        # Get vector store statistics
        stats = rag.get_vectorstore_stats()
        
        return {"stats": stats}
    except Exception as e:
        logger.error(f"Error getting vector store stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vectorstore/reingest")
async def reingest_vectorstore(token: str = Depends(oauth2_scheme)):
    """Re-ingest all documents into vector store"""
    try:
        rag = get_chatpdf_instance()
        
        # Re-ingest all documents
        success = rag.re_ingest_all_documents()
        
        if success:
            return {"message": "Documents re-ingested successfully"}
        else:
            raise HTTPException(status_code=500, detail="Re-ingestion failed")
    except Exception as e:
        logger.error(f"Error re-ingesting documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/vectorstore/clear")
async def clear_vectorstore(token: str = Depends(oauth2_scheme)):
    """Clear the entire vector store"""
    try:
        rag = get_chatpdf_instance()
        
        # Clear vector store
        success = rag.clear_vectorstore()
        
        if success:
            return {"message": "Vector store cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear vector store")
    except Exception as e:
        logger.error(f"Error clearing vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def format_model_size(size_value) -> str:
    """Format model size from bytes or string to human-readable format"""
    if not size_value or size_value == 'Unknown':
        return "Unknown"
    
    # If it's already a formatted string, return as-is
    if isinstance(size_value, str):
        # Check if it contains size indicators
        if any(unit in size_value.lower() for unit in ['gb', 'mb', 'kb', 'various']):
            return size_value
        # Try to parse as numeric bytes
        try:
            size_bytes = int(float(size_value))
        except (ValueError, TypeError):
            return size_value
    else:
        try:
            size_bytes = int(size_value)
        except (ValueError, TypeError):
            return str(size_value) if size_value else "Unknown"
    
    if size_bytes == 0:
        return "Unknown"
        
    if size_bytes >= 1073741824:  # 1GB = 1024^3 bytes
        size_gb = size_bytes / 1073741824
        return f"{size_gb:.1f}GB"
    elif size_bytes >= 1048576:  # 1MB = 1024^2 bytes  
        size_mb = size_bytes / 1048576
        return f"{size_mb:.0f}MB"
    elif size_bytes >= 1024:  # 1KB = 1024 bytes
        size_kb = size_bytes / 1024
        return f"{size_kb:.0f}KB"
    else:
        return f"{size_bytes}B"
