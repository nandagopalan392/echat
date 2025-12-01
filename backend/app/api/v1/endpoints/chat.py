"""
Chat API Endpoints

FastAPI routes for chat functionality including message sending,
session management, RLHF feedback, and file uploads.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse
from typing import Dict, Any

from app.dependencies import get_current_user
from app.db.models.user import User
from app.api.v1.schemas.chat import (
    Message,
    MessageUpdate,
    RLHFFeedback,
    ChatResponse,
    SessionMessagesResponse,
    SessionsResponse,
    FeedbackResponse,
    UploadResponse
)
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)

router = APIRouter()


# Initialize chat service
def get_chat_service() -> ChatService:
    """Dependency to get chat service instance"""
    return ChatService()


@router.post("/chat/send", response_model=ChatResponse)
async def send_message(
    message: Message,
    current_user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service)
) -> Dict[str, Any]:
    """
    Send a chat message and receive RLHF response options.
    
    This endpoint generates two AI responses with different styles (conversational
    and detailed) for the user to choose from as part of RLHF training.
    
    Args:
        message: Message content and optional session_id
        current_user: Authenticated user object
        chat_service: Chat service instance
        
    Returns:
        Dictionary with response options and session information
    """
    try:
        username = current_user.username
        logger.info(f"Processing chat message from user '{username}'")
        
        result = await chat_service.send_message(
            content=message.content,
            session_id=message.session_id,
            username=username
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error in send_message endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/chat/sessions", response_model=SessionsResponse)
async def get_sessions(
    current_user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service)
) -> Dict[str, Any]:
    """
    Get all chat sessions for the current user.
    
    Args:
        current_user: Authenticated user object
        chat_service: Chat service instance
        
    Returns:
        Dictionary containing list of user's chat sessions
    """
    try:
        username = current_user.username
        sessions = chat_service.get_user_sessions(username)
        
        return {"sessions": sessions}
        
    except Exception as e:
        logger.error(f"Error getting sessions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/chat/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
async def get_session_messages(
    session_id: int,
    current_user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service)
) -> Dict[str, Any]:
    """
    Get all messages for a specific chat session.
    
    Args:
        session_id: ID of the session to retrieve messages for
        current_user: Authenticated user object
        chat_service: Chat service instance
        
    Returns:
        Dictionary with messages and session_id
    """
    try:
        username = current_user.username
        result = chat_service.get_session_messages(session_id, username)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting session messages: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/chat/rlhf-feedback", response_model=FeedbackResponse)
async def submit_rlhf_feedback(
    feedback: RLHFFeedback,
    current_user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service)
) -> Dict[str, str]:
    """
    Submit RLHF feedback for response selection.
    
    This endpoint processes user feedback on which AI response they preferred,
    saves the preference for RLHF training, and adds the chosen response to
    the chat history.
    
    Args:
        feedback: Feedback data including session_id and chosen_index
        current_user: Authenticated user object
        chat_service: Chat service instance
        
    Returns:
        Success status message
    """
    try:
        username = current_user.username
        
        result = await chat_service.process_rlhf_feedback(
            session_id=feedback.session_id,
            chosen_index=feedback.chosen_index,
            username=username
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing RLHF feedback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.put("/chat/message/update")
async def update_message(
    message_update: MessageUpdate,
    current_user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service)
) -> Dict[str, Any]:
    """
    Update the latest AI message in a session.
    
    Used after RLHF response selection to update the message content.
    
    Args:
        message_update: Update data including session_id and new content
        current_user: Authenticated user object
        chat_service: Chat service instance
        
    Returns:
        Success status with message_id
    """
    try:
        username = current_user.username
        
        result = chat_service.update_message(
            session_id=message_update.session_id,
            content=message_update.content,
            username=username
        )
        
        return result
        
    except ValueError as ve:
        logger.error(f"Not found: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Error updating message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/chat/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service)
) -> Dict[str, str]:
    """
    Upload a file to be processed for RAG context.
    
    The file is ingested into the vector store and becomes available
    for retrieval-augmented generation in chat responses.
    
    Args:
        file: Uploaded file
        current_user: Authenticated user object
        chat_service: Chat service instance
        
    Returns:
        Success message with file_id
    """
    try:
        username = current_user.username
        
        # Read file content
        contents = await file.read()
        
        result = await chat_service.upload_file(
            filename=file.filename,
            file_content=contents,
            username=username
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
