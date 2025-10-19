"""
Chat API Schemas

Pydantic models for chat-related API requests and responses.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class Message(BaseModel):
    """Chat message request model"""
    content: str = Field(..., description="Message content")
    session_id: Optional[int] = Field(None, description="Chat session ID")


class MessageUpdate(BaseModel):
    """Message update request model"""
    session_id: int = Field(..., description="Chat session ID")
    content: str = Field(..., description="Updated message content")


class RLHFFeedback(BaseModel):
    """RLHF feedback request model"""
    session_id: int = Field(..., description="Chat session ID")
    chosen_index: int = Field(..., description="Index of chosen response (0 or 1)")


class ChatResponse(BaseModel):
    """Chat response model"""
    content: str
    full_response: str
    is_final: bool
    session_id: int
    response_options: Optional[List[Dict[str, Any]]] = None
    rlhf_enabled: Optional[bool] = False
    message: Optional[str] = None
    thinking_included: Optional[bool] = False


class SessionMessage(BaseModel):
    """Single message in a chat session"""
    id: str
    content: str
    isUser: bool
    timestamp: Optional[str] = None


class SessionMessagesResponse(BaseModel):
    """Response containing session messages"""
    messages: List[SessionMessage]
    session_id: int


class SessionsResponse(BaseModel):
    """Response containing user chat sessions"""
    sessions: List[Dict[str, Any]]


class FeedbackResponse(BaseModel):
    """Response for feedback submission"""
    status: str
    message: str


class UploadResponse(BaseModel):
    """Response for file upload"""
    message: str
    file_id: str
