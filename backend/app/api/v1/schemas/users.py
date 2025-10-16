"""
Pydantic schemas for user management API requests and responses
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class UserProfileResponse(BaseModel):
    """User profile response schema"""
    username: str
    email: Optional[str] = None
    role: str
    created_at: str
    last_login: Optional[str] = None
    is_active: bool = True
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "john_doe",
                "email": "john@example.com",
                "role": "Engineer",
                "created_at": "2025-10-16T10:30:00",
                "last_login": "2025-10-16T15:45:00",
                "is_active": True
            }
        }


class ActivityResponse(BaseModel):
    """User activity response schema"""
    id: str
    type: str
    action: str
    details: str
    timestamp: str
    metadata: Dict[str, Any] = {}
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "chat_123",
                "type": "chat",
                "action": "Started conversation",
                "details": "Chat about Python programming",
                "timestamp": "2025-10-16T14:30:00",
                "metadata": {
                    "session_id": 123,
                    "message_count": 5
                }
            }
        }


class UserStatsResponse(BaseModel):
    """User statistics response schema"""
    totalUsers: int = 0
    activeUsers: int = 0
    totalSessions: int = 0
    totalMessages: int = 0
    
    class Config:
        json_schema_extra = {
            "example": {
                "totalUsers": 150,
                "activeUsers": 45,
                "totalSessions": 1234,
                "totalMessages": 5678
            }
        }
