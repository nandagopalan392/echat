"""
Pydantic schemas for admin API requests and responses
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class AddUserRequest(BaseModel):
    """Admin add user request schema"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    role: str = Field(..., description="User role: Engineer, Manager, Business Development, or Associate")
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "new_user",
                "password": "securepass123",
                "role": "Engineer"
            }
        }


class UserListItem(BaseModel):
    """Single user item in list"""
    user_id: Optional[int] = None
    username: str
    role: str
    created_at: Optional[str] = None
    last_login: Optional[str] = None
    is_active: bool = True
    is_admin: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 1,
                "username": "john_doe",
                "role": "Engineer",
                "created_at": "2025-10-16T10:30:00",
                "last_login": "2025-10-16T15:45:00",
                "is_active": True,
                "is_admin": False
            }
        }


class UsersListResponse(BaseModel):
    """Response with list of all users"""
    users: List[UserListItem]
    
    class Config:
        json_schema_extra = {
            "example": {
                "users": [
                    {
                        "user_id": 1,
                        "username": "admin",
                        "role": "Admin",
                        "created_at": "2025-01-01T00:00:00",
                        "is_admin": True
                    },
                    {
                        "user_id": 2,
                        "username": "john_doe",
                        "role": "Engineer",
                        "created_at": "2025-10-16T10:30:00",
                        "is_admin": False
                    }
                ]
            }
        }


class UserStatsDetail(BaseModel):
    """Detailed user statistics"""
    username: str
    total_sessions: int = 0
    total_messages: int = 0
    avg_messages_per_session: float = 0.0
    last_activity: Optional[str] = None
    most_active_day: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "john_doe",
                "total_sessions": 45,
                "total_messages": 234,
                "avg_messages_per_session": 5.2,
                "last_activity": "2025-10-16T15:45:00",
                "most_active_day": "2025-10-15"
            }
        }


class ActivityStats(BaseModel):
    """System activity statistics"""
    daily_active_users: int = 0
    weekly_active_users: int = 0
    monthly_active_users: int = 0
    total_sessions_today: int = 0
    total_messages_today: int = 0
    peak_hour: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "daily_active_users": 12,
                "weekly_active_users": 45,
                "monthly_active_users": 89,
                "total_sessions_today": 67,
                "total_messages_today": 345,
                "peak_hour": "14:00"
            }
        }


class MessageResponse(BaseModel):
    """Generic message response"""
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Operation completed successfully"
            }
        }
