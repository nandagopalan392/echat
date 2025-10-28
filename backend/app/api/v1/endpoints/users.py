"""
User Management API Endpoints
Handles user profile, activities, and statistics
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any
import logging
import datetime
import sqlite3

from app.config import settings
from app.dependencies import get_current_user, get_user_repository
from app.db.models.user import User
from app.db.repositories.user_repository import UserRepository
from app.db.base import get_db, DatabaseConnection
from app.api.v1.schemas.users import (
    UserProfileResponse,
    ActivityResponse,
    UserStatsResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/profile", response_model=Dict[str, UserProfileResponse])
async def get_user_profile(
    current_admin: User = Depends(get_current_admin_user),
    user_repo: UserRepository = Depends(get_user_repository)
):
    """
    Get current user's profile
    
    Returns:
        User profile information including username, email, role, etc.
    """
    try:
        # Get user data from repository
        user = user_repo.get_user_by_username(current_user.username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Return user profile data
        return {
            "user": UserProfileResponse(
                username=user.username,
                email=user.email or "",
                role=user.role,
                created_at=user.created_at.isoformat() if user.created_at else "",
                last_login=user.last_login.isoformat() if user.last_login else "",
                is_active=user.is_active
            )
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving user profile: {str(e)}"
        )


@router.get("/activities", response_model=Dict[str, List[ActivityResponse]])
async def get_user_activities(
    current_admin: User = Depends(get_current_admin_user),
    db: DatabaseConnection = Depends(get_db)
):
    """
    Get user activities/recent actions
    
    Returns:
        List of recent user activities including chats, document uploads, etc.
    """
    try:
        username = current_user.username
        
        # Get recent user activities from chat history and other sources
        activities = []
        
        # Get recent chat sessions using direct database access
        try:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, topic, created_at, last_updated,
                           (SELECT COUNT(*) FROM messages WHERE session_id = chat_sessions.id) as message_count
                    FROM chat_sessions
                    WHERE username = ?
                    ORDER BY last_updated DESC
                    LIMIT 10
                """, (username,))
                
                sessions = cursor.fetchall()
                for session in sessions:
                    activities.append(ActivityResponse(
                        id=f"chat_{session[0]}",
                        type="chat",
                        action="Started conversation",
                        details=(session[1] or "Chat session")[:100],
                        timestamp=session[3] or session[2],  # Use last_updated or created_at
                        metadata={
                            "session_id": session[0],
                            "message_count": session[4]
                        }
                    ))
        except Exception as e:
            logger.warning(f"Error getting chat sessions: {e}")
        
        # Add document upload activities if available
        try:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT filename, upload_date, format
                    FROM files
                    WHERE uploaded_by = ?
                    ORDER BY upload_date DESC
                    LIMIT 5
                """, (username,))
                
                files = cursor.fetchall()
                for file_data in files:
                    activities.append(ActivityResponse(
                        id=f"doc_{file_data[0]}",
                        type="document",
                        action="Uploaded document",
                        details=f"{file_data[0]} ({file_data[2]})",
                        timestamp=file_data[1],
                        metadata={"filename": file_data[0], "format": file_data[2]}
                    ))
        except Exception as e:
            logger.warning(f"Error getting document activities: {e}")
        
        # Sort by timestamp (newest first)
        activities.sort(key=lambda x: x.timestamp, reverse=True)
        
        return {"activities": activities[:20]}  # Return last 20 activities
        
    except Exception as e:
        logger.error(f"Error getting user activities: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving user activities: {str(e)}"
        )


@router.get("/stats", response_model=Dict[str, UserStatsResponse])
async def get_user_stats_general(
    current_admin: User = Depends(get_current_admin_user),
    db: DatabaseConnection = Depends(get_db)
):
    """
    Get general system statistics
    
    Returns:
        System-wide statistics including total users, sessions, and messages
    """
    try:
        # Get overall system statistics
        stats = UserStatsResponse(
            totalUsers=0,
            activeUsers=0,
            totalSessions=0,
            totalMessages=0
        )
        
        try:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Total users
                cursor.execute("SELECT COUNT(*) FROM users")
                stats.totalUsers = cursor.fetchone()[0]
                
                # Active users (users with sessions in last 30 days)
                cursor.execute("""
                    SELECT COUNT(DISTINCT username) 
                    FROM chat_sessions 
                    WHERE created_at >= datetime('now', '-30 days')
                """)
                stats.activeUsers = cursor.fetchone()[0]
                
                # Total sessions
                cursor.execute("SELECT COUNT(*) FROM chat_sessions")
                stats.totalSessions = cursor.fetchone()[0]
                
                # Total messages
                cursor.execute("SELECT COUNT(*) FROM messages")
                stats.totalMessages = cursor.fetchone()[0]
                
        except Exception as e:
            logger.warning(f"Error getting system stats: {e}")
        
        return {"stats": stats}
        
    except Exception as e:
        logger.error(f"Error getting user stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving user statistics: {str(e)}"
        )
