"""
Admin API Endpoints
Handles administrative functions including user management and system statistics
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List
import logging

from app.config import settings
from app.dependencies import get_current_admin_user, get_user_repository
from app.db.models.user import User
from app.db.repositories.user_repository import UserRepository
from app.db.base import get_db, DatabaseConnection
from app.api.v1.schemas.admin import (
    AddUserRequest,
    UsersListResponse,
    UserListItem,
    UserStatsDetail,
    ActivityStats,
    MessageResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/add-user", response_model=MessageResponse)
async def add_user(
    user_data: AddUserRequest,
    admin: User = Depends(get_current_admin_user),
    user_repo: UserRepository = Depends(get_user_repository)
):
    """
    Add a new user (admin only)
    
    Args:
        user_data: User information (username, password, role)
        admin: Current admin user (from dependency)
        user_repo: User repository for database operations
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If username exists or role is invalid
    """
    try:
        # Check if user already exists
        existing_user = user_repo.get_user_by_username(user_data.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        # Validate role
        from app.db.models.user import VALID_ROLES
        if user_data.role not in VALID_ROLES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role. Must be one of: {', '.join(VALID_ROLES)}"
            )
        
        # Add user to database using repository
        success = user_repo.create_user(
            username=user_data.username,
            password=user_data.password,
            role=user_data.role,
            email=user_data.email
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to add user"
            )
        
        logger.info(f"Admin {admin.username} added new user: {user_data.username}")
        return MessageResponse(message="User added successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding user: {str(e)}"
        )


@router.delete("/delete-user/{username}", response_model=MessageResponse)
async def delete_user(
    username: str,
    admin: User = Depends(get_current_admin_user),
    user_repo: UserRepository = Depends(get_user_repository)
):
    """
    Delete a user (admin only)
    
    Args:
        username: Username to delete
        admin: Current admin user (from dependency)
        user_repo: User repository for database operations
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If trying to delete self or user doesn't exist
    """
    try:
        # Prevent admin from deleting themselves
        if username == admin.username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account"
            )
        
        # Check if user exists
        user = user_repo.get_user_by_username(username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User '{username}' not found"
            )
        
        # Delete user
        success = user_repo.delete_user(username)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete user"
            )
        
        logger.info(f"Admin {admin.username} deleted user: {username}")
        return MessageResponse(message=f"User '{username}' deleted successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user {username}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting user: {str(e)}"
        )


@router.get("/users", response_model=UsersListResponse)
async def get_users(
    admin: User = Depends(get_current_admin_user),
    user_repo: UserRepository = Depends(get_user_repository)
):
    """
    Get list of all users (admin only)
    
    Args:
        admin: Current admin user (from dependency)
        user_repo: User repository for database operations
        
    Returns:
        List of all users with their details
    """
    try:
        # Get all users from repository
        users = user_repo.get_all_users()
        
        # Convert User objects to UserListItem schema
        user_list = [
            UserListItem(
                user_id=None,  # Can add user_id to User model if needed
                username=user.username,
                email=user.email,
                role=user.role,
                created_at=user.created_at.isoformat() if user.created_at else None,
                last_login=user.last_login.isoformat() if user.last_login else None,
                is_active=user.is_active,
                is_admin=user.is_admin
            )
            for user in users
        ]
        
        return UsersListResponse(users=user_list)
    except Exception as e:
        logger.error(f"Error getting users list: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving users: {str(e)}"
        )


@router.get("/user-stats/{username}", response_model=Dict[str, Any])
async def get_user_stats(
    username: str,
    admin: User = Depends(get_current_admin_user),
    db: DatabaseConnection = Depends(get_db)
):
    """
    Get detailed statistics for a specific user (admin only)
    
    Args:
        username: Username to get stats for
        admin: Current admin user (from dependency)
        db: Database connection
        
    Returns:
        Detailed user statistics
    """
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get total messages and last message
            cursor.execute("""
                SELECT COUNT(*), MAX(m.timestamp), m.content
                FROM messages m
                JOIN chat_sessions cs ON m.session_id = cs.id
                WHERE cs.username = ?
            """, (username,))
            messages_data = cursor.fetchone()
            total_messages = messages_data[0] if messages_data else 0
            last_active = messages_data[1] if messages_data else None
            last_message = messages_data[2] if messages_data else None

            # Get total sessions
            cursor.execute("""
                SELECT COUNT(*)
                FROM chat_sessions
                WHERE username = ?
            """, (username,))
            total_sessions = cursor.fetchone()[0]

            # Get recent chats with their latest messages
            cursor.execute("""
                SELECT cs.topic, cs.last_updated,
                       (SELECT content FROM messages 
                        WHERE session_id = cs.id 
                        ORDER BY timestamp DESC LIMIT 1) as last_message
                FROM chat_sessions cs
                WHERE cs.username = ?
                ORDER BY cs.last_updated DESC
                LIMIT 5
            """, (username,))
            
            recent_chats = [{
                "topic": row[0] or "New Chat",
                "date": row[1],
                "lastMessage": row[2] or "No messages"
            } for row in cursor.fetchall()]

            stats = {
                "totalMessages": total_messages,
                "totalSessions": total_sessions,
                "lastActive": last_active,
                "lastMessage": last_message,
                "recentChats": recent_chats
            }
            
            return {"data": stats}  # Wrap the stats in a data field
    except Exception as e:
        logger.error(f"Error getting user stats for {username}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving user statistics: {str(e)}"
        )


@router.get("/activity-stats", response_model=Dict[str, Any])
async def get_activity_stats(
    admin: User = Depends(get_current_admin_user),
    db: DatabaseConnection = Depends(get_db)
):
    """
    Get system-wide activity statistics (admin only)
    
    Args:
        admin: Current admin user (from dependency)
        db: Database connection
        
    Returns:
        System activity statistics
    """
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get total stats
            cursor.execute("SELECT COUNT(*) FROM users WHERE is_admin = 0")
            total_users = cursor.fetchone()[0]

            cursor.execute("""
                SELECT COUNT(DISTINCT username) 
                FROM chat_sessions 
                WHERE DATE(last_updated) = DATE('now')
            """)
            active_users = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM messages")
            total_messages = cursor.fetchone()[0]

            # Get recent activities
            cursor.execute("""
                SELECT cs.username, m.content, m.timestamp
                FROM messages m
                JOIN chat_sessions cs ON m.session_id = cs.id
                ORDER BY m.timestamp DESC
                LIMIT 10
            """)
            activities = [{
                "username": row[0],
                "action": "sent a message",
                "timestamp": row[2]
            } for row in cursor.fetchall()]

            stats = {
                "totalUsers": total_users,
                "activeUsers": active_users,
                "totalMessages": total_messages,
                "recentActivities": activities
            }
            
            return {"data": stats}  # Wrap stats in data field
    except Exception as e:
        logger.error(f"Error getting activity stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving activity statistics: {str(e)}"
        )
