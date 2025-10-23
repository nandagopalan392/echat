"""
User Repository
Handles all database operations related to users
"""
import sqlite3
import logging
from typing import Optional, List
from datetime import datetime

from app.db.base import DatabaseConnection
from app.db.models.user import User, VALID_ROLES
from app.core.auth.password import hash_password, verify_password

logger = logging.getLogger(__name__)


class UserRepository:
    """Repository for user data access"""
    
    def __init__(self, db: DatabaseConnection):
        self.db = db
        # Table creation handled by init_db.py
    
    def create_user(self, username: str, password: str, role: str = 'Engineer', 
                   email: str = None, is_admin: bool = False) -> bool:
        """
        Create a new user
        
        Args:
            username: Username
            password: Plain text password (will be hashed)
            role: User role
            email: User email
            is_admin: Whether user is admin
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if role not in VALID_ROLES:
                raise ValueError(f"Invalid role: {role}. Must be one of {VALID_ROLES}")
            
            # Hash password
            password_hash = hash_password(password)
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''INSERT INTO users (username, password_hash, role, email, is_admin) 
                       VALUES (?, ?, ?, ?, ?)''',
                    (username, password_hash, role, email, is_admin)
                )
                logger.info(f"Successfully created user: {username} with role: {role}")
                return True
                
        except sqlite3.IntegrityError as e:
            logger.error(f"User already exists: {username}")
            return False
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            return False
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Get user by username
        
        Args:
            username: Username to search for
            
        Returns:
            User object or None
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''SELECT username, password_hash, created_at, is_admin, is_active, 
                              role, email, last_login
                       FROM users 
                       WHERE username = ?''',
                    (username,)
                )
                row = cursor.fetchone()
                
                if row:
                    return User.from_db_row(row)
                return None
                
        except Exception as e:
            logger.error(f"Error getting user by username: {str(e)}")
            return None
    
    def authenticate_user(self, username: str, password: str) -> bool:
        """
        Authenticate user with username and password
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            True if authenticated, False otherwise
        """
        try:
            user = self.get_user_by_username(username)
            if not user:
                return False
            
            # Verify password
            if verify_password(password, user.password_hash):
                # Update last login
                self.update_last_login(username)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False
    
    def update_last_login(self, username: str):
        """Update user's last login timestamp"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE username = ?',
                    (username,)
                )
        except Exception as e:
            logger.error(f"Error updating last login: {str(e)}")
    
    def get_all_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        """
        Get all users with pagination
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of User objects
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''SELECT username, password_hash, created_at, is_admin, is_active,
                              role, email, last_login
                       FROM users 
                       ORDER BY created_at DESC
                       LIMIT ? OFFSET ?''',
                    (limit, skip)
                )
                rows = cursor.fetchall()
                return [User.from_db_row(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting all users: {str(e)}")
            return []
    
    def update_user(self, username: str, **kwargs) -> bool:
        """
        Update user fields
        
        Args:
            username: Username of user to update
            **kwargs: Fields to update (role, email, is_active, is_admin)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            allowed_fields = ['role', 'email', 'is_active', 'is_admin']
            updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
            
            if not updates:
                return False
            
            # Validate role if being updated
            if 'role' in updates and updates['role'] not in VALID_ROLES:
                raise ValueError(f"Invalid role: {updates['role']}")
            
            # Build UPDATE query
            set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
            values = list(updates.values()) + [username]
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f'UPDATE users SET {set_clause} WHERE username = ?',
                    values
                )
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error updating user: {str(e)}")
            return False
    
    def delete_user(self, username: str) -> bool:
        """
        Delete a user
        
        Args:
            username: Username to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM users WHERE username = ?', (username,))
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error deleting user: {str(e)}")
            return False
    
    def is_admin(self, username: str) -> bool:
        """
        Check if user is an admin
        
        Args:
            username: Username to check
            
        Returns:
            True if admin, False otherwise
        """
        user = self.get_user_by_username(username)
        return user.is_admin if user else False
    
    def change_password(self, username: str, new_password: str) -> bool:
        """
        Change user password
        
        Args:
            username: Username
            new_password: New plain text password
            
        Returns:
            True if successful, False otherwise
        """
        try:
            password_hash = hash_password(new_password)
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'UPDATE users SET password_hash = ? WHERE username = ?',
                    (password_hash, username)
                )
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error changing password: {str(e)}")
            return False
