"""
Chat repository for database operations
"""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.db.base import DatabaseConnection
from app.db.models.chat import ChatSession, Message, ResponseCache

logger = logging.getLogger(__name__)


class ChatRepository:
    """Repository for chat-related database operations"""
    
    def __init__(self, db: DatabaseConnection):
        self.db = db
    
    # Session operations
    def create_session(self, username: str, first_message: str = "") -> Optional[int]:
        """Create a new chat session"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Generate topic from first message
                topic = first_message[:50] + "..." if len(first_message) > 50 else first_message
                if not topic:
                    topic = "New Chat"
                
                now = datetime.now().isoformat()
                cursor.execute(
                    '''INSERT INTO chat_sessions (username, topic, created_at, last_updated)
                       VALUES (?, ?, ?, ?)''',
                    (username, topic, now, now)
                )
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return None
    
    def update_session_topic(self, session_id: int, new_topic: str) -> bool:
        """Update session topic"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'UPDATE chat_sessions SET topic = ? WHERE id = ?',
                    (new_topic, session_id)
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating session topic: {e}")
            return False
    
    def get_user_sessions(self, username: str) -> List[ChatSession]:
        """Get all sessions for a user"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''SELECT id, username, topic, created_at, last_updated
                       FROM chat_sessions
                       WHERE username = ?
                       ORDER BY last_updated DESC''',
                    (username,)
                )
                rows = cursor.fetchall()
                return [ChatSession.from_db_row(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            return []
    
    # Message operations
    def save_message(self, session_id: int, content: str, is_user: bool) -> Optional[int]:
        """Save a message to a session"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                cursor.execute(
                    '''INSERT INTO messages (session_id, content, is_user, timestamp)
                       VALUES (?, ?, ?, ?)''',
                    (session_id, content, is_user, now)
                )
                
                # Update session's last_updated
                cursor.execute(
                    'UPDATE chat_sessions SET last_updated = ? WHERE id = ?',
                    (now, session_id)
                )
                
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error saving message: {e}")
            return None
    
    def update_message(self, message_id: int, content: str) -> bool:
        """Update a message's content"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'UPDATE messages SET content = ? WHERE id = ?',
                    (content, message_id)
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating message: {e}")
            return False
    
    def get_latest_ai_message_id(self, session_id: int) -> Optional[int]:
        """Get the ID of the latest AI message in a session"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''SELECT id FROM messages
                       WHERE session_id = ? AND is_user = 0
                       ORDER BY timestamp DESC
                       LIMIT 1''',
                    (session_id,)
                )
                row = cursor.fetchone()
                return row[0] if row else None
        except Exception as e:
            logger.error(f"Error getting latest AI message: {e}")
            return None
    
    def get_session_messages(self, session_id: int) -> List[Message]:
        """Get all messages for a session"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''SELECT id, session_id, content, is_user, timestamp
                       FROM messages
                       WHERE session_id = ?
                       ORDER BY timestamp ASC''',
                    (session_id,)
                )
                rows = cursor.fetchall()
                return [Message.from_db_row(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting session messages: {e}")
            return []
    
    # Response cache operations
    def get_cached_response(self, query: str) -> Optional[str]:
        """Get cached response for a query"""
        try:
            import hashlib
            query_hash = hashlib.md5(query.encode()).hexdigest()
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT response FROM response_cache WHERE query_hash = ?',
                    (query_hash,)
                )
                row = cursor.fetchone()
                return row[0] if row else None
        except Exception as e:
            logger.error(f"Error getting cached response: {e}")
            return None
    
    def cache_response(self, query: str, response: str) -> bool:
        """Cache a response for a query"""
        try:
            import hashlib
            query_hash = hashlib.md5(query.encode()).hexdigest()
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''INSERT OR REPLACE INTO response_cache (query_hash, query, response, timestamp)
                       VALUES (?, ?, ?, ?)''',
                    (query_hash, query, response, datetime.now().isoformat())
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error caching response: {e}")
            return False
    
    def clear_cache(self) -> bool:
        """Clear all cached responses"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM response_cache')
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
