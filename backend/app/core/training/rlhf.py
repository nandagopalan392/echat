"""
RLHF (Reinforcement Learning from Human Feedback) Manager

Handles saving and retrieving user feedback on AI responses for training purposes.
"""
import os
import sqlite3
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class RLHFManager:
    """
    Manager for RLHF feedback data.
    
    Handles storing response options, user preferences, and retrieving
    feedback data for model training.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize RLHF manager with database.
        
        Args:
            db_path: Optional custom database path
        """
        if db_path is None:
            # Use environment variable with default path
            db_dir = os.getenv('SQLITE_DB_PATH', '/app/data/db')
            db_dir = Path(db_dir).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = os.path.join(db_dir, 'rlhf.db')
        else:
            self.db_path = db_path
        
        # Ensure directory exists and has proper permissions
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        db_dir.chmod(0o777)
        
        logger.info(f"Using RLHF database path: {self.db_path}")
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize RLHF database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create preference data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS rlhf_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        chosen_index INTEGER,
                        username TEXT,
                        comment TEXT,
                        created_at TEXT
                    )
                ''')
                
                # Create response options table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS rlhf_response_options (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER,
                        question TEXT,
                        response_option_0 TEXT,
                        response_option_1 TEXT,
                        chosen_response TEXT,
                        chosen_index INTEGER,
                        username TEXT,
                        created_at TEXT,
                        message_id INTEGER
                    )
                ''')
                
                # Add indexes for faster queries
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_rlhf_feedback_session_id 
                    ON rlhf_feedback (session_id)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_rlhf_feedback_username 
                    ON rlhf_feedback (username)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_rlhf_response_options_session_id 
                    ON rlhf_response_options (session_id)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_rlhf_response_options_username 
                    ON rlhf_response_options (username)
                ''')
                
                conn.commit()
                logger.info("RLHF database initialized successfully")
                
        except Exception as e:
            logger.error(f"RLHF database initialization error: {str(e)}")
            raise
    
    def save_response_options(
        self,
        session_id: int,
        question: str,
        response_option_0: str,
        response_option_1: str,
        username: str,
        message_id: Optional[int] = None
    ) -> bool:
        """
        Save response options before user makes a choice.
        
        Args:
            session_id: Chat session ID
            question: User's question/prompt
            response_option_0: First response option
            response_option_1: Second response option
            username: Username
            message_id: Optional message ID
            
        Returns:
            Success status
        """
        try:
            now = datetime.datetime.now().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    INSERT INTO rlhf_response_options 
                    (session_id, question, response_option_0, response_option_1, 
                     username, created_at, message_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (session_id, question, response_option_0, response_option_1,
                     username, now, message_id)
                )
            
            logger.info(f"Saved RLHF response options for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving RLHF response options: {str(e)}")
            return False
    
    def save_selected_response(
        self,
        session_id: int,
        chosen_index: int,
        user_id: str,
        comment: Optional[str] = None
    ) -> bool:
        """
        Save user's selected response.
        
        Args:
            session_id: Chat session ID
            chosen_index: Index of chosen response (0 or 1)
            user_id: Username
            comment: Optional feedback comment
            
        Returns:
            Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get latest response options for this session
                cursor.execute(
                    '''
                    SELECT id, response_option_0, response_option_1 
                    FROM rlhf_response_options 
                    WHERE session_id = ? AND username = ? AND chosen_response IS NULL
                    ORDER BY created_at DESC LIMIT 1
                    ''',
                    (session_id, user_id)
                )
                
                result = cursor.fetchone()
                if not result:
                    logger.error(
                        f"No response options found for session {session_id} "
                        f"and user {user_id}"
                    )
                    return False
                
                record_id, option_0, option_1 = result
                response_options = [option_0, option_1]
                
                if chosen_index >= len(response_options):
                    logger.error(
                        f"Invalid chosen_index {chosen_index} for session {session_id}"
                    )
                    chosen_index = 0
                
                chosen_response = response_options[chosen_index]
                logger.info(
                    f"Selected response option {chosen_index} for session {session_id}: "
                    f"{chosen_response[:50]}..."
                )
                
                # Update record with chosen response
                cursor.execute(
                    '''
                    UPDATE rlhf_response_options 
                    SET chosen_response = ?, chosen_index = ?
                    WHERE id = ?
                    ''',
                    (chosen_response, chosen_index, record_id)
                )
                
                if cursor.rowcount == 0:
                    logger.error(f"Failed to update response options record {record_id}")
                    return False
                
                # Save to feedback table for backward compatibility
                now = datetime.datetime.now().isoformat()
                cursor.execute(
                    '''
                    INSERT INTO rlhf_feedback 
                    (session_id, chosen_index, username, comment, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ''',
                    (session_id, chosen_index, user_id, comment, now)
                )
                
                conn.commit()
            
            logger.info(
                f"Successfully saved RLHF selected response for session {session_id}: "
                f"option {chosen_index}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error saving RLHF selected response: {str(e)}")
            return False
    
    def get_session_preferences(self, session_id: int) -> List[Dict[str, Any]]:
        """
        Get all preference data for a specific session.
        
        Args:
            session_id: Chat session ID
            
        Returns:
            List of preference records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Check if table exists
                cursor.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name='rlhf_response_options'"
                )
                if not cursor.fetchone():
                    logger.warning("rlhf_response_options table does not exist")
                    return []
                
                # Get all records for this session
                cursor.execute(
                    """
                    SELECT * FROM rlhf_response_options
                    WHERE session_id = ?
                    ORDER BY created_at ASC
                    """,
                    (session_id,)
                )
                
                results = []
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    results.append(row_dict)
                    logger.debug(
                        f"Found preference record: session={row_dict.get('session_id')}, "
                        f"chosen_index={row_dict.get('chosen_index')}, "
                        f"has_chosen_response={bool(row_dict.get('chosen_response'))}"
                    )
                
                logger.info(
                    f"Retrieved {len(results)} preference records for session {session_id}"
                )
                return results
                
        except Exception as e:
            logger.error(f"Error retrieving session preferences: {str(e)}")
            return []
    
    def get_preference_data(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieve preference data for training.
        
        Args:
            limit: Maximum number of records to retrieve
            
        Returns:
            List of preference data records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute(
                    """
                    SELECT * FROM rlhf_response_options
                    WHERE chosen_response IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,)
                )
                
                results = [dict(row) for row in cursor.fetchall()]
                logger.info(f"Retrieved {len(results)} preference records")
                return results
                
        except Exception as e:
            logger.error(f"Error retrieving preference data: {str(e)}")
            return []
