import logging
import sqlite3
import os
from pathlib import Path
import datetime  # Import datetime module correctly
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class RLHF:
    def __init__(self, db_path=None):
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
        self.init_db()

    def init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create preference data table if it doesn't exist
                cursor.execute('''CREATE TABLE IF NOT EXISTS rlhf_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    chosen_index INTEGER,
                    username TEXT,
                    comment TEXT,
                    created_at TEXT
                )''')
                
                # Add indexes for faster queries
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_rlhf_feedback_session_id ON rlhf_feedback (session_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_rlhf_feedback_username ON rlhf_feedback (username)')
                
                conn.commit()
                logger.info("RLHF database initialized successfully")
                
        except Exception as e:
            logger.error(f"RLHF database initialization error: {str(e)}")
            raise

    def save_preference(self, session_id: int, chosen_index: int, user_id: str, comment: Optional[str] = None) -> bool:
        """
        Save user's preference between two response options
        
        Args:
            session_id: The chat session ID
            chosen_index: Which response option was selected (0 or 1)
            user_id: Username of the person providing feedback
            comment: Optional comment on the feedback
            
        Returns:
            bool: Success status
        """
        try:
            # Fix: Use datetime.now() directly
            now = datetime.datetime.now().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    INSERT INTO rlhf_feedback 
                    (session_id, chosen_index, username, comment, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ''',
                    (session_id, chosen_index, user_id, comment, now)
                )
            
            logger.info(f"Saved RLHF preference for session {session_id}: option {chosen_index}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving RLHF preference: {str(e)}")
            return False

    def get_preference_data(self, limit=1000):
        """
        Retrieve preference data for training
        
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
                    """SELECT * FROM rlhf_feedback
                       ORDER BY created_at DESC
                       LIMIT ?""",
                    (limit,)
                )
                
                results = []
                for row in cursor.fetchall():
                    results.append(dict(row))
                
                return results
        except Exception as e:
            logger.error(f"Error retrieving RLHF data: {str(e)}")
            return []

    def get_rlhf_stats(self):
        """
        Get statistics about collected RLHF data
        
        Returns:
            Dictionary with statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total count
                cursor.execute("SELECT COUNT(*) FROM rlhf_feedback")
                total_samples = cursor.fetchone()[0]
                
                # Get user participation stats
                cursor.execute(
                    """SELECT username, COUNT(*) as contribution_count
                       FROM rlhf_feedback
                       GROUP BY username
                       ORDER BY contribution_count DESC
                       LIMIT 10"""
                )
                top_contributors = [
                    {"username": row[0], "count": row[1]} 
                    for row in cursor.fetchall()
                ]
                
                # Get recent samples
                cursor.execute(
                    """SELECT id, username, created_at
                       FROM rlhf_feedback
                       ORDER BY created_at DESC
                       LIMIT 5"""
                )
                recent_samples = [
                    {"id": row[0], "username": row[1], "created_at": row[2]} 
                    for row in cursor.fetchall()
                ]
                
                return {
                    "total_samples": total_samples,
                    "top_contributors": top_contributors,
                    "recent_samples": recent_samples
                }
        except Exception as e:
            logger.error(f"Error getting RLHF stats: {str(e)}")
            return {
                "total_samples": 0,
                "top_contributors": [],
                "recent_samples": []
            }