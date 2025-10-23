"""
RLHF Repository

Data access layer for RLHF (Reinforcement Learning from Human Feedback) operations.
Handles all database operations for user feedback on AI responses.
"""
import sqlite3
import logging
import datetime
from typing import List, Optional

from app.db.connection import DatabaseConnection, get_db_connection
from app.db.models.rlhf import RLHFFeedback, RLHFResponseOption, RLHFTrainingData

logger = logging.getLogger(__name__)


class RLHFRepository:
    """
    Repository for RLHF feedback data operations.
    
    Provides data access methods for storing and retrieving user feedback
    on AI responses for training purposes.
    """
    
    def __init__(self, db: Optional[DatabaseConnection] = None):
        """
        Initialize RLHF repository with database connection.
        
        Args:
            db: Optional DatabaseConnection instance (defaults to singleton)
        """
        if db is None:
            self.db = get_db_connection()
        else:
            self.db = db
        
        logger.debug("RLHFRepository initialized")
    
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
            True if successful, False otherwise
        """
        try:
            now = datetime.datetime.now().isoformat()
            
            conn = self.db.get_connection()
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
            conn.commit()
            
            logger.info(f"Saved RLHF response options for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving RLHF response options: {str(e)}")
            return False
    
    def update_selected_response(
        self,
        session_id: int,
        chosen_index: int,
        username: str,
        comment: Optional[str] = None
    ) -> bool:
        """
        Update response options record with user's selected response.
        
        Args:
            session_id: Chat session ID
            chosen_index: Index of chosen response (0 or 1)
            username: Username
            comment: Optional feedback comment
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Get latest response options for this session
            cursor.execute(
                '''
                SELECT id, response_option_0, response_option_1 
                FROM rlhf_response_options 
                WHERE session_id = ? AND username = ? AND chosen_response IS NULL
                ORDER BY created_at DESC LIMIT 1
                ''',
                (session_id, username)
            )
            
            result = cursor.fetchone()
            if not result:
                logger.error(
                    f"No pending response options found for session {session_id} "
                    f"and user {username}"
                )
                return False
            
            record_id, option_0, option_1 = result
            response_options = [option_0, option_1]
            
            # Validate chosen_index
            if chosen_index >= len(response_options) or chosen_index < 0:
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
            
            conn.commit()
            
            logger.info(
                f"Successfully updated RLHF selected response for session {session_id}: "
                f"option {chosen_index}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error updating RLHF selected response: {str(e)}")
            return False
    
    def save_feedback(
        self,
        session_id: int,
        chosen_index: int,
        username: str,
        comment: Optional[str] = None
    ) -> bool:
        """
        Save user feedback to the feedback table.
        
        Args:
            session_id: Chat session ID
            chosen_index: Index of chosen response (0 or 1)
            username: Username
            comment: Optional feedback comment
            
        Returns:
            True if successful, False otherwise
        """
        try:
            now = datetime.datetime.now().isoformat()
            
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                '''
                INSERT INTO rlhf_feedback 
                (session_id, chosen_index, username, comment, created_at)
                VALUES (?, ?, ?, ?, ?)
                ''',
                (session_id, chosen_index, username, comment, now)
            )
            conn.commit()
            
            logger.info(f"Saved RLHF feedback for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving RLHF feedback: {str(e)}")
            return False
    
    def get_session_response_options(
        self, 
        session_id: int
    ) -> List[RLHFResponseOption]:
        """
        Get all response options for a specific session.
        
        Args:
            session_id: Chat session ID
            
        Returns:
            List of RLHFResponseOption model instances
        """
        try:
            conn = self.db.get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT * FROM rlhf_response_options
                WHERE session_id = ?
                ORDER BY created_at ASC
                """,
                (session_id,)
            )
            
            results = [RLHFResponseOption.from_db_row(row) for row in cursor.fetchall()]
            
            logger.debug(
                f"Retrieved {len(results)} response option records "
                f"for session {session_id}"
            )
            return results
                
        except Exception as e:
            logger.error(f"Error retrieving session response options: {str(e)}")
            return []
    
    def get_session_feedback(
        self, 
        session_id: int
    ) -> List[RLHFFeedback]:
        """
        Get all feedback for a specific session.
        
        Args:
            session_id: Chat session ID
            
        Returns:
            List of RLHFFeedback model instances
        """
        try:
            conn = self.db.get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT * FROM rlhf_feedback
                WHERE session_id = ?
                ORDER BY created_at ASC
                """,
                (session_id,)
            )
            
            results = [RLHFFeedback.from_db_row(row) for row in cursor.fetchall()]
            
            logger.debug(
                f"Retrieved {len(results)} feedback records for session {session_id}"
            )
            return results
                
        except Exception as e:
            logger.error(f"Error retrieving session feedback: {str(e)}")
            return []
    
    def get_training_data(
        self, 
        limit: int = 1000,
        username: Optional[str] = None
    ) -> List[RLHFTrainingData]:
        """
        Retrieve preference data for model training.
        
        Args:
            limit: Maximum number of records to retrieve
            username: Optional filter by username
            
        Returns:
            List of RLHFTrainingData model instances formatted for training
        """
        try:
            conn = self.db.get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if username:
                cursor.execute(
                    """
                    SELECT * FROM rlhf_response_options
                    WHERE chosen_response IS NOT NULL AND username = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (username, limit)
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM rlhf_response_options
                    WHERE chosen_response IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,)
                )
            
            # Convert to RLHFResponseOption first, then to training data
            response_options = [RLHFResponseOption.from_db_row(row) for row in cursor.fetchall()]
            
            # Convert to training data format
            results = []
            for option in response_options:
                training_data = RLHFTrainingData.from_response_option(option)
                if training_data:
                    results.append(training_data)
            
            logger.info(
                f"Retrieved {len(results)} training data records"
                f"{f' for user {username}' if username else ''}"
            )
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving training data: {str(e)}")
            return []
    
    def get_user_feedback_count(self, username: str) -> int:
        """
        Get count of feedback provided by a user.
        
        Args:
            username: Username
            
        Returns:
            Count of feedback records
        """
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT COUNT(*) FROM rlhf_feedback WHERE username = ?",
                (username,)
            )
            
            count = cursor.fetchone()[0]
            logger.debug(f"User {username} has {count} feedback records")
            return count
            
        except Exception as e:
            logger.error(f"Error getting user feedback count: {str(e)}")
            return 0
    
    def get_total_feedback_count(self) -> int:
        """
        Get total count of all feedback records.
        
        Returns:
            Total count of feedback records
        """
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM rlhf_feedback")
            
            count = cursor.fetchone()[0]
            logger.debug(f"Total feedback records: {count}")
            return count
            
        except Exception as e:
            logger.error(f"Error getting total feedback count: {str(e)}")
            return 0


# Global singleton instance
rlhf_repository = RLHFRepository()


def get_rlhf_repository() -> RLHFRepository:
    """
    Get the global RLHFRepository instance (singleton pattern).
    
    Returns:
        RLHFRepository: The global RLHF repository instance
    """
    return rlhf_repository
