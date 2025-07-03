import logging
import datetime  # Import datetime module correctly
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from auth import get_current_user
from models import User
import os
import sqlite3

router = APIRouter()
logger = logging.getLogger("rlhf")

class RLHFFeedback(BaseModel):
    session_id: str
    chosen_index: int
    comment: Optional[str] = None

@router.post("/rlhf-feedback")
async def submit_rlhf_feedback(
    feedback: RLHFFeedback,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Submit RLHF feedback for a chat session
    """
    # Log receipt of feedback immediately
    logger.info(f"Received RLHF feedback for session {feedback.session_id}, processing...")
    
    # Store feedback directly in database rather than using background task
    # This should be fast as it's just a simple database insert
    try:
        # Direct database write instead of background task
        store_rlhf_feedback(
            feedback.session_id, 
            feedback.chosen_index, 
            current_user.username,
            feedback.comment
        )
        logger.info(f"RLHF preference saved for session {feedback.session_id}")
    except Exception as e:
        logger.error(f"Error saving RLHF feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")
    
    # Return success immediately
    return {"status": "success", "message": "Feedback received and processed"}

def store_rlhf_feedback(session_id, chosen_index, username, comment=None):
    """
    Store RLHF feedback directly in the database
    """
    try:
        # Get database path
        db_path = os.getenv('SQLITE_DB_PATH', '/app/data/db/chat.db')
        
        # Use connection context manager for automatic cleanup
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Create the table if it doesn't exist
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
            
            # Insert the feedback - Fix: Use datetime.now() directly
            cursor.execute(
                '''
                INSERT INTO rlhf_feedback 
                (session_id, chosen_index, username, comment, created_at) 
                VALUES (?, ?, ?, ?, ?)
                ''',
                (session_id, chosen_index, username, comment, datetime.datetime.now().isoformat())
            )
            
            # Commit is automatic with the context manager
            
    except Exception as e:
        logger.error(f"Database error in RLHF feedback storage: {str(e)}")
        raise