"""
RLHF (Reinforcement Learning from Human Feedback) Manager

Handles business logic for user feedback on AI responses for training purposes.
"""
import logging
from typing import Dict, List, Any, Optional

from app.db.repositories.rlhf_repository import RLHFRepository, get_rlhf_repository

logger = logging.getLogger(__name__)


class RLHFManager:
    """
    Manager for RLHF feedback operations.
    
    Provides business logic layer for storing and retrieving user feedback
    on AI responses. Uses RLHFRepository for data access.
    """
    
    def __init__(self, repository: Optional[RLHFRepository] = None):
        """
        Initialize RLHF manager with repository.
        
        Args:
            repository: Optional RLHFRepository instance (defaults to singleton)
        """
        if repository is None:
            self.repository = get_rlhf_repository()
        else:
            self.repository = repository
        
        logger.info("RLHF Manager initialized")
    
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
        return self.repository.save_response_options(
            session_id=session_id,
            question=question,
            response_option_0=response_option_0,
            response_option_1=response_option_1,
            username=username,
            message_id=message_id
        )
    
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
        # Update the response options with the chosen response
        success = self.repository.update_selected_response(
            session_id=session_id,
            chosen_index=chosen_index,
            username=user_id,
            comment=comment
        )
        
        if success:
            # Also save to feedback table
            self.repository.save_feedback(
                session_id=session_id,
                chosen_index=chosen_index,
                username=user_id,
                comment=comment
            )
        
        return success
    
    def get_session_preferences(self, session_id: int) -> List[Dict[str, Any]]:
        """
        Get all preference data for a specific session.
        
        Args:
            session_id: Chat session ID
            
        Returns:
            List of preference records
        """
        return self.repository.get_session_response_options(session_id)
    
    def get_preference_data(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieve preference data for training.
        
        Args:
            limit: Maximum number of records to retrieve
            
        Returns:
            List of preference data records
        """
        return self.repository.get_training_data(limit=limit)
    
    def get_user_feedback_stats(self, username: str) -> Dict[str, Any]:
        """
        Get feedback statistics for a specific user.
        
        Args:
            username: Username
            
        Returns:
            Dictionary with user feedback statistics
        """
        feedback_count = self.repository.get_user_feedback_count(username)
        training_data = self.repository.get_training_data(
            limit=1000, 
            username=username
        )
        
        return {
            "username": username,
            "total_feedback": feedback_count,
            "training_samples": len(training_data)
        }
    
    def get_total_feedback_stats(self) -> Dict[str, Any]:
        """
        Get overall feedback statistics.
        
        Returns:
            Dictionary with overall feedback statistics
        """
        total_count = self.repository.get_total_feedback_count()
        
        return {
            "total_feedback": total_count
        }


# Singleton instance
_rlhf_manager: Optional[RLHFManager] = None


def get_rlhf_manager() -> RLHFManager:
    """
    Get singleton RLHF manager instance.
    
    Returns:
        RLHFManager: The global RLHF manager instance
    """
    global _rlhf_manager
    if _rlhf_manager is None:
        _rlhf_manager = RLHFManager()
    return _rlhf_manager

