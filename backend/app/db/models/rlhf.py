"""
RLHF (Reinforcement Learning from Human Feedback) Database Models
Defines data structures for RLHF feedback and response options
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class RLHFFeedback:
    """
    RLHF Feedback model representing user feedback on AI responses.
    
    This stores the simplified feedback record with just the choice made by the user.
    """
    id: Optional[int] = None
    session_id: Optional[str] = None
    chosen_index: Optional[int] = None
    username: Optional[str] = None
    comment: Optional[str] = None
    created_at: Optional[str] = None
    
    @classmethod
    def from_db_row(cls, row) -> 'RLHFFeedback':
        """
        Create RLHFFeedback instance from database row.
        
        Args:
            row: Database row (dict or tuple)
            
        Returns:
            RLHFFeedback instance
        """
        if row is None:
            return None
        
        if hasattr(row, 'keys'):
            # Dictionary-like row (sqlite3.Row)
            return cls(
                id=row.get('id'),
                session_id=row.get('session_id'),
                chosen_index=row.get('chosen_index'),
                username=row.get('username'),
                comment=row.get('comment'),
                created_at=row.get('created_at')
            )
        else:
            # Tuple row
            return cls(
                id=row[0] if len(row) > 0 else None,
                session_id=row[1] if len(row) > 1 else None,
                chosen_index=row[2] if len(row) > 2 else None,
                username=row[3] if len(row) > 3 else None,
                comment=row[4] if len(row) > 4 else None,
                created_at=row[5] if len(row) > 5 else None
            )
    
    def to_dict(self) -> dict:
        """
        Convert feedback to dictionary.
        
        Returns:
            Dictionary representation of the feedback
        """
        return {
            'id': self.id,
            'session_id': self.session_id,
            'chosen_index': self.chosen_index,
            'username': self.username,
            'comment': self.comment,
            'created_at': self.created_at
        }


@dataclass
class RLHFResponseOption:
    """
    RLHF Response Option model representing alternative responses presented to users.
    
    This stores the full context of a choice: the question, both response options,
    and which one was chosen (if any).
    """
    id: Optional[int] = None
    session_id: Optional[int] = None
    question: Optional[str] = None
    response_option_0: Optional[str] = None
    response_option_1: Optional[str] = None
    chosen_response: Optional[str] = None
    chosen_index: Optional[int] = None
    username: Optional[str] = None
    created_at: Optional[str] = None
    message_id: Optional[int] = None
    
    @classmethod
    def from_db_row(cls, row) -> 'RLHFResponseOption':
        """
        Create RLHFResponseOption instance from database row.
        
        Args:
            row: Database row (dict or tuple)
            
        Returns:
            RLHFResponseOption instance
        """
        if row is None:
            return None
        
        if hasattr(row, 'keys'):
            # Dictionary-like row (sqlite3.Row)
            return cls(
                id=row.get('id'),
                session_id=row.get('session_id'),
                question=row.get('question'),
                response_option_0=row.get('response_option_0'),
                response_option_1=row.get('response_option_1'),
                chosen_response=row.get('chosen_response'),
                chosen_index=row.get('chosen_index'),
                username=row.get('username'),
                created_at=row.get('created_at'),
                message_id=row.get('message_id')
            )
        else:
            # Tuple row
            return cls(
                id=row[0] if len(row) > 0 else None,
                session_id=row[1] if len(row) > 1 else None,
                question=row[2] if len(row) > 2 else None,
                response_option_0=row[3] if len(row) > 3 else None,
                response_option_1=row[4] if len(row) > 4 else None,
                chosen_response=row[5] if len(row) > 5 else None,
                chosen_index=row[6] if len(row) > 6 else None,
                username=row[7] if len(row) > 7 else None,
                created_at=row[8] if len(row) > 8 else None,
                message_id=row[9] if len(row) > 9 else None
            )
    
    def to_dict(self) -> dict:
        """
        Convert response option to dictionary.
        
        Returns:
            Dictionary representation of the response option
        """
        return {
            'id': self.id,
            'session_id': self.session_id,
            'question': self.question,
            'response_option_0': self.response_option_0,
            'response_option_1': self.response_option_1,
            'chosen_response': self.chosen_response,
            'chosen_index': self.chosen_index,
            'username': self.username,
            'created_at': self.created_at,
            'message_id': self.message_id
        }
    
    def is_chosen(self) -> bool:
        """
        Check if a choice has been made for this response option.
        
        Returns:
            True if a response has been chosen, False otherwise
        """
        return self.chosen_response is not None
    
    def get_unchosen_option(self) -> Optional[str]:
        """
        Get the response option that was NOT chosen.
        
        Returns:
            The unchosen response text, or None if no choice made
        """
        if not self.is_chosen() or self.chosen_index is None:
            return None
        
        if self.chosen_index == 0:
            return self.response_option_1
        elif self.chosen_index == 1:
            return self.response_option_0
        
        return None


@dataclass
class RLHFTrainingData:
    """
    RLHF Training Data model for model training purposes.
    
    This is a specialized view that combines the question with chosen and rejected responses,
    formatted for training preference models.
    """
    question: str
    chosen_response: str
    rejected_response: str
    username: Optional[str] = None
    session_id: Optional[int] = None
    created_at: Optional[str] = None
    
    @classmethod
    def from_response_option(cls, option: RLHFResponseOption) -> Optional['RLHFTrainingData']:
        """
        Create training data from a response option with a chosen response.
        
        Args:
            option: RLHFResponseOption with a chosen response
            
        Returns:
            RLHFTrainingData instance, or None if no choice was made
        """
        if not option.is_chosen():
            return None
        
        rejected = option.get_unchosen_option()
        if rejected is None:
            return None
        
        return cls(
            question=option.question,
            chosen_response=option.chosen_response,
            rejected_response=rejected,
            username=option.username,
            session_id=option.session_id,
            created_at=option.created_at
        )
    
    def to_dict(self) -> dict:
        """
        Convert training data to dictionary.
        
        Returns:
            Dictionary representation suitable for training
        """
        return {
            'question': self.question,
            'chosen': self.chosen_response,
            'rejected': self.rejected_response,
            'username': self.username,
            'session_id': self.session_id,
            'created_at': self.created_at
        }
    
    def to_training_pair(self) -> dict:
        """
        Convert to simple training pair format.
        
        Returns:
            Dictionary with question, chosen, and rejected keys only
        """
        return {
            'prompt': self.question,
            'chosen': self.chosen_response,
            'rejected': self.rejected_response
        }
