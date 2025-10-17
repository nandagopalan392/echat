"""
Chat-related database models
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class ChatSession:
    """Chat session model"""
    id: Optional[int] = None
    username: str = ""
    topic: str = ""
    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    
    @classmethod
    def from_db_row(cls, row):
        """Create ChatSession from database row"""
        if row is None:
            return None
        return cls(
            id=row['id'] if 'id' in row.keys() else row[0],
            username=row['username'] if 'username' in row.keys() else row[1],
            topic=row['topic'] if 'topic' in row.keys() else row[2],
            created_at=row['created_at'] if 'created_at' in row.keys() else row[3],
            last_updated=row['last_updated'] if 'last_updated' in row.keys() else row[4]
        )


@dataclass
class Message:
    """Chat message model"""
    id: Optional[int] = None
    session_id: int = 0
    content: str = ""
    is_user: bool = True
    timestamp: Optional[datetime] = None
    
    @classmethod
    def from_db_row(cls, row):
        """Create Message from database row"""
        if row is None:
            return None
        return cls(
            id=row['id'] if 'id' in row.keys() else row[0],
            session_id=row['session_id'] if 'session_id' in row.keys() else row[1],
            content=row['content'] if 'content' in row.keys() else row[2],
            is_user=bool(row['is_user'] if 'is_user' in row.keys() else row[3]),
            timestamp=row['timestamp'] if 'timestamp' in row.keys() else row[4]
        )


@dataclass
class ResponseCache:
    """Cached response model"""
    id: Optional[int] = None
    query_hash: str = ""
    query: str = ""
    response: str = ""
    timestamp: Optional[datetime] = None
    
    @classmethod
    def from_db_row(cls, row):
        """Create ResponseCache from database row"""
        if row is None:
            return None
        return cls(
            id=row['id'] if 'id' in row.keys() else row[0],
            query_hash=row['query_hash'] if 'query_hash' in row.keys() else row[1],
            query=row['query'] if 'query' in row.keys() else row[2],
            response=row['response'] if 'response' in row.keys() else row[3],
            timestamp=row['timestamp'] if 'timestamp' in row.keys() else row[4]
        )
