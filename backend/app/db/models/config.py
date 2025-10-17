"""
Configuration-related database models
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class ModelSettings:
    """Model settings model"""
    id: Optional[int] = None
    llm: str = ""
    embedding: str = ""
    parameters: str = ""
    provider: str = "ollama"
    embedding_provider: str = "ollama"
    updated_at: Optional[datetime] = None
    
    @classmethod
    def from_db_row(cls, row):
        """Create ModelSettings from database row"""
        if row is None:
            return None
        
        # Handle both dict-like and tuple-like rows
        if hasattr(row, 'keys'):
            return cls(
                id=row.get('id'),
                llm=row.get('llm', ''),
                embedding=row.get('embedding', ''),
                parameters=row.get('parameters', ''),
                provider=row.get('provider', 'ollama'),
                embedding_provider=row.get('embedding_provider', 'ollama'),
                updated_at=row.get('updated_at')
            )
        else:
            return cls(
                id=row[0] if len(row) > 0 else None,
                llm=row[1] if len(row) > 1 else '',
                embedding=row[2] if len(row) > 2 else '',
                parameters=row[3] if len(row) > 3 else '',
                provider=row[5] if len(row) > 5 else 'ollama',
                embedding_provider=row[6] if len(row) > 6 else 'ollama',
                updated_at=row[4] if len(row) > 4 else None
            )


@dataclass
class RetrievalConfig:
    """Retrieval configuration model"""
    id: Optional[int] = None
    user_id: Optional[str] = None
    config: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @classmethod
    def from_db_row(cls, row):
        """Create RetrievalConfig from database row"""
        if row is None:
            return None
        return cls(
            id=row['id'] if 'id' in row.keys() else row[0],
            user_id=row['user_id'] if 'user_id' in row.keys() else row[1],
            config=row['config'] if 'config' in row.keys() else row[2],
            created_at=row['created_at'] if 'created_at' in row.keys() else row[3],
            updated_at=row['updated_at'] if 'updated_at' in row.keys() else row[4]
        )
