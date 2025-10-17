"""Database models package initialization"""

from app.db.models.user import User
from app.db.models.chat import ChatSession, Message, ResponseCache
from app.db.models.document import FileInfo, ChunkingConfig
from app.db.models.config import ModelSettings, RetrievalConfig
from app.db.models.evaluation import (
    EvaluationMetric,
    EvaluationDataset,
    EvaluationDatasetDocument,
    EvaluationResult,
    EvaluationTask
)

__all__ = [
    # User models
    'User',
    # Chat models
    'ChatSession',
    'Message',
    'ResponseCache',
    # Document models
    'FileInfo',
    'ChunkingConfig',
    # Config models
    'ModelSettings',
    'RetrievalConfig',
    # Evaluation models
    'EvaluationMetric',
    'EvaluationDataset',
    'EvaluationDatasetDocument',
    'EvaluationResult',
    'EvaluationTask',
]
