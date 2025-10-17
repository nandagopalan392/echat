"""
Evaluation-related database models
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class EvaluationMetric:
    """Evaluation metric model"""
    id: Optional[int] = None
    session_id: Optional[int] = None
    message_id: Optional[int] = None
    query: str = ""
    response: str = ""
    context: Optional[str] = None
    groundedness_score: Optional[float] = None
    context_relevance_score: Optional[float] = None
    answer_quality_score: Optional[float] = None
    latency_ms: Optional[int] = None
    evaluated_at: Optional[datetime] = None
    
    @classmethod
    def from_db_row(cls, row):
        """Create EvaluationMetric from database row"""
        if row is None:
            return None
        
        if hasattr(row, 'keys'):
            return cls(
                id=row.get('id'),
                session_id=row.get('session_id'),
                message_id=row.get('message_id'),
                query=row.get('query', ''),
                response=row.get('response', ''),
                context=row.get('context'),
                groundedness_score=row.get('groundedness_score'),
                context_relevance_score=row.get('context_relevance_score'),
                answer_quality_score=row.get('answer_quality_score'),
                latency_ms=row.get('latency_ms'),
                evaluated_at=row.get('evaluated_at')
            )
        else:
            return cls(
                id=row[0] if len(row) > 0 else None,
                session_id=row[1] if len(row) > 1 else None,
                message_id=row[2] if len(row) > 2 else None,
                query=row[3] if len(row) > 3 else '',
                response=row[4] if len(row) > 4 else '',
                context=row[5] if len(row) > 5 else None,
                groundedness_score=row[6] if len(row) > 6 else None,
                context_relevance_score=row[7] if len(row) > 7 else None,
                answer_quality_score=row[8] if len(row) > 8 else None,
                latency_ms=row[9] if len(row) > 9 else None,
                evaluated_at=row[10] if len(row) > 10 else None
            )


@dataclass
class EvaluationDataset:
    """Evaluation dataset model"""
    id: Optional[int] = None
    name: str = ""
    description: Optional[str] = None
    document_count: int = 0
    status: str = "Processing"
    created_by: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    file_path: Optional[str] = None
    
    @classmethod
    def from_db_row(cls, row):
        """Create EvaluationDataset from database row"""
        if row is None:
            return None
        
        if hasattr(row, 'keys'):
            return cls(
                id=row.get('id'),
                name=row.get('name', ''),
                description=row.get('description'),
                document_count=row.get('document_count', 0),
                status=row.get('status', 'Processing'),
                created_by=row.get('created_by', ''),
                created_at=row.get('created_at'),
                updated_at=row.get('updated_at'),
                file_path=row.get('file_path')
            )
        else:
            return cls(
                id=row[0] if len(row) > 0 else None,
                name=row[1] if len(row) > 1 else '',
                description=row[2] if len(row) > 2 else None,
                document_count=row[3] if len(row) > 3 else 0,
                status=row[4] if len(row) > 4 else 'Processing',
                created_by=row[5] if len(row) > 5 else '',
                created_at=row[6] if len(row) > 6 else None,
                updated_at=row[7] if len(row) > 7 else None,
                file_path=row[8] if len(row) > 8 else None
            )


@dataclass
class EvaluationDatasetDocument:
    """Evaluation dataset document model"""
    id: Optional[int] = None
    dataset_id: int = 0
    document_name: str = ""
    document_content: str = ""
    added_at: Optional[datetime] = None
    
    @classmethod
    def from_db_row(cls, row):
        """Create EvaluationDatasetDocument from database row"""
        if row is None:
            return None
        return cls(
            id=row['id'] if 'id' in row.keys() else row[0],
            dataset_id=row['dataset_id'] if 'dataset_id' in row.keys() else row[1],
            document_name=row['document_name'] if 'document_name' in row.keys() else row[2],
            document_content=row['document_content'] if 'document_content' in row.keys() else row[3],
            added_at=row['added_at'] if 'added_at' in row.keys() else row[4]
        )


@dataclass
class EvaluationResult:
    """Evaluation result model"""
    id: Optional[int] = None
    dataset_id: int = 0
    query: str = ""
    expected_answer: str = ""
    actual_answer: str = ""
    context: Optional[str] = None
    groundedness_score: Optional[float] = None
    relevance_score: Optional[float] = None
    quality_score: Optional[float] = None
    latency_ms: Optional[int] = None
    evaluated_at: Optional[datetime] = None
    model_used: Optional[str] = None
    
    @classmethod
    def from_db_row(cls, row):
        """Create EvaluationResult from database row"""
        if row is None:
            return None
        
        if hasattr(row, 'keys'):
            return cls(
                id=row.get('id'),
                dataset_id=row.get('dataset_id', 0),
                query=row.get('query', ''),
                expected_answer=row.get('expected_answer', ''),
                actual_answer=row.get('actual_answer', ''),
                context=row.get('context'),
                groundedness_score=row.get('groundedness_score'),
                relevance_score=row.get('relevance_score'),
                quality_score=row.get('quality_score'),
                latency_ms=row.get('latency_ms'),
                evaluated_at=row.get('evaluated_at'),
                model_used=row.get('model_used')
            )
        else:
            return cls(
                id=row[0] if len(row) > 0 else None,
                dataset_id=row[1] if len(row) > 1 else 0,
                query=row[2] if len(row) > 2 else '',
                expected_answer=row[3] if len(row) > 3 else '',
                actual_answer=row[4] if len(row) > 4 else '',
                context=row[5] if len(row) > 5 else None,
                groundedness_score=row[6] if len(row) > 6 else None,
                relevance_score=row[7] if len(row) > 7 else None,
                quality_score=row[8] if len(row) > 8 else None,
                latency_ms=row[9] if len(row) > 9 else None,
                evaluated_at=row[10] if len(row) > 10 else None,
                model_used=row[11] if len(row) > 11 else None
            )


@dataclass
class EvaluationTask:
    """Evaluation task model"""
    id: Optional[int] = None
    task_id: str = ""
    dataset_id: Optional[int] = None
    status: str = "pending"
    progress: int = 0
    total_queries: int = 0
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    @classmethod
    def from_db_row(cls, row):
        """Create EvaluationTask from database row"""
        if row is None:
            return None
        
        if hasattr(row, 'keys'):
            return cls(
                id=row.get('id'),
                task_id=row.get('task_id', ''),
                dataset_id=row.get('dataset_id'),
                status=row.get('status', 'pending'),
                progress=row.get('progress', 0),
                total_queries=row.get('total_queries', 0),
                created_at=row.get('created_at'),
                started_at=row.get('started_at'),
                completed_at=row.get('completed_at'),
                error_message=row.get('error_message')
            )
        else:
            return cls(
                id=row[0] if len(row) > 0 else None,
                task_id=row[1] if len(row) > 1 else '',
                dataset_id=row[2] if len(row) > 2 else None,
                status=row[3] if len(row) > 3 else 'pending',
                progress=row[4] if len(row) > 4 else 0,
                total_queries=row[5] if len(row) > 5 else 0,
                created_at=row[6] if len(row) > 6 else None,
                started_at=row[7] if len(row) > 7 else None,
                completed_at=row[8] if len(row) > 8 else None,
                error_message=row[9] if len(row) > 9 else None
            )
