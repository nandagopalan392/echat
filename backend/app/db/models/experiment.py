"""
Experiment Database Models
Defines data structures for fine-tuning experiments, training logs, and datasets
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import json


@dataclass
class Experiment:
    """
    Experiment model representing a fine-tuning experiment.
    
    Tracks the configuration, status, and results of model fine-tuning.
    """
    id: str
    name: str
    user_id: str
    base_model: str
    status: str
    config: Dict[str, Any]
    description: str = ""
    model_provider: str = "huggingface"
    dataset_path: Optional[str] = None
    model_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @classmethod
    def from_db_row(cls, row) -> 'Experiment':
        """
        Create Experiment instance from database row.
        
        Args:
            row: Database row (dict or tuple)
            
        Returns:
            Experiment instance
        """
        if row is None:
            return None
        
        # Helper function to safely get values from sqlite3.Row or dict
        def safe_get(key, default=None):
            try:
                value = row[key]
                return value if value is not None else default
            except (KeyError, IndexError, TypeError):
                return default
        
        if hasattr(row, 'keys'):
            # Dictionary-like row (sqlite3.Row or dict)
            # Parse JSON fields
            config = safe_get('config', '{}')
            if isinstance(config, str):
                config = json.loads(config) if config else {}
            
            metrics = safe_get('metrics')
            if isinstance(metrics, str):
                metrics = json.loads(metrics) if metrics else None
            
            return cls(
                id=safe_get('id'),
                name=safe_get('name', ''),
                description=safe_get('description', ''),
                user_id=safe_get('user_id'),
                base_model=safe_get('base_model'),
                model_provider=safe_get('model_provider', 'huggingface'),
                status=safe_get('status'),
                config=config,
                dataset_path=safe_get('dataset_path'),
                model_path=safe_get('model_path'),
                metrics=metrics,
                error_message=safe_get('error_message'),
                created_at=safe_get('created_at'),
                started_at=safe_get('started_at'),
                completed_at=safe_get('completed_at'),
                updated_at=safe_get('updated_at')
            )
        else:
            # Tuple row
            config = json.loads(row[7]) if row[7] else {}
            metrics = json.loads(row[10]) if len(row) > 10 and row[10] else None
            
            return cls(
                id=row[0],
                name=row[1],
                description=row[2] if len(row) > 2 else '',
                user_id=row[3],
                base_model=row[4],
                model_provider=row[5] if len(row) > 5 else 'huggingface',
                status=row[6],
                config=config,
                dataset_path=row[8] if len(row) > 8 else None,
                model_path=row[9] if len(row) > 9 else None,
                metrics=metrics,
                error_message=row[11] if len(row) > 11 else None,
                created_at=row[12] if len(row) > 12 else None,
                started_at=row[13] if len(row) > 13 else None,
                completed_at=row[14] if len(row) > 14 else None,
                updated_at=row[15] if len(row) > 15 else None
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert experiment to dictionary.
        
        Returns:
            Dictionary representation of the experiment
        """
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'user_id': self.user_id,
            'base_model': self.base_model,
            'model_name': self.base_model,  # Alias for frontend compatibility
            'model_provider': self.model_provider,
            'status': self.status,
            'config': self.config,
            'dataset_path': self.dataset_path,
            'model_path': self.model_path,
            'metrics': self.metrics,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'started_at': self.started_at.isoformat() if isinstance(self.started_at, datetime) else self.started_at,
            'completed_at': self.completed_at.isoformat() if isinstance(self.completed_at, datetime) else self.completed_at,
            'updated_at': self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at
        }
    
    def is_running(self) -> bool:
        """Check if experiment is currently running."""
        return self.status.lower() in ['running', 'training', 'processing']
    
    def is_completed(self) -> bool:
        """Check if experiment has completed successfully."""
        return self.status.lower() == 'completed'
    
    def is_failed(self) -> bool:
        """Check if experiment has failed."""
        return self.status.lower() == 'failed'


@dataclass
class TrainingLog:
    """
    Training Log model representing metrics from a training step/epoch.
    
    Tracks detailed metrics during model training.
    """
    id: Optional[int] = None
    experiment_id: Optional[str] = None
    epoch: Optional[int] = None
    step: Optional[int] = None
    loss: Optional[float] = None
    eval_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    accuracy: Optional[float] = None
    timestamp: Optional[datetime] = None
    
    @classmethod
    def from_db_row(cls, row) -> 'TrainingLog':
        """
        Create TrainingLog instance from database row.
        
        Args:
            row: Database row (dict or tuple)
            
        Returns:
            TrainingLog instance
        """
        if row is None:
            return None
        
        # Helper function to safely get values from sqlite3.Row or dict
        def safe_get(key, default=None):
            try:
                value = row[key]
                return value if value is not None else default
            except (KeyError, IndexError, TypeError):
                return default
        
        if hasattr(row, 'keys'):
            # Dictionary-like row (sqlite3.Row or dict)
            return cls(
                id=safe_get('id'),
                experiment_id=safe_get('experiment_id'),
                epoch=safe_get('epoch'),
                step=safe_get('step'),
                loss=safe_get('loss'),
                eval_loss=safe_get('eval_loss'),
                learning_rate=safe_get('learning_rate'),
                accuracy=safe_get('accuracy'),
                timestamp=safe_get('timestamp')
            )
        else:
            # Tuple row
            return cls(
                id=row[0] if len(row) > 0 else None,
                experiment_id=row[1] if len(row) > 1 else None,
                epoch=row[2] if len(row) > 2 else None,
                step=row[3] if len(row) > 3 else None,
                loss=row[4] if len(row) > 4 else None,
                eval_loss=row[5] if len(row) > 5 else None,
                learning_rate=row[6] if len(row) > 6 else None,
                accuracy=row[7] if len(row) > 7 else None,
                timestamp=row[8] if len(row) > 8 else None
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert training log to dictionary.
        
        Returns:
            Dictionary representation of the training log
        """
        return {
            'id': self.id,
            'experiment_id': self.experiment_id,
            'epoch': self.epoch,
            'step': self.step,
            'loss': self.loss,
            'eval_loss': self.eval_loss,
            'learning_rate': self.learning_rate,
            'accuracy': self.accuracy,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp
        }


@dataclass
class Dataset:
    """
    Dataset model representing a training dataset.
    
    Tracks dataset files used for fine-tuning experiments.
    """
    id: str
    name: str
    user_id: str
    file_path: str
    status: str = "Processing"
    file_size: Optional[int] = None
    num_samples: Optional[int] = None
    format: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    
    @classmethod
    def from_db_row(cls, row) -> 'Dataset':
        """
        Create Dataset instance from database row.
        
        Args:
            row: Database row (dict, sqlite3.Row, or tuple)
            
        Returns:
            Dataset instance
        """
        if row is None:
            return None
        
        # Helper function to safely get values from sqlite3.Row or dict
        def safe_get(key, default=None):
            try:
                value = row[key]
                return value if value is not None else default
            except (KeyError, IndexError, TypeError):
                return default
        
        if hasattr(row, 'keys'):
            # Dictionary-like row (sqlite3.Row or dict)
            return cls(
                id=safe_get('id'),
                name=safe_get('name', ''),
                user_id=safe_get('user_id'),
                file_path=safe_get('file_path'),
                file_size=safe_get('file_size'),
                num_samples=safe_get('num_samples'),
                format=safe_get('format'),
                description=safe_get('description'),
                status=safe_get('status', 'Processing'),
                created_at=safe_get('created_at')
            )
        else:
            # Tuple row
            return cls(
                id=row[0],
                name=row[1],
                user_id=row[2],
                file_path=row[3],
                file_size=row[4] if len(row) > 4 else None,
                num_samples=row[5] if len(row) > 5 else None,
                format=row[6] if len(row) > 6 else None,
                description=row[7] if len(row) > 7 else None,
                status=row[8] if len(row) > 8 else 'Processing',
                created_at=row[9] if len(row) > 9 else None
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert dataset to dictionary.
        
        Returns:
            Dictionary representation of the dataset
        """
        return {
            'id': self.id,
            'name': self.name,
            'user_id': self.user_id,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'num_samples': self.num_samples,
            'format': self.format,
            'description': self.description,
            'status': self.status,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at
        }
    
    def is_ready(self) -> bool:
        """Check if dataset is ready for use."""
        return self.status.lower() == 'ready'
    
    def is_processing(self) -> bool:
        """Check if dataset is still being processed."""
        return self.status.lower() == 'processing'


@dataclass
class DatasetSample:
    """
    Dataset Sample model representing an individual training sample.
    
    Stores input-output pairs for training.
    """
    id: Optional[int] = None
    dataset_id: Optional[str] = None
    sample_index: Optional[int] = None
    input_text: Optional[str] = None
    output_text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    
    @classmethod
    def from_db_row(cls, row) -> 'DatasetSample':
        """
        Create DatasetSample instance from database row.
        
        Args:
            row: Database row (dict or tuple)
            
        Returns:
            DatasetSample instance
        """
        if row is None:
            return None
        
        # Helper function to safely get values from sqlite3.Row or dict
        def safe_get(key, default=None):
            try:
                value = row[key]
                return value if value is not None else default
            except (KeyError, IndexError, TypeError):
                return default
        
        if hasattr(row, 'keys'):
            # Dictionary-like row (sqlite3.Row or dict)
            metadata = safe_get('metadata')
            if isinstance(metadata, str):
                metadata = json.loads(metadata) if metadata else None
            
            return cls(
                id=safe_get('id'),
                dataset_id=safe_get('dataset_id'),
                sample_index=safe_get('sample_index'),
                input_text=safe_get('input_text'),
                output_text=safe_get('output_text'),
                metadata=metadata,
                created_at=safe_get('created_at')
            )
        else:
            # Tuple row
            metadata = None
            if len(row) > 5 and row[5]:
                metadata = json.loads(row[5]) if isinstance(row[5], str) else row[5]
            
            return cls(
                id=row[0] if len(row) > 0 else None,
                dataset_id=row[1] if len(row) > 1 else None,
                sample_index=row[2] if len(row) > 2 else None,
                input_text=row[3] if len(row) > 3 else None,
                output_text=row[4] if len(row) > 4 else None,
                metadata=metadata,
                created_at=row[6] if len(row) > 6 else None
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert dataset sample to dictionary.
        
        Returns:
            Dictionary representation of the dataset sample
        """
        return {
            'id': self.id,
            'dataset_id': self.dataset_id,
            'sample_index': self.sample_index,
            'input_text': self.input_text,
            'output_text': self.output_text,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at
        }
    
    def to_training_format(self) -> Dict[str, str]:
        """
        Convert to simple training format.
        
        Returns:
            Dictionary with input and output keys for training
        """
        return {
            'input': self.input_text or '',
            'output': self.output_text or ''
        }
