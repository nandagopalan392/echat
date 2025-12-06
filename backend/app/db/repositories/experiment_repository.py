import sqlite3
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

from app.db.base import DatabaseConnection
from app.db.models.experiment import Experiment, TrainingLog, Dataset, DatasetSample

logger = logging.getLogger(__name__)

class ExperimentStatus(Enum):
    DRAFT = "draft"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ExperimentRepository:
    """Repository for managing finetuning experiments and datasets"""
    
    def __init__(self, db: Optional[DatabaseConnection] = None):
        """
        Initialize ExperimentRepository with DatabaseConnection.
        
        Args:
            db: DatabaseConnection instance. If None, creates a new one.
        """
        if db is None:
            db = DatabaseConnection()
        self.db = db           
        
    def create_experiment(self, experiment_data: Dict[str, Any]) -> str:
        """Create a new experiment"""
        experiment_id = str(uuid.uuid4())
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO experiments (
                    id, name, description, user_id, base_model, model_provider, status, config
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                experiment_id,
                experiment_data['name'],
                experiment_data.get('description', ''),
                experiment_data['user_id'],
                experiment_data['base_model'],
                experiment_data.get('model_provider', 'huggingface'),
                ExperimentStatus.DRAFT.value,
                json.dumps(experiment_data['config'])
            ))
            conn.commit()
        
        logger.info(f"Created experiment {experiment_id}")
        return experiment_id
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """
        Get experiment by ID.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment model instance or None if not found
        """
        with self.db.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM experiments WHERE id = ?', (experiment_id,))
            row = cursor.fetchone()
            
            if row:
                return Experiment.from_db_row(row)
        return None
    
    def get_user_experiments(self, user_id: str, limit: int = 50) -> List[Experiment]:
        """
        Get experiments for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of experiments to return
            
        Returns:
            List of Experiment model instances
        """
        with self.db.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM experiments 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            experiments = [Experiment.from_db_row(row) for row in cursor.fetchall()]
            return experiments
    
    def update_experiment_status(self, experiment_id: str, status: ExperimentStatus, 
                                error_message: str = None):
        """Update experiment status"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            timestamp_field = None
            if status == ExperimentStatus.RUNNING:
                timestamp_field = 'started_at'
            elif status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED]:
                timestamp_field = 'completed_at'
            
            if timestamp_field:
                cursor.execute(f'''
                    UPDATE experiments 
                    SET status = ?, error_message = ?, {timestamp_field} = CURRENT_TIMESTAMP, 
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (status.value, error_message, experiment_id))
            else:
                cursor.execute('''
                    UPDATE experiments 
                    SET status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (status.value, error_message, experiment_id))
            
            conn.commit()
    
    def update_experiment_metrics(self, experiment_id: str, metrics: Dict[str, Any]):
        """Update experiment metrics"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE experiments 
                SET metrics = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (json.dumps(metrics), experiment_id))
            conn.commit()
    
    def update_experiment(self, experiment_id: str, updates: Dict[str, Any]):
        """Update experiment with arbitrary fields"""
        if not updates:
            return
            
        set_clauses = []
        values = []
        
        for field, value in updates.items():
            if field in ['model_path', 'dataset_path', 'name', 'description']:
                set_clauses.append(f"{field} = ?")
                values.append(value)
        
        if set_clauses:
            set_clauses.append("updated_at = CURRENT_TIMESTAMP")
            values.append(experiment_id)
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                query = f"UPDATE experiments SET {', '.join(set_clauses)} WHERE id = ?"
                cursor.execute(query, values)
                conn.commit()
    
    def log_training_step(self, experiment_id: str, epoch: int, step: int, 
                         loss: float, learning_rate: float, accuracy: float = None, eval_loss: float = None):
        """Log training step metrics"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO training_logs (
                    experiment_id, epoch, step, loss, eval_loss, learning_rate, accuracy
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (experiment_id, epoch, step, loss, eval_loss, learning_rate, accuracy))
            conn.commit()
    
    def get_training_logs(self, experiment_id: str) -> List[TrainingLog]:
        """
        Get training logs for an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            List of TrainingLog model instances
        """
        with self.db.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM training_logs 
                WHERE experiment_id = ? 
                ORDER BY epoch, step
            ''', (experiment_id,))
            
            return [TrainingLog.from_db_row(row) for row in cursor.fetchall()]
    
    def create_dataset(self, dataset_data: Dict[str, Any]) -> str:
        """Create a new dataset record"""
        dataset_id = str(uuid.uuid4())
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO datasets (
                    id, name, user_id, file_path, file_size, num_samples, format, description
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                dataset_id,
                dataset_data['name'],
                dataset_data['user_id'],
                dataset_data['file_path'],
                dataset_data.get('file_size'),
                dataset_data.get('num_samples'),
                dataset_data.get('format'),
                dataset_data.get('description')
            ))
            conn.commit()
        
        return dataset_id
    
    def get_user_datasets(self, user_id: str) -> List[Dataset]:
        """
        Get datasets for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of Dataset model instances
        """
        with self.db.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM datasets 
                WHERE user_id = ? 
                ORDER BY created_at DESC
            ''', (user_id,))
            
            return [Dataset.from_db_row(row) for row in cursor.fetchall()]
    
    def delete_experiment(self, experiment_id: str, user_id: str) -> bool:
        """Delete an experiment (only if owned by user)"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            # First check if experiment exists and belongs to user
            cursor.execute(
                'SELECT id FROM experiments WHERE id = ? AND user_id = ?',
                (experiment_id, user_id)
            )
            if not cursor.fetchone():
                return False
            
            # Delete training logs first (foreign key constraint)
            cursor.execute('DELETE FROM training_logs WHERE experiment_id = ?', (experiment_id,))
            # Delete experiment
            cursor.execute('DELETE FROM experiments WHERE id = ?', (experiment_id,))
            conn.commit()
            return True
    
    def create_dataset(self, name: str, description: str, samples: List[Dict], user_id: str) -> str:
        """Create a dataset from samples"""
        dataset_id = str(uuid.uuid4())
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO datasets (
                    id, name, user_id, file_path, file_size, num_samples, format, description, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                dataset_id,
                name,
                user_id,
                f"converted_{dataset_id}.jsonl",  # Virtual file path
                len(json.dumps(samples).encode('utf-8')),  # Estimated size
                len(samples),
                'jsonl',
                description,
                'Processing'  # Set initial status to Processing
            ))
            
            # Store samples in a separate table for converted datasets
            for i, sample in enumerate(samples):
                cursor.execute('''
                    INSERT INTO dataset_samples (dataset_id, sample_index, content)
                    VALUES (?, ?, ?)
                ''', (dataset_id, i, json.dumps(sample)))
            
            conn.commit()
        
        return dataset_id
    
    def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        """
        Get dataset by ID.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            Dataset model instance or None if not found
        """
        with self.db.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM datasets WHERE id = ?
            ''', (dataset_id,))
            
            row = cursor.fetchone()
            return Dataset.from_db_row(row) if row else None
    
    def get_dataset_samples(self, dataset_id: str, limit: int = None) -> List[DatasetSample]:
        """
        Get samples for a dataset.
        
        Args:
            dataset_id: Dataset ID
            limit: Optional limit on number of samples to return
            
        Returns:
            List of DatasetSample model instances
        """
        with self.db.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get from dataset_samples table
            if limit:
                cursor.execute('''
                    SELECT * FROM dataset_samples 
                    WHERE dataset_id = ? 
                    ORDER BY sample_index 
                    LIMIT ?
                ''', (dataset_id, limit))
            else:
                cursor.execute('''
                    SELECT * FROM dataset_samples 
                    WHERE dataset_id = ? 
                    ORDER BY sample_index
                ''', (dataset_id,))
            
            rows = cursor.fetchall()
            if rows:
                return [DatasetSample.from_db_row(row) for row in rows]
            
            # Fallback: try to read from file if it's a file-based dataset
            dataset = self.get_dataset(dataset_id)
            if dataset and dataset.file_path:
                try:
                    file_path = dataset.file_path
                    if os.path.exists(file_path):
                        samples = []
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for i, line in enumerate(f):
                                if limit and i >= limit:
                                    break
                                try:
                                    data = json.loads(line.strip())
                                    # Convert file data to DatasetSample
                                    sample = DatasetSample(
                                        dataset_id=dataset_id,
                                        sample_index=i,
                                        input_text=data.get('input', data.get('text', '')),
                                        output_text=data.get('output', data.get('label', '')),
                                        metadata=data.get('metadata')
                                    )
                                    samples.append(sample)
                                except json.JSONDecodeError:
                                    continue
                        return samples
                except Exception as e:
                    logger.error(f"Error reading dataset file: {e}")
            
            return []
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset and its samples"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if dataset exists
            cursor.execute('SELECT id FROM datasets WHERE id = ?', (dataset_id,))
            if not cursor.fetchone():
                return False
            
            # Delete samples first
            cursor.execute('DELETE FROM dataset_samples WHERE dataset_id = ?', (dataset_id,))
            # Delete dataset
            cursor.execute('DELETE FROM datasets WHERE id = ?', (dataset_id,))
            
            conn.commit()
            return True

    def _deprecated_create_dataset_samples_table(self):
        """Create dataset samples table for storing converted dataset content"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dataset_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT NOT NULL,
                    sample_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (dataset_id) REFERENCES datasets (id) ON DELETE CASCADE
                )
            ''')
            
            # Create index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_dataset_samples_dataset_id 
                ON dataset_samples (dataset_id)
            ''')
            
            conn.commit()

    def update_dataset_samples(self, dataset_id: str, samples: List[Dict]) -> None:
        """Update dataset samples after generation"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Ensure updated_at column exists (migration)
            try:
                cursor.execute('ALTER TABLE datasets ADD COLUMN updated_at TIMESTAMP')
                conn.commit()
            except Exception:
                pass  # Column already exists
            
            # Clear existing samples
            cursor.execute('DELETE FROM dataset_samples WHERE dataset_id = ?', (dataset_id,))
            
            # Insert new samples using correct column names: input_text, output_text, metadata
            for i, sample in enumerate(samples):
                # Extract Q-C-A fields from sample
                input_text = sample.get('question', sample.get('input', ''))
                output_text = sample.get('answer', sample.get('output', ''))
                metadata = json.dumps({
                    'context': sample.get('context', ''),
                    'document_id': sample.get('document_id', ''),
                    'document_name': sample.get('document_name', ''),
                    'chunk_id': sample.get('chunk_id', '')
                })
                
                cursor.execute('''
                    INSERT INTO dataset_samples (dataset_id, sample_index, input_text, output_text, metadata)
                    VALUES (?, ?, ?, ?, ?)
                ''', (dataset_id, i, input_text, output_text, metadata))
            
            # Update dataset metadata
            cursor.execute('''
                UPDATE datasets 
                SET num_samples = ?, file_size = ?
                WHERE id = ?
            ''', (len(samples), len(json.dumps(samples).encode('utf-8')), dataset_id))
            
            conn.commit()

    def update_dataset_status(self, dataset_id: str, status: str) -> None:
        """Update dataset status"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Ensure updated_at column exists (migration)
            try:
                cursor.execute('ALTER TABLE datasets ADD COLUMN updated_at TIMESTAMP')
                conn.commit()
            except Exception:
                pass  # Column already exists
            
            # Update status - use simpler query without updated_at if it fails
            try:
                cursor.execute('''
                    UPDATE datasets 
                    SET status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (status, dataset_id))
            except Exception:
                cursor.execute('''
                    UPDATE datasets 
                    SET status = ?
                    WHERE id = ?
                ''', (status, dataset_id))
            conn.commit()

# Global instance
experiment_repository = ExperimentRepository()


def get_experiment_repository() -> ExperimentRepository:
    """
    Get the global ExperimentRepository instance (singleton pattern).
    
    Returns:
        ExperimentRepository: The global experiment repository instance
    """
    return experiment_repository
