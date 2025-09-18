import sqlite3
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)

class ExperimentStatus(Enum):
    DRAFT = "draft"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ExperimentDB:
    """Database manager for finetuning experiments"""
    
    def __init__(self, db_path: str = "data/experiments.db"):
        self.db_path = db_path
        self.init_db()
        self._create_dataset_samples_table()
    
    def init_db(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Experiments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    user_id TEXT NOT NULL,
                    base_model TEXT NOT NULL,
                    model_provider TEXT DEFAULT 'huggingface',
                    status TEXT NOT NULL,
                    config TEXT NOT NULL,  -- JSON string with training config
                    dataset_path TEXT,
                    model_path TEXT,
                    metrics TEXT,  -- JSON string with training metrics
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Training logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    epoch INTEGER,
                    step INTEGER,
                    loss REAL,
                    eval_loss REAL,
                    learning_rate REAL,
                    accuracy REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            ''')
            
            # Datasets table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    num_samples INTEGER,
                    format TEXT,  -- jsonl, csv, txt
                    description TEXT,
                    status TEXT DEFAULT 'Processing',  -- Processing, Completed, Failed
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Add status column if it doesn't exist (for existing databases)
            try:
                cursor.execute('ALTER TABLE datasets ADD COLUMN status TEXT DEFAULT "Processing"')
                logger.info("Added status column to datasets table")
            except sqlite3.OperationalError:
                # Column already exists
                pass
            
            conn.commit()
            
            # Migration: Add eval_loss column to training_logs if it doesn't exist
            try:
                cursor.execute('ALTER TABLE training_logs ADD COLUMN eval_loss REAL')
                conn.commit()
                logger.info("Added eval_loss column to training_logs table")
            except sqlite3.OperationalError:
                # Column already exists or SQLite limitations; ignore
                pass
            
            # Migration: Add description column if it doesn't exist
            try:
                cursor.execute("ALTER TABLE experiments ADD COLUMN description TEXT DEFAULT ''")
                conn.commit()
                logger.info("Added description column to experiments table")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e) or "already exists" in str(e):
                    pass  # Column already exists
                else:
                    logger.warning(f"Migration warning: {e}")
            
            logger.info("Database initialized successfully")
    
    def create_experiment(self, experiment_data: Dict[str, Any]) -> str:
        """Create a new experiment"""
        experiment_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
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
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM experiments WHERE id = ?', (experiment_id,))
            row = cursor.fetchone()
            
            if row:
                experiment = dict(row)
                experiment['config'] = json.loads(experiment['config']) if experiment['config'] else {}
                experiment['metrics'] = json.loads(experiment['metrics']) if experiment['metrics'] else {}
                return experiment
        return None
    
    def get_user_experiments(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get experiments for a user"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM experiments 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            experiments = []
            for row in cursor.fetchall():
                experiment = dict(row)
                experiment['config'] = json.loads(experiment['config']) if experiment['config'] else {}
                experiment['metrics'] = json.loads(experiment['metrics']) if experiment['metrics'] else {}
                experiments.append(experiment)
            
            return experiments
    
    def update_experiment_status(self, experiment_id: str, status: ExperimentStatus, 
                                error_message: str = None):
        """Update experiment status"""
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
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
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                query = f"UPDATE experiments SET {', '.join(set_clauses)} WHERE id = ?"
                cursor.execute(query, values)
                conn.commit()
    
    def log_training_step(self, experiment_id: str, epoch: int, step: int, 
                         loss: float, learning_rate: float, accuracy: float = None, eval_loss: float = None):
        """Log training step metrics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO training_logs (
                    experiment_id, epoch, step, loss, eval_loss, learning_rate, accuracy
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (experiment_id, epoch, step, loss, eval_loss, learning_rate, accuracy))
            conn.commit()
    
    def get_training_logs(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get training logs for an experiment"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM training_logs 
                WHERE experiment_id = ? 
                ORDER BY epoch, step
            ''', (experiment_id,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def create_dataset(self, dataset_data: Dict[str, Any]) -> str:
        """Create a new dataset record"""
        dataset_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
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
    
    def get_user_datasets(self, user_id: str) -> List[Dict[str, Any]]:
        """Get datasets for a user"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM datasets 
                WHERE user_id = ? 
                ORDER BY created_at DESC
            ''', (user_id,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def delete_experiment(self, experiment_id: str, user_id: str) -> bool:
        """Delete an experiment (only if owned by user)"""
        with sqlite3.connect(self.db_path) as conn:
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
        
        with sqlite3.connect(self.db_path) as conn:
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
    
    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Get dataset by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM datasets WHERE id = ?
            ''', (dataset_id,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_dataset_samples(self, dataset_id: str, limit: int = None) -> List[Dict]:
        """Get samples for a dataset"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Try to get from dataset_samples table first (for converted datasets)
            if limit:
                cursor.execute('''
                    SELECT content FROM dataset_samples 
                    WHERE dataset_id = ? 
                    ORDER BY sample_index 
                    LIMIT ?
                ''', (dataset_id, limit))
            else:
                cursor.execute('''
                    SELECT content FROM dataset_samples 
                    WHERE dataset_id = ? 
                    ORDER BY sample_index
                ''', (dataset_id,))
            
            rows = cursor.fetchall()
            if rows:
                return [json.loads(row['content']) for row in rows]
            
            # Fallback: try to read from file if it's a file-based dataset
            dataset = self.get_dataset(dataset_id)
            if dataset and dataset.get('file_path'):
                try:
                    file_path = dataset['file_path']
                    if os.path.exists(file_path):
                        samples = []
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for i, line in enumerate(f):
                                if limit and i >= limit:
                                    break
                                try:
                                    samples.append(json.loads(line.strip()))
                                except json.JSONDecodeError:
                                    continue
                        return samples
                except Exception as e:
                    logger.error(f"Error reading dataset file: {e}")
            
            return []
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset and its samples"""
        with sqlite3.connect(self.db_path) as conn:
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

    def _create_dataset_samples_table(self):
        """Create dataset samples table for storing converted dataset content"""
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Clear existing samples
            cursor.execute('DELETE FROM dataset_samples WHERE dataset_id = ?', (dataset_id,))
            
            # Insert new samples
            for i, sample in enumerate(samples):
                cursor.execute('''
                    INSERT INTO dataset_samples (dataset_id, sample_index, content)
                    VALUES (?, ?, ?)
                ''', (dataset_id, i, json.dumps(sample)))
            
            # Update dataset metadata
            cursor.execute('''
                UPDATE datasets 
                SET num_samples = ?, file_size = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (len(samples), len(json.dumps(samples).encode('utf-8')), dataset_id))
            
            conn.commit()

    def update_dataset_status(self, dataset_id: str, status: str) -> None:
        """Update dataset status"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE datasets 
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (status, dataset_id))
            conn.commit()

# Global instance
experiment_db = ExperimentDB()
