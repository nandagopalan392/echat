"""
Evaluation repository for database operations
"""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.db.base import DatabaseConnection
from app.db.models.evaluation import (
    EvaluationMetric,
    EvaluationDataset,
    EvaluationDatasetDocument,
    EvaluationResult,
    EvaluationTask
)

logger = logging.getLogger(__name__)


class EvaluationRepository:
    """Repository for evaluation-related database operations"""
    
    def __init__(self, db: DatabaseConnection):
        self.db = db
    
    # Evaluation metrics operations
    def save_evaluation_metric(self, session_id: Optional[int], message_id: Optional[int],
                               query: str, response: str, context: Optional[str],
                               groundedness_score: Optional[float], 
                               context_relevance_score: Optional[float],
                               answer_quality_score: Optional[float],
                               latency_ms: Optional[int]) -> Optional[int]:
        """Save evaluation metric"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                cursor.execute(
                    '''INSERT INTO evaluation_metrics 
                       (session_id, message_id, query, response, context,
                        groundedness_score, context_relevance_score, answer_quality_score,
                        latency_ms, evaluated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (session_id, message_id, query, response, context,
                     groundedness_score, context_relevance_score, answer_quality_score,
                     latency_ms, now)
                )
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error saving evaluation metric: {e}")
            return None
    
    def get_evaluation_metrics(self, limit: int = 100) -> List[EvaluationMetric]:
        """Get recent evaluation metrics"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''SELECT * FROM evaluation_metrics
                       ORDER BY evaluated_at DESC
                       LIMIT ?''',
                    (limit,)
                )
                rows = cursor.fetchall()
                return [EvaluationMetric.from_db_row(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting evaluation metrics: {e}")
            return []
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    '''SELECT 
                           AVG(groundedness_score) as avg_groundedness,
                           AVG(context_relevance_score) as avg_relevance,
                           AVG(answer_quality_score) as avg_quality,
                           AVG(latency_ms) as avg_latency,
                           COUNT(*) as total_evaluations
                       FROM evaluation_metrics
                       WHERE evaluated_at >= datetime('now', '-7 days')'''
                )
                row = cursor.fetchone()
                
                return {
                    'avg_groundedness': row[0] or 0,
                    'avg_relevance': row[1] or 0,
                    'avg_quality': row[2] or 0,
                    'avg_latency': row[3] or 0,
                    'total_evaluations': row[4] or 0
                }
        except Exception as e:
            logger.error(f"Error getting evaluation stats: {e}")
            return {}
    
    # Dataset operations
    def create_dataset(self, name: str, description: Optional[str], 
                      created_by: str, file_path: Optional[str] = None) -> Optional[int]:
        """Create evaluation dataset"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                cursor.execute(
                    '''INSERT INTO evaluation_datasets 
                       (name, description, created_by, created_at, updated_at, file_path, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (name, description, created_by, now, now, file_path, 'Processing')
                )
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            return None
    
    def get_dataset(self, dataset_id: int) -> Optional[EvaluationDataset]:
        """Get evaluation dataset by ID"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT * FROM evaluation_datasets WHERE id = ?',
                    (dataset_id,)
                )
                row = cursor.fetchone()
                return EvaluationDataset.from_db_row(row) if row else None
        except Exception as e:
            logger.error(f"Error getting dataset: {e}")
            return None
    
    def get_all_datasets(self) -> List[EvaluationDataset]:
        """Get all evaluation datasets"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''SELECT * FROM evaluation_datasets
                       ORDER BY created_at DESC'''
                )
                rows = cursor.fetchall()
                return [EvaluationDataset.from_db_row(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting all datasets: {e}")
            return []
    
    def update_dataset_status(self, dataset_id: int, status: str, 
                            document_count: Optional[int] = None) -> bool:
        """Update dataset status"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                if document_count is not None:
                    cursor.execute(
                        '''UPDATE evaluation_datasets
                           SET status = ?, document_count = ?, updated_at = ?
                           WHERE id = ?''',
                        (status, document_count, now, dataset_id)
                    )
                else:
                    cursor.execute(
                        '''UPDATE evaluation_datasets
                           SET status = ?, updated_at = ?
                           WHERE id = ?''',
                        (status, now, dataset_id)
                    )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating dataset status: {e}")
            return False
    
    def update_dataset(self, dataset_id: int, file_path: Optional[str] = None,
                      question_count: Optional[int] = None, document_count: Optional[int] = None,
                      status: Optional[str] = None) -> bool:
        """Update dataset with file path, question count, document count, and/or status"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                # Build dynamic update query based on provided parameters
                updates = []
                params = []
                
                if file_path is not None:
                    updates.append("file_path = ?")
                    params.append(file_path)
                
                if question_count is not None:
                    updates.append("question_count = ?")
                    params.append(question_count)
                
                if document_count is not None:
                    updates.append("document_count = ?")
                    params.append(document_count)
                
                if status is not None:
                    updates.append("status = ?")
                    params.append(status)
                
                updates.append("updated_at = ?")
                params.append(now)
                params.append(dataset_id)
                
                query = f"UPDATE evaluation_datasets SET {', '.join(updates)} WHERE id = ?"
                cursor.execute(query, params)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating dataset: {e}")
            return False
    
    def delete_dataset(self, dataset_id: int) -> bool:
        """Delete evaluation dataset"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Delete related documents and results
                cursor.execute('DELETE FROM evaluation_dataset_documents WHERE dataset_id = ?', (dataset_id,))
                cursor.execute('DELETE FROM evaluation_results WHERE dataset_id = ?', (dataset_id,))
                cursor.execute('DELETE FROM evaluation_datasets WHERE id = ?', (dataset_id,))
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error deleting dataset: {e}")
            return False
    
    # Dataset documents operations
    def add_dataset_document(self, dataset_id: int, document_name: str, 
                           document_content: str) -> Optional[int]:
        """Add document to dataset"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                cursor.execute(
                    '''INSERT INTO evaluation_dataset_documents 
                       (dataset_id, document_name, document_content, added_at)
                       VALUES (?, ?, ?, ?)''',
                    (dataset_id, document_name, document_content, now)
                )
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error adding dataset document: {e}")
            return None
    
    def get_dataset_documents(self, dataset_id: int) -> List[EvaluationDatasetDocument]:
        """Get all documents in a dataset"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''SELECT * FROM evaluation_dataset_documents
                       WHERE dataset_id = ?
                       ORDER BY added_at ASC''',
                    (dataset_id,)
                )
                rows = cursor.fetchall()
                return [EvaluationDatasetDocument.from_db_row(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting dataset documents: {e}")
            return []
    
    # Evaluation results operations
    def save_evaluation_result(self, task_id: str, dataset_id: int, query: str, expected_answer: str,
                              actual_answer: str, context: Optional[str],
                              groundedness_score: Optional[float], relevance_score: Optional[float],
                              quality_score: Optional[float], latency_ms: Optional[int],
                              model_used: Optional[str] = None) -> Optional[int]:
        """Save evaluation result"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                cursor.execute(
                    '''INSERT INTO evaluation_results 
                       (task_id, dataset_id, query, expected_answer, actual_answer, context,
                        groundedness_score, relevance_score, quality_score, latency_ms,
                        model_used, evaluated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (task_id, dataset_id, query, expected_answer, actual_answer, context,
                     groundedness_score, relevance_score, quality_score, latency_ms,
                     model_used, now)
                )
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error saving evaluation result: {e}")
            return None
    
    def get_dataset_results(self, dataset_id: int) -> List[EvaluationResult]:
        """Get all results for a dataset"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''SELECT * FROM evaluation_results
                       WHERE dataset_id = ?
                       ORDER BY evaluated_at DESC''',
                    (dataset_id,)
                )
                rows = cursor.fetchall()
                return [EvaluationResult.from_db_row(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting dataset results: {e}")
            return []
    
    # Evaluation tasks operations
    def create_task(self, task_id: str, dataset_id: Optional[int], 
                   total_queries: int) -> Optional[int]:
        """Create evaluation task"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                cursor.execute(
                    '''INSERT INTO evaluation_tasks 
                       (task_id, dataset_id, status, progress, total_queries, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)''',
                    (task_id, dataset_id, 'pending', 0, total_queries, now)
                )
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error creating task: {e}")
            return None
    
    def get_task(self, task_id: str) -> Optional[EvaluationTask]:
        """Get evaluation task by ID"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT * FROM evaluation_tasks WHERE task_id = ?',
                    (task_id,)
                )
                row = cursor.fetchone()
                return EvaluationTask.from_db_row(row) if row else None
        except Exception as e:
            logger.error(f"Error getting task: {e}")
            return None
    
    def update_task_progress(self, task_id: str, progress: int, status: str) -> bool:
        """Update task progress"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                cursor.execute(
                    '''UPDATE evaluation_tasks
                       SET progress = ?, status = ?, started_at = COALESCE(started_at, ?)
                       WHERE task_id = ?''',
                    (progress, status, now, task_id)
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating task progress: {e}")
            return False
    
    def complete_task(self, task_id: str, status: str = 'completed',
                     error_message: Optional[str] = None,
                     groundedness_score: float = None,
                     answer_relevance_score: float = None,
                     context_relevance_score: float = None,
                     overall_score: float = None,
                     evaluation_time: float = None,
                     metadata: dict = None) -> bool:
        """Complete evaluation task with optional scores and metadata"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                # Build dynamic update query based on provided parameters
                update_fields = ['status = ?', 'completed_at = ?']
                values = [status, now]
                
                if error_message is not None:
                    update_fields.append('error_message = ?')
                    values.append(error_message)
                
                if groundedness_score is not None:
                    update_fields.append('groundedness_score = ?')
                    values.append(groundedness_score)
                
                if answer_relevance_score is not None:
                    update_fields.append('answer_relevance_score = ?')
                    values.append(answer_relevance_score)
                
                if context_relevance_score is not None:
                    update_fields.append('context_relevance_score = ?')
                    values.append(context_relevance_score)
                
                if overall_score is not None:
                    update_fields.append('overall_score = ?')
                    values.append(overall_score)
                
                if evaluation_time is not None:
                    update_fields.append('evaluation_time = ?')
                    values.append(evaluation_time)
                
                if metadata is not None:
                    import json
                    update_fields.append('metadata = ?')
                    values.append(json.dumps(metadata))
                
                values.append(task_id)
                
                cursor.execute(
                    f'''UPDATE evaluation_tasks
                       SET {', '.join(update_fields)}
                       WHERE task_id = ?''',
                    tuple(values)
                )
                conn.commit()
                logger.info(f"Completed task {task_id} with status {status}, scores: g={groundedness_score}, ar={answer_relevance_score}, cr={context_relevance_score}")
                return True
        except Exception as e:
            logger.error(f"Error completing task: {e}")
            return False
    
    def get_all_tasks(self, limit: int = 50) -> List[EvaluationTask]:
        """Get all evaluation tasks"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''SELECT * FROM evaluation_tasks
                       ORDER BY created_at DESC
                       LIMIT ?''',
                    (limit,)
                )
                rows = cursor.fetchall()
                return [EvaluationTask.from_db_row(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting all tasks: {e}")
            return []
