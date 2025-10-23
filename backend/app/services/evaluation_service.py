"""
Evaluation Service
Business logic for evaluation operations
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import sqlite3
import json
from celery.result import AsyncResult

from app.core.config import settings
from app.core.evaluation.system import (
    get_evaluation_manager,
    evaluate_chat_response,
    get_recent_evaluation_stats,
    RAGTriadResult,
    EvaluationResult
)

logger = logging.getLogger(__name__)


class EvaluationService:
    """Service for managing evaluation operations"""
    
    def __init__(self):
        """Initialize evaluation service"""
        self.evaluation_manager = get_evaluation_manager()
        self._db = None
        self._eval_repo = None
        self._chat_repo = None
    
    def get_evaluation_repository(self):
        """Get EvaluationRepository instance"""
        if self._eval_repo is None:
            from app.db import DatabaseConnection
            from app.db.repositories import EvaluationRepository
            if self._db is None:
                self._db = DatabaseConnection()
            self._eval_repo = EvaluationRepository(self._db)
        return self._eval_repo
    
    def get_chat_repository(self):
        """Get ChatRepository instance"""
        if self._chat_repo is None:
            from app.db import DatabaseConnection
            from app.db.repositories import ChatRepository
            if self._db is None:
                self._db = DatabaseConnection()
            self._chat_repo = ChatRepository(self._db)
        return self._chat_repo
    
    # Legacy evaluation functions for backwards compatibility
    def calculate_groundedness(self, response: str, context: str) -> float:
        """Calculate groundedness score using simple keyword overlap"""
        if not response or not context:
            return 0.0
        
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())
        
        if not response_words:
            return 0.0
        
        overlap = len(response_words.intersection(context_words))
        return min(1.0, overlap / len(response_words))
    
    def calculate_relevance(self, query: str, response: str) -> float:
        """Calculate relevance score using simple keyword overlap"""
        if not query or not response:
            return 0.0
        
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words.intersection(response_words))
        return min(1.0, overlap / len(query_words))
    
    def calculate_context_relevance(self, query: str, context: str) -> float:
        """Calculate context relevance using simple keyword overlap"""
        if not query or not context:
            return 0.0
        
        query_words = set(query.lower().split())
        context_words = set(context.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words.intersection(context_words))
        return min(1.0, overlap / len(query_words))
    
    # Task status operations
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get Celery task status"""
        try:
            result = AsyncResult(task_id)
            
            status_info = {
                "task_id": task_id,
                "state": result.state,
                "status": result.state
            }
            
            # Add progress info if available
            if result.state == 'PROGRESS':
                info = result.info or {}
                status_info.update({
                    "current": info.get("current", 0),
                    "total": info.get("total", 0),
                    "progress_percent": (info.get("current", 0) / info.get("total", 1)) * 100 if info.get("total") else 0,
                    "status": info.get("status", "Processing...")
                })
            
            # Add result if completed
            if result.state == 'SUCCESS':
                status_info["result"] = result.result
                if isinstance(result.result, dict):
                    status_info["started_at"] = result.result.get("started_at")
                    status_info["completed_at"] = result.result.get("completed_at")
                    if status_info["started_at"] and status_info["completed_at"]:
                        start = datetime.fromisoformat(status_info["started_at"])
                        end = datetime.fromisoformat(status_info["completed_at"])
                        status_info["execution_time"] = (end - start).total_seconds()
            
            # Add error if failed
            if result.state == 'FAILURE':
                status_info["error"] = str(result.info)
            
            return status_info
            
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            return {
                "task_id": task_id,
                "state": "UNKNOWN",
                "status": "UNKNOWN",
                "error": str(e)
            }
    
    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """Cancel a running task"""
        try:
            result = AsyncResult(task_id)
            result.revoke(terminate=True, signal='SIGKILL')
            
            return {
                "task_id": task_id,
                "status": "cancelled",
                "message": "Task cancellation requested"
            }
        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {e}")
            return {
                "task_id": task_id,
                "status": "error",
                "message": f"Failed to cancel task: {str(e)}"
            }
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get Celery queue status"""
        try:
            from app.workers.celery_app import celery_app
            
            inspect = celery_app.control.inspect()
            
            active = inspect.active()
            scheduled = inspect.scheduled()
            reserved = inspect.reserved()
            
            active_count = sum(len(tasks) for tasks in (active or {}).values())
            scheduled_count = sum(len(tasks) for tasks in (scheduled or {}).values())
            reserved_count = sum(len(tasks) for tasks in (reserved or {}).values())
            
            # Get detailed active task info
            active_details = []
            if active:
                for worker, tasks in active.items():
                    for task in tasks:
                        active_details.append({
                            "worker": worker,
                            "task_id": task.get("id"),
                            "name": task.get("name"),
                            "args": task.get("args"),
                            "kwargs": task.get("kwargs")
                        })
            
            return {
                "active_tasks": active_count,
                "scheduled_tasks": scheduled_count,
                "reserved_tasks": reserved_count,
                "total_tasks": active_count + scheduled_count + reserved_count,
                "active_task_details": active_details
            }
            
        except Exception as e:
            logger.error(f"Error getting queue status: {e}")
            return {
                "active_tasks": 0,
                "scheduled_tasks": 0,
                "reserved_tasks": 0,
                "total_tasks": 0,
                "active_task_details": [],
                "error": str(e)
            }
    
    # Metrics operations
    def get_evaluation_metrics(self, days: int = 7, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get evaluation metrics summary"""
        try:
            chat_db = self.get_chat_db()
            if not chat_db:
                return {"error": "ChatDB not available"}
            
            period_start = datetime.now() - timedelta(days=days)
            
            # Get metrics from database
            conn = chat_db.get_connection()
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    COUNT(*) as total,
                    AVG(groundedness) as avg_groundedness,
                    AVG(answer_relevance) as avg_answer_relevance,
                    AVG(context_relevance) as avg_context_relevance,
                    AVG((groundedness + answer_relevance + context_relevance) / 3.0) as avg_overall,
                    AVG(latency) as avg_latency,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as success_rate
                FROM evaluations
                WHERE timestamp >= ?
            """
            params = [period_start.isoformat()]
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            cursor.execute(query, params)
            result = cursor.fetchone()
            
            return {
                "total_evaluations": result[0] or 0,
                "avg_groundedness": round(result[1] or 0.0, 3),
                "avg_answer_relevance": round(result[2] or 0.0, 3),
                "avg_context_relevance": round(result[3] or 0.0, 3),
                "avg_overall_score": round(result[4] or 0.0, 3),
                "avg_latency": round(result[5] or 0.0, 3),
                "success_rate": round(result[6] or 0.0, 3),
                "period_start": period_start.isoformat(),
                "period_end": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting evaluation metrics: {e}")
            return {"error": str(e)}
    
    def get_historical_metrics(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get historical daily metrics"""
        try:
            chat_db = self.get_chat_db()
            if not chat_db:
                return []
            
            period_start = datetime.now() - timedelta(days=days)
            
            conn = chat_db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    DATE(timestamp) as date,
                    AVG(groundedness) as avg_groundedness,
                    AVG(answer_relevance) as avg_answer_relevance,
                    AVG(context_relevance) as avg_context_relevance,
                    COUNT(*) as total
                FROM evaluations
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, (period_start.isoformat(),))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "date": row[0],
                    "avg_groundedness": round(row[1] or 0.0, 3),
                    "avg_answer_relevance": round(row[2] or 0.0, 3),
                    "avg_context_relevance": round(row[3] or 0.0, 3),
                    "total_evaluations": row[4]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting historical metrics: {e}")
            return []
    
    def get_latency_distribution(self) -> List[Dict[str, Any]]:
        """Get latency distribution"""
        try:
            chat_db = self.get_chat_db()
            if not chat_db:
                return []
            
            conn = chat_db.get_connection()
            cursor = conn.cursor()
            
            # Get total count
            cursor.execute("SELECT COUNT(*) FROM evaluations WHERE latency IS NOT NULL")
            total = cursor.fetchone()[0] or 0
            
            if total == 0:
                return []
            
            # Define buckets
            buckets = [
                ("0-1s", 0, 1),
                ("1-2s", 1, 2),
                ("2-5s", 2, 5),
                ("5-10s", 5, 10),
                ("10s+", 10, float('inf'))
            ]
            
            results = []
            for bucket_name, min_latency, max_latency in buckets:
                if max_latency == float('inf'):
                    cursor.execute(
                        "SELECT COUNT(*) FROM evaluations WHERE latency >= ?",
                        (min_latency,)
                    )
                else:
                    cursor.execute(
                        "SELECT COUNT(*) FROM evaluations WHERE latency >= ? AND latency < ?",
                        (min_latency, max_latency)
                    )
                
                count = cursor.fetchone()[0] or 0
                results.append({
                    "bucket": bucket_name,
                    "count": count,
                    "percentage": round((count / total) * 100, 2)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting latency distribution: {e}")
            return []
    
    def get_quality_breakdown(self) -> List[Dict[str, Any]]:
        """Get quality score breakdown"""
        try:
            chat_db = self.get_chat_db()
            if not chat_db:
                return []
            
            conn = chat_db.get_connection()
            cursor = conn.cursor()
            
            # Get total count
            cursor.execute("SELECT COUNT(*) FROM evaluations")
            total = cursor.fetchone()[0] or 0
            
            if total == 0:
                return []
            
            # Define score ranges
            ranges = [
                ("Poor (0-0.5)", 0, 0.5),
                ("Fair (0.5-0.7)", 0.5, 0.7),
                ("Good (0.7-0.85)", 0.7, 0.85),
                ("Excellent (0.85-1.0)", 0.85, 1.0)
            ]
            
            results = []
            for range_name, min_score, max_score in ranges:
                cursor.execute("""
                    SELECT COUNT(*) FROM evaluations 
                    WHERE ((groundedness + answer_relevance + context_relevance) / 3.0) >= ? 
                    AND ((groundedness + answer_relevance + context_relevance) / 3.0) < ?
                """, (min_score, max_score))
                
                count = cursor.fetchone()[0] or 0
                results.append({
                    "score_range": range_name,
                    "count": count,
                    "percentage": round((count / total) * 100, 2)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting quality breakdown: {e}")
            return []
    
    def get_recent_results(self, limit: int = 10, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent evaluation results"""
        try:
            return get_recent_evaluation_stats(limit=limit, user_id=user_id)
        except Exception as e:
            logger.error(f"Error getting recent results: {e}")
            return []
    
    def get_evaluation_results(
        self, 
        page: int = 1, 
        page_size: int = 20,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get paginated evaluation results"""
        try:
            chat_db = self.get_chat_db()
            if not chat_db:
                return {"results": [], "total_count": 0, "page": page, "page_size": page_size}
            
            conn = chat_db.get_connection()
            cursor = conn.cursor()
            
            # Get total count
            count_query = "SELECT COUNT(*) FROM evaluations"
            params = []
            if user_id:
                count_query += " WHERE user_id = ?"
                params.append(user_id)
            
            cursor.execute(count_query, params)
            total_count = cursor.fetchone()[0] or 0
            
            # Get paginated results
            offset = (page - 1) * page_size
            results_query = """
                SELECT * FROM evaluations
            """
            if user_id:
                results_query += " WHERE user_id = ?"
            results_query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            
            params_with_pagination = params + [page_size, offset]
            cursor.execute(results_query, params_with_pagination)
            
            results = []
            for row in cursor.fetchall():
                results.append(dict(row))
            
            return {
                "results": results,
                "total_count": total_count,
                "page": page,
                "page_size": page_size
            }
            
        except Exception as e:
            logger.error(f"Error getting evaluation results: {e}")
            return {"results": [], "total_count": 0, "page": page, "page_size": page_size}
