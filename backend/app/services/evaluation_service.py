"""
Evaluation Service
Business logic for evaluation operations
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import sqlite3
import json
import redis
import os

from app.config import settings
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
        self._celery_app = None
        self._redis_client = None
    
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
    
    def get_celery_app(self):
        """Get Celery app instance"""
        if self._celery_app is None:
            from app.workers.celery_app import celery_app
            self._celery_app = celery_app
        return self._celery_app
    
    def get_redis_client(self):
        """Get Redis client instance"""
        if self._redis_client is None:
            # Use same Redis instance as Celery broker for consistency
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            self._redis_client = redis.Redis.from_url(redis_url)
        return self._redis_client
    
        
    # Task status operations
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get task status using hybrid approach: Redis (fast) + Celery (reliable).
        
        Architecture:
        1. Primary: Read from Redis cache (fast, real-time progress)
        2. Fallback: Read from Celery backend (reliable, persistent state)
        
        This is production-grade because:
        - Fast response times (Redis cache)
        - Real-time progress updates (published by tasks)
        - Reliable fallback (Celery backend for expired/missing cache)
        - Handles Redis failures gracefully
        
        Used by: HTTP polling fallback when WebSocket connection fails
        """
        try:
            redis_client = self.get_redis_client()
            celery_app = self.get_celery_app()
            
            # Try Redis cache first (fast path - most common case)
            cached_progress = redis_client.get(f"task_progress:{task_id}")
            
            if cached_progress:
                # Found in Redis cache - return real-time progress
                try:
                    progress_data = json.loads(cached_progress)
                    
                    # Enhance with standardized fields
                    status_info = {
                        "task_id": task_id,
                        "state": progress_data.get("status", "UNKNOWN"),
                        "status": progress_data.get("status", "UNKNOWN"),
                        "timestamp": progress_data.get("timestamp", datetime.utcnow().isoformat()),
                        "source": "redis_cache"  # Debug info
                    }
                    
                    # Add progress data if available
                    if "data" in progress_data:
                        data = progress_data["data"]
                        status_info.update({
                            "progress": data.get("progress", 0),
                            "message": data.get("message", ""),
                            "data": data
                        })
                        
                        # Add computed fields
                        if "current" in data and "total" in data and data["total"] > 0:
                            status_info["progress_percent"] = (data["current"] / data["total"]) * 100
                    
                    return status_info
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode Redis cache for task {task_id}: {e}")
                    # Fall through to Celery backend
            
            # Redis cache miss or decode error - fallback to Celery backend (slow but reliable)
            logger.info(f"Redis cache miss for task {task_id}, falling back to Celery backend")
            
            result = celery_app.AsyncResult(task_id)
            
            status_info = {
                "task_id": task_id,
                "state": result.state,
                "status": result.state,
                "timestamp": datetime.utcnow().isoformat(),
                "source": "celery_backend"  # Debug info
            }
            
            # Add result data based on state
            if result.state == 'SUCCESS':
                status_info["result"] = result.result
                status_info["progress"] = 1.0
                
                # Extract timing info if available
                if isinstance(result.result, dict):
                    status_info["started_at"] = result.result.get("started_at")
                    status_info["completed_at"] = result.result.get("completed_at")
                    
                    if status_info.get("started_at") and status_info.get("completed_at"):
                        try:
                            start = datetime.fromisoformat(status_info["started_at"])
                            end = datetime.fromisoformat(status_info["completed_at"])
                            status_info["execution_time"] = (end - start).total_seconds()
                        except (ValueError, TypeError):
                            pass
            
            elif result.state == 'FAILURE':
                status_info["error"] = str(result.info)
                status_info["progress"] = 0
                
                # Check for detailed error in Redis final status cache
                try:
                    final_status = redis_client.get(f"task_final_status:{task_id}")
                    if final_status:
                        final_data = json.loads(final_status)
                        status_info["error"] = final_data.get("data", {}).get("error", status_info["error"])
                        status_info["message"] = final_data.get("data", {}).get("message", "Task failed")
                except Exception:
                    pass
            
            elif result.state == 'PROGRESS':
                # Celery tasks can store progress in meta
                if hasattr(result, 'info') and isinstance(result.info, dict):
                    status_info.update({
                        "current": result.info.get("current", 0),
                        "total": result.info.get("total", 0),
                        "progress": result.info.get("progress", 0),
                        "message": result.info.get("status", "Processing...")
                    })
                    
                    if status_info["total"] > 0:
                        status_info["progress_percent"] = (status_info["current"] / status_info["total"]) * 100
            
            elif result.state in ['PENDING', 'STARTED']:
                status_info["progress"] = 0
                status_info["message"] = "Task queued" if result.state == 'PENDING' else "Task started"
            
            return status_info
            
        except Exception as e:
            logger.error(f"Error getting task status for {task_id}: {e}", exc_info=True)
            return {
                "task_id": task_id,
                "state": "UNKNOWN",
                "status": "UNKNOWN",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "source": "error"
            }
   
      
    def get_recent_results(self, limit: int = 10, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent evaluation tasks with aggregated results"""
        try:
            eval_repo = self.get_evaluation_repository()
            
            with eval_repo.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Query evaluation_tasks with aggregated metrics from evaluation_results
                query = """
                    SELECT 
                        t.task_id,
                        t.dataset_id,
                        t.status,
                        t.created_at,
                        t.completed_at,
                        d.name as dataset_name,
                        AVG(r.groundedness_score) as avg_groundedness,
                        AVG(r.relevance_score) as avg_relevance,
                        AVG(r.quality_score) as avg_quality,
                        AVG(r.latency_ms) as avg_latency,
                        COUNT(r.id) as total_questions,
                        MAX(r.model_used) as model_used
                    FROM evaluation_tasks t
                    LEFT JOIN datasets d ON t.dataset_id = d.id
                    LEFT JOIN evaluation_results r ON r.task_id = t.task_id
                    GROUP BY t.task_id, t.dataset_id, t.status, t.created_at, t.completed_at, d.name
                    ORDER BY t.created_at DESC
                    LIMIT ?
                """
                
                cursor.execute(query, (limit,))
                rows = cursor.fetchall()
                
                results = []
                for row in rows:
                    task_data = {
                        "task_id": row[0],
                        "dataset_id": row[1],
                        "status": row[2],
                        "created_at": row[3],
                        "completed_at": row[4],
                        "metadata": {
                            "dataset_id": row[1],
                            "dataset_name": row[5] or f"Dataset {row[1]}",
                            "model_id": row[11] or "gemma2:2b",
                            "model_name": row[11] or "gemma2:2b",
                            "total_questions": row[10] or 0
                        },
                        "results": {
                            "groundedness": {"score": row[6] or 0},
                            "context_relevance": {"score": row[7] or 0},
                            "answer_relevance": {"score": row[8] or 0},
                            "evaluation_time_seconds": (row[9] or 0) / 1000.0  # Convert ms to seconds
                        },
                        "user_id": "admin"
                    }
                    results.append(task_data)
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting recent results: {e}")
            logger.exception(e)
            return []
    
    def get_evaluation_results(
        self, 
        page: int = 1, 
        page_size: int = 20,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get paginated evaluation results"""
        try:
            eval_repo = self.get_evaluation_repository()
            
            with eval_repo.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get total count
                cursor.execute("SELECT COUNT(*) FROM evaluation_metrics")
                total_count = cursor.fetchone()[0] or 0
                
                # Get paginated results
                offset = (page - 1) * page_size
                cursor.execute("""
                    SELECT 
                        id, session_id, message_id, query, response, context,
                        groundedness_score, context_relevance_score, answer_quality_score,
                        latency_ms, evaluated_at
                    FROM evaluation_metrics
                    ORDER BY evaluated_at DESC 
                    LIMIT ? OFFSET ?
                """, (page_size, offset))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        "id": row[0],
                        "session_id": row[1],
                        "message_id": row[2],
                        "query": row[3],
                        "response": row[4],
                        "context": row[5],
                        "groundedness": row[6],
                        "context_relevance": row[7],
                        "answer_relevance": row[8],
                        "latency": row[9] / 1000 if row[9] else None,  # Convert ms to seconds
                        "timestamp": row[10]
                    })
                
                return {
                    "results": results,
                    "total_count": total_count,
                    "page": page,
                    "page_size": page_size
                }
            
        except Exception as e:
            logger.error(f"Error getting evaluation results: {e}")
            return {"results": [], "total_count": 0, "page": page, "page_size": page_size}
    
    def get_datasets(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all evaluation datasets"""
        try:
            eval_repo = self.get_evaluation_repository()
            
            with eval_repo.db.get_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT 
                        id, name, description, created_at, updated_at,
                        document_count, file_path, status, created_by
                    FROM evaluation_datasets
                    ORDER BY created_at DESC
                """
                
                cursor.execute(query)
                
                datasets = []
                for row in cursor.fetchall():
                    datasets.append({
                        "id": row[0],
                        "name": row[1],
                        "description": row[2],
                        "created_at": row[3],
                        "updated_at": row[4],
                        "document_count": row[5] or 0,
                        "file_path": row[6],
                        "status": row[7] or "Processing",
                        "created_by": row[8]
                    })
                
                return datasets
            
        except Exception as e:
            logger.error(f"Error getting datasets: {e}")
            return []
    
    def get_overview(self, time_range: str = "7d") -> Dict[str, Any]:
        """
        Get evaluation overview with statistics
        
        Args:
            time_range: Time range for stats (7d, 30d, 90d, all)
        """
        try:
            eval_repo = self.get_evaluation_repository()
            
            # Parse time range
            if time_range == "all":
                date_filter = ""
            else:
                days = int(time_range.rstrip('d'))
                cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
                date_filter = f"WHERE evaluated_at >= '{cutoff_date}'"
            
            with eval_repo.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get total evaluations
                cursor.execute(f"SELECT COUNT(*) FROM evaluation_metrics {date_filter}")
                total_evaluations = cursor.fetchone()[0] or 0
                
                # Get average scores
                cursor.execute(f"""
                    SELECT 
                        AVG(groundedness_score) as avg_groundedness,
                        AVG(context_relevance_score) as avg_context_relevance,
                        AVG(answer_quality_score) as avg_answer_quality,
                        AVG(latency_ms) as avg_latency
                    FROM evaluation_metrics 
                    {date_filter}
                """)
                scores = cursor.fetchone()
                
                # Get dataset count
                cursor.execute("SELECT COUNT(*) FROM evaluation_datasets")
                dataset_count = cursor.fetchone()[0] or 0
                
                # Get recent evaluations
                cursor.execute(f"""
                    SELECT 
                        DATE(evaluated_at) as date,
                        COUNT(*) as count
                    FROM evaluation_metrics 
                    {date_filter}
                    GROUP BY DATE(evaluated_at)
                    ORDER BY date DESC
                    LIMIT 30
                """)
                daily_stats = [{"date": row[0], "count": row[1]} for row in cursor.fetchall()]
                
                # Structure response to match frontend expectations
                return {
                    "overall": {
                        "groundedness": round(scores[0], 3) if scores[0] else 0,
                        "contextRelevance": round(scores[1], 3) if scores[1] else 0,
                        "answerQuality": round(scores[2], 3) if scores[2] else 0,
                        "averageLatency": round(scores[3], 2) if scores[3] else 0,
                        "totalEvaluations": total_evaluations
                    },
                    "historical": [],  # Empty for now, can be populated from daily_stats if needed
                    "latencyDistribution": [],  # Empty for now, can be added later
                    "detailed": []  # Empty for now, detailed results are fetched separately
                }
            
        except Exception as e:
            logger.error(f"Error getting evaluation overview: {e}", exc_info=True)
            return {
                "overall": {
                    "groundedness": 0,
                    "contextRelevance": 0,
                    "answerQuality": 0,
                    "averageLatency": 0,
                    "totalEvaluations": 0
                },
                "historical": [],
                "latencyDistribution": [],
                "detailed": []
            }
