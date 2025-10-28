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
