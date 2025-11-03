"""
DEPRECATED: This file has been migrated to app/api/v1/endpoints/evaluation.py

All evaluation endpoints have been moved to the new clean architecture:
- REST endpoints: app/api/v1/endpoints/evaluation.py
- WebSocket endpoints: app/api/v1/endpoints/websocket_evaluation.py
- Service layer: app/services/evaluation_service.py
- Schemas: app/api/v1/schemas/evaluation.py

This file is kept for reference only and will be removed in a future version.
Do not modify this file. Make changes in the new location instead.
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import json
import logging
from datetime import datetime, timedelta
import sqlite3
import random
import asyncio
import time
import redis
import os
from pydantic import BaseModel

# Background task imports
from celery_app import celery_app
from evaluation_tasks import (
    evaluate_rag_response,
    batch_evaluate_conversations,
    evaluate_dataset_with_rag,
    create_dataset_background,
    get_evaluation_status,
    EvaluationTaskStatus
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Redis client for pub/sub notifications - use same DB as Celery broker
redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))

# WebSocket connection manager with reconnection support
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_info: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        self.active_connections[task_id] = websocket
        self.connection_info[task_id] = {
            "connected_at": datetime.utcnow(),
            "last_ping": datetime.utcnow(),
            "reconnect_count": self.connection_info.get(task_id, {}).get("reconnect_count", 0)
        }
        
        # Increment reconnect count if this is a reconnection
        if task_id in self.connection_info and self.connection_info[task_id].get("reconnect_count", 0) > 0:
            self.connection_info[task_id]["reconnect_count"] += 1
            logger.info(f"WebSocket reconnected for task {task_id} (attempt #{self.connection_info[task_id]['reconnect_count']})")
        else:
            logger.info(f"WebSocket connected for task {task_id}")

    def disconnect(self, task_id: str):
        if task_id in self.active_connections:
            del self.active_connections[task_id]
            # Keep connection info for reconnection tracking
            if task_id in self.connection_info:
                self.connection_info[task_id]["disconnected_at"] = datetime.utcnow()
            logger.info(f"WebSocket disconnected for task {task_id}")

    async def send_message(self, task_id: str, message: dict):
        if task_id in self.active_connections:
            try:
                # Update last activity timestamp
                if task_id in self.connection_info:
                    self.connection_info[task_id]["last_activity"] = datetime.utcnow()
                
                await self.active_connections[task_id].send_text(json.dumps(message))
                return True
            except ConnectionResetError:
                logger.info(f"WebSocket connection reset for task {task_id} - client disconnected")
                self.disconnect(task_id)
                return False
            except Exception as e:
                # Log as info instead of error for common disconnection scenarios
                if "disconnected" in str(e).lower() or "closed" in str(e).lower():
                    logger.info(f"WebSocket connection closed for task {task_id}: {e}")
                else:
                    logger.error(f"Failed to send WebSocket message to {task_id}: {e}")
                self.disconnect(task_id)
                return False
        return False
    
    def is_connected(self, task_id: str) -> bool:
        return task_id in self.active_connections
    
    def get_connection_info(self, task_id: str) -> Dict:
        return self.connection_info.get(task_id, {})

manager = ConnectionManager()

# Pydantic models for request/response validation
class EvaluationRequest(BaseModel):
    query: str
    response: str
    context: List[str]
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class BatchEvaluationRequest(BaseModel):
    conversation_ids: List[str]
    user_id: Optional[str] = None

class DatasetEvaluationRequest(BaseModel):
    dataset_id: int
    dataset_name: Optional[str] = None
    model_id: Optional[str] = None
    retrieval_config: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class DatasetCreationRequest(BaseModel):
    name: str
    description: str
    document_ids: List[str]
    num_questions_per_doc: int = 3
    model_name: str = "llama3"
    difficulty_levels: List[str] = ["easy", "medium", "hard"]
    user_id: str = "admin"

class EvaluationResponse(BaseModel):
    task_id: str
    status: str
    message: str
    websocket_url: Optional[str] = None

# Import dependencies that don't cause circular imports
from evaluation_system import (
    get_evaluation_manager, 
    evaluate_chat_response, 
    get_recent_evaluation_stats,
    RAGTriadResult,
    EvaluationResult
)

def get_chat_db():
    """Get ChatDB instance"""
    try:
        from chat_db import ChatDB
        return ChatDB()
    except ImportError:
        # Fallback if chat_db is not available
        return None

# Legacy evaluation functions for backwards compatibility
def calculate_groundedness(response: str, context: str) -> float:
    """Legacy function - calculate groundedness score using simple keyword overlap"""
    if not context or not response:
        return 0.0
    
    context_words = set(context.lower().split())
    response_words = set(response.lower().split())
    
    if not response_words:
        return 0.0
    
    overlap = len(context_words.intersection(response_words))
    score = min(overlap / len(response_words), 1.0)
    
    return min(max(score + random.uniform(-0.2, 0.2), 0.0), 1.0)

def calculate_context_relevance(query: str, context: str) -> float:
    """Legacy function - calculate context relevance using simple keyword overlap"""
    if not query or not context:
        return 0.0
    
    query_words = set(query.lower().split())
    context_words = set(context.lower().split())
    
    if not query_words:
        return 0.0
    
    overlap = len(query_words.intersection(context_words))
    score = min(overlap / len(query_words), 1.0)
    
    return min(max(score + random.uniform(-0.15, 0.15), 0.0), 1.0)

def calculate_answer_quality(query: str, response: str) -> float:
    """Legacy function - calculate answer quality using simple metrics"""
    if not query or not response:
        return 0.0
    
    query_words = set(query.lower().split())
    response_words = set(response.lower().split())
    
    if not query_words or not response_words:
        return 0.0
    
    overlap_score = len(query_words.intersection(response_words)) / len(query_words)
    length_score = min(len(response.split()) / 50.0, 1.0)
    
    quality_score = (overlap_score * 0.7) + (length_score * 0.3)
    
    return min(max(quality_score + random.uniform(-0.2, 0.2), 0.0), 1.0)

# New background evaluation endpoints
@router.post("/evaluate/async", response_model=EvaluationResponse)
async def start_async_evaluation(request: EvaluationRequest):
    """
    Start asynchronous RAG evaluation in the background
    
    This endpoint submits an evaluation task to Celery and returns immediately
    with a task ID that can be used to track progress via WebSocket.
    """
    try:
        # Submit evaluation task to Celery with highest priority
        task = evaluate_rag_response.apply_async(
            args=[],
            kwargs={
                'query': request.query,
                'response': request.response,
                'context': request.context,
                'conversation_id': request.conversation_id,
                'user_id': request.user_id,
                'metadata': request.metadata
            },
            priority=1  # Highest priority for single evaluations
        )
        
        # Create database record for persistence
        chat_db = get_chat_db()
        if chat_db:
            chat_db.create_evaluation_task(
                task_id=task.id,
                task_type="single",
                query=request.query,
                response=request.response,
                context_chunks=len(request.context) if request.context else 0,
                conversation_id=request.conversation_id,
                user_id=request.user_id,
                metadata=request.metadata
            )
        
        logger.info(f"Started async evaluation task: {task.id}")
        
        return EvaluationResponse(
            task_id=task.id,
            status=EvaluationTaskStatus.PENDING,
            message="Evaluation task submitted successfully",
            websocket_url=f"/ws/evaluation/{task.id}"
        )
        
    except Exception as e:
        logger.error(f"Failed to start async evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start evaluation: {str(e)}")

@router.post("/evaluate/batch", response_model=EvaluationResponse)
async def start_batch_evaluation(request: BatchEvaluationRequest):
    """
    Start batch evaluation of multiple conversations
    
    This endpoint submits a batch evaluation task to process multiple
    conversations in the background.
    """
    try:
        if not request.conversation_ids:
            raise HTTPException(status_code=400, detail="No conversation IDs provided")
        
        if len(request.conversation_ids) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 conversations per batch")
        
        # Submit batch evaluation task to Celery
        task = batch_evaluate_conversations.delay(
            conversation_ids=request.conversation_ids,
            user_id=request.user_id
        )
        
        logger.info(f"Started batch evaluation task: {task.id} for {len(request.conversation_ids)} conversations")
        
        return EvaluationResponse(
            task_id=task.id,
            status=EvaluationTaskStatus.PENDING,
            message=f"Batch evaluation task submitted for {len(request.conversation_ids)} conversations",
            websocket_url=f"/ws/evaluation/{task.id}"
        )
        
    except Exception as e:
        logger.error(f"Failed to start batch evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start batch evaluation: {str(e)}")

@router.post("/evaluate/dataset", response_model=EvaluationResponse)
async def start_dataset_evaluation(request: DatasetEvaluationRequest):
    """
    Start real dataset evaluation using RAG system
    
    This endpoint submits a dataset evaluation task that generates real
    question-answer pairs using the RAG system and evaluates them.
    """
    try:
        if not request.dataset_id:
            raise HTTPException(status_code=400, detail="Dataset ID is required")
        
        # Get model_id from metadata if not provided directly
        model_id = request.model_id
        if not model_id and request.metadata:
            model_id = request.metadata.get('model_id')
        
        if not model_id:
            raise HTTPException(status_code=400, detail="Model ID is required")
        
        # Get retrieval_config from metadata if not provided directly
        retrieval_config = request.retrieval_config or {}
        if request.metadata and request.metadata.get('retrieval_config'):
            retrieval_config = request.metadata['retrieval_config']
        
        # Submit dataset evaluation task to Celery with higher priority
        task = evaluate_dataset_with_rag.apply_async(
            args=[],
            kwargs={
                'dataset_id': request.dataset_id,
                'model_id': model_id,
                'retrieval_config': retrieval_config,
                'user_id': request.user_id or "admin"
            },
            priority=3  # Higher priority for evaluations
        )
        
        logger.info(f"Started dataset evaluation task: {task.id} for dataset {request.dataset_id} with model {model_id}")
        
        return EvaluationResponse(
            task_id=task.id,
            status=EvaluationTaskStatus.PENDING,
            message=f"Dataset evaluation task submitted for dataset {request.dataset_id}",
            websocket_url=f"/ws/evaluation/{task.id}"
        )
        
    except Exception as e:
        logger.error(f"Failed to start dataset evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start dataset evaluation: {str(e)}")

@router.post("/create/dataset", response_model=EvaluationResponse)
async def start_dataset_creation(request: DatasetCreationRequest):
    """
    Start background dataset creation using async Celery task
    
    This endpoint submits a dataset creation task that generates evaluation 
    questions from selected documents using LLM with proper task priorities.
    """
    try:
        # Debug logging
        logger.info(f"Received dataset creation request: {request.dict()}")
        logger.info(f"Document IDs type and content: {type(request.document_ids)} - {request.document_ids}")
        logger.info(f"Model name type and content: {type(request.model_name)} - {request.model_name}")
        
        if not request.name:
            raise HTTPException(status_code=400, detail="Dataset name is required")
        
        if not request.document_ids:
            raise HTTPException(status_code=400, detail="At least one document must be selected")
        
        # Create a pending dataset record immediately so it shows up in the UI
        from chat_db import get_chat_db
        chat_db = get_chat_db()
        
        try:
            # Create dataset record with PENDING status
            dataset_id = chat_db.create_evaluation_dataset(
                name=request.name,
                description=request.description,
                document_count=len(request.document_ids),
                created_by=request.user_id
            )
            logger.info(f"Created pending dataset record with ID: {dataset_id}")
        except Exception as e:
            logger.error(f"Failed to create dataset record: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create dataset record: {str(e)}")
        
        # Submit dataset creation task to Celery with lower priority
        task = create_dataset_background.apply_async(
            args=[],
            kwargs={
                'name': request.name,
                'description': request.description,
                'document_ids': request.document_ids,
                'num_questions_per_doc': request.num_questions_per_doc,
                'model_name': request.model_name,
                'difficulty_levels': request.difficulty_levels,
                'user_id': request.user_id,
                'dataset_id': dataset_id  # Pass the dataset_id to the task
            },
            priority=7  # Lower priority for dataset creation
        )
        
        # Update the dataset record with the task_id (just update status for now)
        try:
            chat_db.update_evaluation_dataset_status(
                dataset_id,
                status='Processing'
            )
            logger.info(f"Updated dataset {dataset_id} with status: Processing")
        except Exception as e:
            logger.error(f"Failed to update dataset status: {e}")
        
        logger.info(f"Started async dataset creation task: {task.id} for dataset '{request.name}' with {len(request.document_ids)} documents")
        
        return EvaluationResponse(
            task_id=task.id,
            status=EvaluationTaskStatus.PENDING,
            message=f"Dataset creation task submitted for '{request.name}'",
            websocket_url=f"/ws/evaluation/{task.id}"
        )
        
    except Exception as e:
        logger.error(f"Failed to start dataset creation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start dataset creation: {str(e)}")

@router.get("/evaluate/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Get the current status of an evaluation task
    
    Returns task status, progress information, and results if completed.
    """
    try:
        # Get task status from Celery
        task_result = celery_app.AsyncResult(task_id)
        
        # Get persistent task info from database
        chat_db = get_chat_db()
        db_task = None
        if chat_db:
            db_task = chat_db.get_evaluation_task(task_id)
        
        # Try to get cached result from Redis
        cached_result = redis_client.get(f"evaluation_result:{task_id}")
        cached_batch_result = redis_client.get(f"batch_evaluation_result:{task_id}")
        
        status_info = {
            "task_id": task_id,
            "status": task_result.status,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Include database task info if available
        if db_task:
            status_info.update({
                "task_type": db_task['task_type'],
                "query": db_task['query'],
                "response": db_task['response'],
                "context_chunks": db_task['context_chunks'],
                "conversation_id": db_task['conversation_id'],
                "user_id": db_task['user_id'],
                "metadata": db_task['metadata'],
                "created_at": db_task['created_at'],
                "updated_at": db_task['updated_at'],
                "completed_at": db_task['completed_at']
            })
            
            # Use database status if task is completed there
            if db_task['status'] in ['SUCCESS', 'FAILURE']:
                status_info["status"] = db_task['status']
                
                if db_task['status'] == 'SUCCESS':
                    status_info["result"] = {
                        "groundedness_score": db_task['groundedness_score'],
                        "answer_relevance_score": db_task['answer_relevance_score'],
                        "context_relevance_score": db_task['context_relevance_score'],
                        "overall_score": db_task['overall_score'],
                        "evaluation_time": db_task['evaluation_time']
                    }
                elif db_task['status'] == 'FAILURE':
                    status_info["error"] = db_task['error_message']
        
        if task_result.successful():
            status_info["result"] = task_result.result
            if cached_result:
                status_info["cached_result"] = json.loads(cached_result)
            elif cached_batch_result:
                status_info["cached_result"] = json.loads(cached_batch_result)
        elif task_result.failed():
            status_info["error"] = str(task_result.result)
        elif task_result.status == "PENDING":
            status_info["message"] = "Task is waiting to be processed"
        elif task_result.status == "STARTED":
            status_info["message"] = "Task is currently being processed"
        
        return status_info
        
    except Exception as e:
        logger.error(f"Failed to get task status for {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")

@router.delete("/evaluate/task/{task_id}")
async def cancel_evaluation_task(task_id: str):
    """
    Cancel a running evaluation task
    """
    try:
        # Revoke the task
        celery_app.control.revoke(task_id, terminate=True)
        
        # Clean up cached results
        redis_client.delete(f"evaluation_result:{task_id}")
        redis_client.delete(f"batch_evaluation_result:{task_id}")
        
        logger.info(f"Cancelled evaluation task: {task_id}")
        
        return {
            "task_id": task_id,
            "status": "CANCELLED",
            "message": "Task cancelled successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to cancel task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")

@router.get("/status/{task_id}")
async def get_task_status_polling(task_id: str):
    """
    Get current task status for polling fallback
    
    This endpoint provides comprehensive task status information for clients
    that need to poll for updates when WebSocket connections fail or disconnect.
    Supports both evaluation and dataset creation tasks.
    """
    try:
        # Check if it's a Celery task
        task_result = celery_app.AsyncResult(task_id)
        
        # Try to get task info from database
        db_task = None
        try:
            from chat_db import get_chat_db
            chat_db = get_chat_db()
            
            # Use direct database connection following ChatDB pattern
            with sqlite3.connect(chat_db.db_path) as conn:
                cursor = conn.cursor()
            
            # Check evaluation tasks first
            cursor.execute("""
                SELECT id, task_type, user_id, status, 
                       created_at, updated_at, completed_at, evaluation_time,
                       groundedness_score, answer_relevance_score, context_relevance_score,
                       overall_score, error_message, metadata
                FROM evaluation_tasks 
                WHERE id = ?
            """, (task_id,))
            
            eval_task = cursor.fetchone()
            if eval_task:
                db_task = {
                    'task_id': eval_task[0],
                    'task_type': eval_task[1], 
                    'user_id': eval_task[2],
                    'status': eval_task[3],
                    'created_at': eval_task[4],
                    'updated_at': eval_task[5],
                    'completed_at': eval_task[6],
                    'evaluation_time': eval_task[7],
                    'groundedness_score': eval_task[8],
                    'answer_relevance_score': eval_task[9],
                    'context_relevance_score': eval_task[10],
                    'overall_score': eval_task[11],
                    'error_message': eval_task[12],
                    'metadata': eval_task[13]
                }
            else:
                # Check dataset creation tasks
                cursor.execute("""
                    SELECT id, name, description, status, created_at, updated_at,
                           file_path, document_count
                    FROM evaluation_datasets 
                    WHERE name = ? OR id = ?
                """, (task_id, task_id))
                
                dataset_task = cursor.fetchone()
                if dataset_task:
                    db_task = {
                        'task_id': task_id,
                        'task_type': 'dataset_creation',
                        'dataset_id': dataset_task[0],
                        'name': dataset_task[1],
                        'description': dataset_task[2],
                        'status': dataset_task[3],
                        'created_at': dataset_task[4],
                        'updated_at': dataset_task[5],
                        'file_path': dataset_task[6],
                        'document_count': dataset_task[7]
                    }
            
            cursor.close()
            conn.close()
            
        except Exception as db_error:
            logger.warning(f"Could not fetch task from database: {db_error}")
        
        # Build comprehensive status response
        status_info = {
            "task_id": task_id,
            "status": task_result.status,
            "timestamp": datetime.utcnow().isoformat(),
            "celery_status": task_result.status
        }
        
        # Add database information if available
        if db_task:
            status_info.update({
                "task_type": db_task.get('task_type', 'unknown'),
                "created_at": db_task.get('created_at'),
                "updated_at": db_task.get('updated_at'),
                "completed_at": db_task.get('completed_at')
            })
            
            # Use database status if more recent than Celery status
            if db_task.get('status') in ['SUCCESS', 'FAILURE', 'STARTED', 'PROGRESS']:
                status_info["status"] = db_task['status']
            
            # Add task-specific information
            if db_task.get('task_type') == 'dataset_creation':
                status_info.update({
                    "dataset_id": db_task.get('dataset_id'),
                    "dataset_name": db_task.get('name'),
                    "description": db_task.get('description'),
                    "file_path": db_task.get('file_path'),
                    "question_count": db_task.get('question_count'),
                    "document_count": db_task.get('document_count')
                })
            else:  # evaluation task
                status_info.update({
                    "dataset_id": db_task.get('dataset_id'),
                    "model_id": db_task.get('model_id'),
                    "user_id": db_task.get('user_id'),
                    "evaluation_time": db_task.get('evaluation_time'),
                    "metadata": db_task.get('metadata')
                })
                
                # Add results if completed successfully
                if db_task.get('status') == 'SUCCESS':
                    status_info["results"] = {
                        "groundedness_score": db_task.get('groundedness_score'),
                        "answer_relevance_score": db_task.get('answer_relevance_score'), 
                        "context_relevance_score": db_task.get('context_relevance_score'),
                        "overall_score": db_task.get('overall_score'),
                        "evaluation_time": db_task.get('evaluation_time')
                    }
                elif db_task.get('status') == 'FAILURE':
                    status_info["error"] = db_task.get('error_message')
        
        # Add Celery task result if available
        if task_result.successful():
            status_info["celery_result"] = task_result.result
        elif task_result.failed():
            status_info["celery_error"] = str(task_result.result)
        
        # Add progress information for active tasks
        if status_info["status"] in ["PENDING", "STARTED", "PROGRESS"]:
            if status_info["status"] == "PENDING":
                status_info["message"] = "Task is waiting to be processed"
            elif status_info["status"] == "STARTED":
                status_info["message"] = "Task is currently being processed"
            elif status_info["status"] == "PROGRESS":
                status_info["message"] = "Task is in progress"
        
        # Check for cached progress updates
        try:
            cached_progress = redis_client.get(f"task_progress:{task_id}")
            if cached_progress:
                progress_data = json.loads(cached_progress)
                status_info["progress"] = progress_data
        except Exception as cache_error:
            logger.debug(f"No cached progress for task {task_id}: {cache_error}")
        
        return status_info
        
    except Exception as e:
        logger.error(f"Failed to get task status for {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")

@router.get("/ws/health/{task_id}")
async def get_websocket_health(task_id: str):
    """
    Check WebSocket connection health for a specific task
    
    Returns information about whether a WebSocket connection is active
    and its connection metadata.
    """
    try:
        is_connected = manager.is_connected(task_id)
        connection_info = manager.get_connection_info(task_id)
        
        health_info = {
            "task_id": task_id,
            "websocket_connected": is_connected,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if connection_info:
            health_info["connection_info"] = {
                "connected_at": connection_info.get("connected_at", "").isoformat() if connection_info.get("connected_at") else None,
                "disconnected_at": connection_info.get("disconnected_at", "").isoformat() if connection_info.get("disconnected_at") else None,
                "last_activity": connection_info.get("last_activity", "").isoformat() if connection_info.get("last_activity") else None,
                "last_ping": connection_info.get("last_ping", "").isoformat() if connection_info.get("last_ping") else None,
                "reconnect_count": connection_info.get("reconnect_count", 0)
            }
            
            # Calculate connection uptime if connected
            if is_connected and connection_info.get("connected_at"):
                uptime = datetime.utcnow() - connection_info["connected_at"]
                health_info["connection_info"]["uptime_seconds"] = uptime.total_seconds()
        
        return health_info
        
    except Exception as e:
        logger.error(f"Failed to get WebSocket health for {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get WebSocket health: {str(e)}")

@router.websocket("/ws/evaluation/{task_id}")
async def websocket_evaluation_updates(websocket: WebSocket, task_id: str):
    """
    DEPRECATED: Use /api/evaluation/ws/evaluation/{task_id} from app.api.v1.endpoints.websocket_evaluation instead
    
    WebSocket endpoint for real-time evaluation progress updates with auto-reconnection support
    
    This endpoint will be removed in a future version.
    Please migrate to the new endpoint structure.
    
    Clients can connect to this endpoint to receive real-time updates about their evaluation tasks.
    Supports reconnection and provides connection health monitoring.
    """
    await manager.connect(websocket, task_id)
    
    try:
        # Subscribe to Redis pub/sub for this task
        pubsub = redis_client.pubsub()
        pubsub.subscribe(f"evaluation_updates:{task_id}")
        
        # Get current task status for initial sync
        try:
            # Import here to avoid circular imports
            from chat_db import get_chat_db
            chat_db = get_chat_db()
            
            # Use direct database connection following ChatDB pattern
            with sqlite3.connect(chat_db.db_path) as conn:
                cursor = conn.cursor()
            
            # Check for existing task status
            cursor.execute("""
                SELECT status, metadata, completed_at
                FROM evaluation_tasks 
                WHERE task_id = ?
                UNION
                SELECT status, '{}' as metadata, updated_at as completed_at
                FROM datasets 
                WHERE name = ? OR id = ?
            """, (task_id, task_id, task_id))
            
            task_status = cursor.fetchone()
            cursor.close()
            conn.close()
            
        except Exception as db_error:
            logger.debug(f"Could not fetch initial task status: {db_error}")
            task_status = None
        
        # Send initial connection confirmation with current status
        connection_info = manager.get_connection_info(task_id)
        initial_message = {
            "type": "connection",
            "message": f"Connected to evaluation updates for task {task_id}",
            "timestamp": datetime.utcnow().isoformat(),
            "connection_info": {
                "reconnect_count": connection_info.get("reconnect_count", 0),
                "connected_at": connection_info.get("connected_at", datetime.utcnow()).isoformat()
            }
        }
        
        # Include current status if available
        if task_status:
            initial_message["current_status"] = {
                "status": task_status[0],
                "completed_at": task_status[2]
            }
        
        await manager.send_message(task_id, initial_message)
        
        # Track last ping time for connection health
        last_ping_time = datetime.utcnow()
        ping_interval = 30  # Send ping every 30 seconds
        message_count = 0
        
        # Listen for updates from Redis pub/sub
        while True:
            try:
                # Check for messages from Redis (non-blocking)
                message = pubsub.get_message(timeout=0.1)
                if message and message['type'] == 'message':
                    try:
                        update_data = json.loads(message['data'])
                        message_count += 1
                        
                        # Add message metadata
                        update_data["message_id"] = message_count
                        update_data["received_at"] = datetime.utcnow().isoformat()
                        
                        success = await manager.send_message(task_id, {
                            "type": "evaluation_update",
                            **update_data
                        })
                        
                        if not success:
                            logger.info(f"Failed to send update for task {task_id}, connection likely closed")
                            break
                        
                        # If task is completed or failed, send completion notice and close
                        if update_data.get("status") in [EvaluationTaskStatus.SUCCESS, EvaluationTaskStatus.FAILURE]:
                            await asyncio.sleep(0.5)  # Small delay to ensure message is received
                            await manager.send_message(task_id, {
                                "type": "completion",
                                "message": f"Task {update_data.get('status').lower()}, connection will close",
                                "final_status": update_data.get("status"),
                                "timestamp": datetime.utcnow().isoformat()
                            })
                            break
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode Redis message: {e}")
                
                # Send periodic ping to check connection health
                current_time = datetime.utcnow()
                if (current_time - last_ping_time).total_seconds() > ping_interval:
                    try:
                        success = await manager.send_message(task_id, {
                            "type": "ping",
                            "timestamp": current_time.isoformat(),
                            "connection_health": {
                                "uptime_seconds": (current_time - connection_info.get("connected_at", current_time)).total_seconds(),
                                "messages_sent": message_count
                            }
                        })
                        
                        if success:
                            last_ping_time = current_time
                        else:
                            logger.info(f"Ping failed for task {task_id}, connection closed")
                            break
                            
                    except Exception as ping_error:
                        logger.info(f"Ping failed for task {task_id}: {ping_error}")
                        break
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in WebSocket loop for task {task_id}: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for task {task_id}")
    except Exception as e:
        logger.error(f"WebSocket error for task {task_id}: {e}")
    finally:
        manager.disconnect(task_id)
        try:
            pubsub.close()
        except:
            pass

@router.get("/evaluate/queue/status")
async def get_queue_status():
    """
    Get current queue status including active tasks and concurrency limits
    """
    try:
        # Import concurrency manager
        from evaluation_tasks import concurrency_manager
        
        # Get current slot usage
        dataset_slots_used = redis_client.scard("dataset_creation_slots")
        evaluation_slots_used = redis_client.scard("evaluation_slots")
        
        # Get active task IDs
        dataset_tasks = list(redis_client.smembers("dataset_creation_slots"))
        evaluation_tasks = list(redis_client.smembers("evaluation_slots"))
        
        # Convert bytes to strings
        dataset_tasks = [task.decode('utf-8') if isinstance(task, bytes) else task for task in dataset_tasks]
        evaluation_tasks = [task.decode('utf-8') if isinstance(task, bytes) else task for task in evaluation_tasks]
        
        # Get Celery queue stats
        inspect = celery_app.control.inspect()
        active_tasks = inspect.active() or {}
        scheduled_tasks = inspect.scheduled() or {}
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "concurrency_limits": {
                "dataset_creation": {
                    "limit": concurrency_manager.dataset_creation_limit,
                    "used": dataset_slots_used,
                    "available": concurrency_manager.dataset_creation_limit - dataset_slots_used,
                    "active_tasks": dataset_tasks
                },
                "evaluation": {
                    "limit": concurrency_manager.evaluation_limit,
                    "used": evaluation_slots_used,
                    "available": concurrency_manager.evaluation_limit - evaluation_slots_used,
                    "active_tasks": evaluation_tasks
                }
            },
            "celery_queue": {
                "active": sum(len(tasks) for tasks in active_tasks.values()),
                "scheduled": sum(len(tasks) for tasks in scheduled_tasks.values()),
                "workers": list(active_tasks.keys())
            },
            "task_priorities": {
                "single_evaluation": 1,  # Highest
                "dataset_evaluation": 3,  # Higher
                "dataset_creation": 7    # Lower
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get queue status: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.get("/results")
async def get_evaluation_results(
    model_filter: Optional[str] = Query(None, description="Filter by model name"),
    dataset_filter: Optional[str] = Query(None, description="Filter by dataset name"),
    limit: int = Query(50, description="Number of results to return", ge=1, le=100)
):
    """
    Get evaluation results from database
    
    Returns both traditional evaluation results and background task results
    """
    try:
        from chat_db import get_chat_db
        chat_db = get_chat_db()
        
        if not chat_db:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        # Get traditional evaluation results
        results = chat_db.get_evaluation_results(
            model_filter=model_filter,
            dataset_filter=dataset_filter,
            limit=limit
        )
        
        # Also get recent evaluation tasks (background evaluations)
        tasks = chat_db.get_evaluation_tasks(status="SUCCESS", limit=limit)
        
        # Convert tasks to results format for consistency
        for task in tasks:
            if task.get('task_type') == 'dataset_evaluation' and task.get('status') == 'SUCCESS':
                task_result = {
                    'id': f"task_{task['id']}",
                    'test_case_id': task['id'],
                    'dataset_id': task.get('metadata', {}).get('dataset_id', 'unknown'),
                    'dataset': task.get('metadata', {}).get('dataset_name', 'Unknown Dataset'),
                    'model': task.get('metadata', {}).get('model_name', 'Unknown Model'),
                    'groundedness': task.get('groundedness_score', 0.0),
                    'context_relevance': task.get('context_relevance_score', 0.0),
                    'answer_quality': task.get('answer_relevance_score', 0.0),
                    'avg_latency': task.get('evaluation_time', 0.0),
                    'total_questions': task.get('metadata', {}).get('total_questions', 0),
                    'status': 'Completed',
                    'run_by': task.get('user_id', 'system'),
                    'run_date': task.get('completed_at', '').split('T')[0] if task.get('completed_at') else '',
                    'started_at': task.get('created_at'),
                    'completed_at': task.get('completed_at')
                }
                results.append(task_result)
        
        # Sort by completion date, most recent first
        results.sort(key=lambda x: x.get('completed_at', x.get('run_date', '')), reverse=True)
        
        return {
            "success": True,
            "results": results[:limit],
            "total": len(results)
        }
        
    except Exception as e:
        logger.error(f"Failed to get evaluation results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get evaluation results: {str(e)}")

@router.get("/evaluate/results/recent")
async def get_recent_evaluation_results(
    limit: int = Query(10, description="Number of recent results to return", ge=1, le=100),
    task_type: Optional[str] = Query(None, description="Filter by task type: single or batch")
):
    """
    Get recent evaluation results from database and Redis cache
    
    Returns the most recent evaluation results including persistent database records.
    """
    try:
        results = []
        
        # Get results from database first (persistent)
        chat_db = get_chat_db()
        if chat_db:
            db_tasks = chat_db.get_evaluation_tasks(limit=limit)
            for task in db_tasks:
                if task_type and task['task_type'] != task_type:
                    continue
                    
                result = {
                    "task_id": task['id'],
                    "type": task['task_type'],
                    "status": task['status'],
                    "query": task['query'],
                    "response": task['response'],
                    "context_chunks": task['context_chunks'],
                    "conversation_id": task['conversation_id'],
                    "user_id": task['user_id'],
                    "metadata": task['metadata'],
                    "timestamp": task['created_at'],
                    "updated_at": task['updated_at'],
                    "completed_at": task['completed_at']
                }
                
                # Add scores if available
                if task['status'] == 'SUCCESS':
                    result["results"] = {
                        "groundedness": {"score": task['groundedness_score']},
                        "answer_relevance": {"score": task['answer_relevance_score']},
                        "context_relevance": {"score": task['context_relevance_score']},
                        "overall_score": task['overall_score'],
                        "evaluation_time_seconds": task['evaluation_time']
                    }
                elif task['status'] == 'FAILURE':
                    result["error"] = task['error_message']
                
                results.append(result)
        
        # Also get any recent results from Redis cache that might not be in DB yet
        if task_type != "batch":
            eval_keys = redis_client.keys("evaluation_result:*")
            for key in eval_keys[:limit]:
                try:
                    result_data = redis_client.get(key)
                    if result_data:
                        result = json.loads(result_data)
                        result["type"] = "single"
                        # Only add if not already in database results
                        if not any(r['task_id'] == result.get('task_id') for r in results):
                            results.append(result)
                except json.JSONDecodeError:
                    continue
        
        # Get batch evaluation result keys
        if task_type != "single":
            batch_keys = redis_client.keys("batch_evaluation_result:*")
            for key in batch_keys[:limit]:
                try:
                    result_data = redis_client.get(key)
                    if result_data:
                        result = json.loads(result_data)
                        result["type"] = "batch"
                        # Only add if not already in database results
                        if not any(r['task_id'] == result.get('task_id') for r in results):
                            results.append(result)
                except json.JSONDecodeError:
                    continue
        
        # Sort by timestamp (most recent first)
        results.sort(key=lambda x: x.get("timestamp", x.get("created_at", "")), reverse=True)
        
        return {
            "results": results[:limit],
            "total_found": len(results),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get recent evaluation results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recent results: {str(e)}")

@router.get("/metrics")
async def get_evaluation_metrics(
    timeframe: str = Query("7d", description="Time frame: 1d, 7d, 30d")
):
    """Get current evaluation metrics"""
    try:
        # TODO: Add authentication back when circular import is resolved
        # if not current_user.is_admin:
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        # Calculate date range
        end_date = datetime.now()
        if timeframe == "1d":
            start_date = end_date - timedelta(days=1)
        elif timeframe == "7d":
            start_date = end_date - timedelta(days=7)
        elif timeframe == "30d":
            start_date = end_date - timedelta(days=30)
        else:
            start_date = end_date - timedelta(days=7)
        
        db = get_chat_db()
        
        # Get recent chat sessions and messages for evaluation
        if db:
            with sqlite3.connect(db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get recent chat interactions
                cursor.execute("""
                    SELECT cs.id as session_id, cs.username, cs.topic, cs.created_at,
                           m.content, m.is_user, m.timestamp
                    FROM chat_sessions cs
                    JOIN messages m ON cs.id = m.session_id
                    WHERE cs.created_at >= ? AND cs.created_at <= ?
                    ORDER BY cs.created_at DESC, m.timestamp ASC
                """, (start_date.isoformat(), end_date.isoformat()))
                
                messages = cursor.fetchall()
        else:
            messages = []
        
        # Process messages and calculate metrics
        total_interactions = 0
        groundedness_scores = []
        context_relevance_scores = []
        answer_quality_scores = []
        latency_scores = []
        
        current_session = None
        user_query = None
        
        for message in messages:
            if message['is_user']:
                user_query = message['content']
                current_session = message['session_id']
            else:
                if user_query and current_session == message['session_id']:
                    # Calculate metrics for this Q&A pair
                    total_interactions += 1
                    
                    # Mock context for demonstration (in real implementation, retrieve from RAG)
                    mock_context = f"Retrieved context for query: {user_query[:100]}..."
                    
                    # Calculate evaluation metrics
                    groundedness = calculate_groundedness(message['content'], mock_context)
                    context_relevance = calculate_context_relevance(user_query, mock_context)
                    answer_quality = calculate_answer_quality(user_query, message['content'])
                    
                    # Mock latency (in real implementation, track actual response times)
                    latency = random.uniform(0.5, 3.0)  # seconds
                    
                    groundedness_scores.append(groundedness)
                    context_relevance_scores.append(context_relevance)
                    answer_quality_scores.append(answer_quality)
                    latency_scores.append(latency)
                    
                    user_query = None
        
        # Calculate averages
        if total_interactions > 0:
            avg_groundedness = sum(groundedness_scores) / len(groundedness_scores)
            avg_context_relevance = sum(context_relevance_scores) / len(context_relevance_scores)
            avg_answer_quality = sum(answer_quality_scores) / len(answer_quality_scores)
            avg_latency = sum(latency_scores) / len(latency_scores)
        else:
            # Default values for demo
            avg_groundedness = 0.85
            avg_context_relevance = 0.78
            avg_answer_quality = 0.82
            avg_latency = 1.2
        
        return {
            "timeframe": timeframe,
            "total_interactions": total_interactions if total_interactions > 0 else 150,  # Mock for demo
            "metrics": {
                "groundedness": {
                    "score": round(avg_groundedness, 3),
                    "description": "How well responses are grounded in retrieved context",
                    "threshold": 0.7
                },
                "context_relevance": {
                    "score": round(avg_context_relevance, 3),
                    "description": "Relevance of retrieved context to user queries",
                    "threshold": 0.7
                },
                "answer_quality": {
                    "score": round(avg_answer_quality, 3),
                    "description": "Overall quality and completeness of answers",
                    "threshold": 0.75
                },
                "latency": {
                    "score": round(avg_latency, 2),
                    "description": "Average response time in seconds",
                    "threshold": 2.0,
                    "unit": "seconds"
                }
            },
            "calculated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting evaluation metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/historical")
async def get_historical_metrics(
    days: int = Query(30, description="Number of days to look back")
):
    """Get historical evaluation metrics"""
    try:
        # TODO: Add authentication back when circular import is resolved
        # if not current_user.is_admin:
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        # Generate mock historical data (in real implementation, store these metrics in DB)
        historical_data = []
        
        for i in range(days):
            date = datetime.now() - timedelta(days=days - i - 1)
            
            # Generate realistic trending data
            base_groundedness = 0.85
            base_context_relevance = 0.78
            base_answer_quality = 0.82
            base_latency = 1.2
            
            # Add some trend and noise
            trend_factor = (i / days) * 0.1  # Slight improvement over time
            noise_factor = random.uniform(-0.05, 0.05)
            
            historical_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "groundedness": round(max(min(base_groundedness + trend_factor + noise_factor, 1.0), 0.0), 3),
                "context_relevance": round(max(min(base_context_relevance + trend_factor + noise_factor, 1.0), 0.0), 3),
                "answer_quality": round(max(min(base_answer_quality + trend_factor + noise_factor, 1.0), 0.0), 3),
                "latency": round(max(base_latency - (trend_factor * 0.5) + (noise_factor * 0.3), 0.1), 2),
                "total_queries": random.randint(10, 50)
            })
        
        return {
            "period": f"{days} days",
            "data": historical_data
        }
        
    except Exception as e:
        logger.error(f"Error getting historical metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/latency-distribution")
async def get_latency_distribution(
    timeframe: str = Query("7d", description="Time frame: 1d, 7d, 30d")
):
    """Get latency distribution data"""
    try:
        # TODO: Add authentication back when circular import is resolved
        # if not current_user.is_admin:
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        # Generate mock latency distribution (in real implementation, get from actual data)
        latency_ranges = [
            {"range": "0-0.5s", "count": random.randint(20, 40)},
            {"range": "0.5-1s", "count": random.randint(30, 60)},
            {"range": "1-2s", "count": random.randint(40, 80)},
            {"range": "2-3s", "count": random.randint(20, 50)},
            {"range": "3-5s", "count": random.randint(10, 30)},
            {"range": "5s+", "count": random.randint(5, 15)}
        ]
        
        return {
            "timeframe": timeframe,
            "distribution": latency_ranges
        }
        
    except Exception as e:
        logger.error(f"Error getting latency distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quality-breakdown")
async def get_quality_breakdown(
    metric: str = Query("answer_quality", description="Metric to analyze: groundedness, context_relevance, answer_quality")
):
    """Get detailed quality breakdown by score ranges"""
    try:
        # TODO: Add authentication back when circular import is resolved
        # if not current_user.is_admin:
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        # Generate mock quality breakdown (in real implementation, analyze actual scores)
        breakdown = [
            {"range": "0.9-1.0", "count": random.randint(40, 70), "percentage": 0},
            {"range": "0.8-0.9", "count": random.randint(30, 50), "percentage": 0},
            {"range": "0.7-0.8", "count": random.randint(20, 40), "percentage": 0},
            {"range": "0.6-0.7", "count": random.randint(10, 25), "percentage": 0},
            {"range": "0.5-0.6", "count": random.randint(5, 15), "percentage": 0},
            {"range": "0.0-0.5", "count": random.randint(2, 10), "percentage": 0}
        ]
        
        # Calculate percentages
        total_count = sum(item["count"] for item in breakdown)
        for item in breakdown:
            item["percentage"] = round((item["count"] / total_count) * 100, 1) if total_count > 0 else 0
        
        return {
            "metric": metric,
            "breakdown": breakdown,
            "total_evaluations": total_count
        }
        
    except Exception as e:
        logger.error(f"Error getting quality breakdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate-response")
async def evaluate_single_response(
    background_tasks: BackgroundTasks,
    question: str,
    answer: str,
    context: str,
    session_id: Optional[str] = None
):
    """Evaluate a single chat response using RAG evaluation metrics"""
    try:
        # TODO: Add authentication back when circular import is resolved
        # if not current_user.is_admin:
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        # Perform evaluation asynchronously
        result = await evaluate_chat_response(
            question=question,
            answer=answer,
            context=context,
            session_id=session_id,
            user_id="anonymous"  # TODO: Add proper user tracking when auth is restored
        )
        
        return {
            "evaluation_id": f"eval_{int(time.time())}",
            "overall_score": result.overall_score,
            "metrics": {
                "groundedness": {
                    "score": result.groundedness.score,
                    "raw_score": result.groundedness.raw_score,
                    "reasoning": result.groundedness.reasoning[:500] + "..." if len(result.groundedness.reasoning) > 500 else result.groundedness.reasoning
                },
                "answer_relevance": {
                    "score": result.answer_relevance.score,
                    "raw_score": result.answer_relevance.raw_score,
                    "reasoning": result.answer_relevance.reasoning[:500] + "..." if len(result.answer_relevance.reasoning) > 500 else result.answer_relevance.reasoning
                },
                "context_relevance": {
                    "score": result.context_relevance.score,
                    "raw_score": result.context_relevance.raw_score,
                    "reasoning": result.context_relevance.reasoning[:500] + "..." if len(result.context_relevance.reasoning) > 500 else result.context_relevance.reasoning
                }
            },
            "evaluation_time_seconds": result.evaluation_time_seconds,
            "timestamp": result.groundedness.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error evaluating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/evaluation-trends")
async def get_evaluation_trends(
    days: int = Query(7, description="Number of days for trend analysis")
):
    """Get evaluation trends over time"""
    try:
        # TODO: Add authentication back when circular import is resolved
        # if not current_user.is_admin:
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        # Get evaluation manager to access history
        manager = get_evaluation_manager()
        evaluations = manager.evaluation_history
        
        if not evaluations:
            # Return mock trend data for demo
            trend_data = []
            for i in range(days):
                date = datetime.now() - timedelta(days=days - i - 1)
                trend_data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "groundedness": round(0.85 + random.uniform(-0.1, 0.1), 3),
                    "answer_relevance": round(0.82 + random.uniform(-0.1, 0.1), 3),
                    "context_relevance": round(0.78 + random.uniform(-0.1, 0.1), 3),
                    "overall_score": round(0.82 + random.uniform(-0.1, 0.1), 3),
                    "evaluation_count": random.randint(5, 25)
                })
            
            return {
                "period": f"{days} days",
                "data": trend_data,
                "note": "Mock data - start using the system to see real trends"
            }
        
        # Group evaluations by date
        from collections import defaultdict
        daily_data = defaultdict(list)
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_evaluations = [
            e for e in evaluations 
            if e.groundedness.timestamp >= cutoff_date
        ]
        
        for evaluation in recent_evaluations:
            date_key = evaluation.groundedness.timestamp.strftime("%Y-%m-%d")
            daily_data[date_key].append(evaluation)
        
        # Calculate daily averages
        trend_data = []
        for i in range(days):
            date = datetime.now() - timedelta(days=days - i - 1)
            date_key = date.strftime("%Y-%m-%d")
            
            if date_key in daily_data:
                day_evaluations = daily_data[date_key]
                trend_data.append({
                    "date": date_key,
                    "groundedness": round(sum(e.groundedness.score for e in day_evaluations) / len(day_evaluations), 3),
                    "answer_relevance": round(sum(e.answer_relevance.score for e in day_evaluations) / len(day_evaluations), 3),
                    "context_relevance": round(sum(e.context_relevance.score for e in day_evaluations) / len(day_evaluations), 3),
                    "overall_score": round(sum(e.overall_score for e in day_evaluations) / len(day_evaluations), 3),
                    "evaluation_count": len(day_evaluations)
                })
            else:
                # No data for this day
                trend_data.append({
                    "date": date_key,
                    "groundedness": None,
                    "answer_relevance": None,
                    "context_relevance": None,
                    "overall_score": None,
                    "evaluation_count": 0
                })
        
        return {
            "period": f"{days} days",
            "data": trend_data,
            "total_evaluations": len(recent_evaluations)
        }
        
    except Exception as e:
        logger.error(f"Error getting evaluation trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/detailed-evaluation/{evaluation_id}")
async def get_detailed_evaluation(
    evaluation_id: str
):
    """Get detailed information about a specific evaluation"""
    try:
        # TODO: Add authentication back when circular import is resolved
        # if not current_user.is_admin:
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        # In a real implementation, you would store evaluations in a database
        # For now, return a mock detailed evaluation
        return {
            "evaluation_id": evaluation_id,
            "timestamp": datetime.now().isoformat(),
            "input": {
                "question": "What are the main benefits of renewable energy?",
                "context": "Renewable energy sources include solar, wind, hydroelectric, and geothermal power. These sources are sustainable because they are naturally replenished and do not deplete finite resources like fossil fuels.",
                "answer": "The main benefits of renewable energy include environmental sustainability, reduced greenhouse gas emissions, energy independence, and long-term cost savings."
            },
            "evaluation_results": {
                "groundedness": {
                    "score": 0.85,
                    "raw_score": 8.5,
                    "reasoning": "The answer is well-grounded in the provided context. All mentioned benefits (sustainability, reduced emissions, independence, cost savings) are either directly stated or logically derivable from the context about renewable energy sources being naturally replenished and not depleting finite resources.",
                    "criteria": "Evaluating how well the answer is supported by the provided context",
                    "supporting_evidence": "Context mentions renewable sources are 'sustainable' and 'naturally replenished', supporting the answer's claims about sustainability and environmental benefits."
                },
                "answer_relevance": {
                    "score": 0.92,
                    "raw_score": 9.2,
                    "reasoning": "The answer directly addresses the question about benefits of renewable energy. Each point mentioned (environmental sustainability, reduced emissions, energy independence, cost savings) is a relevant benefit that answers the question comprehensively.",
                    "criteria": "Evaluating how well the answer addresses the specific question asked",
                    "supporting_evidence": "Question asks for 'main benefits' and answer provides specific, relevant benefits without going off-topic."
                },
                "context_relevance": {
                    "score": 0.78,
                    "raw_score": 7.8,
                    "reasoning": "The context provides good information about what renewable energy sources are and their sustainable nature, which is relevant to understanding their benefits. However, it could be more specific about the actual benefits mentioned in the answer.",
                    "criteria": "Evaluating how relevant the provided context is to answering the question",
                    "supporting_evidence": "Context explains renewable energy characteristics that support benefit claims, though it doesn't explicitly list all benefits mentioned in the answer."
                }
            },
            "overall_score": 0.85,
            "evaluation_time_seconds": 2.3,
            "metadata": {
                "llm_model": "llama3",
                "evaluation_version": "1.0",
                "session_id": "demo_session"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting detailed evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
