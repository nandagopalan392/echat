"""
WebSocket endpoints for fine-tuning real-time updates
"""
import asyncio
import json
import logging
import os
import redis
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Optional

from app.core.websocket import get_finetuning_manager
from app.db.repositories.experiment_repository import get_experiment_repository, ExperimentStatus
from app.dependencies import authenticate_websocket_token, authenticate_websocket_cookie, authenticate_websocket_admin_token, authenticate_websocket_admin_cookie
from app.db.base import DatabaseConnection

logger = logging.getLogger(__name__)

router = APIRouter()

# Redis client for Q-C-A progress tracking
redis_client = redis.Redis(host=os.getenv('REDIS_HOST', 'redis'), port=6379, db=0)


@router.websocket("/ws/finetuning/{experiment_id}")
async def websocket_experiment_progress(
    websocket: WebSocket,
    experiment_id: str,
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for real-time fine-tuning experiment progress updates
    
    **Authentication:** Admin only - Supports both cookie-based (preferred) and query parameter authentication
    - Cookie: access_token cookie (automatic, secure)
    - Query param: ?token=your_jwt_token (backward compatibility)
    
    Provides real-time updates including:
    - Experiment status changes
    - Training metrics (loss, accuracy, etc.)
    - Training logs
    - Error messages
    
    Connection automatically closes when experiment reaches terminal state (completed/failed/cancelled)
    """
    db = DatabaseConnection()
    
    # Try cookie authentication first (preferred) - ADMIN ONLY
    admin_user = await authenticate_websocket_admin_cookie(websocket, db)
    
    # Fallback to query parameter authentication - ADMIN ONLY
    if not admin_user and token:
        admin_user = await authenticate_websocket_admin_token(token, db)
    
    # Admin authentication required
    if not admin_user:
        await websocket.close(code=1008, reason="Admin authentication required: provide access_token cookie or token parameter")
        return
    
    logger.info(f"WebSocket authenticated for admin user: {admin_user.username}, experiment: {experiment_id}")
    
    manager = get_finetuning_manager()
    await manager.connect(websocket, experiment_id)
    
    try:
        experiment_repo = get_experiment_repository()
        
        while True:
            # Check experiment status - experiment is a dataclass
            experiment = experiment_repo.get_experiment(experiment_id)
            
            if experiment:
                # Get latest logs - logs are TrainingLog dataclass objects
                logs = experiment_repo.get_training_logs(experiment_id)
                latest_logs = [log.to_dict() if hasattr(log, 'to_dict') else log for log in logs[-10:]] if logs else []
                
                # Check Redis for Celery task progress updates
                redis_progress = None
                try:
                    progress_data = redis_client.get(f"training_progress_{experiment_id}")
                    if progress_data:
                        redis_progress = json.loads(progress_data)
                except Exception as e:
                    logger.debug(f"No Redis progress data for {experiment_id}: {e}")
                
                # Get live training metrics if available
                try:
                    from app.core.training.metrics import get_metrics_collector
                    metrics_collector = get_metrics_collector(experiment_id)
                    training_metrics = None
                    if metrics_collector:
                        training_metrics = metrics_collector.get_metrics_summary()
                except ImportError:
                    logger.warning("training_metrics module not available")
                    training_metrics = None
                
                # Use attribute access for Experiment dataclass
                status = experiment.status
                status_lower = (status or "").lower()
                
                # If Redis has progress, include it with loss data for live charts
                progress_info = {}
                if redis_progress:
                    redis_data = redis_progress.get("data", {})
                    progress_info = {
                        "celery_status": redis_progress.get("status"),
                        "progress": redis_data.get("progress", 0),
                        "message": redis_data.get("message", ""),
                        "current_epoch": redis_data.get("current_epoch", 0),
                        "total_epochs": redis_data.get("total_epochs", 0),
                        "loss": redis_data.get("loss"),
                        "step": redis_data.get("step"),
                    }
                
                message_data = {
                    "type": "experiment_update",
                    "experiment_id": experiment_id,
                    "status": status_lower,
                    "metrics": experiment.metrics or {},
                    "latest_logs": latest_logs,
                    "error_message": experiment.error_message,
                    "training_metrics": training_metrics,
                    "progress_info": progress_info
                }
                
                success = await manager.send_message(experiment_id, message_data)
                
                if not success:
                    logger.info(f"Failed to send update for experiment {experiment_id}, connection closed")
                    break
                
                logger.debug(f"WebSocket sent update for experiment {experiment_id}: status={status_lower}")
                
                # Close connection if experiment is completed or failed
                terminal_statuses = ["completed", "failed", "cancelled"]
                terminal_status_values = [
                    ExperimentStatus.COMPLETED.value,
                    ExperimentStatus.FAILED.value,
                    ExperimentStatus.CANCELLED.value
                ]
                
                if status_lower in terminal_statuses or status in terminal_status_values:
                    logger.info(f"Experiment {experiment_id} finished with status {status_lower}, closing WebSocket")
                    
                    # Send final message
                    await manager.send_message(experiment_id, {
                        "type": "completion",
                        "message": f"Experiment {status_lower}",
                        "final_status": status_lower
                    })
                    
                    await asyncio.sleep(0.5)  # Ensure message is received
                    await websocket.close()
                    break
            else:
                # Experiment not found
                await manager.send_message(experiment_id, {
                    "type": "error",
                    "message": f"Experiment {experiment_id} not found"
                })
                break
            
            # Update every 2 seconds for responsive UI
            await asyncio.sleep(2)
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for experiment {experiment_id}")
    
    except Exception as e:
        logger.error(f"WebSocket error for experiment {experiment_id}: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
    
    finally:
        manager.disconnect(experiment_id)


@router.websocket("/ws/qca-dataset/{task_id}")
async def websocket_qca_dataset_progress(
    websocket: WebSocket,
    task_id: str,
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for real-time Q-C-A dataset creation progress updates
    
    **Authentication:** Admin only - Supports both cookie-based (preferred) and query parameter authentication
    - Cookie: access_token cookie (automatic, secure)
    - Query param: ?token=your_jwt_token (backward compatibility)
    
    Provides real-time updates including:
    - Dataset creation progress (percentage complete)
    - Status changes (started, processing, completed, failed)
    - Generated Q-C-A items count
    - Error messages
    
    Connection automatically closes when dataset creation reaches terminal state (success/failure)
    """
    db = DatabaseConnection()
    
    # Try cookie authentication first (preferred) - ADMIN ONLY
    admin_user = await authenticate_websocket_admin_cookie(websocket, db)
    
    # Fallback to query parameter authentication - ADMIN ONLY
    if not admin_user and token:
        admin_user = await authenticate_websocket_admin_token(token, db)
    
    # Admin authentication required
    if not admin_user:
        await websocket.close(code=1008, reason="Admin authentication required: provide access_token cookie or token parameter")
        return
    
    logger.info(f"WebSocket authenticated for admin user: {admin_user.username}, Q-C-A task: {task_id}")
    
    await websocket.accept()
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "message": f"Connected to Q-C-A dataset updates for task {task_id}",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Poll Redis for progress updates
        last_status = None
        while True:
            try:
                # Get progress from Redis
                progress_data = redis_client.get(f"qca_progress_{task_id}")
                
                if progress_data:
                    progress = json.loads(progress_data)
                    status = progress.get("status", "UNKNOWN")
                    data = progress.get("data", {})
                    
                    # Only send if status changed
                    if progress != last_status:
                        await websocket.send_json({
                            "type": "progress",
                            "task_id": task_id,
                            "status": status,
                            "data": data,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        last_status = progress
                        
                        logger.debug(f"Q-C-A WebSocket sent update for task {task_id}: status={status}")
                    
                    # Check for terminal states
                    if status in ["SUCCESS", "FAILURE", "ERROR"]:
                        logger.info(f"Q-C-A task {task_id} finished with status {status}, closing WebSocket")
                        
                        # Send final message
                        await websocket.send_json({
                            "type": "completion",
                            "message": f"Q-C-A dataset creation {status.lower()}",
                            "final_status": status,
                            "data": data,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        
                        await asyncio.sleep(0.5)  # Ensure message is received
                        await websocket.close()
                        break
                else:
                    # No progress data yet, check Celery task state
                    try:
                        from app.workers.celery_app import celery_app
                        result = celery_app.AsyncResult(task_id)
                        
                        if result.state == 'PENDING':
                            await websocket.send_json({
                                "type": "progress",
                                "task_id": task_id,
                                "status": "PENDING",
                                "data": {"message": "Task is queued", "progress": 0.0},
                                "timestamp": datetime.utcnow().isoformat()
                            })
                        elif result.state == 'FAILURE':
                            await websocket.send_json({
                                "type": "completion",
                                "task_id": task_id,
                                "status": "FAILURE",
                                "message": str(result.result),
                                "timestamp": datetime.utcnow().isoformat()
                            })
                            await asyncio.sleep(0.5)
                            await websocket.close()
                            break
                    except Exception as e:
                        logger.debug(f"Could not check Celery task state: {e}")
                
                # Poll every 1 second for responsive UI
                await asyncio.sleep(1)
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in Redis for task {task_id}: {e}")
                await asyncio.sleep(1)
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for Q-C-A task {task_id}")
    
    except Exception as e:
        logger.error(f"WebSocket error for Q-C-A task {task_id}: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
