"""
WebSocket endpoints for evaluation real-time updates
"""
import asyncio
import json
import logging
import sqlite3
import os
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Optional
import redis

from app.core.websocket import get_evaluation_manager
from app.workers.tasks.evaluation_tasks import EvaluationTaskStatus
from app.dependencies import authenticate_websocket_token, authenticate_websocket_cookie, authenticate_websocket_admin_token, authenticate_websocket_admin_cookie
from app.db.base import DatabaseConnection

logger = logging.getLogger(__name__)

router = APIRouter()

# Redis client for pub/sub notifications
redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))


@router.websocket("/ws/evaluation/{task_id}")
async def websocket_evaluation_updates(
    websocket: WebSocket,
    task_id: str,
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for real-time evaluation progress updates with auto-reconnection support
    
    **Authentication:** Admin only - Supports both cookie-based (preferred) and query parameter authentication
    - Cookie: access_token cookie (automatic, secure)
    - Query param: ?token=your_jwt_token (backward compatibility)
    
    Provides real-time updates including:
    - Evaluation progress (percentage complete)
    - Status changes (pending, running, completed, failed)
    - Evaluation results and metrics
    - Error messages
    
    Features:
    - Auto-reconnection support with connection health tracking
    - Redis pub/sub integration for distributed updates
    - Periodic ping messages for connection health monitoring
    - Graceful completion handling
    
    Connection automatically closes when evaluation reaches terminal state (success/failure)
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
    
    logger.info(f"WebSocket authenticated for admin user: {admin_user.username}, task: {task_id}")
    
    manager = get_evaluation_manager()
    await manager.connect(websocket, task_id)
    
    try:
        # Subscribe to Redis pub/sub for this task
        pubsub = redis_client.pubsub()
        pubsub.subscribe(f"evaluation_updates:{task_id}")
        
        # Get current task status for initial sync
        try:
            from app.db import DatabaseConnection
            from app.db.repositories import EvaluationRepository
            
            db = DatabaseConnection()
            eval_repo = EvaluationRepository(db)
            task = eval_repo.get_task(task_id)
            
            if task:
                task_status = (task.status, task.metadata or '{}', task.completed_at)
            else:
                # Check if it's a dataset ID
                try:
                    dataset_id = int(task_id)
                    dataset = eval_repo.get_dataset(dataset_id)
                    if dataset:
                        task_status = (dataset.status, '{}', dataset.updated_at)
                    else:
                        task_status = None
                except ValueError:
                    task_status = None
        
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
                            logger.info(f"Failed to send update for task {task_id}, connection closed")
                            break
                        
                        # If task is completed or failed, send completion notice and close
                        if update_data.get("status") in [EvaluationTaskStatus.SUCCESS, EvaluationTaskStatus.FAILURE]:
                            await asyncio.sleep(0.5)  # Ensure message is received
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
            pubsub.unsubscribe(f"evaluation_updates:{task_id}")
            pubsub.close()
        except Exception as cleanup_error:
            logger.debug(f"Error during WebSocket cleanup for task {task_id}: {cleanup_error}")
