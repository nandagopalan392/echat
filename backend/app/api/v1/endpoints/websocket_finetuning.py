"""
WebSocket endpoints for fine-tuning real-time updates
"""
import asyncio
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.websocket import get_finetuning_manager
from experiment_db import get_experiment_db, ExperimentStatus

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/finetuning/{experiment_id}")
async def websocket_experiment_progress(websocket: WebSocket, experiment_id: str):
    """
    WebSocket endpoint for real-time fine-tuning experiment progress updates
    
    Provides real-time updates including:
    - Experiment status changes
    - Training metrics (loss, accuracy, etc.)
    - Training logs
    - Error messages
    
    Connection automatically closes when experiment reaches terminal state (completed/failed/cancelled)
    """
    manager = get_finetuning_manager()
    await manager.connect(websocket, experiment_id)
    
    try:
        experiment_db = get_experiment_db()
        
        while True:
            # Check experiment status
            experiment = experiment_db.get_experiment(experiment_id)
            
            if experiment:
                # Get latest logs
                logs = experiment_db.get_training_logs(experiment_id)
                latest_logs = logs[-10:] if logs else []  # Last 10 entries
                
                # Get live training metrics if available
                try:
                    from training_metrics import get_metrics_collector
                    metrics_collector = get_metrics_collector(experiment_id)
                    training_metrics = None
                    if metrics_collector:
                        training_metrics = metrics_collector.get_metrics_summary()
                except ImportError:
                    logger.warning("training_metrics module not available")
                    training_metrics = None
                
                status = experiment.get('status')
                status_lower = (status or "").lower()
                
                message_data = {
                    "type": "experiment_update",
                    "experiment_id": experiment_id,
                    "status": status_lower,
                    "metrics": experiment.get('metrics', {}),
                    "latest_logs": latest_logs,
                    "error_message": experiment.get('error_message'),
                    "training_metrics": training_metrics
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
