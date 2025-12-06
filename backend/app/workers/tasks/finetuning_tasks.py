"""
Finetuning Celery Tasks

This module contains Celery tasks for model fine-tuning with priority support.
Following the same pattern as evaluation_tasks.py and qca_tasks.py.
"""

import os
import json
import time
import logging
import redis
from typing import Dict, Any, Optional
from datetime import datetime

from celery import current_task
from app.workers.celery_app import celery_app
from app.db.repositories.experiment_repository import get_experiment_repository, ExperimentStatus

logger = logging.getLogger(__name__)

# Redis client for progress updates
redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"))


def publish_training_update(experiment_id: str, status: str, data: Dict[str, Any]):
    """Publish training progress update via Redis for WebSocket consumption"""
    try:
        progress_data = {
            "status": status,
            "data": data,
            "timestamp": time.time()
        }
        
        # Store in Redis for WebSocket endpoint to poll
        redis_client.set(
            f"training_progress_{experiment_id}", 
            json.dumps(progress_data),
            ex=3600  # Expire after 1 hour
        )
        
        # Also publish to pub/sub channel for real-time updates
        redis_client.publish(
            f"training:{experiment_id}",
            json.dumps(progress_data)
        )
        
        logger.info(f"Training Progress Update: {experiment_id} - {status} - {data.get('message', '')}")
    except Exception as e:
        logger.error(f"Failed to publish training progress update: {e}")


@celery_app.task(bind=True, name="finetuning_tasks.start_training_background")
def start_training_background(
    self,
    experiment_id: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Start model fine-tuning in background with progress tracking
    
    This task runs the actual training process in a Celery worker,
    publishing progress updates via Redis for WebSocket consumption.
    
    Args:
        experiment_id: The experiment ID to train
        config: Training configuration including:
            - base_model: Model to fine-tune
            - dataset_path: Path to training data
            - dataset_id: Dataset ID
            - epochs: Number of training epochs
            - learning_rate: Learning rate
            - batch_size: Batch size
            - use_lora: Whether to use LoRA
            - lora_r, lora_alpha, lora_dropout: LoRA config
            
    Returns:
        Dict containing training results
    """
    task_id = self.request.id
    start_time = time.time()
    
    logger.info(f"🚀 TRAINING START: Starting training for experiment {experiment_id}, task {task_id}")
    
    try:
        # Get experiment repository
        experiment_repo = get_experiment_repository()
        
        # Update experiment status to RUNNING
        experiment_repo.update_experiment_status(experiment_id, ExperimentStatus.RUNNING)
        
        # Publish initial status
        publish_training_update(experiment_id, "STARTED", {
            "message": "Training started",
            "experiment_id": experiment_id,
            "progress": 0.0,
            "current_epoch": 0,
            "total_epochs": config.get('epochs', 3)
        })
        
        # Import the finetuner (heavy imports done here to avoid startup delay)
        from app.core.training.hf_finetuner import HuggingFaceFineTuner
        from app.core.training.metrics import create_metrics_collector, cleanup_metrics_collector
        
        # Create finetuner instance
        finetuner = HuggingFaceFineTuner()
        
        # Create metrics collector for this training run
        metrics_collector = create_metrics_collector(experiment_id)
        
        publish_training_update(experiment_id, "PROGRESS", {
            "message": "Loading model and dataset",
            "progress": 0.05,
            "current_epoch": 0,
            "total_epochs": config.get('epochs', 3)
        })
        
        # Run the training (this is the heavy lifting)
        success = finetuner._train_model_with_progress(
            experiment_id=experiment_id,
            config=config,
            progress_callback=lambda progress, message, epoch, total_epochs, metrics=None: publish_training_update(
                experiment_id, 
                "PROGRESS", 
                {
                    "message": message,
                    "progress": progress,
                    "current_epoch": epoch,
                    "total_epochs": total_epochs,
                    "loss": metrics.get("loss") if metrics else None,
                    "step": metrics.get("step") if metrics else None
                }
            )
        )
        
        execution_time = time.time() - start_time
        
        if success:
            # Update experiment status to COMPLETED
            experiment_repo.update_experiment_status(experiment_id, ExperimentStatus.COMPLETED)
            
            publish_training_update(experiment_id, "SUCCESS", {
                "message": "Training completed successfully",
                "experiment_id": experiment_id,
                "progress": 1.0,
                "execution_time": execution_time
            })
            
            logger.info(f"✅ TRAINING COMPLETE: Experiment {experiment_id} completed in {execution_time:.2f}s")
            
            return {
                "status": "success",
                "experiment_id": experiment_id,
                "execution_time": execution_time
            }
        else:
            # Update experiment status to FAILED
            experiment_repo.update_experiment_status(
                experiment_id, 
                ExperimentStatus.FAILED, 
                "Training failed"
            )
            
            publish_training_update(experiment_id, "FAILURE", {
                "message": "Training failed",
                "experiment_id": experiment_id,
                "progress": 0.0
            })
            
            return {
                "status": "failed",
                "experiment_id": experiment_id,
                "error": "Training failed"
            }
            
    except Exception as e:
        execution_time = time.time() - start_time
        error_message = str(e)
        
        logger.error(f"❌ TRAINING ERROR: Experiment {experiment_id} failed: {error_message}")
        
        try:
            experiment_repo = get_experiment_repository()
            experiment_repo.update_experiment_status(
                experiment_id, 
                ExperimentStatus.FAILED, 
                error_message
            )
        except Exception as db_error:
            logger.error(f"Failed to update experiment status: {db_error}")
        
        publish_training_update(experiment_id, "FAILURE", {
            "message": f"Training failed: {error_message}",
            "experiment_id": experiment_id,
            "error": error_message,
            "progress": 0.0
        })
        
        return {
            "status": "failed",
            "experiment_id": experiment_id,
            "error": error_message,
            "execution_time": execution_time
        }


@celery_app.task(bind=True, name="finetuning_tasks.stop_training_background")
def stop_training_background(
    self,
    experiment_id: str
) -> Dict[str, Any]:
    """
    Stop a running training task
    
    Args:
        experiment_id: The experiment ID to stop
        
    Returns:
        Dict containing stop result
    """
    task_id = self.request.id
    
    logger.info(f"🛑 TRAINING STOP: Stopping training for experiment {experiment_id}")
    
    try:
        experiment_repo = get_experiment_repository()
        
        # Update experiment status to CANCELLED
        experiment_repo.update_experiment_status(experiment_id, ExperimentStatus.CANCELLED)
        
        # Publish stop status
        publish_training_update(experiment_id, "CANCELLED", {
            "message": "Training cancelled by user",
            "experiment_id": experiment_id,
            "progress": 0.0
        })
        
        # TODO: Actually interrupt the running training task
        # This would require storing the task ID and using celery.control.revoke
        
        return {
            "status": "cancelled",
            "experiment_id": experiment_id
        }
        
    except Exception as e:
        logger.error(f"Error stopping training: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@celery_app.task(name="finetuning_tasks.cleanup_training_artifacts")
def cleanup_training_artifacts() -> Dict[str, Any]:
    """
    Periodic task to clean up old training artifacts
    
    Runs hourly to clean up:
    - Old training logs
    - Orphaned model files
    - Expired Redis progress keys
    """
    logger.info("Running training artifacts cleanup")
    
    cleaned_count = 0
    
    try:
        # Clean up expired Redis keys
        for key in redis_client.scan_iter("training_progress_*"):
            ttl = redis_client.ttl(key)
            if ttl == -1:  # No expiry set
                redis_client.expire(key, 3600)  # Set 1 hour expiry
                cleaned_count += 1
        
        logger.info(f"Cleanup completed: {cleaned_count} items processed")
        
        return {
            "status": "success",
            "cleaned_count": cleaned_count
        }
        
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
