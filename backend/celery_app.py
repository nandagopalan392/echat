"""
Celery application configuration for background evaluation tasks
"""

import os
from celery import Celery
from celery.signals import worker_ready, worker_shutting_down
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Celery configuration
redis_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")

# Create Celery instance
celery_app = Celery(
    "echat_evaluation",
    broker=redis_url,
    backend=os.getenv("CELERY_RESULT_BACKEND", redis_url),
    include=["evaluation_tasks"]  # Import tasks module
)

# Configure Celery
celery_app.conf.update(
    # Task routing and execution
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task execution settings
    task_always_eager=False,  # Set to True for testing
    task_eager_propagates=True,
    task_ignore_result=False,
    task_store_eager_result=True,
    
    # Result backend settings
    result_expires=3600,  # 1 hour
    result_backend_transport_options={
        "master_name": "mymaster",
        "visibility_timeout": 3600,
    },
    
    # Worker settings with priority support and async execution
    worker_prefetch_multiplier=1,  # Important: Only fetch 1 task at a time for priority
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks
    worker_disable_rate_limits=False,  # Enable rate limiting
    worker_concurrency=4,   # Number of async workers
    worker_pool='threads',  # Use thread pool for async compatibility
    
    # Task priority settings (0=highest, 9=lowest)
    task_inherit_parent_priority=True,
    task_default_priority=5,  # Medium priority
    worker_direct=True,  # Enable direct task routing
    task_routes={
        'evaluation_tasks.create_dataset_background': {
            'priority': 7,  # Lower priority for dataset creation
            'rate_limit': '2/m',  # Max 2 dataset creations per minute
        },
        'evaluation_tasks.evaluate_dataset_with_rag': {
            'priority': 3,  # Higher priority for evaluations
            'rate_limit': '5/m',  # Max 5 evaluations per minute
        },
        'evaluation_tasks.evaluate_rag_response': {
            'priority': 1,  # Highest priority for single evaluations
            'rate_limit': '10/m',  # Max 10 single evaluations per minute
        },
        'qca_tasks.create_qca_dataset_background': {
            'priority': 6,  # Medium-low priority for Q-C-A dataset creation
            'rate_limit': '1/m',  # Max 1 Q-C-A dataset creation per minute
        },
    },
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Task routing - use default queue for simplicity
    # task_routes={
    #     "evaluation_tasks.evaluate_rag_response": {"queue": "evaluation"},
    #     "evaluation_tasks.batch_evaluate_conversations": {"queue": "evaluation"},
    #     "evaluation_tasks.get_evaluation_status": {"queue": "evaluation"},
    # },
    
    # Beat schedule (for periodic tasks if needed)
    beat_schedule={
        "cleanup-old-results": {
            "task": "evaluation_tasks.cleanup_old_evaluation_results",
            "schedule": 3600.0,  # Run every hour
        },
    },
)

@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    logger.info("Celery worker ready and listening for tasks")

@worker_shutting_down.connect  
def worker_shutting_down_handler(sender=None, **kwargs):
    logger.info("Celery worker shutting down")

if __name__ == "__main__":
    celery_app.start()
