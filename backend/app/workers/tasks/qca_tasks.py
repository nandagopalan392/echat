"""
Q-C-A Dataset Generation Celery Tasks

This module contains Celery tasks for Q-C-A dataset generation.
"""

import asyncio
import os
import time
import datetime
from typing import List, Dict, Any

from celery import current_task
from app.workers.celery_app import celery_app
from app.core.datasets.qca_dataset_generator import QCADatasetGenerator
from app.db.repositories.experiment_repository import get_experiment_repository

import logging
logger = logging.getLogger(__name__)

def publish_qca_update(task_id: str, status: str, data: Dict[str, Any]):
    """Publish Q-C-A dataset progress update via Redis"""
    try:
        import json
        import redis
        import os
        
        # Connect to Redis directly
        redis_client = redis.Redis(host=os.getenv('REDIS_HOST', 'redis'), port=6379, db=0)
        
        progress_data = {
            "status": status,
            "data": data,
            "timestamp": time.time()
        }
        
        # Serialize to JSON string before storing in Redis
        redis_client.set(f"qca_progress_{task_id}", json.dumps(progress_data))
        logger.info(f"Q-C-A Progress Update: {task_id} - {status} - {data}")
    except Exception as e:
        logger.error(f"Failed to publish Q-C-A progress update: {e}")

@celery_app.task(bind=True, name="qca_tasks.create_qca_dataset_background")
def create_qca_dataset_background(
    self,
    name: str,
    description: str,
    document_ids: List[str],
    questions_per_doc: int = 5,
    model_name: str = "gemma2:2b",
    user_id: str = "admin"
) -> Dict[str, Any]:
    """
    Create Q-C-A dataset in background with progress tracking
    
    Args:
        name: Dataset name
        description: Dataset description
        document_ids: List of document IDs to use
        questions_per_doc: Number of questions per document
        model_name: Model to use for generation
        user_id: User creating the dataset
        
    Returns:
        Dict containing dataset creation results
    """
    task_id = self.request.id
    start_time = time.time()
    
    logger.info(f"🚀 Q-C-A DATASET CREATE: Starting async dataset creation for task {task_id}")
    logger.info(f"🚀 Q-C-A DATASET CREATE: Name '{name}', Documents: {len(document_ids)}, User: {user_id}")
    
    try:
        # Initialize progress update
        publish_qca_update(task_id, "STARTED", {
            "message": "Starting Q-C-A dataset creation",
            "dataset_name": name,
            "total_documents": len(document_ids),
            "progress": 0.0
        })
        
        # Get experiment repository instance
        experiment_repo = get_experiment_repository()
        
        # Create dataset record in database
        dataset_id = experiment_repo.create_dataset(
            name=name,
            description=description,
            samples=[],  # Empty initially, will be populated after generation
            user_id=user_id
        )
        
        logger.info(f"Created Q-C-A dataset record: {name} with ID {dataset_id}")
        
        publish_qca_update(task_id, "PROGRESS", {
            "message": "Created dataset record",
            "dataset_id": dataset_id,
            "progress": 0.05
        })
        
        # Fetch documents
        publish_qca_update(task_id, "PROGRESS", {
            "message": "Fetching documents",
            "progress": 0.1
        })
        
        from document_storage import get_document_storage
        doc_storage = get_document_storage()
        
        documents = []
        for doc_id in document_ids:
            try:
                doc_info = doc_storage.get_document_info(doc_id)
                if doc_info:
                    documents.append(doc_info)
            except Exception as e:
                logger.warning(f"Could not fetch document {doc_id}: {e}")
                continue
        
        if not documents:
            raise ValueError("No valid documents found")
        
        # Start Q-C-A generation
        publish_qca_update(task_id, "PROGRESS", {
            "message": "Starting Q-C-A generation",
            "progress": 0.2
        })
        
        # Generate dataset using Q-C-A pipeline
        generator = QCADatasetGenerator(ollama_base_url=os.getenv('OLLAMA_HOST', 'http://ollama:11434'))
        
        # Use asyncio.run instead of mixing sync/async
        def run_async_generation():
            """Run the async generation in a new event loop"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(generator.generate_dataset_from_documents(
                    documents=documents,
                    dataset_name=name,
                    dataset_description=description,
                    questions_per_doc=questions_per_doc,
                    model_name=model_name
                ))
            finally:
                # Properly close the generator without using the loop
                try:
                    if hasattr(generator, 'client') and generator.client:
                        # Close synchronously if possible, or skip if it causes issues
                        pass
                except Exception as e:
                    logger.warning(f"Could not close generator client: {e}")
                loop.close()
        
        dataset = run_async_generation()
        
        # Save dataset to JSONL file
        publish_qca_update(task_id, "PROGRESS", {
            "message": "Saving dataset",
            "progress": 0.8
        })
        
        datasets_dir = "/app/data/finetuning_datasets"
        os.makedirs(datasets_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_name}_{timestamp}.jsonl"
        file_path = os.path.join(datasets_dir, filename)
        
        # Save using the generator's save method
        asyncio.run(generator.save_dataset_to_jsonl(dataset, file_path))
        
        logger.info(f"Dataset saved to {file_path}")
        
        # Update database with the generated samples
        samples = [{
            "instruction": item.instruction,
            "input": item.input,
            "output": item.output
        } for item in dataset.items]
        
        # Update the dataset in the database
        experiment_repo.update_dataset_samples(dataset_id, samples)
        experiment_repo.update_dataset_status(dataset_id, "Completed")
        
        logger.info(f"Updated dataset {dataset_id} status to Completed")
        
        execution_time = time.time() - start_time
        
        # Final success update
        publish_qca_update(task_id, "SUCCESS", {
            "message": "Q-C-A dataset created successfully",
            "dataset_id": dataset_id,
            "file_path": file_path,
            "total_items": len(dataset.items),
            "execution_time": execution_time,
            "progress": 1.0
        })
        
        logger.info(f"Successfully generated Q-C-A dataset {name} with {len(dataset.items)} items in {execution_time:.2f}s")
        
        return {
            "status": "success",
            "dataset_id": dataset_id,
            "file_path": file_path,
            "total_items": len(dataset.items),
            "execution_time": execution_time
        }
        
    except Exception as e:
        error_msg = str(e)
        execution_time = time.time() - start_time
        
        logger.error(f"Error in Q-C-A dataset creation: {error_msg}")
        
        # Update dataset status to failed if dataset was created
        try:
            if 'dataset_id' in locals():
                experiment_repo = get_experiment_repository()
                experiment_repo.update_dataset_status(dataset_id, "Failed")
                logger.info(f"Updated dataset {dataset_id} status to Failed")
        except Exception as db_error:
            logger.error(f"Failed to update dataset status: {db_error}")
        
        # Publish failure notification
        publish_qca_update(task_id, "FAILURE", {
            "message": f"Q-C-A dataset creation failed: {error_msg}",
            "error": error_msg,
            "execution_time": execution_time,
            "progress": 0.0
        })
        
        raise Exception(error_msg)
