"""
Fine-tuning Service
Business logic for fine-tuning operations
"""
from typing import Dict, List, Any, Optional
import logging
import os
import uuid
import json

logger = logging.getLogger(__name__)


class FinetuningService:
    """Service for managing fine-tuning operations"""
    
    def __init__(self):
        """Initialize fine-tuning service"""
        from app.db.repositories.experiment_repository import experiment_repository, ExperimentStatus
        from app.core.training.hf_finetuner import hf_finetuner
        
        self.experiment_db = experiment_repository
        self.ExperimentStatus = ExperimentStatus
        self.hf_finetuner = hf_finetuner
    
    # Model operations
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models for fine-tuning from cache"""
        try:
            from app.core.providers import get_model_cache
            model_cache = get_model_cache()
            finetuning_models = model_cache.get_finetuning_models()
            
            # Ensure each model has required fields for ModelInfo schema
            result = []
            for model in finetuning_models:
                result.append({
                    "name": model.get("name", ""),
                    "description": model.get("description", ""),
                    "size": model.get("size", "unknown"),
                    "type": model.get("type", "text-generation")
                })
            return result
        except Exception as e:
            logger.warning(f"Could not get models from cache: {e}")
            # Fallback to direct fetch
            try:
                model_names = self.hf_finetuner.get_available_models()
                # Convert to dict format for schema compatibility
                return [{"name": name, "description": "", "size": "unknown", "type": "text-generation"} for name in model_names]
            except Exception as inner_e:
                logger.error(f"Error getting available models: {inner_e}")
                raise
    
    # Experiment operations
    def create_experiment(
        self,
        user_id: str,
        name: str,
        description: str,
        model_name: str,
        dataset_id: str,
        config: Dict[str, Any]
    ) -> str:
        """Create a new fine-tuning experiment"""
        try:
            # Get dataset info
            dataset_info = self.experiment_db.get_dataset(dataset_id)
            if not dataset_info:
                raise ValueError("Dataset not found")
            
            # Add dataset path to config
            # dataset_info is a Dataset dataclass, access attributes directly
            config['dataset_path'] = dataset_info.file_path
            config['dataset_id'] = dataset_id
            
            # Create experiment data
            experiment_data = {
                'name': name,
                'description': description,
                'user_id': user_id,
                'base_model': model_name,
                'model_provider': 'huggingface',
                'config': config
            }
            
            # Create experiment in database
            experiment_id = self.experiment_db.create_experiment(experiment_data)
            logger.info(f"Created experiment {experiment_id} for user {user_id}")
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"Error creating experiment: {e}")
            raise
    
    def start_training(self, experiment_id: str, config: Dict[str, Any]) -> str:
        """
        Start training for an experiment using Celery background task.
        
        Returns the Celery task ID for tracking progress.
        """
        try:
            from app.workers.tasks.finetuning_tasks import start_training_background
            
            # Submit training to Celery queue
            task = start_training_background.delay(experiment_id, config)
            
            logger.info(f"Started training task {task.id} for experiment {experiment_id}")
            return task.id
            
        except Exception as e:
            logger.error(f"Error starting training for {experiment_id}: {e}")
            raise
    
    def stop_training(self, experiment_id: str) -> bool:
        """Stop a running training task"""
        try:
            from app.workers.tasks.finetuning_tasks import stop_training_background
            
            # Submit stop request to Celery
            task = stop_training_background.delay(experiment_id)
            
            logger.info(f"Submitted stop request for experiment {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping training for {experiment_id}: {e}")
            raise
    
    def get_user_experiments(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get experiments for a user"""
        try:
            experiments = self.experiment_db.get_user_experiments(user_id, limit)
            # Convert Experiment dataclass objects to dictionaries
            return [exp.to_dict() for exp in experiments]
        except Exception as e:
            logger.error(f"Error getting experiments for user {user_id}: {e}")
            raise
    
    def get_experiment(self, experiment_id: str, user_id: str) -> Dict[str, Any]:
        """Get experiment details"""
        try:
            experiment = self.experiment_db.get_experiment(experiment_id)
            if not experiment:
                raise ValueError("Experiment not found")
            
            # Check ownership - experiment is a dataclass, use attribute access
            if experiment.user_id != user_id:
                raise PermissionError("Not authorized")
            
            # Convert to dict and add training logs
            result = experiment.to_dict()
            training_logs = self.experiment_db.get_training_logs(experiment_id)
            # Convert TrainingLog objects to dicts
            result['training_logs'] = [log.to_dict() if hasattr(log, 'to_dict') else log for log in training_logs]
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting experiment {experiment_id}: {e}")
            raise
    
    def delete_experiment(self, experiment_id: str, user_id: str) -> bool:
        """Delete an experiment"""
        try:
            return self.experiment_db.delete_experiment(experiment_id, user_id)
        except Exception as e:
            logger.error(f"Error deleting experiment {experiment_id}: {e}")
            raise
    
    def start_experiment_training(self, experiment_id: str, user_id: str) -> str:
        """
        Start training for an existing experiment using Celery task.
        
        Returns the Celery task ID for tracking progress.
        """
        try:
            # Get experiment - returns Experiment dataclass
            experiment = self.experiment_db.get_experiment(experiment_id)
            if not experiment:
                raise ValueError("Experiment not found")
            
            # Check ownership - use attribute access
            if experiment.user_id != user_id:
                raise PermissionError("Not authorized")
            
            # Check status - use attribute access
            if experiment.status not in [self.ExperimentStatus.DRAFT.value, self.ExperimentStatus.FAILED.value]:
                raise ValueError(f"Cannot start experiment in {experiment.status} status")
            
            # Start training via Celery task - use attribute access for config
            from app.workers.tasks.finetuning_tasks import start_training_background
            task = start_training_background.delay(experiment_id, experiment.config)
            
            logger.info(f"Started training task {task.id} for experiment {experiment_id}")
            return task.id
            
        except Exception as e:
            logger.error(f"Error starting experiment training {experiment_id}: {e}")
            raise
            raise
    
    def get_training_logs(self, experiment_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Get training logs for an experiment"""
        try:
            experiment = self.experiment_db.get_experiment(experiment_id)
            if not experiment:
                raise ValueError("Experiment not found")
            
            # Use attribute access for Experiment dataclass
            if experiment.user_id != user_id:
                raise PermissionError("Not authorized")
            
            logs = self.experiment_db.get_training_logs(experiment_id)
            # Convert TrainingLog objects to dicts
            return [log.to_dict() if hasattr(log, 'to_dict') else log for log in logs]
            
        except Exception as e:
            logger.error(f"Error getting logs for {experiment_id}: {e}")
            raise
    
    def get_training_metrics(self, experiment_id: str, user_id: str) -> Dict[str, Any]:
        """Get live training metrics for an experiment"""
        try:
            from app.core.training.metrics import get_metrics_collector
            
            # Check if experiment belongs to user - experiment is a dataclass
            experiment = self.experiment_db.get_experiment(experiment_id)
            if not experiment:
                raise ValueError("Experiment not found")
            
            # Use attribute access for Experiment dataclass
            if experiment.user_id != user_id:
                raise PermissionError("Access denied")
            
            # Get metrics collector if training is active
            metrics_collector = get_metrics_collector(experiment_id)
            if metrics_collector:
                return metrics_collector.get_metrics_summary()
            else:
                # Return historical data from database if available
                # Use attribute access for metrics
                if experiment.metrics and experiment.metrics.get('training_history'):
                    training_history = experiment.metrics['training_history']
                    final_progress = experiment.metrics.get('final_progress', {})
                    
                    logs = self.experiment_db.get_training_logs(experiment_id)
                    return {
                        "experiment_id": experiment_id,
                        "progress": final_progress,
                        "metrics": training_history,
                        "system": {"current": {}, "history": []},
                        "training_logs": [log.to_dict() if hasattr(log, 'to_dict') else log for log in logs[-20:]],
                        "training_completed": True
                    }
                else:
                    # Return empty structure if no training data available
                    logs = self.experiment_db.get_training_logs(experiment_id)
                    return {
                        "experiment_id": experiment_id,
                        "progress": {"current_epoch": 0, "total_epochs": 0, "current_step": 0, "total_steps": 0},
                        "metrics": {"train_losses": [], "eval_losses": [], "learning_rates": [], "accuracies": []},
                        "system": {"current": {}, "history": []},
                        "training_logs": [log.to_dict() if hasattr(log, 'to_dict') else log for log in logs[-20:]] if logs else [],
                        "training_completed": False
                    }
                    
        except Exception as e:
            logger.error(f"Error getting metrics for {experiment_id}: {e}")
            raise
    
    # Dataset operations
    def get_user_datasets(self, user_id: str) -> List[Dict[str, Any]]:
        """Get datasets for a user"""
        try:
            datasets = self.experiment_db.get_user_datasets(user_id)
            # Convert Dataset model instances to dictionaries
            return [dataset.to_dict() if hasattr(dataset, 'to_dict') else dataset for dataset in datasets]
        except Exception as e:
            logger.error(f"Error getting datasets for user {user_id}: {e}")
            raise
    
    def validate_dataset_file(self, file_path: str) -> Dict[str, Any]:
        """Validate a dataset file"""
        try:
            return self.hf_finetuner.validate_dataset(file_path)
        except Exception as e:
            logger.error(f"Error validating dataset: {e}")
            raise
    
    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Get dataset details"""
        try:
            dataset = self.experiment_db.get_dataset(dataset_id)
            if not dataset:
                raise ValueError("Dataset not found")
            
            # Get sample data (first 10 samples for preview)
            samples = self.experiment_db.get_dataset_samples(dataset_id, limit=10)
            # Convert DatasetSample objects to dicts
            samples_list = [s.to_dict() if hasattr(s, 'to_dict') else s for s in samples]
            
            # Dataset is a dataclass - use attribute access
            return {
                "id": dataset.id,
                "name": dataset.name,
                "description": dataset.description,
                "num_samples": dataset.num_samples,
                "file_size": getattr(dataset, 'file_size', 0) or 0,
                "format": getattr(dataset, 'format', 'jsonl') or 'jsonl',
                "created_at": dataset.created_at,
                "samples": samples_list
            }
            
        except Exception as e:
            logger.error(f"Error getting dataset {dataset_id}: {e}")
            raise
    
    def get_dataset_download(self, dataset_id: str) -> tuple[str, str]:
        """Get dataset content for download (content, filename)"""
        try:
            dataset = self.experiment_db.get_dataset(dataset_id)
            if not dataset:
                raise ValueError("Dataset not found")
            
            # Get all samples
            samples = self.experiment_db.get_dataset_samples(dataset_id)
            
            # Create JSONL content
            jsonl_lines = []
            for sample in samples:
                # Convert DatasetSample to dict if needed
                sample_dict = sample.to_dict() if hasattr(sample, 'to_dict') else sample
                jsonl_lines.append(json.dumps(sample_dict, ensure_ascii=False))
            
            jsonl_content = '\n'.join(jsonl_lines)
            
            # Create filename - Dataset is a dataclass, use attribute access
            safe_name = "".join(c for c in dataset.name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{safe_name}.jsonl"
            
            return jsonl_content, filename
            
        except Exception as e:
            logger.error(f"Error downloading dataset {dataset_id}: {e}")
            raise
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset"""
        try:
            return self.experiment_db.delete_dataset(dataset_id)
        except Exception as e:
            logger.error(f"Error deleting dataset {dataset_id}: {e}")
            raise
    
    def create_dataset_from_documents(
        self,
        user_id: str,
        name: str,
        description: str,
        document_ids: List[str],
        questions_per_doc: int,
        model_name: str = "gemma2:2b"
    ) -> str:
        """Create dataset from documents (returns task_id for background processing)"""
        try:
            # Validate documents exist
            from document_storage import get_document_storage
            doc_storage = get_document_storage()
            
            valid_docs = []
            for doc_id in document_ids:
                try:
                    doc = doc_storage.get_document(doc_id)
                    if doc:
                        valid_docs.append(doc)
                except Exception as e:
                    logger.error(f"Error loading document {doc_id}: {e}")
                    continue
            
            if not valid_docs:
                raise ValueError("No valid documents found")
            
            # Start background task for dataset generation
            from app.workers.tasks.qca_tasks import create_qca_dataset_background
            
            task = create_qca_dataset_background.delay(
                name=name,
                description=description,
                document_ids=document_ids,
                questions_per_doc=questions_per_doc,
                model_name=model_name,
                user_id=user_id
            )
            
            logger.info(f"Started Q-C-A dataset generation task: {task.id}")
            
            return task.id
            
        except Exception as e:
            logger.error(f"Error creating dataset from documents: {e}")
            raise
