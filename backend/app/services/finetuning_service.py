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
    def get_available_models(self) -> List[str]:
        """Get list of available models for fine-tuning"""
        try:
            return self.hf_finetuner.get_available_models()
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
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
            config['dataset_path'] = dataset_info['file_path']
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
    
    def start_training(self, experiment_id: str, config: Dict[str, Any]) -> None:
        """Start training for an experiment (background task)"""
        try:
            self.hf_finetuner.start_training(experiment_id, config)
        except Exception as e:
            logger.error(f"Error starting training for {experiment_id}: {e}")
            raise
    
    def get_user_experiments(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get experiments for a user"""
        try:
            return self.experiment_db.get_user_experiments(user_id, limit)
        except Exception as e:
            logger.error(f"Error getting experiments for user {user_id}: {e}")
            raise
    
    def get_experiment(self, experiment_id: str, user_id: str) -> Dict[str, Any]:
        """Get experiment details"""
        try:
            experiment = self.experiment_db.get_experiment(experiment_id)
            if not experiment:
                raise ValueError("Experiment not found")
            
            # Check ownership
            if experiment['user_id'] != user_id:
                raise PermissionError("Not authorized")
            
            # Get training logs
            training_logs = self.experiment_db.get_training_logs(experiment_id)
            experiment['training_logs'] = training_logs
            
            return experiment
            
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
    
    def start_experiment_training(self, experiment_id: str, user_id: str) -> None:
        """Start training for an existing experiment"""
        try:
            # Get experiment
            experiment = self.experiment_db.get_experiment(experiment_id)
            if not experiment:
                raise ValueError("Experiment not found")
            
            # Check ownership
            if experiment['user_id'] != user_id:
                raise PermissionError("Not authorized")
            
            # Check status
            if experiment['status'] not in [self.ExperimentStatus.DRAFT.value, self.ExperimentStatus.FAILED.value]:
                raise ValueError(f"Cannot start experiment in {experiment['status']} status")
            
            # Start training
            self.hf_finetuner.start_training(experiment_id, experiment['config'])
            
        except Exception as e:
            logger.error(f"Error starting experiment training {experiment_id}: {e}")
            raise
    
    def get_training_logs(self, experiment_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Get training logs for an experiment"""
        try:
            experiment = self.experiment_db.get_experiment(experiment_id)
            if not experiment:
                raise ValueError("Experiment not found")
            
            if experiment['user_id'] != user_id:
                raise PermissionError("Not authorized")
            
            return self.experiment_db.get_training_logs(experiment_id)
            
        except Exception as e:
            logger.error(f"Error getting logs for {experiment_id}: {e}")
            raise
    
    def get_training_metrics(self, experiment_id: str, user_id: str) -> Dict[str, Any]:
        """Get live training metrics for an experiment"""
        try:
            from app.core.training.metrics import get_metrics_collector
            
            # Check if experiment belongs to user
            experiment = self.experiment_db.get_experiment(experiment_id)
            if not experiment:
                raise ValueError("Experiment not found")
            
            if experiment.get('user_id') != user_id:
                raise PermissionError("Access denied")
            
            # Get metrics collector if training is active
            metrics_collector = get_metrics_collector(experiment_id)
            if metrics_collector:
                return metrics_collector.get_metrics_summary()
            else:
                # Return historical data from database if available
                if experiment.get('metrics') and experiment['metrics'].get('training_history'):
                    training_history = experiment['metrics']['training_history']
                    final_progress = experiment['metrics'].get('final_progress', {})
                    
                    return {
                        "experiment_id": experiment_id,
                        "progress": final_progress,
                        "metrics": training_history,
                        "system": {"current": {}, "history": []},
                        "training_logs": self.experiment_db.get_training_logs(experiment_id)[-20:],
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
                        "training_logs": logs[-20:] if logs else [],
                        "training_completed": False
                    }
                    
        except Exception as e:
            logger.error(f"Error getting metrics for {experiment_id}: {e}")
            raise
    
    # Dataset operations
    def get_user_datasets(self, user_id: str) -> List[Dict[str, Any]]:
        """Get datasets for a user"""
        try:
            return self.experiment_db.get_user_datasets(user_id)
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
            
            return {
                "id": dataset['id'],
                "name": dataset['name'],
                "description": dataset['description'],
                "num_samples": dataset['num_samples'],
                "file_size": dataset.get('file_size', 0),
                "format": dataset.get('format', 'jsonl'),
                "created_at": dataset['created_at'],
                "samples": samples
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
                jsonl_lines.append(json.dumps(sample, ensure_ascii=False))
            
            jsonl_content = '\n'.join(jsonl_lines)
            
            # Create filename
            safe_name = "".join(c for c in dataset['name'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
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
