"""
Fine-tuning API Endpoints
All fine-tuning-related endpoints migrated from main.py
"""
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, BackgroundTasks, Form
from fastapi.responses import Response
from typing import List, Dict, Any, Optional
import logging
import os
import uuid
import asyncio

from app.api.v1.schemas.finetuning import (
    ExperimentCreateRequest,
    ExperimentResponse,
    ExperimentListResponse,
    ExperimentDetailsResponse,
    TrainingLogsResponse,
    TrainingMetricsResponse,
    DatasetListResponse,
    DatasetCreateRequest,
    DatasetCreateResponse,
    DatasetDetailsResponse,
    DatasetValidationResult,
    ModelListResponse,
    DeleteResponse,
    StartTrainingResponse
)
from app.services.finetuning_service import FinetuningService
from app.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/finetuning", tags=["finetuning"])

# Initialize service
finetuning_service = FinetuningService()


# ==================== Model Endpoints ====================

@router.get("/models", response_model=ModelListResponse)
async def get_available_models(current_admin: dict = Depends(get_current_admin_user)):
    """
    Get list of available models for fine-tuning
    """
    try:
        models = finetuning_service.get_available_models()
        return ModelListResponse(models=models)
    
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Experiment Endpoints ====================

@router.post("/experiments", response_model=ExperimentResponse)
async def create_experiment(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    description: str = Form(""),
    model_name: str = Form(...),
    dataset_id: str = Form(...),
    learning_rate: float = Form(0.0001),
    num_epochs: int = Form(3),
    batch_size: int = Form(4),
    lora_r: int = Form(16),
    lora_alpha: int = Form(32),
    lora_dropout: float = Form(0.1),
    target_modules: str = Form("q_proj,v_proj"),
    max_seq_length: int = Form(512),
    warmup_ratio: float = Form(0.03),
    weight_decay: float = Form(0.01),
    gradient_accumulation_steps: int = Form(1),
    logging_steps: int = Form(10),
    save_steps: int = Form(500),
    eval_steps: int = Form(100),
    save_total_limit: int = Form(2),
    load_best_model_at_end: bool = Form(True),
    metric_for_best_model: str = Form("eval_loss"),
    greater_is_better: bool = Form(False),
    evaluation_strategy: str = Form("steps"),
    save_strategy: str = Form("steps"),
    current_admin: dict = Depends(get_current_admin_user)
):
    """
    Create a new fine-tuning experiment
    """
    try:
        user_id = current_user['sub']
        
        # Create experiment configuration
        config = {
            'base_model': model_name,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': num_epochs,
            'use_lora': True,  # Always use LoRA for now
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'lora_dropout': lora_dropout,
            'target_modules': target_modules,
            'max_length': max_seq_length,
            'warmup_ratio': warmup_ratio,
            'weight_decay': weight_decay,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'logging_steps': logging_steps,
            'save_steps': save_steps,
            'eval_steps': eval_steps,
            'save_total_limit': save_total_limit,
            'load_best_model_at_end': load_best_model_at_end,
            'metric_for_best_model': metric_for_best_model,
            'greater_is_better': greater_is_better,
            'evaluation_strategy': evaluation_strategy,
            'save_strategy': save_strategy
        }
        
        # Create experiment
        experiment_id = finetuning_service.create_experiment(
            user_id=user_id,
            name=name,
            description=description,
            model_name=model_name,
            dataset_id=dataset_id,
            config=config
        )
        
        logger.info(f"Created experiment {experiment_id} for user {user_id}")
        
        # Automatically start training after creation
        training_status = "draft"
        try:
            background_tasks.add_task(finetuning_service.start_training, experiment_id, config)
            logger.info(f"Auto-started training for experiment {experiment_id}")
            training_status = "training_started"
        except Exception as e:
            logger.error(f"Failed to start training for experiment {experiment_id}: {e}")
        
        return ExperimentResponse(
            experiment={
                "id": experiment_id,
                "name": name,
                "description": description,
                "base_model": model_name,
                "dataset_id": dataset_id
            },
            status="created",
            training_status=training_status
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments", response_model=ExperimentListResponse)
async def get_user_experiments(
    limit: int = 50,
    current_admin: dict = Depends(get_current_admin_user)
):
    """
    Get experiments for the current user
    """
    try:
        user_id = current_user['sub']
        experiments = finetuning_service.get_user_experiments(user_id, limit)
        return ExperimentListResponse(experiments=experiments)
    
    except Exception as e:
        logger.error(f"Error getting experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}")
async def get_experiment_details(
    experiment_id: str,
    current_admin: dict = Depends(get_current_admin_user)
):
    """
    Get detailed information about an experiment
    """
    try:
        user_id = current_user['sub']
        experiment = finetuning_service.get_experiment(experiment_id, user_id)
        return experiment
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting experiment details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/{experiment_id}/start", response_model=StartTrainingResponse)
async def start_experiment(
    experiment_id: str,
    current_admin: dict = Depends(get_current_admin_user)
):
    """
    Start training for an experiment
    """
    try:
        user_id = current_user['sub']
        
        # Start training
        asyncio.create_task(
            asyncio.to_thread(finetuning_service.start_experiment_training, experiment_id, user_id)
        )
        
        return StartTrainingResponse(
            status="training_started",
            experiment_id=experiment_id
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/{experiment_id}/stop")
async def stop_experiment(
    experiment_id: str,
    current_admin: dict = Depends(get_current_admin_user)
):
    """
    Stop training for an experiment
    Note: This endpoint is a placeholder - actual implementation requires
    stopping the background training task
    """
    try:
        user_id = current_user['sub']
        
        # TODO: Implement actual training stop logic
        # This would require tracking training tasks and sending stop signals
        
        return {
            "status": "stop_requested",
            "experiment_id": experiment_id,
            "message": "Stop request sent (not yet implemented)"
        }
    
    except Exception as e:
        logger.error(f"Error stopping experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}/logs", response_model=TrainingLogsResponse)
async def get_experiment_logs(
    experiment_id: str,
    current_admin: dict = Depends(get_current_admin_user)
):
    """
    Get training logs for an experiment
    """
    try:
        user_id = current_user['sub']
        logs = finetuning_service.get_training_logs(experiment_id, user_id)
        return TrainingLogsResponse(logs=logs)
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting experiment logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}/metrics", response_model=TrainingMetricsResponse)
async def get_experiment_metrics(
    experiment_id: str,
    current_admin: dict = Depends(get_current_admin_user)
):
    """
    Get live training metrics for an experiment
    """
    try:
        user_id = current_user['sub']
        metrics = finetuning_service.get_training_metrics(experiment_id, user_id)
        return TrainingMetricsResponse(**metrics)
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting experiment metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/experiments/{experiment_id}")
async def update_experiment(
    experiment_id: str,
    current_admin: dict = Depends(get_current_admin_user)
):
    """
    Update experiment details
    Note: This endpoint is a placeholder for future implementation
    """
    try:
        user_id = current_user['sub']
        
        # TODO: Implement experiment update logic
        
        return {
            "status": "update_not_implemented",
            "message": "Experiment update not yet implemented"
        }
    
    except Exception as e:
        logger.error(f"Error updating experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/experiments/{experiment_id}", response_model=DeleteResponse)
async def delete_experiment(
    experiment_id: str,
    current_admin: dict = Depends(get_current_admin_user)
):
    """
    Delete an experiment
    """
    try:
        user_id = current_user['sub']
        success = finetuning_service.delete_experiment(experiment_id, user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Experiment not found or not authorized")
        
        return DeleteResponse(status="deleted")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Dataset Endpoints ====================

@router.get("/datasets", response_model=DatasetListResponse)
async def get_user_datasets(current_admin: dict = Depends(get_current_admin_user)):
    """
    Get datasets for the current user
    """
    try:
        user_id = current_user['sub']
        datasets = finetuning_service.get_user_datasets(user_id)
        return DatasetListResponse(datasets=datasets)
    
    except Exception as e:
        logger.error(f"Error getting datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate-dataset")
async def validate_dataset_file(
    dataset: UploadFile = File(...),
    current_admin: dict = Depends(get_current_admin_user)
):
    """
    Validate a dataset file without creating an experiment
    """
    try:
        if not dataset.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not dataset.filename.endswith(('.jsonl', '.csv')):
            raise HTTPException(
                status_code=400,
                detail="Dataset must be in JSONL or CSV format"
            )
        
        # Save temporary file
        temp_id = str(uuid.uuid4())
        temp_path = f"/tmp/{temp_id}_{dataset.filename}"
        
        try:
            with open(temp_path, "wb") as buffer:
                content = await dataset.read()
                buffer.write(content)
            
            # Validate
            result = finetuning_service.validate_dataset_file(temp_path)
            return result
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-dataset")
async def upload_dataset(
    dataset: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(""),
    current_admin: dict = Depends(get_current_admin_user)
):
    """
    Upload a dataset file for fine-tuning
    Note: This endpoint is a placeholder for future implementation
    """
    try:
        user_id = current_user['sub']
        
        # TODO: Implement dataset upload and storage
        
        return {
            "status": "upload_not_implemented",
            "message": "Dataset upload not yet fully implemented"
        }
    
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/create", response_model=DatasetCreateResponse)
async def create_finetuning_dataset(
    request: DatasetCreateRequest,
    current_admin: dict = Depends(get_current_admin_user)
):
    """
    Create a new fine-tuning dataset from documents using Q-C-A format
    """
    try:
        user_id = current_user['sub']
        
        # Create dataset from documents
        task_id = finetuning_service.create_dataset_from_documents(
            user_id=user_id,
            name=request.name,
            description=request.description,
            document_ids=request.document_ids,
            questions_per_doc=request.questions_per_doc,
            model_name=request.model_name
        )
        
        return DatasetCreateResponse(
            success=True,
            task_id=task_id,
            message=f"Dataset creation started. Processing {len(request.document_ids)} documents with {request.questions_per_doc} questions per document.",
            total_documents=len(request.document_ids)
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating finetuning dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/convert")
async def convert_dataset(
    current_admin: dict = Depends(get_current_admin_user)
):
    """
    Convert dataset between formats
    Note: This endpoint is a placeholder for future implementation
    """
    try:
        return {
            "status": "convert_not_implemented",
            "message": "Dataset conversion not yet implemented"
        }
    
    except Exception as e:
        logger.error(f"Error converting dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/create/{dataset_id}/progress")
async def get_dataset_creation_progress(
    dataset_id: str,
    current_admin: dict = Depends(get_current_admin_user)
):
    """
    Get progress of dataset creation
    Note: Progress tracking should be done via Redis/WebSocket for real implementations
    """
    try:
        # TODO: Implement proper progress tracking via Redis
        return {
            "dataset_id": dataset_id,
            "status": "progress_not_implemented",
            "message": "Use WebSocket /api/ws/qca-dataset/{task_id} for real-time progress"
        }
    
    except Exception as e:
        logger.error(f"Error getting dataset creation progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{dataset_id}", response_model=DatasetDetailsResponse)
async def get_dataset_details(
    dataset_id: str,
    current_admin: dict = Depends(get_current_admin_user)
):
    """
    Get detailed information about a fine-tuning dataset
    """
    try:
        dataset = finetuning_service.get_dataset(dataset_id)
        return DatasetDetailsResponse(**dataset)
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting dataset details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{dataset_id}/download")
async def download_dataset(
    dataset_id: str,
    current_admin: dict = Depends(get_current_admin_user)
):
    """
    Download a fine-tuning dataset in JSONL format
    """
    try:
        jsonl_content, filename = finetuning_service.get_dataset_download(dataset_id)
        
        # Return as file download
        return Response(
            content=jsonl_content.encode('utf-8'),
            media_type='application/jsonl',
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"'
            }
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    current_admin: dict = Depends(get_current_admin_user)
):
    """
    Delete a fine-tuning dataset
    """
    try:
        success = finetuning_service.delete_dataset(dataset_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        return {"success": True, "message": "Dataset deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))
