"""
Evaluation API Endpoints
All evaluation-related endpoints migrated from routes/evaluation.py
"""
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from app.api.v1.schemas.evaluation import (
    DatasetEvaluationRequest,
    DatasetCreationRequest,
    EvaluationResponse,
    EvaluationResultsResponse
)
from app.services.evaluation_service import EvaluationService
from app.dependencies import get_current_admin_user

# Background task imports
from app.workers.celery_app import celery_app
from app.workers.tasks.evaluation_tasks import (
    evaluate_dataset_with_rag,
    create_dataset_background
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

# Initialize service
evaluation_service = EvaluationService()


# ==================== Evaluation Execution Endpoints ====================

@router.post("/evaluate/dataset", response_model=EvaluationResponse)
async def evaluate_dataset(
    request: DatasetEvaluationRequest,
    current_admin: Dict = Depends(get_current_admin_user)
):
    """
    Evaluate RAG system against a dataset
    """
    try:
        task = evaluate_dataset_with_rag.delay(
            dataset_id=request.dataset_id,
            dataset_name=request.dataset_name,
            model_id=request.model_id,
            retrieval_config=request.retrieval_config,
            user_id=request.user_id or current_admin.get("username"),
            metadata=request.metadata
        )
        
        websocket_url = f"/api/evaluation/ws/evaluation/{task.id}"
        
        return EvaluationResponse(
            task_id=task.id,
            status="PENDING",
            message="Dataset evaluation task submitted",
            websocket_url=websocket_url
        )
    
    except Exception as e:
        logger.error(f"Error submitting dataset evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create/dataset", response_model=EvaluationResponse)
async def create_dataset(
    request: DatasetCreationRequest,
    current_admin: Dict = Depends(get_current_admin_user)
):
    """
    Create evaluation dataset from documents (Admin only)
    """
    try:
        # Create dataset record in database BEFORE queueing task
        # This ensures the dataset appears immediately in the frontend with "Processing" status
        user_id = request.user_id or current_admin.get("username")
        
        from app.db import DatabaseConnection
        from app.db.repositories import EvaluationRepository
        
        db = DatabaseConnection()
        eval_repo = EvaluationRepository(db)
        
        # Create dataset record synchronously
        dataset_id = eval_repo.create_dataset(
            name=request.name,
            description=request.description,
            created_by=user_id,
            file_path=None  # Will be updated by background task
        )
        
        if not dataset_id:
            raise HTTPException(status_code=500, detail="Failed to create dataset record")
        
        logger.info(f"Created dataset record with ID {dataset_id}, queuing background task")
        
        # Now queue the background task with the dataset_id
        task = create_dataset_background.delay(
            name=request.name,
            description=request.description,
            document_ids=request.document_ids,
            num_questions_per_doc=request.num_questions_per_doc,
            model_name=request.model_name,
            difficulty_levels=request.difficulty_levels,
            user_id=user_id,
            dataset_id=dataset_id  # Pass the existing dataset_id
        )
        
        websocket_url = f"/api/evaluation/ws/evaluation/{task.id}"
        
        return EvaluationResponse(
            task_id=task.id,
            status="PENDING",
            message="Dataset creation task submitted",
            websocket_url=websocket_url,
            dataset_id=dataset_id  # Return the dataset_id
        )
    
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Task Status Endpoints ====================

@router.get("/status/{task_id}")
async def get_task_status_polling(
    task_id: str,
    current_admin: Dict = Depends(get_current_admin_user)
):
    """
    Get detailed status of any task (dataset creation or evaluation)
    
    **Production-Grade HTTP Polling Fallback**
    This endpoint serves as a fallback when WebSocket connection fails.
    
    Architecture:
    - Primary: WebSocket for real-time updates (low latency, push-based)
    - Fallback: This HTTP endpoint for polling (when WS fails/disconnects)
    
    Data Source: Redis cache (populated by Celery workers)
    """
    try:
        status = evaluation_service.get_task_status(task_id)
        return JSONResponse(content=status)
    
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Results Endpoints ====================

@router.get("/evaluate/results/recent")
async def get_recent_results(
    limit: int = Query(10, ge=1, le=100, description="Number of recent results"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    current_admin: Dict = Depends(get_current_admin_user)
):
    """
    Get recent evaluation results
    """
    try:
        results = evaluation_service.get_recent_results(limit=limit, user_id=user_id)
        return JSONResponse(content={"results": results, "count": len(results)})
    
    except Exception as e:
        logger.error(f"Error getting recent results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Results Endpoints ====================

@router.get("/results", response_model=EvaluationResultsResponse)
async def get_evaluation_results(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    current_admin: Dict = Depends(get_current_admin_user)
):
    """
    Get paginated evaluation results
    """
    try:
        results = evaluation_service.get_evaluation_results(
            page=page,
            page_size=page_size,
            user_id=user_id
        )
        return EvaluationResultsResponse(**results)
    
    except Exception as e:
        logger.error(f"Error getting evaluation results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Dataset Endpoints ====================

@router.get("/datasets")
async def get_datasets(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    current_admin: Dict = Depends(get_current_admin_user)
):
    """
    Get all evaluation datasets
    """
    try:
        datasets = evaluation_service.get_datasets(user_id=user_id)
        return JSONResponse(content={"datasets": datasets, "count": len(datasets)})
    
    except Exception as e:
        logger.error(f"Error getting datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{dataset_id}")
async def get_dataset_details(
    dataset_id: int,
    current_admin: Dict = Depends(get_current_admin_user)
):
    """
    Get detailed information about a specific dataset including questions
    """
    try:
        import json
        import os
        from app.db import DatabaseConnection
        from app.db.repositories import EvaluationRepository
        
        # Get dataset metadata from database
        db = DatabaseConnection()
        eval_repo = EvaluationRepository(db)
        dataset = eval_repo.get_dataset(dataset_id)
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Convert dataset model to dict
        dataset_dict = {
            "id": dataset.id,
            "name": dataset.name,
            "description": dataset.description,
            "created_at": dataset.created_at,
            "updated_at": dataset.updated_at,
            "document_count": dataset.document_count,
            "file_path": dataset.file_path,
            "status": dataset.status,
            "created_by": dataset.created_by
        }
        
        # Load questions from JSON file if it exists
        if dataset.file_path and os.path.exists(dataset.file_path):
            try:
                with open(dataset.file_path, 'r', encoding='utf-8') as f:
                    dataset_data = json.load(f)
                    items = dataset_data.get("items", [])
                    dataset_dict["questions"] = items
                    dataset_dict["question_count"] = len(items)
                    dataset_dict["generation_metadata"] = dataset_data.get("generation_metadata", {})
                    
                    # Group questions by source_file for frontend display
                    documents_data = {}
                    for item in items:
                        source_file = item.get("source_file", "Unknown Document")
                        if source_file not in documents_data:
                            documents_data[source_file] = []
                        documents_data[source_file].append(item)
                    
                    dataset_dict["documents_data"] = documents_data
            except Exception as e:
                logger.error(f"Error loading dataset file {dataset.file_path}: {e}")
                dataset_dict["questions"] = []
                dataset_dict["question_count"] = 0
                dataset_dict["documents_data"] = {}
        else:
            dataset_dict["questions"] = []
            dataset_dict["question_count"] = 0
            dataset_dict["documents_data"] = {}
        
        return JSONResponse(content=dataset_dict)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Overview Endpoints ====================

@router.get("/overview")
async def get_evaluation_overview(
    time_range: str = Query("7d", description="Time range: 7d, 30d, 90d, all"),
    current_admin: Dict = Depends(get_current_admin_user)
):
    """
    Get evaluation overview with statistics
    """
    try:
        overview = evaluation_service.get_overview(time_range=time_range)
        return JSONResponse(content=overview)
    
    except Exception as e:
        logger.error(f"Error getting evaluation overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))
