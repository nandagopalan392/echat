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
        task = create_dataset_background.delay(
            name=request.name,
            description=request.description,
            document_ids=request.document_ids,
            num_questions_per_doc=request.num_questions_per_doc,
            model_name=request.model_name,
            difficulty_levels=request.difficulty_levels,
            user_id=request.user_id or current_admin.get("username")
        )
        
        websocket_url = f"/api/evaluation/ws/evaluation/{task.id}"
        
        return EvaluationResponse(
            task_id=task.id,
            status="PENDING",
            message="Dataset creation task submitted",
            websocket_url=websocket_url
        )
    
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Task Status Endpoints ====================

@router.get("/evaluate/status/{task_id}")
async def get_evaluation_status(
    task_id: str,
    current_admin: Dict = Depends(get_current_admin_user)
):
    """
    Get detailed status of an evaluation task
    """
    try:
        status = evaluation_service.get_task_status(task_id)
        return JSONResponse(content=status)
    
    except Exception as e:
        logger.error(f"Error getting evaluation status: {e}")
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
