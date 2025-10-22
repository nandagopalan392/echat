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
    EvaluationRequest,
    BatchEvaluationRequest,
    DatasetEvaluationRequest,
    DatasetCreationRequest,
    EvaluationResponse,
    EvaluationStatusResponse,
    QueueStatusResponse,
    EvaluationMetrics,
    HistoricalMetrics,
    LatencyDistribution,
    QualityBreakdown,
    DetailedEvaluation,
    TestCaseRunRequest,
    EvaluationResultsResponse
)
from app.services.evaluation_service import EvaluationService
from app.dependencies import get_current_user, get_current_admin_user

# Background task imports
from app.workers.celery_app import celery_app
from app.workers.tasks.evaluation_tasks import (
    evaluate_rag_response,
    batch_evaluate_conversations,
    evaluate_dataset_with_rag,
    create_dataset_background
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

# Initialize service
evaluation_service = EvaluationService()


# ==================== Evaluation Execution Endpoints ====================

@router.post("/evaluate/async", response_model=EvaluationResponse)
async def evaluate_async(
    request: EvaluationRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Submit an asynchronous evaluation task
    Evaluates a single RAG response against the RAG Triad metrics
    """
    try:
        # Submit Celery task
        task = evaluate_rag_response.delay(
            query=request.query,
            response=request.response,
            context=request.context,
            conversation_id=request.conversation_id,
            user_id=request.user_id or current_user.get("username"),
            metadata=request.metadata
        )
        
        websocket_url = f"/api/evaluation/ws/evaluation/{task.id}"
        
        return EvaluationResponse(
            task_id=task.id,
            status="PENDING",
            message="Evaluation task submitted successfully",
            websocket_url=websocket_url
        )
    
    except Exception as e:
        logger.error(f"Error submitting async evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate/batch", response_model=EvaluationResponse)
async def evaluate_batch(
    request: BatchEvaluationRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Submit a batch evaluation task for multiple conversations
    """
    try:
        task = batch_evaluate_conversations.delay(
            conversation_ids=request.conversation_ids,
            user_id=request.user_id or current_user.get("username")
        )
        
        websocket_url = f"/api/evaluation/ws/evaluation/{task.id}"
        
        return EvaluationResponse(
            task_id=task.id,
            status="PENDING",
            message=f"Batch evaluation submitted for {len(request.conversation_ids)} conversations",
            websocket_url=websocket_url
        )
    
    except Exception as e:
        logger.error(f"Error submitting batch evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate/dataset", response_model=EvaluationResponse)
async def evaluate_dataset(
    request: DatasetEvaluationRequest,
    current_user: Dict = Depends(get_current_user)
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
            user_id=request.user_id or current_user.get("username"),
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
    current_user: Dict = Depends(get_current_admin_user)
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
            user_id=request.user_id or current_user.get("username")
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
    current_user: Dict = Depends(get_current_user)
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


@router.get("/status/{task_id}")
async def get_task_status_legacy(
    task_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Legacy endpoint for task status (for backwards compatibility)
    """
    return await get_evaluation_status(task_id, current_user)


@router.delete("/evaluate/task/{task_id}")
async def cancel_evaluation_task(
    task_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Cancel a running evaluation task
    """
    try:
        result = evaluation_service.cancel_task(task_id)
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error cancelling task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluate/queue/status", response_model=QueueStatusResponse)
async def get_queue_status(
    current_user: Dict = Depends(get_current_admin_user)
):
    """
    Get Celery queue status (Admin only)
    """
    try:
        status = evaluation_service.get_queue_status()
        return QueueStatusResponse(**status)
    
    except Exception as e:
        logger.error(f"Error getting queue status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Metrics Endpoints ====================

@router.get("/metrics", response_model=EvaluationMetrics)
async def get_evaluation_metrics(
    days: int = Query(7, ge=1, le=90, description="Number of days to aggregate"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get evaluation metrics summary
    """
    try:
        metrics = evaluation_service.get_evaluation_metrics(days=days, user_id=user_id)
        return EvaluationMetrics(**metrics)
    
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/historical", response_model=List[HistoricalMetrics])
async def get_historical_metrics(
    days: int = Query(30, ge=1, le=365, description="Number of days of history"),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get historical daily metrics
    """
    try:
        metrics = evaluation_service.get_historical_metrics(days=days)
        return [HistoricalMetrics(**m) for m in metrics]
    
    except Exception as e:
        logger.error(f"Error getting historical metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latency-distribution", response_model=List[LatencyDistribution])
async def get_latency_distribution(
    current_user: Dict = Depends(get_current_user)
):
    """
    Get latency distribution across buckets
    """
    try:
        distribution = evaluation_service.get_latency_distribution()
        return [LatencyDistribution(**d) for d in distribution]
    
    except Exception as e:
        logger.error(f"Error getting latency distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quality-breakdown", response_model=List[QualityBreakdown])
async def get_quality_breakdown(
    current_user: Dict = Depends(get_current_user)
):
    """
    Get quality score breakdown
    """
    try:
        breakdown = evaluation_service.get_quality_breakdown()
        return [QualityBreakdown(**b) for b in breakdown]
    
    except Exception as e:
        logger.error(f"Error getting quality breakdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluate/results/recent")
async def get_recent_results(
    limit: int = Query(10, ge=1, le=100, description="Number of recent results"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    current_user: Dict = Depends(get_current_user)
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
    current_user: Dict = Depends(get_current_user)
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


# ==================== Test Case Execution ====================

@router.post("/test-cases/run")
async def run_test_cases(
    request: TestCaseRunRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Run evaluation test cases
    """
    try:
        # This would be implemented based on your test case execution logic
        # For now, return a placeholder response
        return JSONResponse(content={
            "status": "success",
            "message": f"Running {len(request.test_cases)} test cases",
            "test_count": len(request.test_cases)
        })
    
    except Exception as e:
        logger.error(f"Error running test cases: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Legacy/Compatibility Endpoints ====================

@router.post("/evaluate-response")
async def evaluate_response_legacy(
    request: EvaluationRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Legacy synchronous evaluation endpoint (for backwards compatibility)
    Note: Use /evaluate/async for better performance
    """
    try:
        from app.core.evaluation.system import evaluate_chat_response
        
        result = evaluate_chat_response(
            query=request.query,
            response=request.response,
            context="\n\n".join(request.context)
        )
        
        return JSONResponse(content={
            "status": "success",
            "result": result.dict() if hasattr(result, 'dict') else result
        })
    
    except Exception as e:
        logger.error(f"Error in legacy evaluate response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trulens-metrics")
async def get_trulens_metrics(
    days: int = Query(7, ge=1, le=90),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get TruLens-style metrics (compatibility endpoint)
    """
    try:
        metrics = evaluation_service.get_evaluation_metrics(days=days)
        
        # Format in TruLens style
        return JSONResponse(content={
            "groundedness": {
                "mean": metrics.get("avg_groundedness", 0),
                "std": 0.0,  # Would need to calculate from data
                "count": metrics.get("total_evaluations", 0)
            },
            "answer_relevance": {
                "mean": metrics.get("avg_answer_relevance", 0),
                "std": 0.0,
                "count": metrics.get("total_evaluations", 0)
            },
            "context_relevance": {
                "mean": metrics.get("avg_context_relevance", 0),
                "std": 0.0,
                "count": metrics.get("total_evaluations", 0)
            }
        })
    
    except Exception as e:
        logger.error(f"Error getting TruLens metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluation-trends")
async def get_evaluation_trends(
    days: int = Query(30, ge=1, le=365),
    current_user: Dict = Depends(get_current_user)
):
    """
    Get evaluation trends over time
    """
    try:
        trends = evaluation_service.get_historical_metrics(days=days)
        
        return JSONResponse(content={
            "trends": trends,
            "summary": {
                "total_days": len(trends),
                "total_evaluations": sum(t.get("total_evaluations", 0) for t in trends),
                "avg_overall_score": sum(
                    (t.get("avg_groundedness", 0) + 
                     t.get("avg_answer_relevance", 0) + 
                     t.get("avg_context_relevance", 0)) / 3.0 
                    for t in trends
                ) / len(trends) if trends else 0
            }
        })
    
    except Exception as e:
        logger.error(f"Error getting evaluation trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/detailed-evaluation/{evaluation_id}")
async def get_detailed_evaluation(
    evaluation_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get detailed evaluation result by ID
    """
    try:
        # Mock detailed evaluation for demonstration
        # In production, fetch from database
        return DetailedEvaluation(
            evaluation_id=evaluation_id,
            timestamp=datetime.now().isoformat(),
            input={
                "question": "Sample question",
                "context": "Sample context",
                "answer": "Sample answer"
            },
            evaluation_results={
                "groundedness": {
                    "score": 0.85,
                    "reasoning": "Well grounded in context"
                },
                "answer_relevance": {
                    "score": 0.92,
                    "reasoning": "Highly relevant answer"
                },
                "context_relevance": {
                    "score": 0.78,
                    "reasoning": "Context is relevant"
                }
            },
            overall_score=0.85,
            evaluation_time_seconds=2.3,
            metadata={"model": "llama3", "version": "1.0"}
        )
    
    except Exception as e:
        logger.error(f"Error getting detailed evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
