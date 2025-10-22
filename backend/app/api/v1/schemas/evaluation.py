"""
Evaluation API Schemas
Pydantic models for evaluation endpoints request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class EvaluationRequest(BaseModel):
    """Single evaluation request"""
    query: str = Field(..., description="The user query")
    response: str = Field(..., description="The RAG system response")
    context: List[str] = Field(..., description="Retrieved context chunks")
    conversation_id: Optional[str] = Field(None, description="Associated conversation ID")
    user_id: Optional[str] = Field(None, description="User who initiated the evaluation")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class BatchEvaluationRequest(BaseModel):
    """Batch evaluation request for multiple conversations"""
    conversation_ids: List[str] = Field(..., description="List of conversation IDs to evaluate")
    user_id: Optional[str] = Field(None, description="User who initiated the batch evaluation")


class DatasetEvaluationRequest(BaseModel):
    """Dataset-based evaluation request"""
    dataset_id: int = Field(..., description="ID of the evaluation dataset")
    dataset_name: Optional[str] = Field(None, description="Name of the dataset")
    model_id: Optional[str] = Field(None, description="Model to use for evaluation")
    retrieval_config: Optional[Dict[str, Any]] = Field(None, description="Custom retrieval configuration")
    user_id: Optional[str] = Field(None, description="User who initiated the evaluation")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class DatasetCreationRequest(BaseModel):
    """Dataset creation request"""
    name: str = Field(..., description="Dataset name")
    description: str = Field(..., description="Dataset description")
    document_ids: List[str] = Field(..., description="List of document IDs to generate questions from")
    num_questions_per_doc: int = Field(3, ge=1, le=10, description="Questions per document")
    model_name: str = Field("llama3", description="Model to use for question generation")
    difficulty_levels: List[str] = Field(
        ["easy", "medium", "hard"], 
        description="Difficulty levels to generate"
    )
    user_id: str = Field("admin", description="User creating the dataset")


class EvaluationResponse(BaseModel):
    """Standard evaluation task response"""
    task_id: str = Field(..., description="Celery task ID")
    status: str = Field(..., description="Task status (PENDING, STARTED, SUCCESS, FAILURE)")
    message: str = Field(..., description="Human-readable status message")
    websocket_url: Optional[str] = Field(None, description="WebSocket URL for real-time updates")


class EvaluationStatusResponse(BaseModel):
    """Detailed evaluation task status"""
    task_id: str
    state: str
    status: str
    current: Optional[int] = None
    total: Optional[int] = None
    progress_percent: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    execution_time: Optional[float] = None


class QueueStatusResponse(BaseModel):
    """Celery queue status"""
    active_tasks: int = Field(..., description="Number of active tasks")
    scheduled_tasks: int = Field(..., description="Number of scheduled tasks")
    reserved_tasks: int = Field(..., description="Number of reserved tasks")
    total_tasks: int = Field(..., description="Total tasks in queue")
    active_task_details: List[Dict[str, Any]] = Field(default_factory=list)


class EvaluationMetrics(BaseModel):
    """Evaluation metrics summary"""
    total_evaluations: int
    avg_groundedness: float
    avg_answer_relevance: float
    avg_context_relevance: float
    avg_overall_score: float
    avg_latency: float
    success_rate: float
    period_start: Optional[str] = None
    period_end: Optional[str] = None


class HistoricalMetrics(BaseModel):
    """Historical evaluation metrics"""
    date: str
    avg_groundedness: float
    avg_answer_relevance: float
    avg_context_relevance: float
    total_evaluations: int


class LatencyDistribution(BaseModel):
    """Latency distribution buckets"""
    bucket: str
    count: int
    percentage: float


class QualityBreakdown(BaseModel):
    """Quality score breakdown"""
    score_range: str
    count: int
    percentage: float


class DetailedEvaluation(BaseModel):
    """Detailed evaluation result"""
    evaluation_id: str
    timestamp: str
    input: Dict[str, Any]
    evaluation_results: Dict[str, Any]
    overall_score: float
    evaluation_time_seconds: float
    metadata: Optional[Dict[str, Any]] = None


class RAGTriadScores(BaseModel):
    """RAG Triad evaluation scores"""
    groundedness: float = Field(..., ge=0, le=1, description="Groundedness score (0-1)")
    answer_relevance: float = Field(..., ge=0, le=1, description="Answer relevance score (0-1)")
    context_relevance: float = Field(..., ge=0, le=1, description="Context relevance score (0-1)")
    overall_score: float = Field(..., ge=0, le=1, description="Overall score (0-1)")


class TestCaseRunRequest(BaseModel):
    """Test case execution request"""
    test_cases: List[Dict[str, Any]] = Field(..., description="List of test cases to execute")
    model_id: Optional[str] = Field(None, description="Model to use for testing")
    retrieval_config: Optional[Dict[str, Any]] = Field(None, description="Retrieval configuration")


class EvaluationResultsResponse(BaseModel):
    """Evaluation results listing"""
    results: List[Dict[str, Any]]
    total_count: int
    page: int
    page_size: int
