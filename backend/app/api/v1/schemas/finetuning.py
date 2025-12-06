"""
Fine-tuning API Schemas
Pydantic models for fine-tuning endpoints request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class ExperimentCreateRequest(BaseModel):
    """Create fine-tuning experiment request"""
    name: str = Field(..., description="Experiment name")
    description: str = Field("", description="Experiment description")
    model_name: str = Field(..., description="Base model to fine-tune")
    dataset_id: str = Field(..., description="Dataset ID to use for training")
    learning_rate: float = Field(0.0001, ge=0.00001, le=0.01, description="Learning rate")
    num_epochs: int = Field(3, ge=1, le=100, description="Number of training epochs")
    batch_size: int = Field(4, ge=1, le=64, description="Training batch size")
    lora_r: int = Field(16, ge=4, le=128, description="LoRA rank")
    lora_alpha: int = Field(32, ge=8, le=256, description="LoRA alpha")
    lora_dropout: float = Field(0.1, ge=0.0, le=0.5, description="LoRA dropout")
    target_modules: str = Field("q_proj,v_proj", description="Comma-separated target modules")
    max_seq_length: int = Field(512, ge=128, le=4096, description="Maximum sequence length")
    warmup_ratio: float = Field(0.03, ge=0.0, le=0.5, description="Warmup ratio")
    weight_decay: float = Field(0.01, ge=0.0, le=0.1, description="Weight decay")
    gradient_accumulation_steps: int = Field(1, ge=1, le=16, description="Gradient accumulation steps")
    logging_steps: int = Field(10, ge=1, le=100, description="Logging frequency")
    save_steps: int = Field(500, ge=10, le=5000, description="Save checkpoint frequency")
    eval_steps: int = Field(100, ge=10, le=1000, description="Evaluation frequency")
    save_total_limit: int = Field(2, ge=1, le=10, description="Max checkpoints to keep")
    load_best_model_at_end: bool = Field(True, description="Load best model at end")
    metric_for_best_model: str = Field("eval_loss", description="Metric for best model")
    greater_is_better: bool = Field(False, description="Whether higher metric is better")
    evaluation_strategy: str = Field("steps", description="Evaluation strategy")
    save_strategy: str = Field("steps", description="Save strategy")


class ExperimentResponse(BaseModel):
    """Experiment creation response"""
    experiment: Dict[str, Any]
    status: str
    training_status: Optional[str] = None


class ExperimentListResponse(BaseModel):
    """List experiments response"""
    experiments: List[Dict[str, Any]]


class ExperimentDetailsResponse(BaseModel):
    """Experiment details with logs"""
    id: str
    name: str
    description: str
    base_model: str
    status: str
    user_id: str
    created_at: str
    training_logs: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None


class TrainingLogsResponse(BaseModel):
    """Training logs response"""
    logs: List[Dict[str, Any]]


class TrainingMetricsResponse(BaseModel):
    """Live training metrics response"""
    experiment_id: str
    progress: Dict[str, Any]
    metrics: Dict[str, Any]
    system: Dict[str, Any]
    training_logs: List[Dict[str, Any]]
    training_completed: bool


class DatasetListResponse(BaseModel):
    """List datasets response"""
    datasets: List[Dict[str, Any]]


class DatasetCreateRequest(BaseModel):
    """Create dataset from documents request"""
    name: str = Field(..., description="Dataset name")
    description: str = Field("", description="Dataset description")
    document_ids: List[str] = Field(..., min_length=1, description="Document IDs to process")
    questions_per_doc: int = Field(5, ge=1, le=20, description="Questions per document")
    model_name: str = Field("gemma2:2b", description="Model for question generation")


class DatasetCreateResponse(BaseModel):
    """Dataset creation task response"""
    success: bool
    task_id: str
    message: str
    total_documents: int
    websocket_url: Optional[str] = None


class DatasetDetailsResponse(BaseModel):
    """Dataset details with samples"""
    id: str
    name: str
    description: str
    num_samples: int
    file_size: int
    format: str
    created_at: str
    samples: List[Dict[str, Any]]


class DatasetValidationResult(BaseModel):
    """Dataset validation result"""
    valid: bool
    num_samples: int
    sample_preview: List[Dict[str, Any]]
    errors: Optional[List[str]] = None


class ModelInfo(BaseModel):
    """Model information"""
    name: str
    description: str
    size: str
    type: str


class ModelListResponse(BaseModel):
    """Available models response"""
    models: List[ModelInfo]


class DeleteResponse(BaseModel):
    """Generic delete response"""
    status: str
    message: Optional[str] = None


class StartTrainingResponse(BaseModel):
    """Start training response"""
    status: str
    experiment_id: str
    task_id: Optional[str] = None  # Celery task ID for WebSocket tracking
