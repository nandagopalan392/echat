"""
WebSocket message schemas for real-time updates
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime


class WebSocketMessage(BaseModel):
    """Base WebSocket message schema"""
    type: str = Field(..., description="Message type")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ConnectionMessage(WebSocketMessage):
    """Connection confirmation message"""
    message: str = Field(..., description="Connection message")
    connection_info: Dict[str, Any] = Field(default_factory=dict)
    current_status: Optional[Dict[str, Any]] = None


class PingMessage(WebSocketMessage):
    """Ping message for connection health"""
    connection_health: Dict[str, Any] = Field(default_factory=dict)


class CompletionMessage(WebSocketMessage):
    """Task completion message"""
    message: str = Field(..., description="Completion message")
    final_status: str = Field(..., description="Final task status")


class ErrorMessage(WebSocketMessage):
    """Error message"""
    message: str = Field(..., description="Error message")


# Fine-tuning WebSocket messages
class ExperimentUpdateMessage(WebSocketMessage):
    """Fine-tuning experiment update message"""
    experiment_id: str = Field(..., description="Experiment ID")
    status: str = Field(..., description="Experiment status")
    metrics: Dict[str, Any] = Field(default_factory=dict)
    latest_logs: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    training_metrics: Optional[Dict[str, Any]] = None


# Evaluation WebSocket messages
class EvaluationUpdateMessage(WebSocketMessage):
    """Evaluation task update message"""
    task_id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Task status")
    progress: Optional[float] = Field(None, ge=0.0, le=100.0)
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    message_id: Optional[int] = None
    received_at: Optional[str] = None
