"""
Retrieval Configuration API Schemas

Pydantic models for retrieval configuration API requests and responses.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class RetrievalConfigData(BaseModel):
    """Retrieval configuration data model"""
    similarity_threshold: float = Field(0.2, ge=0.0, le=1.0, description="Minimum similarity score for retrieving chunks")
    keyword_similarity_weight: float = Field(0.7, ge=0.0, le=1.0, description="Weight of keyword similarity")
    reranker_enabled: bool = Field(False, description="Whether to use reranker model")
    reranker_model: str = Field("", description="Name of the reranker model")
    reranker_provider: str = Field("ollama", description="Provider for the reranker model (ollama, huggingface)")
    max_chunks: int = Field(5, ge=1, le=20, description="Maximum number of chunks to retrieve")
    search_type: str = Field("similarity", description="Type of search (similarity, mmr, hybrid)")
    auto_merging_enabled: bool = Field(False, description="Whether to enable auto merging retrieval")
    auto_merging_similarity_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Similarity threshold for merging chunks")


class RetrievalConfigResponse(BaseModel):
    """Response containing retrieval configuration"""
    config: Dict[str, Any]


class RetrievalConfigUpdateRequest(BaseModel):
    """Request to update retrieval configuration"""
    config: Dict[str, Any]


class RetrievalConfigUpdateResponse(BaseModel):
    """Response after updating retrieval configuration"""
    success: bool
    config: Dict[str, Any]
    warnings: List[str] = Field(default_factory=list)
    reranker_download: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class RerankerModel(BaseModel):
    """Reranker model information"""
    name: str
    display_name: str
    description: str
    provider: str
    is_local: Optional[bool] = None
    size: Optional[str] = None


class RerankerModelsResponse(BaseModel):
    """Response containing available reranker models"""
    models: List[RerankerModel]


class RerankerDownloadStatus(BaseModel):
    """Status of reranker model download"""
    model_name: str
    downloading: bool
    completed: bool
    message: str


class DownloadResult(BaseModel):
    """Result of a download operation"""
    success: bool
    downloaded: bool
    message: str
    downloading: Optional[bool] = None
