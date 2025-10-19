"""
Chunking Configuration API Schemas

Pydantic models for chunking configuration API requests and responses.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ChunkingConfigData(BaseModel):
    """Chunking configuration data model"""
    method: str = Field(..., description="Chunking method")
    chunk_token_num: int = Field(1000, ge=128, le=8192, description="Token threshold for chunk size")
    max_token: int = Field(8192, ge=128, le=16384, description="Maximum tokens per chunk")
    chunk_overlap: int = Field(200, ge=0, description="Overlap between chunks")
    delimiter: Optional[str] = Field(None, description="Text delimiters")
    layout_recognize: Optional[str] = Field("auto", description="Layout recognition method")
    preserve_formatting: bool = Field(True, description="Whether to preserve document formatting")
    extract_tables: bool = Field(True, description="Whether to extract tables separately")
    extract_images: bool = Field(False, description="Whether to extract images")
    language: Optional[str] = Field("auto", description="Document language detection")


class ChunkingMethodInfo(BaseModel):
    """Information about a chunking method"""
    name: str
    description: str
    supported_formats: List[str]


class ChunkingMethodsResponse(BaseModel):
    """Response containing available chunking methods"""
    methods: Dict[str, ChunkingMethodInfo]


class ChunkingConfigResponse(BaseModel):
    """Response containing chunking configuration"""
    config: Dict[str, Any]


class ChunkingConfigUpdateRequest(BaseModel):
    """Request to update chunking configuration"""
    config: ChunkingConfigData


class ChunkingConfigUpdateResponse(BaseModel):
    """Response after updating chunking configuration"""
    message: str
    warnings: List[str] = Field(default_factory=list)
    config: Dict[str, Any]


class OptimalChunkingMethodResponse(BaseModel):
    """Response containing optimal chunking method for a file type"""
    file_extension: str
    optimal_method: str
    available_methods: List[str]
