"""
Pydantic schemas for document management API requests and responses
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class DocumentInfo(BaseModel):
    """Single document information"""
    document_id: int
    filename: str
    content_type: str
    file_size: int
    upload_date: str
    user_id: Optional[str] = None
    status: str = "completed"
    error_message: Optional[str] = None
    chunking_method: Optional[str] = None
    embedding_model: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": 1,
                "filename": "document.pdf",
                "content_type": "application/pdf",
                "file_size": 1024000,
                "upload_date": "2025-10-17T10:30:00",
                "user_id": "admin",
                "status": "completed",
                "chunking_method": "general",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        }


class DocumentListResponse(BaseModel):
    """Response with list of documents"""
    documents: List[DocumentInfo]


class DocumentUploadResponse(BaseModel):
    """Response for document upload"""
    message: str
    file_id: str
    processed_files: List[str]
    failed_files: List[str]
    folder_path: Optional[str] = None
    error: Optional[str] = None


class DocumentReingestRequest(BaseModel):
    """Request to reingest documents for new embedding model"""
    embedding_model: str = Field(..., description="New embedding model to use")


class DocumentReingestSpecificRequest(BaseModel):
    """Request to reingest specific documents with custom config"""
    documents: List[Dict[str, Any]] = Field(..., description="List of documents with their configurations")
    
    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "document_id": 1,
                        "chunking_method": "general",
                        "chunking_config": {
                            "chunk_token_num": 1000,
                            "chunk_overlap": 200,
                            "delimiter": "\\n\\n|\\n|\\.|\\!|\\?"
                        }
                    }
                ]
            }
        }


class DocumentBulkDeleteRequest(BaseModel):
    """Request to bulk delete documents"""
    document_ids: List[int] = Field(..., description="List of document IDs to delete")
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_ids": [1, 2, 3]
            }
        }


class DocumentBulkDeleteResponse(BaseModel):
    """Response for bulk delete operation"""
    message: str
    results: Dict[str, Any]


class DocumentCheckDuplicateRequest(BaseModel):
    """Request to check if document is duplicate"""
    filename: str = Field(..., description="Filename to check")
    hash: str = Field(..., description="File hash (SHA256)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "filename": "document.pdf",
                "hash": "a1b2c3d4e5f6..."
            }
        }


class DocumentCheckDuplicateResponse(BaseModel):
    """Response for duplicate check"""
    exists: bool
    existing_file: Optional[Dict[str, Any]] = None


class DocumentChunksResponse(BaseModel):
    """Response with document chunks"""
    filename: str
    total_chunks: int
    chunks: List[Dict[str, Any]]
    embedding_model: str


class DocumentPreviewResponse(BaseModel):
    """Response for document preview"""
    type: str  # pdf, image, text, html, etc.
    content_type: str
    filename: str
    content: Optional[str] = None
    image_url: Optional[str] = None
    pdf_url: Optional[str] = None
    pages_info: Optional[List[Dict[str, Any]]] = None
    total_pages: Optional[int] = None
    truncated: bool = False


class MessageResponse(BaseModel):
    """Generic message response"""
    message: str
    count: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Operation completed successfully",
                "count": 5
            }
        }


class UploadProgressResponse(BaseModel):
    """Response for upload progress tracking"""
    file_id: str = Field(..., description="Unique upload identifier")
    filename: str = Field(..., description="Name of the file being uploaded")
    total_size: int = Field(..., description="Total file size in bytes")
    progress: int = Field(..., ge=0, le=100, description="Upload progress percentage (0-100)")
    status: str = Field(..., description="Upload status: initializing, uploading, processing, completed, failed")
    message: str = Field(..., description="Status message")
    created_at: str = Field(..., description="Upload start timestamp (ISO format)")
    updated_at: str = Field(..., description="Last update timestamp (ISO format)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "upload_1729330000.123",
                "filename": "document.pdf",
                "total_size": 1024000,
                "progress": 75,
                "status": "processing",
                "message": "Processing document...",
                "created_at": "2025-10-19T10:00:00.000Z",
                "updated_at": "2025-10-19T10:00:15.000Z"
            }
        }
