"""
Document Management API Endpoints
Handles document upload, ingestion, deletion, and retrieval
"""
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from typing import List, Dict, Any, Optional
import logging
import os

from app.dependencies import get_current_user, get_current_admin_user
from app.db.models.user import User
from app.services.document_service import get_document_service
from app.api.v1.schemas.documents import (
    DocumentListResponse,
    DocumentUploadResponse,
    DocumentReingestRequest,
    DocumentReingestSpecificRequest,
    DocumentBulkDeleteRequest,
    DocumentBulkDeleteResponse,
    DocumentCheckDuplicateRequest,
    DocumentCheckDuplicateResponse,
    DocumentChunksResponse,
    DocumentPreviewResponse,
    MessageResponse,
    UploadProgressResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/admin/upload", response_model=DocumentUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    is_folder: str = Form(default="false"),
    folder_path: str = Form(default=""),
    chunking_method: str = Form(default="auto"),
    chunk_token_num: int = Form(default=1000),
    chunk_overlap: int = Form(default=200),
    delimiter: str = Form(default="\\n\\n|\\n|\\.|\\!|\\?"),
    max_token: int = Form(default=4096),
    layout_recognize: str = Form(default="auto"),
    preserve_formatting: bool = Form(default=True),
    extract_tables: bool = Form(default=True),
    extract_images: bool = Form(default=False),
    admin: User = Depends(get_current_admin_user)
):
    """
    Upload and process a file (admin only)
    
    Supports various document formats including PDF, DOCX, images, etc.
    Files are processed, chunked, and stored in vector database.
    
    Returns a file_id that can be used to track upload progress via
    GET /api/upload-progress/{file_id}
    """
    try:
        # Read file contents
        contents = await file.read()
        
        # Prepare chunking config
        chunking_config = {
            "chunk_token_num": chunk_token_num,
            "chunk_overlap": chunk_overlap,
            "delimiter": delimiter,
            "max_token": max_token,
            "layout_recognize": layout_recognize,
            "preserve_formatting": preserve_formatting,
            "extract_tables": extract_tables,
            "extract_images": extract_images
        }
        
        # Process upload (progress tracking handled inside service)
        doc_service = get_document_service()
        result = doc_service.process_upload(
            file_contents=contents,
            filename=file.filename,
            user_id=admin.username,
            is_folder=is_folder.lower() == "true",
            folder_path=folder_path,
            chunking_method=chunking_method,
            chunking_config=chunking_config
        )
        
        return DocumentUploadResponse(**result)
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(current_admin: User = Depends(get_current_admin_user)):
    """
    List all documents with their ingestion status
    
    Requires authentication to view documents.
    """
    try:
        doc_service = get_document_service()
        documents = doc_service.get_all_documents()
        return DocumentListResponse(documents=documents)
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/documents/ingested")
async def list_ingested_documents(current_admin: User = Depends(get_current_admin_user)):
    """
    List documents ingested for current embedding model
    
    Returns only documents that have been successfully ingested
    and are available for RAG queries.
    """
    try:
        doc_service = get_document_service()
        result = doc_service.get_ingested_documents()
        return result
    except Exception as e:
        logger.error(f"Error listing ingested documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/documents/reingest")
async def reingest_documents(
    request: DocumentReingestRequest,
    admin: User = Depends(get_current_admin_user)
):
    """
    Re-ingest all documents for a new embedding model (admin only)
    
    This is used when switching embedding models to rebuild the vector store.
    """
    try:
        doc_service = get_document_service()
        success = doc_service.reingest_documents(request.embedding_model)
        
        if success:
            return {
                "message": f"Documents re-ingested for model: {request.embedding_model}",
                "embedding_model": request.embedding_model
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Re-ingestion failed"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reingesting documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/documents/reingest-specific")
async def reingest_specific_documents(
    request: DocumentReingestSpecificRequest,
    admin: User = Depends(get_current_admin_user)
):
    """
    Re-ingest specific documents with per-document chunking configuration (admin only)
    
    Allows fine-grained control over how individual documents are processed.
    """
    try:
        if not request.documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No documents provided"
            )
        
        # Parse and validate each document configuration
        from app.config.chunking import ChunkingMethod, ChunkingConfig
        
        parsed_documents = []
        for doc_config in request.documents:
            document_id = doc_config.get('document_id')
            chunking_method_str = doc_config.get('chunking_method')
            chunking_config_dict = doc_config.get('chunking_config', {})
            
            if not document_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Document ID is required for each document"
                )
            
            # Parse chunking method
            chunking_method = None
            if chunking_method_str:
                try:
                    chunking_method = ChunkingMethod(chunking_method_str)
                except ValueError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid chunking method: {chunking_method_str}"
                    )
            
            # Parse chunking config
            chunking_config = None
            if chunking_config_dict:
                try:
                    chunking_config = ChunkingConfig.from_dict(chunking_config_dict)
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid chunking config: {str(e)}"
                    )
            
            parsed_documents.append({
                'document_id': document_id,
                'chunking_method': chunking_method,
                'chunking_config': chunking_config
            })
        
        # Perform reingestion
        doc_service = get_document_service()
        results = doc_service.reingest_specific_documents(parsed_documents)
        
        return {
            "message": f"Reingestion completed: {results['successful']}/{results['total']} successful",
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in specific document reingestion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/documents/{document_id}", response_model=MessageResponse)
async def delete_document(
    document_id: int,
    admin: User = Depends(get_current_admin_user)
):
    """
    Delete a document by ID (admin only)
    
    Removes the document from storage and vector database.
    """
    try:
        doc_service = get_document_service()
        success = doc_service.delete_document(document_id)
        
        if success:
            return MessageResponse(message=f"Document {document_id} deleted successfully")
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/documents/bulk-delete", response_model=DocumentBulkDeleteResponse)
async def bulk_delete_documents(
    request: DocumentBulkDeleteRequest,
    admin: User = Depends(get_current_admin_user)
):
    """
    Bulk delete multiple documents from storage (admin only)
    
    Efficiently delete multiple documents in a single operation.
    """
    try:
        if not request.document_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No document IDs provided"
            )
        
        doc_service = get_document_service()
        results = doc_service.bulk_delete_documents(request.document_ids)
        
        return DocumentBulkDeleteResponse(
            message=f"Bulk deletion completed: {results['successful']} successful, {results['failed']} failed",
            results=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk delete: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/admin/clear-all-documents", response_model=MessageResponse)
async def clear_all_documents(
    admin: User = Depends(get_current_admin_user)
):
    """
    Clear all documents from storage (admin only)
    
    WARNING: This will delete ALL documents in the system!
    """
    try:
        doc_service = get_document_service()
        success = doc_service.clear_all_documents()
        
        if success:
            return MessageResponse(message="All documents cleared successfully")
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to clear documents"
            )
    except Exception as e:
        logger.error(f"Error clearing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/admin/cleanup-orphaned", response_model=MessageResponse)
async def cleanup_orphaned_documents(
    admin: User = Depends(get_current_admin_user)
):
    """
    Cleanup orphaned documents (admin only)
    
    Removes documents that exist in storage but not in vector database,
    or vice versa.
    """
    try:
        doc_service = get_document_service()
        count = doc_service.cleanup_orphaned_documents()
        return MessageResponse(
            message=f"Cleaned up {count} orphaned documents",
            count=count
        )
    except Exception as e:
        logger.error(f"Error cleaning up orphaned documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/files/check-duplicate", response_model=DocumentCheckDuplicateResponse)
async def check_file_duplicate(
    request: DocumentCheckDuplicateRequest,
    current_admin: User = Depends(get_current_admin_user)
):
    """
    Check if a file with the same filename and hash already exists
    
    Useful for preventing duplicate uploads.
    """
    try:
        doc_service = get_document_service()
        result = doc_service.check_duplicate(request.filename, request.hash)
        return DocumentCheckDuplicateResponse(**result)
    except Exception as e:
        logger.error(f"Error checking file duplicate: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/files/{filename}", response_model=MessageResponse)
async def delete_file_by_filename(
    filename: str,
    admin: User = Depends(get_current_admin_user)
):
    """
    Delete a document by filename (admin only)
    
    Alternative to delete by ID when you only have the filename.
    """
    try:
        doc_service = get_document_service()
        success = doc_service.delete_by_filename(filename)
        
        if success:
            return MessageResponse(message=f"File {filename} deleted successfully")
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/files/{filename}/chunks", response_model=DocumentChunksResponse)
async def get_document_chunks(
    filename: str,
    current_admin: User = Depends(get_current_admin_user)
):
    """
    Get all chunks for a document by filename
    
    Returns the chunked content stored in the vector database.
    """
    try:
        doc_service = get_document_service()
        result = doc_service.get_document_chunks(filename)
        return DocumentChunksResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting document chunks: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/documents/{document_id}/image")
async def get_document_image(
    document_id: int,
    background_tasks: BackgroundTasks,
    current_admin: User = Depends(get_current_admin_user)
):
    """
    Serve document image if it's an image file
    
    Returns the raw image file for display in the UI.
    """
    try:
        from app.services.storage_service import get_storage_service
        storage = get_storage_service()
        
        # Get document info first
        doc_info = storage._get_document_by_id(document_id)
        if not doc_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Check if it's an image file
        is_image_by_content_type = doc_info['content_type'].startswith('image/')
        is_image_by_extension = doc_info['filename'].lower().endswith(
            ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg')
        )
        
        if not (is_image_by_content_type or is_image_by_extension):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document is not an image"
            )
        
        # Get the document content from MinIO
        temp_file_path = storage.get_document_file(document_id)
        
        # Schedule cleanup
        background_tasks.add_task(os.remove, temp_file_path)
        
        # Determine correct content type
        response_content_type = doc_info['content_type']
        if not response_content_type.startswith('image/'):
            import mimetypes
            guessed_type, _ = mimetypes.guess_type(doc_info['filename'])
            if guessed_type and guessed_type.startswith('image/'):
                response_content_type = guessed_type
        
        # Return the image file
        return FileResponse(
            temp_file_path,
            media_type=response_content_type,
            filename=doc_info['filename']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving document image {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/documents/{document_id}/preview", response_model=DocumentPreviewResponse)
async def get_document_preview(
    document_id: int,
    background_tasks: BackgroundTasks,
    current_admin: User = Depends(get_current_admin_user)
):
    """
    Get document preview content for side-by-side viewing
    
    Extracts and returns preview content based on document type.
    """
    try:
        from app.services.storage_service import get_storage_service
        storage = get_storage_service()
        
        # Get document info first
        doc_info = storage._get_document_by_id(document_id)
        if not doc_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # For images, return image metadata
        is_image_by_content_type = doc_info['content_type'].startswith('image/')
        is_image_by_extension = doc_info['filename'].lower().endswith(
            ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg')
        )
        
        if is_image_by_content_type or is_image_by_extension:
            return DocumentPreviewResponse(
                type="image",
                content_type=doc_info['content_type'],
                filename=doc_info['filename'],
                image_url=f"/api/documents/{document_id}/image"
            )
        
        # Get the document content from MinIO
        temp_file_path = storage.get_document_file(document_id)
        
        try:
            # Extract text content based on file type
            content = ""
            content_type = doc_info['content_type'].lower()
            
            if content_type == 'text/plain' or doc_info['filename'].endswith('.txt'):
                with open(temp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                return DocumentPreviewResponse(
                    type="text",
                    content_type=doc_info['content_type'],
                    filename=doc_info['filename'],
                    content=content[:50000],  # Limit to 50KB
                    truncated=len(content) >= 50000
                )
            
            elif content_type == 'application/pdf' or doc_info['filename'].endswith('.pdf'):
                import PyPDF2
                pages_info = []
                with open(temp_file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    content = ""
                    for page_num in range(min(20, len(pdf_reader.pages))):  # Limit to first 20 pages
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        content += f"--- Page {page_num + 1} ---\n"
                        content += page_text + "\n\n"
                        
                        pages_info.append({
                            "page_number": page_num + 1,
                            "text": page_text,
                            "text_length": len(page_text)
                        })
                
                return DocumentPreviewResponse(
                    type="pdf",
                    content_type=doc_info['content_type'],
                    filename=doc_info['filename'],
                    content=content,
                    pdf_url=f"/api/documents/{document_id}/raw",
                    pages_info=pages_info,
                    total_pages=len(pdf_reader.pages),
                    truncated=len(content) >= 50000
                )
            
            else:
                # For other types, return basic info
                return DocumentPreviewResponse(
                    type="unknown",
                    content_type=doc_info['content_type'],
                    filename=doc_info['filename'],
                    content="Preview not available for this file type"
                )
                
        finally:
            # Schedule cleanup
            background_tasks.add_task(os.remove, temp_file_path)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document preview {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/documents/{document_id}/raw")
async def get_document_raw(
    document_id: int,
    background_tasks: BackgroundTasks,
    current_admin: User = Depends(get_current_admin_user)
):
    """
    Get raw document file for download or viewing
    
    Returns the original uploaded file.
    """
    try:
        from app.services.storage_service import get_storage_service
        storage = get_storage_service()
        
        # Get document info
        doc_info = storage._get_document_by_id(document_id)
        if not doc_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Get the document file from MinIO
        temp_file_path = storage.get_document_file(document_id)
        
        # Schedule cleanup
        background_tasks.add_task(os.remove, temp_file_path)
        
        # Return the file
        return FileResponse(
            temp_file_path,
            media_type=doc_info['content_type'],
            filename=doc_info['filename']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving raw document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/upload-progress/{file_id}", response_model=UploadProgressResponse)
async def get_upload_progress(
    file_id: str,
    current_admin: User = Depends(get_current_admin_user)
):
    """
    Get real-time upload progress for a specific file
    
    Track the progress of an ongoing file upload by its unique file_id.
    Returns current progress percentage, status, and timing information.
    
    Args:
        file_id: Unique identifier for the upload (returned from upload endpoint)
        current_user: Authenticated user
        
    Returns:
        UploadProgressResponse with progress details
        
    Raises:
        404: Upload not found or expired
        
    Example:
        GET /api/upload-progress/upload_1729330000.123
        
        Response:
        {
            "file_id": "upload_1729330000.123",
            "filename": "document.pdf",
            "total_size": 1024000,
            "progress": 75,
            "status": "processing",
            "message": "Processing document...",
            "created_at": "2025-10-19T10:00:00.000Z",
            "updated_at": "2025-10-19T10:00:15.000Z"
        }
    """
    try:
        doc_service = get_document_service()
        
        # Clean up stale uploads periodically
        doc_service.cleanup_stale_uploads()
        
        # Get progress data
        progress_data = doc_service.get_upload_progress(file_id)
        
        if progress_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Upload progress not found for file_id: {file_id}. Upload may have completed or expired."
            )
        
        return UploadProgressResponse(
            file_id=file_id,
            **progress_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting upload progress for {file_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
