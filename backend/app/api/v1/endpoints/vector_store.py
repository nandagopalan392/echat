"""
Vector Store Endpoints - API routes for vector store operations

This module provides REST API endpoints for managing vector stores including:
- Getting statistics
- Clearing vector stores (admin only)
- Debugging collection information
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import logging

from app.services.vector_store_service import get_vector_store_service, VectorStoreService
from app.dependencies import get_current_user, get_current_admin_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["vector-store"])


@router.get("/vector-store/stats")
async def get_vector_store_stats(
    current_admin: dict = Depends(get_current_admin_user),
    vector_store_service: VectorStoreService = Depends(get_vector_store_service)
) -> Dict[str, Any]:
    """
    Get detailed statistics about the vector store and collections.
    
    Returns comprehensive statistics including:
    - Collection information (names, counts)
    - Document counts and storage metrics
    - Embedding information
    - Performance statistics
    
    Args:
        current_user: Authenticated user (from JWT token)
        vector_store_service: Vector store service dependency
        
    Returns:
        Dictionary containing:
        - success: Boolean indicating success
        - stats: Detailed vector store statistics
        
    Raises:
        HTTPException 401: If user is not authenticated
        HTTPException 503: If vector store is not available
        HTTPException 500: If there's an error retrieving stats
        
    Example response:
    ```json
    {
        "success": true,
        "stats": {
            "total_documents": 150,
            "total_collections": 2,
            "storage_size": "45.2 MB",
            "collections": [...]
        }
    }
    ```
    """
    logger.info(f"User '{current_user.get('username')}' requested vector store stats")
    return await vector_store_service.get_vector_store_stats()


@router.delete("/vector-store/clear")
async def clear_vector_store(
    admin: dict = Depends(get_current_admin_user),
    vector_store_service: VectorStoreService = Depends(get_vector_store_service)
) -> Dict[str, Any]:
    """
    Clear the entire vector store - admin only.
    
    This is a DESTRUCTIVE operation that removes all vectors and documents
    from the vector store. Use with extreme caution.
    
    Only administrators can perform this operation.
    
    Args:
        admin: Authenticated admin user (from JWT token)
        vector_store_service: Vector store service dependency
        
    Returns:
        Dictionary containing:
        - success: Boolean indicating success
        - message: Confirmation message
        
    Raises:
        HTTPException 401: If user is not authenticated
        HTTPException 403: If user is not an administrator
        HTTPException 500: If clearing fails
        
    Example response:
    ```json
    {
        "success": true,
        "message": "Vector store cleared successfully"
    }
    ```
    """
    logger.warning(
        f"Admin user '{admin.get('username')}' is clearing the entire vector store"
    )
    result = await vector_store_service.clear_vector_store()
    logger.info(f"Vector store cleared by admin '{admin.get('username')}'")
    return result


@router.get("/debug/collection-info")
async def get_collection_debug_info(
    current_admin: dict = Depends(get_current_admin_user),
    vector_store_service: VectorStoreService = Depends(get_vector_store_service)
) -> Dict[str, Any]:
    """
    Debug endpoint to inspect collection structure and contents.
    
    Provides detailed diagnostic information about the vector store collection
    including sample documents and metadata. Useful for troubleshooting
    and monitoring.
    
    Args:
        current_user: Authenticated user (from JWT token)
        vector_store_service: Vector store service dependency
        
    Returns:
        Dictionary containing:
        - collection_name: Name of the collection
        - total_documents: Total number of documents
        - sample_metadata: List of sample metadata entries
        - sample_document_previews: List of document preview strings
        
    Raises:
        HTTPException 401: If user is not authenticated
        HTTPException 404: If collection is not found
        HTTPException 503: If vector store is not available
        HTTPException 500: If there's an error retrieving debug info
        
    Example response:
    ```json
    {
        "collection_name": "chatpdf_collection",
        "total_documents": 150,
        "sample_metadata": [
            {"source": "document1.pdf", "page": 1},
            {"source": "document2.pdf", "page": 1}
        ],
        "sample_document_previews": [
            "This is the first document content...",
            "This is the second document content..."
        ]
    }
    ```
    """
    logger.info(
        f"User '{current_user.get('username')}' requested collection debug info"
    )
    return await vector_store_service.get_collection_debug_info()


@router.get("/vector-store/collections")
async def get_collection_metadata(
    current_admin: dict = Depends(get_current_admin_user),
    vector_store_service: VectorStoreService = Depends(get_vector_store_service)
) -> Dict[str, Any]:
    """
    Get metadata about all vector store collections.
    
    Returns information about all collections in the vector store including
    their names, document counts, and metadata.
    
    Args:
        current_user: Authenticated user (from JWT token)
        vector_store_service: Vector store service dependency
        
    Returns:
        Dictionary containing:
        - success: Boolean indicating success
        - total_collections: Number of collections
        - collections: List of collection information
        
    Raises:
        HTTPException 401: If user is not authenticated
        HTTPException 503: If vector store is not available
        HTTPException 500: If there's an error retrieving metadata
        
    Example response:
    ```json
    {
        "success": true,
        "total_collections": 2,
        "collections": [
            {
                "name": "collection1",
                "count": 100,
                "metadata": {...}
            },
            {
                "name": "collection2",
                "count": 50,
                "metadata": {...}
            }
        ]
    }
    ```
    """
    logger.info(
        f"User '{current_user.get('username')}' requested collection metadata"
    )
    return await vector_store_service.get_collection_metadata()
