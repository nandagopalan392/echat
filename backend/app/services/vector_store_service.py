"""
Vector Store Service - Business logic for vector store operations

This service handles all vector store and RAG-related operations including
statistics, clearing, and debugging collection information.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import HTTPException

from app.core.rag import get_rag_engine, RAGEngine

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Service for managing vector store operations.
    
    Handles operations like:
    - Getting vector store statistics
    - Clearing the vector store
    - Debugging collection information
    - Managing RAG instance
    """
    
    def __init__(self):
        """Initialize the vector store service."""
        self.logger = logging.getLogger(__name__)
    
    def _get_rag_engine(self) -> RAGEngine:
        """
        Get the RAG engine instance.
        
        Returns:
            RAGEngine instance
            
        Raises:
            HTTPException: If RAG engine is not available
        """
        try:
            rag_engine = get_rag_engine()
            
            if not rag_engine:
                raise HTTPException(
                    status_code=503, 
                    detail="RAG engine not available"
                )
            
            return rag_engine
            
        except ImportError as e:
            self.logger.error(f"Failed to import RAG engine: {e}")
            raise HTTPException(
                status_code=500,
                detail="RAG engine module not available"
            )
        except Exception as e:
            self.logger.error(f"Failed to get RAG engine: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Failed to initialize RAG engine: {str(e)}"
            )
    
    async def get_vector_store_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the vector store and collections.
        
        Returns:
            Dictionary containing vector store statistics:
            - Collection information
            - Document counts
            - Storage metrics
            - Performance statistics
            
        Raises:
            HTTPException: If unable to retrieve stats
        """
        try:
            rag_engine = self._get_rag_engine()
            stats = rag_engine.get_vector_store_stats()
            
            self.logger.info("Retrieved vector store statistics successfully")
            
            return {
                "success": True,
                "stats": stats
            }
            
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            error_msg = f"Error getting vector store stats: {str(e)}"
            self.logger.error(error_msg)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get vector store stats: {str(e)}"
            )
    
    async def clear_vector_store(self) -> Dict[str, Any]:
        """
        Clear the entire vector store.
        
        This is a destructive operation that removes all vectors and documents
        from the vector store. Should only be called by admin users.
        
        Returns:
            Dictionary with success status and message
            
        Raises:
            HTTPException: If clearing fails
        """
        try:
            rag_engine = self._get_rag_engine()
            success = rag_engine.clear_vector_store()
            
            if success:
                self.logger.info("Vector store cleared successfully")
                return {
                    "success": True,
                    "message": "Vector store cleared successfully"
                }
            else:
                error_msg = "Failed to clear vector store - operation returned False"
                self.logger.error(error_msg)
                raise HTTPException(
                    status_code=500,
                    detail="Failed to clear vector store"
                )
            
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            error_msg = f"Error clearing vector store: {str(e)}"
            self.logger.error(error_msg)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to clear vector store: {str(e)}"
            )
    
    async def get_collection_debug_info(self) -> Dict[str, Any]:
        """
        Get debug information about the vector store collection.
        
        This endpoint provides detailed diagnostic information including:
        - Collection name and configuration
        - Total document count
        - Sample metadata
        - Sample document previews
        
        Returns:
            Dictionary containing debug information
            
        Raises:
            HTTPException: If unable to retrieve debug info
        """
        try:
            rag_engine = self._get_rag_engine()
            
            # Check if vector store is available
            if not rag_engine.vector_store:
                raise HTTPException(
                    status_code=503,
                    detail="Vector store not available"
                )
            
            # Get Chroma client and collection
            chroma_client = rag_engine.vector_store._client
            collection_name = rag_engine._get_collection_name()
            
            try:
                collection = chroma_client.get_collection(collection_name)
                
                # Get basic collection info
                count = collection.count()
                
                # Get sample documents (limit to 10 for debugging)
                sample_results = collection.get(
                    limit=10,
                    include=["metadatas", "documents"]
                )
                
                # Format sample documents with preview (first 100 chars)
                sample_previews = [
                    doc[:100] + "..." if len(doc) > 100 else doc
                    for doc in sample_results.get('documents', [])
                ]
                
                debug_info = {
                    "collection_name": collection_name,
                    "total_documents": count,
                    "sample_metadata": sample_results.get('metadatas', []),
                    "sample_document_previews": sample_previews
                }
                
                self.logger.info(
                    f"Retrieved debug info for collection '{collection_name}' "
                    f"with {count} documents"
                )
                
                return debug_info
                
            except Exception as e:
                error_msg = f"Collection error: {str(e)}"
                self.logger.error(error_msg)
                raise HTTPException(
                    status_code=404,
                    detail=error_msg
                )
            
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            error_msg = f"Error getting collection debug info: {str(e)}"
            self.logger.error(error_msg)
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
    
    async def get_collection_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the vector store collections.
        
        Returns:
            Dictionary with collection metadata including:
            - Collection names
            - Document counts
            - Embedding dimensions
            - Storage information
            
        Raises:
            HTTPException: If unable to retrieve metadata
        """
        try:
            rag_engine = self._get_rag_engine()
            
            if not rag_engine.vector_store:
                raise HTTPException(
                    status_code=503,
                    detail="Vector store not available"
                )
            
            chroma_client = rag_engine.vector_store._client
            collections = chroma_client.list_collections()
            
            collection_info = []
            for collection in collections:
                count = collection.count()
                collection_info.append({
                    "name": collection.name,
                    "count": count,
                    "metadata": collection.metadata if hasattr(collection, 'metadata') else {}
                })
            
            return {
                "success": True,
                "total_collections": len(collections),
                "collections": collection_info
            }
            
        except HTTPException:
            raise
        except Exception as e:
            error_msg = f"Error getting collection metadata: {str(e)}"
            self.logger.error(error_msg)
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )


# Global service instance
_vector_store_service: Optional[VectorStoreService] = None


def get_vector_store_service() -> VectorStoreService:
    """
    Get or create the vector store service instance.
    
    Returns:
        VectorStoreService instance
    """
    global _vector_store_service
    
    if _vector_store_service is None:
        _vector_store_service = VectorStoreService()
    
    return _vector_store_service
