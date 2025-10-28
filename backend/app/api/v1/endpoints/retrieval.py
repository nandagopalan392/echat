"""
Retrieval Configuration API Endpoints

FastAPI routes for retrieval configuration management including
config get/update, reranker model listing, and download status.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import Dict, Any, Optional

from app.dependencies import get_current_user, get_document_repository
from app.api.v1.schemas.retrieval import (
    RetrievalConfigResponse,
    RetrievalConfigUpdateResponse,
    RerankerModelsResponse,
    RerankerModel,
    RerankerDownloadStatus
)
from app.services.retrieval_config_service import RetrievalConfigService
from app.db.repositories import DocumentRepository

logger = logging.getLogger(__name__)

router = APIRouter()


def get_retrieval_service(
    document_repository: DocumentRepository = Depends(get_document_repository)
) -> RetrievalConfigService:
    """Dependency to get retrieval config service instance"""
    return RetrievalConfigService(document_repository)


@router.get("/config", response_model=RetrievalConfigResponse)
async def get_retrieval_config(
    current_admin: dict = Depends(get_current_admin_user),
    service: RetrievalConfigService = Depends(get_retrieval_service)
):
    """
    Get current retrieval configuration.
    
    Returns the current retrieval configuration (user-specific if available,
    otherwise default configuration).
    """
    try:
        user_id = current_user.get('sub')
        config = service.get_config(user_id)
        
        return RetrievalConfigResponse(config=config)
        
    except Exception as e:
        logger.error(f"Error getting retrieval config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get retrieval configuration: {str(e)}"
        )


@router.put("/config", response_model=RetrievalConfigUpdateResponse)
async def update_retrieval_config(
    config_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_admin: dict = Depends(get_current_admin_user),
    service: RetrievalConfigService = Depends(get_retrieval_service)
):
    """
    Update retrieval configuration with auto-download for reranker models.
    
    Updates the user-specific retrieval configuration. If a reranker model
    is specified, the system will automatically check for availability and
    download it if needed.
    
    Args:
        config_data: New configuration data
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user
        service: Retrieval config service
    """
    try:
        user_id = current_user.get('sub')
        
        # Update configuration
        updated_config, warnings, download_result, message = await service.update_config(
            config_data,
            user_id
        )
        
        response_data = {
            "success": True,
            "config": updated_config,
            "warnings": warnings,
            "reranker_download": download_result
        }
        
        if message:
            response_data["message"] = message
        
        return RetrievalConfigUpdateResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error updating retrieval config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update retrieval configuration: {str(e)}"
        )


@router.get("/reranker-models", response_model=RerankerModelsResponse)
async def get_available_reranker_models(
    provider: Optional[str] = None,
    current_admin: dict = Depends(get_current_admin_user),
    service: RetrievalConfigService = Depends(get_retrieval_service)
):
    """
    Get list of available reranker models from Ollama and HuggingFace.
    
    Returns a list of all available reranker models including both locally
    installed models and models available for download.
    
    Args:
        provider: Optional filter by provider ('ollama' or 'huggingface')
    """
    try:
        models_data = await service.get_available_reranker_models(provider)
        
        # Convert to response model format
        models = [RerankerModel(**model) for model in models_data]
        
        return RerankerModelsResponse(models=models)
        
    except Exception as e:
        logger.error(f"Error getting reranker models: {e}")
        # Return basic fallback on error
        fallback_models = [
            RerankerModel(
                name="",
                display_name="None (Vector + Keyword)",
                description="Use weighted combination of vector similarity and keyword matching",
                provider="none"
            ),
            RerankerModel(
                name="BAAI/bge-reranker-v2-m3",
                display_name="BGE Reranker V2 M3",
                description="🤗 Multilingual BGE reranking model (error occurred)",
                provider="huggingface",
                is_local=False
            )
        ]
        return RerankerModelsResponse(models=fallback_models)


@router.get("/reranker-download-status", response_model=RerankerDownloadStatus)
async def get_reranker_download_status(
    model_name: str,
    current_admin: dict = Depends(get_current_admin_user),
    service: RetrievalConfigService = Depends(get_retrieval_service)
):
    """
    Get the download status of a reranker model.
    
    Returns the current download status for a specific reranker model,
    including whether it's downloading, completed, or failed.
    
    Args:
        model_name: Name of the reranker model
    """
    try:
        status_data = service.get_reranker_download_status(model_name)
        return RerankerDownloadStatus(**status_data)
        
    except Exception as e:
        logger.error(f"Error getting download status for {model_name}: {e}")
        return RerankerDownloadStatus(
            model_name=model_name,
            downloading=False,
            completed=False,
            message=f"Error getting status: {str(e)}"
        )
