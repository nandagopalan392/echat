"""
Chunking Configuration API Endpoints

FastAPI routes for chunking configuration management including
method information, config get/update, and optimal method selection.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any

from app.dependencies import get_current_user, get_document_repository
from app.api.v1.schemas.chunking import (
    ChunkingMethodsResponse,
    ChunkingMethodInfo,
    ChunkingConfigResponse,
    ChunkingConfigUpdateRequest,
    ChunkingConfigUpdateResponse,
    OptimalChunkingMethodResponse
)
from app.services.chunking_config_service import ChunkingConfigService
from app.db.repositories import DocumentRepository

logger = logging.getLogger(__name__)

router = APIRouter()


def get_chunking_service(
    document_repository: DocumentRepository = Depends(get_document_repository)
) -> ChunkingConfigService:
    """Dependency to get chunking config service instance"""
    return ChunkingConfigService(document_repository)


@router.get("/methods", response_model=ChunkingMethodsResponse)
async def get_chunking_methods(
    current_admin: dict = Depends(get_current_admin_user),
    service: ChunkingConfigService = Depends(get_chunking_service)
):
    """
    Get available chunking methods and their supported file formats.
    
    Returns information about all available chunking methods including:
    - Method name
    - Description
    - Supported file formats
    """
    try:
        methods_data = service.get_available_methods()
        
        # Convert to response model format
        methods = {
            name: ChunkingMethodInfo(**info)
            for name, info in methods_data.items()
        }
        
        return ChunkingMethodsResponse(methods=methods)
        
    except Exception as e:
        logger.error(f"Error getting chunking methods: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get chunking methods: {str(e)}"
        )


@router.get("/config/{method}", response_model=ChunkingConfigResponse)
async def get_chunking_config(
    method: str,
    current_admin: dict = Depends(get_current_admin_user),
    service: ChunkingConfigService = Depends(get_chunking_service)
):
    """
    Get chunking configuration for a specific method.
    
    Returns the current configuration (user-specific if available,
    otherwise default configuration) for the specified chunking method.
    
    Args:
        method: Chunking method name (e.g., 'general', 'qa', 'table')
    """
    try:
        user_id = current_user.get('sub')
        config = service.get_config(method, user_id)
        
        return ChunkingConfigResponse(config=config)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting chunking config for {method}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get chunking configuration: {str(e)}"
        )


@router.post("/config/{method}", response_model=ChunkingConfigUpdateResponse)
async def update_chunking_config(
    method: str,
    config_data: Dict[str, Any],
    current_admin: dict = Depends(get_current_admin_user),
    service: ChunkingConfigService = Depends(get_chunking_service)
):
    """
    Update chunking configuration for a specific method.
    
    Updates the user-specific configuration for the specified chunking method.
    The configuration is validated before saving, and any warnings are returned.
    
    Args:
        method: Chunking method name (e.g., 'general', 'qa', 'table')
        config_data: New configuration data
    """
    try:
        user_id = current_user.get('sub')
        
        # Update configuration
        updated_config, warnings = service.update_config(method, config_data, user_id)
        
        return ChunkingConfigUpdateResponse(
            message="Configuration updated successfully",
            warnings=warnings,
            config=updated_config
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error updating chunking config for {method}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update chunking configuration: {str(e)}"
        )


@router.get("/optimal/{file_extension}", response_model=OptimalChunkingMethodResponse)
async def get_optimal_chunking_method(
    file_extension: str,
    current_admin: dict = Depends(get_current_admin_user),
    service: ChunkingConfigService = Depends(get_chunking_service)
):
    """
    Get optimal chunking method for a file extension.
    
    Returns the recommended chunking method and all available methods
    for the specified file type.
    
    Args:
        file_extension: File extension (with or without dot, e.g., 'pdf' or '.pdf')
    """
    try:
        optimal_method, available_methods = service.get_optimal_method(file_extension)
        
        # Remove dot for consistent response
        clean_extension = file_extension.lstrip('.')
        
        return OptimalChunkingMethodResponse(
            file_extension=clean_extension,
            optimal_method=optimal_method,
            available_methods=available_methods
        )
        
    except Exception as e:
        logger.error(f"Error getting optimal chunking method for {file_extension}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get optimal chunking method: {str(e)}"
        )
