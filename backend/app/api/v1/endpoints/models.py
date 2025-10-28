"""
Model Management Endpoints (Refactored)

This module provides a thin HTTP layer for model management endpoints,
delegating all business logic to the ModelService.

Clean Architecture Pattern:
- Endpoints: HTTP layer (this file) - handles requests/responses and authentication
- Services: Business logic layer (ModelService) - handles model operations
- Core/DB: Data layer - handles persistence
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends

# Import dependencies
from app.dependencies import get_current_user
from app.services import ModelService

# Initialize logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/models", tags=["models"])

# Initialize service (will be dependency-injected in production)
model_service = ModelService()


@router.get("/status")
async def get_model_status(current_admin: dict = Depends(get_current_admin_user)):
    """
    Get status of models in Ollama.
    
    Returns:
        Dict with current model status and availability
    """
    try:
        return await model_service.get_model_status()
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")


@router.get("/available")
async def get_available_models(current_admin: dict = Depends(get_current_admin_user)):
    """
    Get list of available models from both local Ollama and Ollama library.
    
    Returns:
        Dict with categorized models and metadata
    """
    try:
        return await model_service.get_available_models()
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get available models: {str(e)}")


@router.get("/current")
async def get_current_models(current_admin: dict = Depends(get_current_admin_user)):
    """
    Get currently configured models and their parameters.
    
    Returns:
        Dict with current LLM and embedding models plus parameters
    """
    try:
        return await model_service.get_current_models()
    except Exception as e:
        logger.error(f"Error getting current models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get current models: {str(e)}")


@router.get("/providers")
async def get_model_providers(
    provider: str = "all",
    current_admin: dict = Depends(get_current_admin_user)
):
    """
    Get available models from different providers (Ollama, HuggingFace).
    
    Args:
        provider: Filter by provider ('all', 'ollama', 'huggingface')
        
    Returns:
        Dict with models grouped by provider
    """
    try:
        return await model_service.get_model_providers(provider)
    except Exception as e:
        logger.error(f"Error getting model providers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model providers: {str(e)}")


@router.post("/check-gpu")
async def check_gpu_compatibility(
    request: dict,
    current_admin: dict = Depends(get_current_admin_user)
):
    """
    Check GPU compatibility for specified models.
    
    Request body:
        {
            "models": [{"name": "model-name", "size": "7B"}],
            "mode": "combined" or "individual"
        }
        
    Returns:
        Dict with compatibility status and recommendations
    """
    try:
        models = request.get('models', [])
        mode = request.get('mode', 'combined')
        
        if not models:
            raise HTTPException(status_code=400, detail="No models specified for GPU check")
        
        return await model_service.check_gpu_compatibility(models, mode)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking GPU compatibility: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to check GPU compatibility: {str(e)}")


@router.post("/settings")
async def update_models_settings(
    request: dict,
    current_admin: dict = Depends(get_current_admin_user)
):
    """
    Update model settings with full validation and GPU compatibility checks.
    
    Request body:
        {
            "llm": "model-name",
            "embedding": "model-name",
            "llm_size": "7B" (optional),
            "embedding_size": "335m" (optional),
            "force": false (optional),
            "parameters": {
                "temperature": 0.7,
                "max_tokens": 2048,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            } (optional),
            "provider": "ollama" (optional),
            "embedding_provider": "ollama" (optional)
        }
        
    Returns:
        Dict with update status and downloaded models
    """
    try:
        llm_model = request.get('llm')
        embedding_model = request.get('embedding')
        llm_size = request.get('llm_size')
        embedding_size = request.get('embedding_size')
        force_update = request.get('force', False)
        model_parameters = request.get('parameters', {})
        provider = request.get('provider', 'ollama')
        embedding_provider = request.get('embedding_provider', provider)
        
        if not llm_model or not embedding_model:
            raise HTTPException(status_code=400, detail="Both LLM and embedding models are required")
        
        result = await model_service.update_model_settings(
            llm_model=llm_model,
            embedding_model=embedding_model,
            llm_size=llm_size,
            embedding_size=embedding_size,
            force_update=force_update,
            model_parameters=model_parameters,
            provider=provider,
            embedding_provider=embedding_provider
        )
        
        return result
        
    except ValueError as e:
        # Handle GPU compatibility errors
        import json
        try:
            error_data = json.loads(str(e))
            raise HTTPException(status_code=400, detail=error_data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update models: {str(e)}")


@router.post("/download-huggingface")
async def download_huggingface_model(
    request: dict,
    current_admin: dict = Depends(get_current_admin_user)
):
    """
    Download and validate HuggingFace model before adding to settings.
    
    Request body:
        {
            "model_name": "organization/model-name",
            "model_type": "llm" or "embedding"
        }
        
    Returns:
        Dict with download status and model details
    """
    try:
        model_name = request.get('model_name')
        model_type = request.get('model_type', 'llm')
        
        if not model_name:
            raise HTTPException(status_code=400, detail="Model name is required")
        
        if model_type not in ['llm', 'embedding']:
            raise HTTPException(status_code=400, detail="model_type must be 'llm' or 'embedding'")
        
        result = await model_service.download_huggingface_model(
            model_name=model_name,
            model_type=model_type
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading HuggingFace model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download model: {str(e)}")


@router.post("/simple-settings")
async def update_simple_models_settings(
    request: dict,
    current_admin: dict = Depends(get_current_admin_user)
):
    """
    Update model settings with simplified validation - for basic UI.
    
    This is a simplified version of /settings that is more permissive and designed
    for basic frontend model selection without extensive validation.
    
    Request body:
        {
            "llm": "model-name",
            "embedding": "model-name",
            "parameters": {...} (optional),
            "provider": "ollama" (optional),
            "embedding_provider": "ollama" (optional)
        }
        
    Returns:
        Dict with update status and warnings
    """
    try:
        llm_model = request.get('llm')
        embedding_model = request.get('embedding')
        model_parameters = request.get('parameters', {})
        provider = request.get('provider')
        embedding_provider = request.get('embedding_provider', provider)
        
        if not llm_model or not embedding_model:
            raise HTTPException(status_code=400, detail="Both LLM and embedding models are required")
        
        result = await model_service.update_simple_settings(
            llm_model=llm_model,
            embedding_model=embedding_model,
            model_parameters=model_parameters,
            provider=provider,
            embedding_provider=embedding_provider
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating models (simple): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update models: {str(e)}")

