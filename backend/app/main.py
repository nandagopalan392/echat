"""
FastAPI Application Entry Point

This is the main application file that initializes the FastAPI app,
configures middleware, and registers all API routers.
"""

import os
import logging

from fastapi import FastAPI

# Import centralized configuration
from app.config import settings

# Import middleware
from app.api.middleware import setup_cors

# Import routers
from app.api.v1.endpoints.auth import router as auth_router
from app.api.v1.endpoints.users import router as users_router
from app.api.v1.endpoints.admin import router as admin_router
from app.api.v1.endpoints.documents import router as documents_router
from app.api.v1.endpoints.vector_store import router as vector_store_router
from app.api.v1.endpoints.chat import router as chat_router
from app.api.v1.endpoints.chunking import router as chunking_router
from app.api.v1.endpoints.retrieval import router as retrieval_router
from app.api.v1.endpoints.evaluation import router as evaluation_router
from app.api.v1.endpoints.finetuning import router as finetuning_router
from app.api.v1.endpoints.websocket_finetuning import router as websocket_finetuning_router
from app.api.v1.endpoints.websocket_evaluation import router as websocket_evaluation_router
from app.api.v1.endpoints.models import router as models_router

# Import database and core services from clean architecture
from app.db import DatabaseConnection, initialize_database
from app.services.rag_service import get_rag_service
from app.core.training.rlhf import RLHFManager
from app.core.providers import get_model_cache

# Initialize FastAPI app
app = FastAPI(
    title="Knowledge API",
    description="API for knowledge application",
    version="1.0.0"
)

# Initialize core services using clean architecture
db = DatabaseConnection()
rlhf_manager = RLHFManager()

def get_rag_service_instance():
    """Get RAG service instance using singleton"""
    return get_rag_service()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set specific loggers to WARNING to suppress their output
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('rag').setLevel(logging.WARNING)
logging.getLogger('chat_db').setLevel(logging.WARNING)
logging.getLogger('ollama_scraper').setLevel(logging.WARNING)
logging.getLogger('document_storage').setLevel(logging.WARNING)
logging.getLogger('enhanced_document_processor').setLevel(logging.DEBUG)
logging.getLogger('table_extraction').setLevel(logging.DEBUG)
logging.getLogger('chunking_config').setLevel(logging.WARNING)
logging.getLogger('reranker').setLevel(logging.WARNING)
logging.getLogger('fastapi').setLevel(logging.WARNING)
logging.getLogger('uvicorn').setLevel(logging.WARNING)
logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
logging.getLogger('uvicorn.error').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('langchain').setLevel(logging.WARNING)
logging.getLogger('langchain_core').setLevel(logging.WARNING)
logging.getLogger('langchain_ollama').setLevel(logging.WARNING)
logging.getLogger('langchain_community').setLevel(logging.WARNING)
logging.getLogger('langchain_chroma').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure CORS middleware
setup_cors(app)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    import time
    total_startup_time = time.time()
    logger.info("🚀 Starting up the application...")
    
    try:
        # Initialize database schema with clean architecture
        db_start = time.time()
        initialize_database()  # Clean architecture - no more legacy code!
        db_time = time.time() - db_start
        logger.info(f"🚀 Database initialized successfully in {db_time:.2f} seconds")
        
        # Check vector store status
        vs_start = time.time()
        chroma_path = os.getenv('CHROMA_DB_PATH', '/app/data/chroma_db')
        if os.path.exists(chroma_path):
            logger.info("🚀 Found existing vector store")
            rag_service = get_rag_service_instance()
            if rag_service.rag_engine.vector_store:
                vs_time = time.time() - vs_start
                logger.info(f"🚀 Vector store loaded successfully in {vs_time:.2f} seconds")
            else:
                logger.warning("Vector store exists but couldn't be loaded")
        else:
            logger.info("No existing vector store found")
        
        # Initialize Docling processors to preload models
        logger.info("🚀 Initializing Docling processors to preload models...")
        docling_start = time.time()
        
        try:
            from app.core.rag.chunking.enhanced_document_processor import get_document_processor
            doc_processor = get_document_processor()
            docling_init_time = time.time() - docling_start
            logger.info(f"🚀 Docling document processor initialized in {docling_init_time:.2f} seconds")
            
            # Initialize table extractor
            table_start = time.time()
            from app.core.rag.chunking.table_extraction import get_table_extractor
            table_extractor = get_table_extractor()
            table_init_time = time.time() - table_start
            logger.info(f"🚀 Docling table extractor initialized in {table_init_time:.2f} seconds")
            
        except Exception as e:
            logger.warning(f"🚀 Error initializing Docling processors: {e}")
        
        # Initialize model cache and warm it up
        logger.info("🚀 Initializing model cache...")
        cache_start = time.time()
        try:
            model_cache = get_model_cache()
            model_cache.warm_up()
            cache_time = time.time() - cache_start
            logger.info(f"🚀 Model cache warmed up in {cache_time:.2f} seconds")
        except Exception as e:
            logger.warning(f"🚀 Error warming up model cache: {e}")
        
        total_time = time.time() - total_startup_time
        logger.info(f"🚀 Application startup completed in {total_time:.2f} seconds")
            
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Shutting down the application...")
    try:
        # Clean up GPU memory if available
        rag_service = get_rag_service_instance()
        if rag_service and rag_service.rag_engine:
            rag_service.rag_engine.clear_gpu_memory()
        
        # Note: Vector store is persisted to disk, no need to clear it
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

# Register API routers
app.include_router(auth_router, prefix="/api/auth", tags=["authentication"])
app.include_router(users_router, prefix="/api/users", tags=["users"])
app.include_router(admin_router, prefix="/api/admin", tags=["admin"])
app.include_router(documents_router, prefix="/api", tags=["documents"])
app.include_router(vector_store_router, tags=["vector-store"])
app.include_router(chat_router, prefix="/api", tags=["chat"])
app.include_router(chunking_router, prefix="/api/chunking", tags=["chunking"])
app.include_router(retrieval_router, prefix="/api/retrieval", tags=["retrieval"])
app.include_router(evaluation_router, prefix="/api", tags=["evaluation"])
app.include_router(finetuning_router, prefix="/api", tags=["finetuning"])
app.include_router(models_router, tags=["models"])
app.include_router(websocket_finetuning_router, prefix="/api", tags=["websocket", "finetuning"])
app.include_router(websocket_evaluation_router, prefix="/api/evaluation", tags=["websocket", "evaluation"])

# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint - Health check"""
    return {"status": "healthy", "message": "Knowledge API is running"}
