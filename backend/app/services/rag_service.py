"""
RAG Service - Business logic for RAG operations
Orchestrates RAG engine with storage service and document processing
Migrated from rag.py ChatPDF class - service orchestration layer
"""

import os
import logging
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from app.core.rag import get_rag_engine, RAGEngine
from app.services.storage_service import get_storage_service, StorageService
from app.config.chunking import ChunkingMethod, ChunkingConfig, get_chunking_config_manager, FileFormatSupport
from app.core.rag.chunking.enhanced_document_processor import get_document_processor
from app.db import DatabaseConnection
from app.db.repositories import ConfigRepository
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RAGService:
    """
    Service layer for RAG operations
    Orchestrates RAG engine with storage and document processing
    """
    
    def __init__(self):
        """Initialize RAG service"""
        # Get core components
        self.rag_engine = get_rag_engine()
        self.storage_service = get_storage_service()
        self.doc_processor = get_document_processor()
        self.config_manager = get_chunking_config_manager()
        
        # Initialize database connection and repositories
        self.db = DatabaseConnection()
        self.config_repo = ConfigRepository(self.db)
        
        # Load model settings from database
        self._load_model_settings()

    def _load_model_settings(self):
        """Load model settings from database"""
        try:
            db_settings = self.config_repo.get_model_settings()
            
            if db_settings:
                llm_model = db_settings.get('llm')
                embedding_model = db_settings.get('embedding')
                parameters = db_settings.get('parameters', {})
                
                # Update RAG engine models
                if llm_model or embedding_model:
                    self.rag_engine.reload_models(llm_model, embedding_model)
                
                # Update parameters
                if parameters:
                    self.rag_engine.update_model_parameters(parameters)
                
                logger.info(f"Loaded model settings: LLM={llm_model}, Embedding={embedding_model}")
            
        except Exception as e:
            logger.warning(f"Could not load model settings from database: {e}")

    @property
    def llm_model(self) -> str:
        """Get current LLM model"""
        return self.rag_engine.llm_model

    @property
    def embedding_model(self) -> str:
        """Get current embedding model"""
        return self.rag_engine.embedding_model

    @property
    def models_loaded(self) -> bool:
        """Check if models are loaded"""
        return self.rag_engine.models_loaded

    def ensure_models_loaded(self):
        """Ensure models are loaded"""
        self.rag_engine.ensure_models_loaded()

    def ingest_with_storage_and_chunking(
        self,
        file_path: str,
        original_filename: str = None,
        chunking_method: ChunkingMethod = None,
        chunking_config: ChunkingConfig = None,
        user_id: str = None
    ) -> bool:
        """
        Ingest a document with storage, chunking, and vector indexing
        This is the main ingestion pipeline
        """
        try:
            logger.info(f"Starting document ingestion: {file_path}")
            
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False

            # Use provided filename or extract from path
            if original_filename is None:
                original_filename = Path(file_path).name
            
            # Determine file type and optimal chunking method
            file_ext = Path(file_path).suffix[1:]  # Remove dot
            if chunking_method is None:
                chunking_method = FileFormatSupport.get_optimal_method(file_ext)
            
            # Get chunking configuration
            if chunking_config is None:
                chunking_config = self.config_manager.get_config(chunking_method, user_id)
            
            # Validate method supports file type
            if not FileFormatSupport.is_supported(chunking_method, file_ext):
                logger.warning(f"Method {chunking_method.value} not supported for {file_ext}, using general")
                chunking_method = ChunkingMethod.GENERAL
                chunking_config = self.config_manager.get_config(chunking_method, user_id)
            
            # Determine content type
            content_type_map = {
                'pdf': 'application/pdf',
                'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'doc': 'application/msword',
                'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'xls': 'application/vnd.ms-excel',
                'ppt': 'application/vnd.ms-powerpoint',
                'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                'txt': 'text/plain',
                'csv': 'text/csv',
                'html': 'text/html',
                'json': 'application/json',
                'eml': 'message/rfc822'
            }
            content_type = content_type_map.get(file_ext, 'application/octet-stream')
            
            # Step 1: Store document in storage service
            doc_info = self.storage_service.store_document(
                file_path,
                original_filename,
                content_type,
                chunking_method.value,
                chunking_config.to_dict()
            )
            
            logger.info(f"Document stored with ID: {doc_info['id']}")
            
            # Step 2: Process document with enhanced processor
            try:
                chunking_result = self.doc_processor.process_document(
                    file_path,
                    chunking_method,
                    chunking_config,
                    user_id,
                    original_filename
                )
                
                logger.info(f"Document processed: {len(chunking_result.chunks)} chunks created using {chunking_result.method_used.value}")
                
                # Log warnings
                if chunking_result.warnings:
                    for warning in chunking_result.warnings:
                        logger.warning(f"Chunking warning: {warning}")
                
                # Step 3: Add chunks to vector store
                collection_name = self.rag_engine._get_collection_name()
                success = self.rag_engine.add_chunks_to_vector_store(chunking_result.chunks, collection_name)
                
                if success:
                    # Track successful ingestion
                    simple_metadata = {
                        'total_chunks': len(chunking_result.chunks),
                        'method_used': chunking_result.method_used.value,
                        'processing_time': chunking_result.metadata.get('processing_time', 0) if chunking_result.metadata else 0,
                        'file_size': chunking_result.metadata.get('file_size', 0) if chunking_result.metadata else 0,
                        'warnings_count': len(chunking_result.warnings)
                    }
                    
                    if chunking_result.warnings:
                        simple_metadata['warnings'] = '; '.join(chunking_result.warnings[:3])
                    
                    self.storage_service.track_ingestion(
                        document_id=doc_info['id'],
                        embedding_model=self.rag_engine.embedding_model,
                        vector_store_collection=collection_name,
                        chunk_count=len(chunking_result.chunks),
                        metadata=simple_metadata,
                        chunking_method=chunking_result.method_used.value,
                        chunking_config=chunking_result.config_used.to_dict()
                    )
                    
                    logger.info(f"Successfully ingested document {doc_info['id']} with {chunking_method.value} chunking")
                    return True
                else:
                    self.storage_service.mark_ingestion_failed(
                        doc_info['id'],
                        self.rag_engine.embedding_model,
                        "Failed to add chunks to vector store"
                    )
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to process document: {e}")
                self.storage_service.mark_ingestion_failed(
                    doc_info['id'],
                    self.rag_engine.embedding_model,
                    f"Document processing failed: {str(e)}"
                )
                return False
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {str(e)}", exc_info=True)
            return False

    def get_all_documents(self) -> List[Dict]:
        """Get list of all stored documents with ingestion status"""
        return self.storage_service.list_all_documents()

    def get_ingested_documents(self) -> List[Dict]:
        """Get list of successfully ingested documents for current model"""
        all_docs = self.storage_service.list_all_documents()
        return [doc for doc in all_docs if doc.get('indexed', False)]

    def remove_document_by_id(self, document_id: int, embedding_models: List[str] = None) -> bool:
        """
        Remove document from vector store and storage
        Handles deletion across multiple embedding models
        """
        try:
            # If no models specified, get all models that have this document
            if not embedding_models:
                embedding_models = self._get_document_embedding_models(document_id)
            
            if not embedding_models:
                embedding_models = [self.rag_engine.embedding_model]
            
            total_deleted = 0
            
            # Remove from vector stores
            chroma_client = self.rag_engine.vector_store._client if self.rag_engine.vector_store else None
            
            if chroma_client:
                for model in embedding_models:
                    try:
                        collection_name = f"embeddings_{model.replace('/', '_').replace('-', '_').replace(':', '_')}"
                        deleted = self.rag_engine.remove_chunks_by_document_id(document_id, collection_name)
                        total_deleted += deleted
                        logger.info(f"Removed {deleted} chunks for document {document_id} from model {model}")
                    except Exception as e:
                        logger.error(f"Error removing from vector store {model}: {e}")
            
            # Remove from storage
            storage_success = self.storage_service.delete_document(document_id)
            
            logger.info(f"Document {document_id} removal: {total_deleted} chunks deleted, storage={'success' if storage_success else 'failed'}")
            return storage_success
            
        except Exception as e:
            logger.error(f"Error removing document {document_id}: {e}")
            return False

    def _get_document_embedding_models(self, document_id: int) -> List[str]:
        """Get all embedding models that have ingested this document"""
        try:
            conn = sqlite3.connect(self.storage_service.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT DISTINCT embedding_model 
                FROM ingestion_metadata 
                WHERE document_id = ? AND ingestion_status = 'completed'
            ''', (document_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [row[0] for row in rows] if rows else []
            
        except Exception as e:
            logger.warning(f"Could not determine embedding models for document {document_id}: {e}")
            return []

    def reingest_for_model_switch(self, new_embedding_model: str) -> Dict:
        """Re-ingest all documents for a new embedding model"""
        try:
            logger.info(f"Re-ingesting all documents for new embedding model: {new_embedding_model}")
            
            # Update RAG engine model
            self.rag_engine.reload_models(embedding_model=new_embedding_model)
            
            # Get all documents
            all_docs = self.storage_service.list_all_documents()
            
            results = {
                'total': len(all_docs),
                'successful': 0,
                'failed': 0,
                'failed_ids': []
            }
            
            for doc in all_docs:
                try:
                    # Get document file
                    temp_file = self.storage_service.get_document_file(doc['id'])
                    
                    # Re-ingest with same chunking config
                    chunking_method = ChunkingMethod(doc.get('chunking_method', 'general'))
                    success = self.ingest_with_storage_and_chunking(
                        temp_file,
                        doc['filename'],
                        chunking_method=chunking_method
                    )
                    
                    # Cleanup temp file
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                    
                    if success:
                        results['successful'] += 1
                    else:
                        results['failed'] += 1
                        results['failed_ids'].append(doc['id'])
                        
                except Exception as e:
                    logger.error(f"Error re-ingesting document {doc['id']}: {e}")
                    results['failed'] += 1
                    results['failed_ids'].append(doc['id'])
            
            logger.info(f"Re-ingestion complete: {results['successful']} successful, {results['failed']} failed")
            return results
            
        except Exception as e:
            logger.error(f"Error during model switch re-ingestion: {e}")
            return {'total': 0, 'successful': 0, 'failed': 0, 'failed_ids': []}

    def reingest_specific_documents_with_config(
        self,
        document_ids: List[int],
        chunking_method: ChunkingMethod,
        chunking_config: ChunkingConfig,
        user_id: str = None
    ) -> Dict:
        """Re-ingest specific documents with new chunking configuration"""
        try:
            logger.info(f"Re-ingesting {len(document_ids)} documents with {chunking_method.value} chunking")
            
            results = {
                'total': len(document_ids),
                'successful': 0,
                'failed': 0,
                'failed_ids': []
            }
            
            for doc_id in document_ids:
                try:
                    # Get document info
                    doc_info = self.storage_service.get_document(doc_id)
                    if not doc_info:
                        logger.warning(f"Document {doc_id} not found")
                        results['failed'] += 1
                        results['failed_ids'].append(doc_id)
                        continue
                    
                    # Remove existing chunks from vector store
                    self.rag_engine.remove_chunks_by_document_id(doc_id)
                    
                    # Get document file
                    temp_file = self.storage_service.get_document_file(doc_id)
                    
                    # Process with new config
                    chunking_result = self.doc_processor.process_document(
                        temp_file,
                        chunking_method,
                        chunking_config,
                        user_id,
                        doc_info['filename']
                    )
                    
                    # Add to vector store
                    collection_name = self.rag_engine._get_collection_name()
                    success = self.rag_engine.add_chunks_to_vector_store(chunking_result.chunks, collection_name)
                    
                    # Cleanup temp file
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                    
                    if success:
                        # Update ingestion tracking
                        self.storage_service.track_ingestion(
                            document_id=doc_id,
                            embedding_model=self.rag_engine.embedding_model,
                            vector_store_collection=collection_name,
                            chunk_count=len(chunking_result.chunks),
                            chunking_method=chunking_result.method_used.value,
                            chunking_config=chunking_result.config_used.to_dict()
                        )
                        results['successful'] += 1
                    else:
                        results['failed'] += 1
                        results['failed_ids'].append(doc_id)
                        
                except Exception as e:
                    logger.error(f"Error re-ingesting document {doc_id}: {e}")
                    results['failed'] += 1
                    results['failed_ids'].append(doc_id)
            
            logger.info(f"Re-ingestion complete: {results['successful']} successful, {results['failed']} failed")
            return results
            
        except Exception as e:
            logger.error(f"Error during re-ingestion: {e}")
            return {'total': 0, 'successful': 0, 'failed': 0, 'failed_ids': []}

    def get_vector_store_stats(self) -> Dict:
        """Get vector store statistics"""
        return self.rag_engine.get_vector_store_stats()

    def query(self, question: str, k: int = 4):
        """Query documents and generate answer"""
        return self.rag_engine.query(question, k)

    def reload_model_settings(self):
        """Reload model settings from database"""
        self._load_model_settings()


# Singleton instance
_rag_service = None


def get_rag_service() -> RAGService:
    """Get RAG service singleton"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
# TODO: Update main.py and other files to use this service
# TODO: Keep all LangChain, ChromaDB, and Ollama integrations
