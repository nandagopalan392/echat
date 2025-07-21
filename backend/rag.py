# rag.py
from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings
import os
import gc
import tempfile
import asyncio
import httpx
import requests
import traceback
import sqlite3
from typing import List
# Set up some reasonable defaults for ChromaDB
os.environ["CHROMA_SERVER_NOFILE"] = "65536"
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, List, Dict, Tuple
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import aiohttp
import json
from chunking_config import ChunkingMethod, ChunkingConfig
import time
from PyPDF2 import PdfReader  # Add this import
import time
import functools
from langchain.schema import Document
from document_storage import get_document_storage
import subprocess
import re
from enhanced_document_processor import get_document_processor
from chunking_config import ChunkingMethod, ChunkingConfig, get_chunking_config_manager, FileFormatSupport

# Add stub for reranker that will be dynamically loaded
_reranker_instance = None

def get_reranker():
    """Lazy import get_reranker function to avoid circular imports"""
    global _reranker_instance
    if (_reranker_instance is None):
        # Import at runtime to avoid circular imports
        from reranker import get_reranker as _get_reranker
        _reranker_instance = _get_reranker()
    return _reranker_instance

set_debug(True)
set_verbose(True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def estimate_query_complexity(query):
    """Estimate the complexity of a user query"""
    # Simple heuristic based on query length
    complexity = min(10, max(1, len(query.split()) / 5))
    return complexity

def estimate_hallucination(response, context_docs):
    """Estimate if a response might be a hallucination"""
    # Simple heuristic: check if response tokens appear in context docs
    if not context_docs:
        # No context = higher hallucination risk
        return 0.8
        
    # Extract words from context
    context_words = set()
    for doc in context_docs:
        if hasattr(doc, 'page_content'):
            context_words.update(doc.page_content.lower().split())
    
    # Extract words from response
    response_words = set(response.lower().split())
    
    # Calculate overlap
    if not context_words:
        return 0.5  # No context words to compare against
        
    overlap = len(context_words.intersection(response_words)) / len(response_words)
    
    # Higher overlap = lower hallucination risk
    hallucination_score = max(0.0, min(1.0, 1.0 - overlap))
    
    return hallucination_score

class ChatPDF:
    """A class for handling PDF ingestion and question answering using RAG."""

    def __init__(self, llm_model: str = "deepseek-r1:latest", embedding_model: str = "mxbai-embed-large"):
        # Try to load model settings from config file first
        try:
            config_path = "model_settings.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    settings = json.load(f)
                    llm_model = settings.get('llm', llm_model)
                    embedding_model = settings.get('embedding', embedding_model)
                    logger.info(f"Loaded model settings from config: LLM={llm_model}, Embedding={embedding_model}")
        except Exception as e:
            logger.warning(f"Could not load model settings from config: {e}")
        
        # Initialize base attributes
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.model = None
        self.embeddings = None
        
        # Initialize document storage service
        self.doc_storage = get_document_storage()
        self.vector_store = None
        self.retriever = None
        self.documents = []
        
        # Add a flag to track if models are loaded - set to False initially
        self.models_loaded = False
        
        # Update text splitter settings for better handling of technical documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased chunk size
            chunk_overlap=200,  # Increased overlap
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # More granular separators
        )
        
        # Default prompt with standard professional tone
        self.prompt = ChatPromptTemplate.from_template("""
            Answer the following question based on the provided context:
            Context: {context}
            Question: {question}
            Answer concisely and accurately.
        """)
        
        # Alternative prompt with more conversational tone
        self.conversational_prompt = ChatPromptTemplate.from_template("""
            Answer the following question in a friendly, conversational tone:
            Context: {context}
            Question: {question}
            Be warm, personable, and engaging in your response.
        """)
        
        # Get Chroma DB path from environment
        self.chroma_path = os.getenv('CHROMA_DB_PATH', '/app/data/chroma_db')
        self._ensure_chroma_dir()
        
        # Initialize models with retries
        self.ensure_models_loaded()
        
        # Try to load existing vector store with embedding-specific collection
        self._initialize_vector_store()

    def _get_collection_name(self):
        """Get collection name based on current embedding model"""
        # Create a collection name based on the embedding model
        # Replace problematic characters with underscores
        safe_model_name = self.embedding_model.replace(":", "_").replace("-", "_").replace("/", "_")
        return f"embeddings_{safe_model_name}"

    def _ensure_chroma_dir(self):
        """Ensure Chroma directory exists with proper permissions"""
        try:
            path = Path(self.chroma_path)
            path.mkdir(parents=True, exist_ok=True)
            
            # Set permissions recursively for the entire directory tree
            self._fix_permissions_recursive(path)
                
            logger.info(f"Ensured Chroma directory at {self.chroma_path} with proper permissions")
        except Exception as e:
            logger.error(f"Error creating Chroma directory: {str(e)}")
            raise

    def _fix_permissions_recursive(self, path):
        """Fix permissions recursively for ChromaDB directories and files"""
        try:
            path = Path(path)
            
            # Fix directory permissions
            if path.is_dir():
                os.chmod(path, 0o777)
                logger.debug(f"Set directory permissions for {path}")
                
                # Recursively fix all subdirectories and files
                for item in path.rglob("*"):
                    try:
                        if item.is_dir():
                            os.chmod(item, 0o777)
                        else:
                            os.chmod(item, 0o666)
                    except Exception as e:
                        logger.warning(f"Could not set permissions for {item}: {e}")
            else:
                # Fix file permissions
                os.chmod(path, 0o666)
                logger.debug(f"Set file permissions for {path}")
                
        except Exception as e:
            logger.warning(f"Could not fix permissions for {path}: {e}")

    def _ensure_writable_before_operation(self):
        """Ensure ChromaDB directory is writable before any write operation"""
        try:
            if os.path.exists(self.chroma_path):
                self._fix_permissions_recursive(self.chroma_path)
                
            # Test write access by creating a temporary file
            test_file = Path(self.chroma_path) / ".write_test"
            test_file.touch()
            test_file.unlink()
            
            # Also check for the SQLite database specifically
            if not self._test_chromadb_write_access():
                raise RuntimeError("ChromaDB database is not writable")
            
        except Exception as e:
            logger.error(f"ChromaDB directory not writable: {e}")
            # Try to fix permissions one more time
            try:
                os.makedirs(self.chroma_path, mode=0o777, exist_ok=True)
                self._fix_permissions_recursive(self.chroma_path)
                
                # Try the write test again
                test_file = Path(self.chroma_path) / ".write_test"
                test_file.touch()
                test_file.unlink()
                
                # Test ChromaDB access again
                if not self._test_chromadb_write_access():
                    raise RuntimeError("ChromaDB database is still not writable after permission fix")
                
            except Exception as fix_error:
                logger.error(f"Failed to fix permissions: {fix_error}")
                raise RuntimeError(f"ChromaDB directory is not writable: {e}")

    def _initialize_vector_store(self):
        """Initialize vector store with existing data if available"""
        try:
            if not self.embeddings:
                logger.warning("Embeddings not loaded, skipping vector store initialization")
                return
                
            # Get collection name first
            collection_name = self._get_collection_name()
            
            # Only try to restore from MinIO if local ChromaDB doesn't exist or is empty
            should_restore = False
            target_path = Path(self.chroma_path)
            if not target_path.exists():
                should_restore = True
            else:
                sqlite_file = target_path / "chroma.sqlite3"
                if not sqlite_file.exists() or sqlite_file.stat().st_size == 0:
                    should_restore = True
            
            if should_restore:
                # Try to restore from MinIO BEFORE creating any ChromaDB client
                restored = self._restore_vector_store_from_minio(collection_name)
                if restored:
                    logger.info(f"Restored vector store collection '{collection_name}' from MinIO")
            else:
                logger.info(f"Local ChromaDB exists, skipping MinIO restore")
            
            # Ensure proper permissions before any ChromaDB operations
            self._ensure_writable_before_operation()
                
            # Create settings for ChromaDB client
            settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
            
            # Create client AFTER restore and permission fixing
            client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=settings
            )
            
            try:
                # Try to load existing collection
                existing_collection = client.get_collection(collection_name)
                logger.info(f"Found existing collection '{collection_name}' with {existing_collection.count()} documents")
                
                # Initialize vector store with existing collection
                self.vector_store = Chroma(
                    client=client,
                    collection_name=collection_name,
                    embedding_function=self.embeddings,
                )
                
                # Initialize retriever if vector store has data
                if existing_collection.count() > 0:
                    self.retriever = self.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 3}
                    )
                    logger.info("Initialized retriever with existing data")
                else:
                    logger.info("Collection exists but is empty")
                    
            except Exception as e:
                # Collection doesn't exist, that's fine - it will be created when needed
                logger.info(f"No existing collection '{collection_name}' found: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            # Don't raise the error - let the system continue without existing data

    def _migrate_or_recreate_vector_store(self):
        """Migrate or recreate vector store when embedding model changes"""
        try:
            logger.info("Migrating or recreating vector store")
            
            # Reset current vector store and retriever
            self.vector_store = None
            self.retriever = None
            
            # Get new collection name
            collection_name = self._get_collection_name()
            
            # Ensure proper permissions before any ChromaDB operations
            self._ensure_writable_before_operation()
            
            # Create settings for ChromaDB client
            settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
            
            # Create client
            client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=settings
            )
            
            try:
                # Try to delete old collection if it exists with different embedding
                existing_collections = client.list_collections()
                for collection in existing_collections:
                    if collection.name != collection_name:
                        try:
                            client.delete_collection(collection.name)
                            logger.info(f"Deleted old collection: {collection.name}")
                        except Exception as e:
                            logger.warning(f"Could not delete collection {collection.name}: {str(e)}")
            except Exception as e:
                logger.warning(f"Error cleaning up old collections: {str(e)}")
            
            # Create new vector store with current embedding model
            if self.embeddings:
                self.vector_store = Chroma(
                    client=client,
                    collection_name=collection_name,
                    embedding_function=self.embeddings,
                )
                logger.info(f"Created new vector store with collection '{collection_name}'")
                
                # Check if collection has existing data
                try:
                    existing_collection = client.get_collection(collection_name)
                    doc_count = existing_collection.count()
                    logger.info(f"Collection '{collection_name}' has {doc_count} existing documents")
                    
                    if doc_count > 0:
                        # Initialize retriever for existing data
                        self.retriever = self.vector_store.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 3}
                        )
                        logger.info("Initialized retriever with existing vector data")
                    else:
                        logger.info("Collection is empty, will need to ingest documents")
                        
                except Exception as e:
                    logger.warning(f"Could not check existing collection: {e}")
                
                # Automatically re-ingest all documents for the new embedding model
                logger.info("Starting automatic re-ingestion of documents for new embedding model")
                success = self.reingest_all_documents_for_current_model()
                
                if success:
                    # Reinitialize retriever after re-ingestion
                    try:
                        self.retriever = self.vector_store.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 3}
                        )
                        logger.info("Retriever reinitialized after successful re-ingestion")
                    except Exception as e:
                        logger.error(f"Failed to reinitialize retriever: {e}")
                else:
                    logger.warning("Re-ingestion failed or no documents were processed")
                
            else:
                logger.error("Cannot create vector store: embeddings not loaded")
                raise Exception("Embeddings not available for vector store creation")
                
        except Exception as e:
            logger.error(f"Error migrating vector store: {str(e)}")
            raise

    async def pull_models(self):
        """Pull required models if not available"""
        try:
            import httpx
            ollama_host = os.getenv('OLLAMA_HOST', 'http://ollama:11434')
            
            async with httpx.AsyncClient(timeout=600.0) as client:
                # Pull embedding model first (usually smaller)
                logger.info(f"Pulling embedding model: {self.embedding_model}")
                try:
                    response = await client.post(
                        f"{ollama_host}/api/pull",
                        json={"name": self.embedding_model},
                        timeout=600.0  # 10 minutes for large models
                    )
                    if response.status_code == 200:
                        logger.info(f"Successfully pulled embedding model: {self.embedding_model}")
                    else:
                        logger.warning(f"Failed to pull embedding model: {response.status_code}")
                except Exception as e:
                    logger.error(f"Error pulling embedding model: {str(e)}")
                    
                # Pull LLM model
                logger.info(f"Pulling LLM model: {self.llm_model}")
                try:
                    response = await client.post(
                        f"{ollama_host}/api/pull",
                        json={"name": self.llm_model},
                        timeout=600.0  # 10 minutes for large models
                    )
                    if response.status_code == 200:
                        logger.info(f"Successfully pulled LLM model: {self.llm_model}")
                    else:
                        logger.warning(f"Failed to pull LLM model: {response.status_code}")
                except Exception as e:
                    logger.error(f"Error pulling LLM model: {str(e)}")
                    
                logger.info(f"Model pulling completed")
                
        except Exception as e:
            logger.error(f"Error pulling models: {str(e)}")
            raise

    def update_models(self, llm_model: str, embedding_model: str):
        """Update models and reload them, handling embedding dimension changes"""
        try:
            logger.info(f"Updating models - LLM: {llm_model}, Embedding: {embedding_model}")
            
            # Check if embedding model is changing
            embedding_changed = self.embedding_model != embedding_model
            
            # Clear GPU memory first to prevent out-of-memory issues
            logger.info("Clearing GPU memory before model switch...")
            self.clear_gpu_memory()
            
            # Update model names
            self.llm_model = llm_model
            self.embedding_model = embedding_model
            
            # Force reload with new models
            self.ensure_models_loaded()
            
            # Handle vector store migration if embedding model changed
            if embedding_changed:
                logger.info("Embedding model changed, migrating vector store")
                self._migrate_or_recreate_vector_store()
            elif self.vector_store and self.embeddings:
                # If only LLM changed, just reinitialize retriever
                try:
                    self.retriever = self.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 3}
                    )
                    logger.info("Vector store retriever updated")
                except Exception as e:
                    logger.warning(f"Could not update vector store retriever: {str(e)}")
            
            logger.info("Models updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error updating models: {str(e)}")
            raise

    def ensure_models_loaded(self, max_retries=3):
        """Initialize models with retries and proper memory management"""
        # DON'T check models_loaded here to prevent infinite recursion
        
        for attempt in range(max_retries):
            try:
                # Clear memory before loading if this is a retry
                if attempt > 0:
                    logger.info(f"Retry {attempt}: Clearing memory before model load")
                    self.clear_gpu_memory()
                    import time
                    time.sleep(2)  # Brief pause after memory clearing
                
                if not self.embeddings:
                    logger.info(f"Loading embedding model: {self.embedding_model}")
                    # Try to initialize embeddings
                    self.embeddings = OllamaEmbeddings(
                        model=self.embedding_model,
                        base_url=os.getenv('OLLAMA_HOST', 'http://ollama:11434')
                    )
                    # Test embeddings
                    self.embeddings.embed_query("test")
                    logger.info("Embedding model loaded successfully")
                
                if not self.model:
                    logger.info(f"Loading LLM model: {self.llm_model}")
                    self.model = ChatOllama(
                        model=self.llm_model,
                        base_url=os.getenv('OLLAMA_HOST', 'http://ollama:11434'),
                        temperature=0.1,
                        num_ctx=2048,
                        timeout=120,  # Increase timeout to 2 minutes
                        streaming=True,  # Enable streaming
                        seed=42  # Add seed for consistent responses
                    )
                    logger.info("LLM model loaded successfully")
                
                # Mark models as loaded after successful initialization
                self.models_loaded = True
                logger.info("All models loaded successfully")
                return True
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    logger.warning("Memory-related error detected, clearing GPU memory")
                    self.clear_gpu_memory()
                
                if attempt == max_retries - 1:
                    logger.error("Failed to load models after maximum retries")
                    raise
                
                # Try to pull models if they might not be available
                try:
                    import asyncio
                    asyncio.run(self.pull_models())
                except Exception as pull_e:
                    logger.warning(f"Failed to pull models: {pull_e}")
                
                # Wait before retry
                import time
                time.sleep(5)
                
        return False

    def check_models(self):
        """Check if models are loaded and reload if necessary"""
        if not self.models_loaded:
            logger.info("Models not loaded, loading them now...")
            self.ensure_models_loaded()
        return self.models_loaded

    def ingest(self, pdf_file_path: str) -> bool:
        """Ingest a PDF file into the vector store with improved error handling"""
        try:
            logger.info(f"Starting ingestion of PDF: {pdf_file_path}")
            
            if not os.path.exists(pdf_file_path):
                logger.error(f"File not found: {pdf_file_path}")
                return False

            # First try PyPDF2
            try:
                logger.debug("Attempting to read PDF with PyPDF2")
                pdf = PdfReader(pdf_file_path)
                text_content = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text.strip():  # Only add non-empty pages
                        text_content.append(text)
                
                if not text_content:
                    raise ValueError("No text content extracted from PDF")
                
                combined_text = "\n\n".join(text_content)
                logger.debug(f"Extracted {len(text_content)} pages of text")
                
                # Create documents from extracted text
                docs = [Document(page_content=combined_text)]
                
            except Exception as e:
                logger.warning(f"PyPDF2 extraction failed, trying langchain loader: {str(e)}")
                # Fallback to langchain's PyPDFLoader
                docs = PyPDFLoader(file_path=pdf_file_path).load()

            if not docs:
                logger.error("No documents created from PDF")
                return False

            # Create chunks with detailed logging
            logger.debug("Creating text chunks...")
            chunks = self.text_splitter.split_documents(docs)
            logger.info(f"Created {len(chunks)} chunks from PDF")

            if not chunks:
                logger.error("No chunks created from text splitting")
                return False

            # Add filename to chunk metadata 
            filename = Path(pdf_file_path).name
            for chunk in chunks:
                if not chunk.metadata:
                    chunk.metadata = {}
                chunk.metadata['source'] = filename

            # Debug: Log metadata before filtering
            logger.info(f"Before filtering - first chunk metadata: {chunks[0].metadata if chunks else 'No chunks'}")

            # Use our custom metadata filtering instead of langchain's filter_complex_metadata
            chunks = self._filter_metadata_for_chromadb(chunks)
            
            # Debug: Log metadata after filtering
            logger.info(f"After filtering - first chunk metadata: {chunks[0].metadata if chunks else 'No chunks'}")
            logger.info(f"Total chunks after filtering: {len(chunks)}")

            # Create or update vector store with embedding-specific collection
            try:
                if not self.vector_store:
                    logger.debug("Creating new vector store with embedding-specific collection")
                    
                    # Ensure proper permissions before any ChromaDB operations
                    self._ensure_writable_before_operation()
                    
                    # Use PersistentClient with Python 3.10+ which supports the required type annotations
                    settings = Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                    client = chromadb.PersistentClient(
                        path=self.chroma_path,
                        settings=settings
                    )
                    
                    # Use embedding-specific collection name
                    collection_name = self._get_collection_name()
                    
                    self.vector_store = Chroma(
                        client=client,
                        collection_name=collection_name,
                        embedding_function=self.embeddings,
                    )

                # CRITICAL: Always ensure permissions before write operations
                self._ensure_writable_before_operation()
                logger.info(f"Created new vector store with collection '{collection_name}'")
                
                # Add documents to vector store
                logger.debug("Adding documents to vector store")
                
                # Ensure permissions are still correct before adding documents
                self._periodic_permission_check()
                self._handle_chromadb_operation(self.vector_store.add_documents, chunks)
                
                # Initialize or update retriever
                if not self.retriever:
                    self.retriever = self.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 3}
                    )
                    logger.debug("Initialized retriever")
                
                # No need to call persist() with PersistentClient - it auto-persists
                logger.info(f"Successfully processed PDF: {pdf_file_path}")
                return True
                
            except Exception as e:
                logger.error(f"Vector store operation failed: {str(e)}")
                # Check if this is a dimension mismatch error
                if "dimension" in str(e).lower():
                    logger.info("Dimension mismatch detected, attempting to migrate vector store")
                    try:
                        # Force recreation of vector store
                        self._migrate_or_recreate_vector_store()
                        
                        # Retry ingestion with new vector store
                        if self.vector_store:
                            self._handle_chromadb_operation(self.vector_store.add_documents, chunks)
                            if not self.retriever:
                                self.retriever = self.vector_store.as_retriever(
                                    search_type="similarity",
                                    search_kwargs={"k": 3}
                                )
                            logger.info(f"Successfully processed PDF after migration: {pdf_file_path}")
                            return True
                    except Exception as migration_error:
                        logger.error(f"Vector store migration failed: {str(migration_error)}")
                        raise
                raise

        except Exception as e:
            logger.error(f"PDF ingestion failed: {str(e)}", exc_info=True)
            return False

    def ingest_with_storage(self, pdf_file_path: str, original_filename: str = None) -> bool:
        """
        Ingest a PDF file using the new document storage system
        Stores raw file in MinIO and tracks ingestion per embedding model
        """
        try:
            logger.info(f"Starting ingestion of PDF with storage: {pdf_file_path}")
            
            if not os.path.exists(pdf_file_path):
                logger.error(f"File not found: {pdf_file_path}")
                return False

            # Use provided original filename or extract from path
            if original_filename is None:
                original_filename = Path(pdf_file_path).name
            content_type = "application/pdf"
            
            # Store document in MinIO and get document info
            doc_info = self.doc_storage.store_document(
                pdf_file_path, 
                original_filename, 
                content_type
            )
            
            logger.info(f"Document stored with ID: {doc_info['id']}")
            
            # Process and ingest for current embedding model
            return self._ingest_for_current_model(doc_info['id'], pdf_file_path)
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {str(e)}", exc_info=True)
            return False
    
    def _ingest_for_current_model(self, document_id: int, file_path: str = None) -> bool:
        """Ingest a specific document for the current embedding model"""
        try:
            # Get file path if not provided
            if file_path is None:
                file_path = self.doc_storage.get_document_file(document_id)
                cleanup_temp_file = True
            else:
                cleanup_temp_file = False
            
            # Extract text from PDF
            text_content = self._extract_text_from_pdf(file_path)
            if not text_content:
                self.doc_storage.mark_ingestion_failed(
                    document_id, 
                    self.embedding_model, 
                    "No text content extracted from PDF"
                )
                return False
             # Create document and split into chunks
            docs = [Document(page_content=text_content)]
            chunks = self.text_splitter.split_documents(docs)
            
            if not chunks:
                self.doc_storage.mark_ingestion_failed(
                    document_id, 
                    self.embedding_model, 
                    "No chunks created from text splitting"
                )
                return False

            # Get document info to add filename to chunk metadata
            doc_info = self.doc_storage._get_document_by_id(document_id)
            filename = doc_info['filename'] if doc_info else f"document_{document_id}"
            
            # Add filename to chunk metadata
            for chunk in chunks:
                if not chunk.metadata:
                    chunk.metadata = {}
                chunk.metadata['source'] = filename
                chunk.metadata['document_id'] = document_id

            # Debug: Log metadata before filtering
            logger.info(f"Before filtering - first chunk metadata: {chunks[0].metadata if chunks else 'No chunks'}")

            # Use our custom metadata filtering instead of langchain's filter_complex_metadata
            chunks = self._filter_metadata_for_chromadb(chunks)
            
            # Debug: Log metadata after filtering
            logger.info(f"After filtering - first chunk metadata: {chunks[0].metadata if chunks else 'No chunks'}")
            logger.info(f"Total chunks after filtering: {len(chunks)}")
            
            # Get collection name for current model
            collection_name = self._get_collection_name()
            
            # Add to vector store
            success = self._add_chunks_to_vector_store(chunks, collection_name)
            
            if success:
                # Track successful ingestion
                self.doc_storage.track_ingestion(
                    document_id=document_id,
                    embedding_model=self.embedding_model,
                    vector_store_collection=collection_name,
                    chunk_count=len(chunks),
                    metadata={
                        'text_length': len(text_content),
                        'chunk_size': self.text_splitter._chunk_size,
                        'chunk_overlap': self.text_splitter._chunk_overlap
                    }
                )
                logger.info(f"Successfully ingested document {document_id} for model {self.embedding_model}")
            else:
                self.doc_storage.mark_ingestion_failed(
                    document_id, 
                    self.embedding_model, 
                    "Failed to add chunks to vector store"
                )
            
            # Cleanup temporary file if we created it
            if cleanup_temp_file and os.path.exists(file_path):
                os.remove(file_path)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to ingest document {document_id} for model {self.embedding_model}: {e}")
            self.doc_storage.mark_ingestion_failed(
                document_id, 
                self.embedding_model, 
                str(e)
            )
            return False
    
    def _extract_text_from_pdf(self, pdf_file_path: str) -> str:
        """Extract text from PDF using multiple methods with enhanced error handling"""
        
        # Check if file exists and has valid size
        if not os.path.exists(pdf_file_path):
            raise ValueError(f"PDF file not found: {pdf_file_path}")
            
        file_size = os.path.getsize(pdf_file_path)
        if file_size == 0:
            raise ValueError("PDF file is empty")
            
        if file_size < 10:  # Minimum PDF size
            raise ValueError("PDF file too small to be valid")
        
        logger.debug(f"Processing PDF file: {pdf_file_path} (size: {file_size} bytes)")
        
        # Check file header for valid PDF signature
        try:
            with open(pdf_file_path, 'rb') as f:
                header = f.read(8)
                if not header.startswith(b'%PDF-'):
                    logger.warning(f"Invalid PDF header detected: {header}")
                    # Try to recover if it's a text file misnamed as PDF
                    try:
                        with open(pdf_file_path, 'r', encoding='utf-8') as text_file:
                            content = text_file.read()
                            if content.strip():
                                logger.info("File appears to be text, treating as text content")
                                return content
                    except:
                        pass
                    raise ValueError(f"File does not appear to be a valid PDF (header: {header})")
        except Exception as e:
            logger.warning(f"Could not check PDF header: {e}")
        
        extraction_errors = []
        
        try:
            # First try PyPDF2
            logger.debug("Attempting to read PDF with PyPDF2")
            pdf = PdfReader(pdf_file_path)
            
            # Check if PDF is encrypted
            if pdf.is_encrypted:
                logger.warning("PDF is encrypted, attempting to decrypt with empty password")
                if not pdf.decrypt(""):
                    raise ValueError("PDF is encrypted and cannot be decrypted")
            
            text_content = []
            total_pages = len(pdf.pages)
            logger.debug(f"PDF has {total_pages} pages")
            
            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text()
                    if text and text.strip():  # Only add non-empty pages
                        text_content.append(text.strip())
                        logger.debug(f"Extracted text from page {i+1}/{total_pages}")
                except Exception as page_error:
                    logger.warning(f"Failed to extract text from page {i+1}: {page_error}")
                    continue
            
            if text_content:
                combined_text = "\n\n".join(text_content)
                logger.debug(f"Successfully extracted {len(text_content)} pages of text with PyPDF2")
                if len(combined_text.strip()) > 10:  # Ensure we have meaningful content
                    return combined_text
                else:
                    logger.warning("PyPDF2 extracted text is too short, trying alternative method")
            else:
                logger.warning("PyPDF2 extracted no readable text")
            
        except Exception as e:
            error_msg = f"PyPDF2 extraction failed: {str(e)}"
            logger.warning(error_msg)
            extraction_errors.append(error_msg)
        
        try:
            # Fallback to langchain's PyPDFLoader
            logger.debug("Attempting to read PDF with langchain PyPDFLoader")
            docs = PyPDFLoader(file_path=pdf_file_path).load()
            if docs:
                page_contents = []
                for i, doc in enumerate(docs):
                    if doc.page_content and doc.page_content.strip():
                        page_contents.append(doc.page_content.strip())
                        logger.debug(f"Extracted content from document {i+1}/{len(docs)}")
                
                if page_contents:
                    combined_text = "\n\n".join(page_contents)
                    logger.debug(f"Successfully extracted text from {len(page_contents)} pages with PyPDFLoader")
                    if len(combined_text.strip()) > 10:
                        return combined_text
                    else:
                        logger.warning("PyPDFLoader extracted text is too short")
                else:
                    logger.warning("PyPDFLoader extracted no readable content")
                
        except Exception as e:
            error_msg = f"PyPDFLoader extraction failed: {str(e)}"
            logger.warning(error_msg)
            extraction_errors.append(error_msg)
        
        # Try one more fallback - treat as binary and search for text patterns
        try:
            logger.debug("Attempting binary text extraction as last resort")
            with open(pdf_file_path, 'rb') as f:
                content = f.read()
                # Look for readable text in the binary content
                text_parts = []
                # Extract strings that look like readable text
                import re
                text_matches = re.findall(rb'[a-zA-Z0-9\s\.,!?;:()]{10,}', content)
                for match in text_matches[:50]:  # Limit to first 50 matches
                    try:
                        decoded = match.decode('utf-8', errors='ignore').strip()
                        if len(decoded) > 10 and decoded.count(' ') > 2:  # Basic text validation
                            text_parts.append(decoded)
                    except:
                        continue
                
                if text_parts:
                    combined_text = "\n".join(text_parts)
                    logger.info(f"Extracted {len(text_parts)} text segments from binary content")
                    if len(combined_text.strip()) > 50:
                        return combined_text
                        
        except Exception as e:
            error_msg = f"Binary extraction failed: {str(e)}"
            logger.warning(error_msg)
            extraction_errors.append(error_msg)
        
        # All methods failed
        all_errors = "; ".join(extraction_errors)
        error_message = f"Failed to extract text from PDF using any method. Errors: {all_errors}"
        logger.error(error_message)
        raise ValueError(error_message)
    
    def _filter_metadata_for_chromadb(self, chunks: List[Document]) -> List[Document]:
        """Filter metadata to keep only ChromaDB-compatible fields"""
        logger.info(f"🔍 METADATA DEBUG: Filtering {len(chunks)} chunks")
        
        filtered_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info(f"🔍 CHUNK {i}: Original metadata: {chunk.metadata}")
            
            # Create a new document with filtered metadata
            filtered_metadata = {}
            if chunk.metadata:
                for key, value in chunk.metadata.items():
                    logger.debug(f"🔍 Processing metadata key '{key}': {type(value)} = {value}")
                    # Keep simple types that ChromaDB can handle
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        filtered_metadata[key] = value
                        logger.debug(f"✅ Kept metadata '{key}': {value}")
                    elif isinstance(value, dict) and len(str(value)) < 1000:  # Small dicts
                        filtered_metadata[key] = str(value)
                        logger.debug(f"✅ Converted dict metadata '{key}': {str(value)}")
                    else:
                        logger.debug(f"❌ Skipped complex metadata '{key}': {type(value)}")
            
            logger.info(f"🔍 CHUNK {i}: Filtered metadata: {filtered_metadata}")
            
            filtered_chunk = Document(
                page_content=chunk.page_content,
                metadata=filtered_metadata
            )
            filtered_chunks.append(filtered_chunk)
        
        logger.info(f"🔍 METADATA DEBUG: Returning {len(filtered_chunks)} filtered chunks")
        return filtered_chunks

    def _add_chunks_to_vector_store(self, chunks: List[Document], collection_name: str) -> bool:
        """Add document chunks to the vector store"""
        try:
            if not chunks:
                logger.warning("No chunks provided for vector store addition")
                return False
            
            # Ensure ChromaDB directory has proper permissions
            self._ensure_writable_before_operation()
            
            logger.info(f"🔍 DEBUG: Adding {len(chunks)} chunks to vector store collection '{collection_name}'")
            
            # Log original metadata before filtering
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                logger.info(f"🔍 DEBUG: Original chunk {i} metadata: {chunk.metadata}")
                logger.info(f"🔍 DEBUG: Original chunk {i} content preview: {chunk.page_content[:100]}...")
            
            # Use our custom metadata filtering instead of langchain's filter_complex_metadata
            filtered_chunks = self._filter_metadata_for_chromadb(chunks)
            
            # Log filtered metadata
            for i, chunk in enumerate(filtered_chunks[:3]):  # Show first 3 chunks
                logger.info(f"🔍 DEBUG: Filtered chunk {i} metadata: {chunk.metadata}")
            
            if not self.vector_store:
                logger.debug("Creating new vector store with embedding-specific collection")
                settings = Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
                
                client = chromadb.PersistentClient(
                    path=self.chroma_path,
                    settings=settings
                )
                
                self.vector_store = Chroma(
                    client=client,
                    collection_name=collection_name,
                    embedding_function=self.embeddings,
                )
            
            # Before adding to vector store, let's manually verify what we're sending
            logger.info(f"🔍 DEBUG: About to add {len(filtered_chunks)} filtered chunks to ChromaDB")
            
            # Add documents to vector store
            success = self._handle_chromadb_operation(self.vector_store.add_documents, filtered_chunks)
            
            if success:
                # After successful addition, let's verify what was actually stored
                logger.info("🔍 DEBUG: Verifying chunks were stored with metadata...")
                try:
                    # Get the collection directly and check what was stored
                    chroma_client = self.vector_store._client
                    collection = chroma_client.get_collection(collection_name)
                    
                    # Get the last few documents to see their metadata
                    recent_docs = collection.get(limit=3, include=["metadatas", "documents"])
                    
                    logger.info(f"🔍 DEBUG: Recent documents in collection: {len(recent_docs.get('ids', []))}")
                    for i, metadata in enumerate(recent_docs.get('metadatas', [])[:3]):
                        logger.info(f"🔍 DEBUG: Recently stored document {i} metadata: {metadata}")
                        
                except Exception as verify_error:
                    logger.error(f"🔍 DEBUG: Could not verify stored chunks: {verify_error}")
                
                # Update retriever
                self.retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                )
                
                # Backup to MinIO after successful ingestion
                backup_success = self._backup_vector_store_to_minio(collection_name)
                if backup_success:
                    logger.info(f"Vector store backed up to MinIO after adding {len(chunks)} chunks")
                else:
                    logger.warning("Failed to backup vector store to MinIO")
                
                logger.info(f"✅ Added {len(filtered_chunks)} chunks to vector store collection '{collection_name}'")
                return True
            else:
                logger.error("❌ Failed to add chunks to vector store")
                return False
            
        except Exception as e:
            logger.error(f"Failed to add chunks to vector store: {e}")
            return False
    
    def clear_gpu_memory(self):
        """Clear GPU memory by unloading models and forcing garbage collection"""
        try:
            logger.info("Clearing GPU memory...")
            
            # Clear the models
            self.model = None
            self.embeddings = None
            self.models_loaded = False
            
            # Clear vector store references
            if self.vector_store:
                try:
                    # Clear any cached embeddings in the vector store
                    self.vector_store = None
                except Exception as e:
                    logger.warning(f"Error clearing vector store: {e}")
            
            self.retriever = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Try to clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.info("CUDA cache cleared")
            except ImportError:
                logger.info("PyTorch not available, skipping CUDA cache clear")
            except Exception as e:
                logger.warning(f"Error clearing CUDA cache: {e}")
            
            logger.info("GPU memory clearing completed")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing GPU memory: {e}")
            return False

    def reingest_for_model_switch(self, new_embedding_model: str) -> bool:
        """Re-ingest all documents when switching embedding models"""
        try:
            logger.info(f"Re-ingesting documents for new embedding model: {new_embedding_model}")
            
            # Update embedding model
            old_model = self.embedding_model
            self.embedding_model = new_embedding_model
            
            # Clear current models to force reload with new embedding model
            self.clear_gpu_memory()
            
            # Reload models with new embedding model
            self.ensure_models_loaded()
            
            # Recreate vector store for new embedding model
            self._migrate_or_recreate_vector_store()
            
            logger.info(f"Successfully switched from {old_model} to {new_embedding_model}")
            return True
            
        except Exception as e:
            logger.error(f"Error during model switch re-ingestion: {e}")
            return False

    def reingest_all_documents_for_current_model(self) -> bool:
        """Re-ingest all documents for the current embedding model"""
        try:
            logger.info(f"Re-ingesting all documents for embedding model: {self.embedding_model}")
            
            # First, clean up any orphaned documents (in DB but not in MinIO)
            orphaned_count = self.doc_storage.cleanup_orphaned_documents()
            if orphaned_count > 0:
                logger.info(f"Cleaned up {orphaned_count} orphaned documents before re-ingestion")
            
            # Get ALL documents (not just pending ones) for forced re-ingestion
            all_docs = self.doc_storage.list_all_documents()
            
            if not all_docs:
                logger.info("No documents found to re-ingest")
                return True
            
            logger.info(f"Found {len(all_docs)} documents to re-ingest")
            
            success_count = 0
            total_docs = len(all_docs)
            
            for doc in all_docs:
                try:
                    # First mark as pending for this model to ensure it gets re-ingested
                    self.doc_storage.mark_ingestion_pending(doc['id'], self.embedding_model)
                    
                    # Check if document exists in MinIO before attempting ingestion
                    if not self.doc_storage.document_exists_in_storage(doc['id']):
                        logger.warning(f"Document {doc['filename']} (ID: {doc['id']}) not found in MinIO storage - skipping")
                        self.doc_storage.mark_ingestion_failed(
                            doc['id'], 
                            self.embedding_model, 
                            "Document not found in MinIO storage"
                        )
                        continue
                    
                    # Attempt ingestion with detailed error handling
                    try:
                        success = self._ingest_for_current_model(doc['id'])
                        if success:
                            success_count += 1
                            logger.info(f"Successfully re-ingested: {doc['filename']} ({success_count}/{total_docs})")
                        else:
                            logger.warning(f"Failed to re-ingest: {doc['filename']}")
                            self.doc_storage.mark_ingestion_failed(
                                doc['id'], 
                                self.embedding_model, 
                                "Ingestion failed - see logs for details"
                            )
                    except ValueError as ve:
                        # Handle PDF extraction and validation errors more gracefully
                        error_msg = str(ve)
                        if "PDF" in error_msg or "extract text" in error_msg:
                            logger.warning(f"PDF processing failed for {doc['filename']}: {error_msg}")
                            self.doc_storage.mark_ingestion_failed(
                                doc['id'], 
                                self.embedding_model, 
                                f"PDF processing error: {error_msg}"
                            )
                        else:
                            logger.error(f"Document validation failed for {doc['filename']}: {error_msg}")
                            self.doc_storage.mark_ingestion_failed(
                                doc['id'], 
                                self.embedding_model, 
                                f"Document validation error: {error_msg}"
                            )
                    except Exception as ie:
                        # Handle other ingestion errors
                        logger.error(f"Unexpected error re-ingesting {doc['filename']}: {str(ie)}")
                        self.doc_storage.mark_ingestion_failed(
                            doc['id'], 
                            self.embedding_model, 
                            f"Ingestion error: {str(ie)}"
                        )
                        
                except Exception as e:
                    logger.error(f"Error processing document {doc.get('filename', 'unknown')}: {str(e)}")
                    if doc.get('id'):
                        self.doc_storage.mark_ingestion_failed(
                            doc['id'], 
                            self.embedding_model, 
                            f"Processing error: {str(e)}"
                        )
            
            logger.info(f"Re-ingestion complete: {success_count}/{total_docs} documents successful")
            
            # Ensure retriever is properly initialized after re-ingestion
            if success_count > 0:
                retriever_initialized = self._ensure_retriever_initialized()
                if retriever_initialized:
                    logger.info("Retriever successfully reinitialized after re-ingestion")
                else:
                    logger.warning("Failed to reinitialize retriever after re-ingestion")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error during re-ingestion: {str(e)}")
            return False

    def get_ingested_documents(self) -> List[dict]:
        """Get list of documents ingested for current embedding model"""
        return self.doc_storage.get_documents_for_model(self.embedding_model)
    
    def get_all_documents(self) -> List[dict]:
        """Get list of all stored documents with their ingestion status"""
        return self.doc_storage.list_all_documents()
    
    def get_vector_store_stats(self) -> dict:
        """Get statistics about the vector store and collections"""
        try:
            if not self.vector_store:
                logger.warning("Vector store not initialized")
                return {
                    "error": "No vector store initialized",
                    "collections": 0,
                    "total_documents": 0,
                    "models_loaded": self.models_loaded,
                    "embedding_model": self.embedding_model
                }
            
            # Get ChromaDB client
            try:
                chroma_client = self.vector_store._client
            except AttributeError as e:
                logger.error(f"Vector store client not accessible: {e}")
                return {
                    "error": "Vector store client not accessible",
                    "collections": 0,
                    "total_documents": 0,
                    "vector_store_type": type(self.vector_store).__name__
                }
            
            # Get all collections
            try:
                collections = chroma_client.list_collections()
            except Exception as e:
                logger.error(f"Failed to list collections: {e}")
                return {
                    "error": f"Failed to list collections: {str(e)}",
                    "collections": 0,
                    "total_documents": 0
                }
            
            stats = {
                "total_collections": len(collections),
                "current_collection": self._get_collection_name(),
                "current_embedding_model": self.embedding_model,
                "collections": [],
                "size_bytes": 0  # Initialize size counter
            }
            
            total_docs = 0
            for collection in collections:
                try:
                    # Get collection info
                    collection_count = collection.count()
                    total_docs += collection_count
                    
                    stats["collections"].append({
                        "name": collection.name,
                        "count": collection_count,
                        "embedding_model": collection.name.replace("docs_", "").replace("_", "-")
                    })
                except Exception as e:
                    logger.warning(f"Error getting stats for collection {collection.name}: {e}")
                    stats["collections"].append({
                        "name": collection.name,
                        "count": 0,
                        "error": str(e)
                    })
            
            stats["total_documents"] = total_docs
            
            # Calculate approximate size by checking ChromaDB directory
            try:
                import os
                chroma_path = Path(self.chroma_path)
                if chroma_path.exists():
                    total_size = 0
                    for root, dirs, files in os.walk(chroma_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                total_size += os.path.getsize(file_path)
                            except OSError:
                                pass  # Skip files we can't access
                    stats["size_bytes"] = total_size
            except Exception as e:
                logger.warning(f"Could not calculate vector store size: {e}")
                stats["size_bytes"] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "error": str(e),
                "collections": 0,
                "total_documents": 0,
                "traceback": traceback.format_exc()
            }
    
    def get_available_embedding_models(self) -> List[str]:
        """Get list of available embedding models from Ollama"""
        try:
            # Get all models from Ollama
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get('models', [])
                
                # Filter for embedding models
                embedding_models = []
                for model in models:
                    model_name = model.get('name', '').lower()
                    # Check if it's an embedding model based on name patterns
                    if any(keyword in model_name for keyword in [
                        'embed', 'embedding', 'bge', 'e5', 'sentence-transformer',
                        'all-minilm', 'multilingual-e5'
                    ]):
                        embedding_models.append(model.get('name', ''))
                
                return embedding_models
            else:
                logger.error(f"Failed to fetch models from Ollama: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error getting available embedding models: {e}")
            return []
    
    def get_current_embedding_model(self) -> str:
        """Get the currently configured embedding model"""
        return self.embedding_model if self.embedding_model else ""
    
    def switch_embedding_model(self, model_name: str) -> bool:
        """Switch to a different embedding model"""
        try:
            # Validate model is available
            available_models = self.get_available_embedding_models()
            if model_name not in available_models:
                logger.error(f"Model {model_name} is not available")
                return False
            
            # Update the embedding model
            old_model = self.embedding_model
            self.embedding_model = model_name
            
            # Reinitialize the embeddings with new model
            try:
                self.embeddings = OllamaEmbeddings(
                    model=model_name,
                    base_url=self.ollama_url
                )
                
                # Clear the vector store to force reinitialization with new embeddings
                self.vector_store = None
                self.retriever = None
                
                # Update config if available
                if hasattr(self, 'config'):
                    self.config['embedding_model'] = model_name
                
                logger.info(f"Successfully switched embedding model from {old_model} to {model_name}")
                return True
                
            except Exception as e:
                # Rollback on failure
                self.embedding_model = old_model
                logger.error(f"Failed to switch embedding model: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error switching embedding model: {e}")
            return False

    async def stream_response(self, question: str, style: str = "standard"):
        """
        Generate a streaming response to a question based on the ingested documents
        
        Args:
            question: The user's question
            style: Response style ("standard", "conversational", "detailed")
        """
        try:
            # Ensure models are loaded
            self.ensure_models_loaded()
            
            # Try to ensure retriever is initialized before checking
            if not self.retriever:
                logger.info("Retriever not available, attempting to initialize from vector store")
                retriever_initialized = self._ensure_retriever_initialized()
                if not retriever_initialized:
                    logger.warning("No retriever available and could not initialize - no documents have been ingested")
                    yield "I don't have any documents to reference. Please upload some documents first."
                    return
            
            # Retrieve relevant documents
            relevant_docs = self.retriever.get_relevant_documents(question)
            
            if not relevant_docs:
                logger.warning("No relevant documents found for query")
                yield "I couldn't find any relevant information in the uploaded documents to answer your question."
                return
            
            # Create context from relevant documents
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Choose prompt based on style
            if style == "conversational":
                prompt = self.conversational_prompt
            else:
                prompt = self.prompt
            
            # Create the chain for streaming
            chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | prompt 
                | self.model
                | StrOutputParser()
            )
            
            # Stream the response
            async for chunk in chain.astream({"context": context, "question": question}):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in stream_response: {e}")
            yield f"An error occurred while processing your question: {str(e)}"

    def query(self, question: str, style: str = "standard") -> str:
        """
        Generate a non-streaming response to a question based on the ingested documents
        
        Args:
            question: The user's question
            style: Response style ("standard", "conversational", "detailed")
        """
        try:
            # Ensure models are loaded
            self.ensure_models_loaded()
            
            # Try to ensure retriever is initialized before checking
            if not self.retriever:
                logger.info("Retriever not available, attempting to initialize from vector store")
                retriever_initialized = self._ensure_retriever_initialized()
                if not retriever_initialized:
                    logger.warning("No retriever available and could not initialize - no documents have been ingested")
                    return "I don't have any documents to reference. Please upload some documents first."
            
            # Retrieve relevant documents
            relevant_docs = self.retriever.get_relevant_documents(question)
            
            if not relevant_docs:
                logger.warning("No relevant documents found for query")
                return "I couldn't find any relevant information in the uploaded documents to answer your question."
            
            # Create context from relevant documents
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Choose prompt based on style
            if style == "conversational":
                prompt = self.conversational_prompt
            else:
                prompt = self.prompt
            
            # Create the chain
            chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | prompt 
                | self.model
                | StrOutputParser()
            )
            
            # Get the response
            response = chain.invoke({"context": context, "question": question})
            return response
                
        except Exception as e:
            logger.error(f"Error in query: {e}")
            return f"An error occurred while processing your question: {str(e)}"
    
    def _backup_vector_store_to_minio(self, collection_name: str):
        """Backup vector store collection to MinIO"""
        try:
            import shutil
            import tempfile
            from pathlib import Path
            
            logger.info(f"Backing up vector store collection '{collection_name}' to MinIO")
            
            # Create temp directory for backup
            with tempfile.TemporaryDirectory() as temp_dir:
                backup_path = Path(temp_dir) / f"{collection_name}_backup"
                
                # Copy entire ChromaDB directory (ChromaDB stores everything in the main directory)
                chroma_db_path = Path(self.chroma_path)
                if chroma_db_path.exists():
                    shutil.copytree(chroma_db_path, backup_path)
                    
                    # Create tar archive
                    import tarfile
                    archive_path = Path(temp_dir) / f"{collection_name}.tar.gz"
                    with tarfile.open(archive_path, 'w:gz') as tar:
                        tar.add(backup_path, arcname=collection_name)
                    
                    # Upload to MinIO
                    minio_key = f"vector_stores/{collection_name}.tar.gz"
                    self.doc_storage.minio_client.fput_object(
                        self.doc_storage.bucket_name,
                        minio_key,
                        str(archive_path)
                    )
                    
                    logger.info(f"Vector store backed up to MinIO: {minio_key}")
                    return True
                else:
                    logger.warning(f"ChromaDB path not found: {chroma_db_path}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to backup vector store to MinIO: {e}")
            return False

    def _restore_vector_store_from_minio(self, collection_name: str):
        """Restore vector store collection from MinIO"""
        try:
            import tempfile
            import tarfile
            import shutil
            from pathlib import Path
            
            # Check if ChromaDB directory already exists and has data
            target_path = Path(self.chroma_path)
            if target_path.exists():
                # Check if it has a valid ChromaDB database
                sqlite_file = target_path / "chroma.sqlite3"
                if sqlite_file.exists() and sqlite_file.stat().st_size > 0:
                    logger.info(f"ChromaDB directory already exists with data, skipping restore")
                    # Still fix permissions just in case
                    self._fix_permissions_recursive(target_path)
                    return False  # Not restored from MinIO, but existing data found
            
            logger.info(f"Restoring vector store collection '{collection_name}' from MinIO")
            
            minio_key = f"vector_stores/{collection_name}.tar.gz"
            
            # Check if backup exists in MinIO
            try:
                self.doc_storage.minio_client.stat_object(self.doc_storage.bucket_name, minio_key)
            except Exception:
                logger.info(f"No backup found in MinIO for collection: {collection_name}")
                return False
            
            # Create temp directory for restoration
            with tempfile.TemporaryDirectory() as temp_dir:
                archive_path = Path(temp_dir) / f"{collection_name}.tar.gz"
                
                # Download from MinIO
                self.doc_storage.minio_client.fget_object(
                    self.doc_storage.bucket_name,
                    minio_key,
                    str(archive_path)
                )
                
                # Extract archive
                with tarfile.open(archive_path, 'r:gz') as tar:
                    tar.extractall(temp_dir)
                
                # Move to ChromaDB directory
                extracted_path = Path(temp_dir) / collection_name
                
                if extracted_path.exists():
                    # Remove existing ChromaDB directory if it exists
                    if target_path.exists():
                        shutil.rmtree(target_path)
                    
                    # Create parent directories
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Move extracted data
                    shutil.move(str(extracted_path), str(target_path))
                    
                    # CRITICAL: Fix permissions after restore to ensure writability
                    self._fix_permissions_recursive(target_path)
                    
                    logger.info(f"Vector store restored from MinIO with proper permissions: {collection_name}")
                    return True
                else:
                    logger.warning(f"Extracted path not found: {extracted_path}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to restore vector store from MinIO: {e}")
            return False

    def _ensure_retriever_initialized(self):
        """Ensure retriever is properly initialized with current vector store"""
        try:
            # Ensure embeddings are loaded first
            if not self.embeddings:
                logger.warning("Embeddings not loaded, cannot initialize retriever")
                return False
            
            # Get collection name for current model
            collection_name = self._get_collection_name()
            
            # Ensure proper permissions before any ChromaDB operations
            self._ensure_writable_before_operation()
            
            # Create settings for ChromaDB client
            settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
            
            client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=settings
            )
            
            try:
                # Check if collection exists and has data
                collection = client.get_collection(collection_name)
                doc_count = collection.count()
                
                if doc_count > 0:
                    # Initialize vector store if not already done
                    if not self.vector_store:
                        self.vector_store = Chroma(
                            client=client,
                            collection_name=collection_name,
                            embedding_function=self.embeddings,
                        )
                        logger.info(f"Vector store initialized for collection '{collection_name}'")
                    
                    # Initialize retriever
                    self.retriever = self.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 3}
                    )
                    logger.info(f"Retriever reinitialized with {doc_count} documents")
                    return True
                else:
                    logger.warning(f"Collection '{collection_name}' is empty")
                    return False
                    
            except Exception as e:
                logger.warning(f"Collection '{collection_name}' does not exist: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error ensuring retriever initialization: {e}")
            return False

    def _periodic_permission_check(self):
        """Periodically check and fix ChromaDB permissions"""
        try:
            if os.path.exists(self.chroma_path):
                # Check if we can write to the directory
                test_file = Path(self.chroma_path) / ".permission_test"
                try:
                    test_file.touch()
                    test_file.unlink()
                    logger.debug("ChromaDB permissions are correct")
                except (PermissionError, OSError) as e:
                    logger.warning(f"ChromaDB permission issue detected: {e}")
                    self._fix_permissions_recursive(self.chroma_path)
                    logger.info("Fixed ChromaDB permissions")
        except Exception as e:
            logger.error(f"Error during permission check: {e}")

    def _handle_chromadb_operation(self, operation_func, *args, **kwargs):
        """Wrapper to handle ChromaDB operations with automatic permission fixing"""
        # Debug the documents being added
        if hasattr(operation_func, '__name__') and operation_func.__name__ == 'add_documents':
            if args and len(args) > 0:
                docs = args[0]
                logger.info(f"🔍 ADD_DOCUMENTS DEBUG: Adding {len(docs)} documents")
                for i, doc in enumerate(docs[:3]):  # Log first 3 documents
                    logger.info(f"🔍 DOC {i}: metadata = {doc.metadata}")
                    logger.info(f"🔍 DOC {i}: content preview = {doc.page_content[:100]}...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = operation_func(*args, **kwargs)
                
                # Log result if it's add_documents
                if hasattr(operation_func, '__name__') and operation_func.__name__ == 'add_documents':
                    logger.info(f"✅ ADD_DOCUMENTS SUCCESS: Operation completed")
                
                return True  # Return True for success
            except Exception as e:
                error_msg = str(e).lower()
                if any(perm_error in error_msg for perm_error in 
                       ['readonly database', 'permission denied', 'database is locked', 'attempt to write a readonly database']):
                    logger.warning(f"ChromaDB permission error detected on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        # Try to fix permissions and retry
                        try:
                            self._fix_permissions_recursive(self.chroma_path)
                            logger.info(f"Fixed permissions, retrying operation (attempt {attempt + 2})")
                            time.sleep(1)  # Small delay before retry
                            continue
                        except Exception as fix_error:
                            logger.error(f"Failed to fix permissions: {fix_error}")
                    else:
                        # Last resort: recreate ChromaDB directory
                        logger.warning("Attempting to recreate ChromaDB directory as last resort")
                        if self._recreate_chromadb_directory():
                            # Try to reinitialize vector store
                            try:
                                self._initialize_vector_store()
                                # Retry the operation once more
                                operation_func(*args, **kwargs)
                                return True
                            except Exception as reinit_error:
                                logger.error(f"Failed to reinitialize after directory recreation: {reinit_error}")
                        
                        logger.error(f"ChromaDB operation failed after {max_retries} attempts: {e}")
                        return False
                else:
                    # Non-permission error, re-raise immediately
                    logger.error(f"ChromaDB operation failed: {e}")
                    return False
        
        # This should never be reached
        return False

    def _recreate_chromadb_directory(self):
        """Recreate ChromaDB directory from scratch as a last resort"""
        try:
            import shutil
            logger.warning("Recreating ChromaDB directory due to persistent permission issues")
            
            # Remove the entire ChromaDB directory
            if os.path.exists(self.chroma_path):
                shutil.rmtree(self.chroma_path)
            
            # Recreate with proper permissions
            os.makedirs(self.chroma_path, mode=0o777, exist_ok=True)
            self._fix_permissions_recursive(self.chroma_path)
            
            # Reset vector store and retriever since we cleared the data
            self.vector_store = None
            self.retriever = None
            
            logger.info("ChromaDB directory recreated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to recreate ChromaDB directory: {e}")
            return False

    def _test_chromadb_write_access(self):
        """Test if ChromaDB database is writable"""
        try:
            sqlite_file = Path(self.chroma_path) / "chroma.sqlite3"
            if sqlite_file.exists():
                # Try to open the database in write mode
                import sqlite3
                conn = sqlite3.connect(str(sqlite_file), timeout=5.0)
                # Try a simple write operation
                cursor = conn.cursor()
                cursor.execute("PRAGMA user_version")
                conn.close()
                return True
            return True  # File doesn't exist yet, should be writable
        except Exception as e:
            logger.error(f"ChromaDB write test failed: {e}")
            return False

    def remove_document_from_vectorstore(self, filename: str) -> bool:
        """Remove document chunks from vector store by filename"""
        try:
            if not self.vector_store:
                logger.warning("Vector store not initialized")
                return False
            
            # Get collection name for current embedding model
            collection_name = f"collection_{self.embedding_model.replace('/', '_').replace('-', '_')}"
            
            # Get ChromaDB client
            chroma_client = self.vector_store._client
            
            # Get the collection
            try:
                collection = chroma_client.get_collection(collection_name)
            except Exception as e:
                logger.warning(f"Collection {collection_name} not found: {e}")
                return True  # Consider it success if collection doesn't exist
            
            # Query for documents with this filename in metadata
            try:
                results = collection.get(
                    where={"source": filename}
                )
                
                if results and results.get('ids'):
                    # Delete all chunks for this document
                    collection.delete(ids=results['ids'])
                    logger.info(f"Removed {len(results['ids'])} chunks for document {filename}")
                    return True
                else:
                    logger.info(f"No chunks found for document {filename}")
                    return True
                    
            except Exception as e:
                logger.error(f"Error removing document {filename} from vector store: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error removing document from vectorstore: {e}")
            return False

    def remove_document_by_id(self, document_id: int, embedding_models: List[str] = None) -> bool:
        """Remove document chunks from vector store by document ID across embedding models"""
        try:
            if not self.vector_store:
                logger.warning("Vector store not initialized")
                return False
            
            # If no specific models provided, get all models that have ingested this document
            if not embedding_models:
                try:
                    from document_storage import get_document_storage
                    doc_storage = get_document_storage()
                    
                    # Get all embedding models that have ingested this document
                    conn = sqlite3.connect(doc_storage.db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        SELECT DISTINCT embedding_model 
                        FROM ingestion_metadata 
                        WHERE document_id = ? AND ingestion_status = 'completed'
                    ''', (document_id,))
                    
                    rows = cursor.fetchall()
                    conn.close()
                    
                    embedding_models = [row[0] for row in rows] if rows else [self.embedding_model]
                    
                except Exception as e:
                    logger.warning(f"Could not determine embedding models for document {document_id}: {e}")
                    # Fallback to current embedding model
                    embedding_models = [self.embedding_model]
            
            total_deleted = 0
            
            # Get ChromaDB client
            chroma_client = self.vector_store._client
            
            # Remove from all relevant embedding model collections
            for model in embedding_models:
                try:
                    collection_name = f"collection_{model.replace('/', '_').replace('-', '_')}"
                    
                    # Get the collection
                    try:
                        collection = chroma_client.get_collection(collection_name)
                    except Exception as e:
                        logger.warning(f"Collection {collection_name} not found: {e}")
                        continue
                    
                    # Query for documents with this document_id in metadata
                    try:
                        results = collection.get(
                            where={"document_id": str(document_id)}
                        )
                        
                        if results and results.get('ids'):
                            # Delete all chunks for this document
                            collection.delete(ids=results['ids'])
                            deleted_count = len(results['ids'])
                            total_deleted += deleted_count
                            logger.info(f"Removed {deleted_count} chunks for document {document_id} from model {model}")
                        else:
                            logger.info(f"No chunks found for document {document_id} in model {model}")
                            
                    except Exception as e:
                        logger.error(f"Error removing document {document_id} from vector store {model}: {e}")
                        continue
                        
                except Exception as e:
                    logger.error(f"Error processing model {model} for document {document_id}: {e}")
                    continue
            
            logger.info(f"Successfully removed document {document_id} from vector stores. Total chunks deleted: {total_deleted}")
            return True
                
        except Exception as e:
            logger.error(f"Error removing document {document_id} from vectorstore: {e}")
            return False

    def clear_vectorstore(self):
        """Clear the entire vector store"""
        try:
            if self.vector_store and hasattr(self.vector_store, '_collection'):
                collection = self.vector_store._collection
                if collection:
                    # Delete all documents from the collection
                    collection.delete()
                    logger.info("Vector store cleared successfully")
                    return True
            logger.warning("Vector store not available for clearing")
            return False
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False

    def get_vectorstore_stats(self) -> dict:
        """Get statistics about the vector store - alias for get_vector_store_stats with debug info"""
        logger.info("🔍 DEBUG: get_vectorstore_stats() called")
        try:
            logger.info(f"🔍 DEBUG: vector_store exists: {self.vector_store is not None}")
            logger.info(f"🔍 DEBUG: models_loaded: {self.models_loaded}")
            logger.info(f"🔍 DEBUG: embedding_model: {self.embedding_model}")
            
            # Force initialization if not done
            if not self.vector_store:
                logger.warning("🔍 DEBUG: Vector store not initialized, attempting initialization")
                self._initialize_vector_store()
                logger.info(f"🔍 DEBUG: After initialization, vector_store exists: {self.vector_store is not None}")
            
            logger.info("🔍 DEBUG: Calling get_vector_store_stats()")
            result = self.get_vector_store_stats()
            logger.info(f"🔍 DEBUG: get_vector_store_stats() returned: {type(result)}")
            return result
        except Exception as e:
            logger.error(f"🔍 DEBUG: Error in get_vectorstore_stats: {e}")
            import traceback
            logger.error(f"🔍 DEBUG: Traceback: {traceback.format_exc()}")
            return {
                "error": f"Failed to get vectorstore stats: {str(e)}",
                "collections": 0,
                "total_documents": 0,
                "vector_store_initialized": self.vector_store is not None,
                "models_loaded": self.models_loaded
            }
    
    def reingest_specific_documents(self, document_ids: List[int], 
                                   chunking_method: ChunkingMethod = None,
                                   chunking_config: ChunkingConfig = None) -> Dict[str, any]:
        """
        Reingest specific documents with optional new chunking configuration
        
        Flow:
        1. Check if documents exist in DB and MinIO
        2. Delete old chunks from vector store for each document+model
        3. Download files from MinIO (no re-upload needed)
        4. Re-apply chunking using stored or provided chunking configuration
        5. Regenerate embeddings using current embedding model
        6. Re-index chunks into vector store
        7. Update ingestion metadata in SQLite DB
        """
        try:
            logger.info(f"Starting reingestion of {len(document_ids)} documents")
            
            results = {
                'total': len(document_ids),
                'successful': 0,
                'failed': 0,
                'failed_ids': [],
                'details': []
            }
            
            for doc_id in document_ids:
                try:
                    # Step 1: Check if document exists in DB and MinIO
                    doc_info = self.doc_storage.get_document_with_config(doc_id)
                    if not doc_info:
                        logger.warning(f"Document {doc_id} not found in database")
                        results['failed'] += 1
                        results['failed_ids'].append(doc_id)
                        results['details'].append({
                            'document_id': doc_id,
                            'status': 'failed',
                            'error': 'Document not found in database'
                        })
                        continue
                    
                    if not self.doc_storage.document_exists_in_storage(doc_id):
                        logger.warning(f"Document {doc_id} not found in MinIO storage")
                        results['failed'] += 1
                        results['failed_ids'].append(doc_id)
                        results['details'].append({
                            'document_id': doc_id,
                            'filename': doc_info['filename'],
                            'status': 'failed',
                            'error': 'Document not found in MinIO storage'
                        })
                        continue
                    
                    # Step 2: Delete old chunks from vector store for this document+model
                    logger.info(f"Removing old chunks for document {doc_id}")
                    try:
                        self.remove_document_by_id(doc_id, [self.embedding_model])
                    except Exception as e:
                        logger.warning(f"Could not remove old chunks for document {doc_id}: {e}")
                    
                    # Step 3: Download file from MinIO (temporary file)
                    temp_file_path = self.doc_storage.get_document_file(doc_id)
                    
                    try:
                        # Step 4: Determine chunking configuration to use
                        if chunking_method is None:
                            # Use stored chunking method
                            stored_method = doc_info.get('chunking_method', 'general')
                            try:
                                chunking_method = ChunkingMethod(stored_method)
                            except ValueError:
                                logger.warning(f"Unknown stored chunking method '{stored_method}', using general")
                                chunking_method = ChunkingMethod.GENERAL
                        
                        if chunking_config is None:
                            # Use stored chunking config if available
                            stored_config = doc_info.get('chunking_config')
                            if stored_config:
                                chunking_config = ChunkingConfig.from_dict(stored_config)
                            else:
                                # Get default config for the method
                                from chunking_config import get_chunking_config_manager
                                config_manager = get_chunking_config_manager()
                                chunking_config = config_manager.get_config(chunking_method)
                        
                        logger.info(f"Reingesting document {doc_id} with method {chunking_method.value}")
                        
                        # Step 5-7: Process document with enhanced processor
                        from enhanced_document_processor import get_document_processor
                        doc_processor = get_document_processor()
                        
                        chunking_result = doc_processor.process_document(
                            temp_file_path,
                            chunking_method,
                            chunking_config,
                            None,  # user_id not needed for reingestion
                            doc_info['filename']
                        )
                        
                        if not chunking_result.chunks:
                            raise ValueError("No chunks created during reprocessing")
                        
                        logger.info(f"Document {doc_id} reprocessed: {len(chunking_result.chunks)} chunks created")
                        
                        # Add chunks to vector store
                        collection_name = self._get_collection_name()
                        success = self._add_chunks_to_vector_store(chunking_result.chunks, collection_name)
                        
                        if success:
                            # Update ingestion metadata
                            simple_metadata = {
                                'total_chunks': len(chunking_result.chunks),
                                'method_used': chunking_result.method_used.value,
                                'processing_time': chunking_result.metadata.get('processing_time', 0) if chunking_result.metadata else 0,
                                'file_size': chunking_result.metadata.get('file_size', 0) if chunking_result.metadata else 0,
                                'warnings_count': len(chunking_result.warnings),
                                'reingestion': True
                            }
                            
                            if chunking_result.warnings:
                                simple_metadata['warnings'] = '; '.join(chunking_result.warnings[:3])
                            
                            self.doc_storage.track_ingestion(
                                document_id=doc_id,
                                embedding_model=self.embedding_model,
                                vector_store_collection=collection_name,
                                chunk_count=len(chunking_result.chunks),
                                metadata=simple_metadata,
                                chunking_method=chunking_result.method_used.value,
                                chunking_config=chunking_result.config_used.to_dict()
                            )
                            
                            results['successful'] += 1
                            results['details'].append({
                                'document_id': doc_id,
                                'filename': doc_info['filename'],
                                'status': 'success',
                                'chunks_created': len(chunking_result.chunks),
                                'method_used': chunking_result.method_used.value
                            })
                            
                            logger.info(f"Successfully reingested document {doc_id}: {doc_info['filename']}")
                        else:
                            raise ValueError("Failed to add chunks to vector store")
                    
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                
                except Exception as e:
                    logger.error(f"Error reingesting document {doc_id}: {e}")
                    results['failed'] += 1
                    results['failed_ids'].append(doc_id)
                    results['details'].append({
                        'document_id': doc_id,
                        'filename': doc_info.get('filename', f'document_{doc_id}'),
                        'status': 'failed',
                        'error': str(e)
                    })
                    
                    # Mark as failed in database
                    self.doc_storage.mark_ingestion_failed(
                        doc_id,
                        self.embedding_model,
                        f"Reingestion failed: {str(e)}"
                    )
            
            logger.info(f"Reingestion complete: {results['successful']}/{results['total']} successful")
            
            # Ensure retriever is updated if any documents were successfully reingested
            if results['successful'] > 0:
                try:
                    self._ensure_retriever_initialized()
                    logger.info("Retriever reinitialized after reingestion")
                except Exception as e:
                    logger.warning(f"Failed to reinitialize retriever: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during document reingestion: {e}")
            return {
                'total': len(document_ids),
                'successful': 0,
                'failed': len(document_ids),
                'failed_ids': document_ids,
                'error': str(e)
            }

    def reingest_specific_documents_with_config(self, documents_config: List[dict]) -> dict:
        """
        Re-ingest specific documents with per-document chunking configuration
        Each document can have its own chunking method and config
        
        Args:
            documents_config: List of dicts with keys:
                - document_id: int
                - chunking_method: ChunkingMethod (optional)
                - chunking_config: ChunkingConfig (optional)
        
        Returns:
            Dict with reingestion results
        """
        try:
            logger.info(f"Starting reingestion of {len(documents_config)} documents with per-document configuration")
            
            results = {
                'total': len(documents_config),
                'successful': 0,
                'failed': 0,
                'failed_ids': [],
                'details': []
            }
            
            for doc_config in documents_config:
                doc_id = doc_config['document_id']
                chunking_method = doc_config.get('chunking_method')
                chunking_config = doc_config.get('chunking_config')
                
                try:
                    # Step 1: Check if document exists in DB and MinIO
                    doc_info = self.doc_storage.get_document_with_config(doc_id)
                    if not doc_info:
                        logger.warning(f"Document {doc_id} not found in database")
                        results['failed'] += 1
                        results['failed_ids'].append(doc_id)
                        results['details'].append({
                            'document_id': doc_id,
                            'status': 'failed',
                            'error': 'Document not found in database'
                        })
                        continue
                    
                    if not self.doc_storage.document_exists_in_storage(doc_id):
                        logger.warning(f"Document {doc_id} not found in MinIO storage")
                        results['failed'] += 1
                        results['failed_ids'].append(doc_id)
                        results['details'].append({
                            'document_id': doc_id,
                            'filename': doc_info['filename'],
                            'status': 'failed',
                            'error': 'Document not found in MinIO storage'
                        })
                        continue
                    
                    # Step 2: Delete old chunks from vector store for this document+model
                    logger.info(f"Removing old chunks for document {doc_id}")
                    try:
                        self.remove_document_by_id(doc_id, [self.embedding_model])
                    except Exception as e:
                        logger.warning(f"Could not remove old chunks for document {doc_id}: {e}")
                    
                    # Step 3: Download file from MinIO (temporary file)
                    temp_file_path = self.doc_storage.get_document_file(doc_id)
                    
                    try:
                        # Step 4: Determine chunking configuration to use
                        if chunking_method is None:
                            # Use stored chunking method
                            stored_method = doc_info.get('chunking_method', 'general')
                            try:
                                chunking_method = ChunkingMethod(stored_method)
                            except ValueError:
                                logger.warning(f"Unknown stored chunking method '{stored_method}', using general")
                                chunking_method = ChunkingMethod.GENERAL
                        
                        if chunking_config is None:
                            # Use stored chunking config if available
                            stored_config = doc_info.get('chunking_config')
                            if stored_config:
                                chunking_config = ChunkingConfig.from_dict(stored_config)
                            else:
                                # Get default config for the method
                                from chunking_config import get_chunking_config_manager
                                config_manager = get_chunking_config_manager()
                                chunking_config = config_manager.get_config(chunking_method)
                        
                        logger.info(f"Reingesting document {doc_id} with method {chunking_method.value}")
                        
                        # Step 5-7: Process document with enhanced processor
                        from enhanced_document_processor import get_document_processor
                        doc_processor = get_document_processor()
                        
                        chunking_result = doc_processor.process_document(
                            temp_file_path,
                            chunking_method,
                            chunking_config,
                            None,  # user_id not needed for reingestion
                            doc_info['filename']
                        )
                        
                        if not chunking_result.chunks:
                            results['failed'] += 1
                            results['failed_ids'].append(doc_id)
                            results['details'].append({
                                'document_id': doc_id,
                                'filename': doc_info['filename'],
                                'status': 'failed',
                                'error': 'No chunks generated during processing'
                            })
                            continue
                        
                        # Add to vector store using existing ingest flow
                        collection_name = f"embeddings_{self.embedding_model.replace('-', '_')}"
                        
                        # Track ingestion metadata
                        simple_metadata = {
                            'method_used': chunking_result.method_used.value,
                            'chunk_count': len(chunking_result.chunks),
                            'file_format': chunking_result.metadata.get('file_format', 'unknown') if chunking_result.metadata else 'unknown',
                            'processing_time': chunking_result.metadata.get('processing_time', 0) if chunking_result.metadata else 0,
                            'file_size': chunking_result.metadata.get('file_size', 0) if chunking_result.metadata else 0,
                            'warnings_count': len(chunking_result.warnings) if chunking_result.warnings else 0
                        }
                        
                        if chunking_result.warnings:
                            simple_metadata['warnings'] = '; '.join(chunking_result.warnings[:3])
                        
                        # Add chunks directly to vector store
                        success = self._add_chunks_to_vector_store(chunking_result.chunks, collection_name)
                        
                        if success:
                            # Update document metadata in storage
                            try:
                                self.storage.update_document_metadata(
                                    doc_id,
                                    chunking_method=chunking_result.method_used.value,
                                    chunk_count=len(chunking_result.chunks),
                                    metadata=simple_metadata
                                )
                            except Exception as storage_error:
                                logger.warning(f"Failed to update document metadata: {storage_error}")
                                # Continue anyway as the reingestion was successful
                        
                        if success:
                            results['successful'] += 1
                            results['details'].append({
                                'document_id': doc_id,
                                'filename': doc_info['filename'],
                                'status': 'success',
                                'chunks_created': len(chunking_result.chunks),
                                'method_used': chunking_result.method_used.value
                            })
                            logger.info(f"Successfully reingested document {doc_id}: {doc_info['filename']}")
                        else:
                            results['failed'] += 1
                            results['failed_ids'].append(doc_id)
                            results['details'].append({
                                'document_id': doc_id,
                                'filename': doc_info['filename'],
                                'status': 'failed',
                                'error': 'Failed during vector store ingestion'
                            })
                    
                    finally:
                        # Clean up temporary file
                        if temp_file_path and os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                
                except Exception as e:
                    logger.error(f"Error reingesting document {doc_id}: {e}")
                    results['failed'] += 1
                    results['failed_ids'].append(doc_id)
                    results['details'].append({
                        'document_id': doc_id,
                        'status': 'failed',
                        'error': str(e)
                    })
            
            logger.info(f"Reingestion complete: {results['successful']}/{results['total']} successful")
            
            # Ensure retriever is updated if any documents were successfully reingested
            if results['successful'] > 0:
                try:
                    self._ensure_retriever_initialized()
                    logger.info("Retriever reinitialized after reingestion")
                except Exception as e:
                    logger.warning(f"Failed to reinitialize retriever: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during document reingestion with config: {e}")
            return {
                'total': len(documents_config),
                'successful': 0,
                'failed': len(documents_config),
                'failed_ids': [doc['document_id'] for doc in documents_config],
                'error': str(e)
            }

    def reingest_all_documents(self) -> bool:
        """Re-ingest all documents into vector store"""
        logger.info("🔍 DEBUG: reingest_all_documents() called")
        return self.reingest_all_documents_for_current_model()

    def ingest_with_storage_and_chunking(self, file_path: str, original_filename: str = None,
                                        chunking_method: ChunkingMethod = None, 
                                        chunking_config: ChunkingConfig = None,
                                        user_id: str = None) -> bool:
        """
        Ingest a document using enhanced processing with configurable chunking methods
        """
        try:
            logger.info(f"Starting enhanced ingestion: {file_path}")
            
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False

            # Use provided original filename or extract from path
            if original_filename is None:
                original_filename = Path(file_path).name
            
            # Determine file type and optimal chunking method
            file_ext = Path(file_path).suffix[1:]  # Remove dot
            if chunking_method is None:
                chunking_method = FileFormatSupport.get_optimal_method(file_ext)
            
            # Get chunking configuration
            config_manager = get_chunking_config_manager()
            if chunking_config is None:
                chunking_config = config_manager.get_config(chunking_method, user_id)
            
            # Validate that the method supports this file type
            if not FileFormatSupport.is_supported(chunking_method, file_ext):
                logger.warning(f"Method {chunking_method.value} not supported for {file_ext}, using naive")
                chunking_method = ChunkingMethod.NAIVE
                chunking_config = config_manager.get_config(chunking_method, user_id)
            
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
            
            # Store document in MinIO with chunking configuration
            doc_info = self.doc_storage.store_document(
                file_path, 
                original_filename, 
                content_type,
                chunking_method.value,
                chunking_config.to_dict()
            )
            
            logger.info(f"Document stored with ID: {doc_info['id']}")
            
            # Process document with enhanced processor
            doc_processor = get_document_processor()
            
            try:
                chunking_result = doc_processor.process_document(
                    file_path, 
                    chunking_method, 
                    chunking_config, 
                    user_id,
                    original_filename  # Pass the original filename
                )
                
                logger.info(f"Document processed: {len(chunking_result.chunks)} chunks created using {chunking_result.method_used.value}")
                
                # Log any warnings
                if chunking_result.warnings:
                    for warning in chunking_result.warnings:
                        logger.warning(f"Chunking warning: {warning}")
                
                # Add chunks to vector store
                collection_name = self._get_collection_name()
                success = self._add_chunks_to_vector_store(chunking_result.chunks, collection_name)
                
                if success:
                    # Track successful ingestion with chunking information
                    # Create simple metadata instead of spreading complex objects
                    simple_metadata = {
                        'total_chunks': len(chunking_result.chunks),
                        'method_used': chunking_result.method_used.value,
                        'processing_time': chunking_result.metadata.get('processing_time', 0) if chunking_result.metadata else 0,
                        'file_size': chunking_result.metadata.get('file_size', 0) if chunking_result.metadata else 0,
                        'warnings_count': len(chunking_result.warnings)
                    }
                    
                    # Add warning messages as a simple string
                    if chunking_result.warnings:
                        simple_metadata['warnings'] = '; '.join(chunking_result.warnings[:3])  # Limit to first 3 warnings
                    
                    self.doc_storage.track_ingestion(
                        document_id=doc_info['id'],
                        embedding_model=self.embedding_model,
                        vector_store_collection=collection_name,
                        chunk_count=len(chunking_result.chunks),
                        metadata=simple_metadata,
                        chunking_method=chunking_result.method_used.value,
                        chunking_config=chunking_result.config_used.to_dict()
                    )
                    logger.info(f"Successfully ingested document {doc_info['id']} with {chunking_method.value} chunking")
                    return True
                else:
                    self.doc_storage.mark_ingestion_failed(
                        doc_info['id'], 
                        self.embedding_model, 
                        "Failed to add chunks to vector store"
                    )
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to process document with enhanced processor: {e}")
                self.doc_storage.mark_ingestion_failed(
                    doc_info['id'], 
                    self.embedding_model, 
                    f"Document processing failed: {str(e)}"
                )
                return False
            
        except Exception as e:
            logger.error(f"Enhanced document ingestion failed: {str(e)}", exc_info=True)
            return False


def check_model_compatibility(model_name, model_size=None):
    """
    Check if a model is compatible with the current system.
    Returns: (is_compatible, message, details)
    """
    try:
        # Import the detailed compatibility check from main
        from main import check_model_compatibility_detailed
        return check_model_compatibility_detailed(model_name, model_size)
    except ImportError:
        # Fallback implementation if main functions are not available
        logger.warning("Could not import detailed GPU compatibility check, using fallback")
        
        if not model_name:
            return False, "No model specified", {}
        
        # Basic compatibility check - if model name is valid, consider it compatible
        return True, f"Model {model_name} is compatible (fallback check)", {
            "model": model_name,
            "size": model_size or "Unknown",
            "status": "compatible_fallback"
        }
    except Exception as e:
        logger.error(f"Error checking model compatibility for {model_name}: {e}")
        return False, f"Error checking compatibility: {str(e)}", {}


# Global singleton instance
_chatpdf_instance = None

def get_chatpdf_instance():
    """Get singleton instance of ChatPDF"""
    global _chatpdf_instance
    if _chatpdf_instance is None:
        _chatpdf_instance = ChatPDF()
    return _chatpdf_instance

def reset_chatpdf_instance():
    """Reset the singleton instance (useful for testing or model changes)"""
    global _chatpdf_instance
    if _chatpdf_instance:
        # Clear GPU memory before resetting
        try:
            _chatpdf_instance.clear_gpu_memory()
        except Exception as e:
            logger.warning(f"Error clearing GPU memory during reset: {e}")
    _chatpdf_instance = None