# rag.py
from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.config import Settings
import os
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
import time
from PyPDF2 import PdfReader  # Add this import
import time
import functools
from langchain.schema import Document
from document_storage import get_document_storage
import subprocess
import re

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

            # Clean metadata
            chunks = filter_complex_metadata(chunks)

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
            
            # Clean metadata
            chunks = filter_complex_metadata(chunks)
            
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
        """Extract text from PDF using multiple methods"""
        try:
            # First try PyPDF2
            logger.debug("Attempting to read PDF with PyPDF2")
            pdf = PdfReader(pdf_file_path)
            text_content = []
            for page in pdf.pages:
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    text_content.append(text)
            
            if text_content:
                combined_text = "\n\n".join(text_content)
                logger.debug(f"Extracted {len(text_content)} pages of text with PyPDF2")
                return combined_text
            
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {str(e)}")
        
        try:
            # Fallback to langchain's PyPDFLoader
            logger.debug("Attempting to read PDF with langchain PyPDFLoader")
            docs = PyPDFLoader(file_path=pdf_file_path).load()
            if docs:
                combined_text = "\n\n".join([doc.page_content for doc in docs])
                logger.debug(f"Extracted text from {len(docs)} pages with PyPDFLoader")
                return combined_text
                
        except Exception as e:
            logger.warning(f"PyPDFLoader extraction failed: {str(e)}")
        
        raise ValueError("Failed to extract text from PDF using any method")
    
    def _add_chunks_to_vector_store(self, chunks: List[Document], collection_name: str) -> bool:
        """Add document chunks to the vector store"""
        try:
            # Ensure ChromaDB directory has proper permissions
            self._ensure_writable_before_operation()
            
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
            
            # Add documents to vector store
            self._handle_chromadb_operation(self.vector_store.add_documents, chunks)
            
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
            
            logger.info(f"Added {len(chunks)} chunks to vector store collection '{collection_name}'")
            return True
            
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
                    
                    success = self._ingest_for_current_model(doc['id'])
                    if success:
                        success_count += 1
                        logger.info(f"Successfully re-ingested: {doc['filename']} ({success_count}/{total_docs})")
                    else:
                        logger.warning(f"Failed to re-ingest: {doc['filename']}")
                except Exception as e:
                    logger.error(f"Error re-ingesting {doc['filename']}: {str(e)}")
            
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
                return {
                    "error": "No vector store initialized",
                    "collections": 0,
                    "total_documents": 0
                }
            
            # Get ChromaDB client
            chroma_client = self.vector_store._client
            
            # Get all collections
            collections = chroma_client.list_collections()
            
            stats = {
                "total_collections": len(collections),
                "current_collection": self._get_collection_name(),
                "current_embedding_model": self.embedding_model,
                "collections": []
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
            return stats
            
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {
                "error": str(e),
                "collections": 0,
                "total_documents": 0
            }

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
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return operation_func(*args, **kwargs)
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
                                return operation_func(*args, **kwargs)
                            except Exception as reinit_error:
                                logger.error(f"Failed to reinitialize after directory recreation: {reinit_error}")
                        
                        logger.error(f"ChromaDB operation failed after {max_retries} attempts: {e}")
                        raise
                else:
                    # Non-permission error, re-raise immediately
                    raise
        
        # This should never be reached
        raise RuntimeError("Unexpected state in ChromaDB operation handler")

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

def get_gpu_memory_info() -> Dict[str, int]:
    """Get GPU memory information in MB"""
    try:
        # Try nvidia-smi first
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines:
                # Take the first GPU
                memory_info = lines[0].split(', ')
                if len(memory_info) >= 3:
                    total_mb = int(memory_info[0])
                    used_mb = int(memory_info[1])
                    free_mb = int(memory_info[2])
                    return {
                        'total': total_mb,
                        'used': used_mb,
                        'free': free_mb,
                        'available': free_mb
                    }
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass
    
    try:
        # Fallback: Try to get info from /proc/driver/nvidia/gpus/
        gpu_dirs = [d for d in os.listdir('/proc/driver/nvidia/gpus/') if os.path.isdir(f'/proc/driver/nvidia/gpus/{d}')]
        if gpu_dirs:
            # Read memory info from the first GPU
            gpu_dir = gpu_dirs[0]
            with open(f'/proc/driver/nvidia/gpus/{gpu_dir}/information', 'r') as f:
                content = f.read()
                # Extract memory info
                memory_match = re.search(r'Video Memory:\s+(\d+)\s+MB', content)
                if memory_match:
                    total_mb = int(memory_match.group(1))
                    # Estimate available as 80% of total (conservative)
                    available_mb = int(total_mb * 0.8)
                    return {
                        'total': total_mb,
                        'used': total_mb - available_mb,
                        'free': available_mb,
                        'available': available_mb
                    }
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    
    # If no GPU info available, return default values
    logger.warning("Could not determine GPU memory, using default estimates")
    return {
        'total': 8192,  # 8GB default
        'used': 2048,   # 2GB used
        'free': 6144,   # 6GB free
        'available': 6144
    }

def estimate_model_memory_requirement(model_name: str, model_size: str = None) -> int:
    """Estimate memory requirement for a model in MB"""
    name_lower = model_name.lower()
    
    # If size is provided, try to parse it
    if model_size and isinstance(model_size, str):
        size_lower = model_size.lower()
        # Extract numeric value from size string
        size_match = re.search(r'(\d+\.?\d*)', size_lower)
        if size_match:
            size_value = float(size_match.group(1))
            
            # Convert based on unit
            if 'gb' in size_lower:
                return int(size_value * 1024)  # Convert GB to MB
            elif 'mb' in size_lower:
                return int(size_value)
            elif 'b' in size_lower and 'gb' not in size_lower and 'mb' not in size_lower:
                # Assume it's parameters (e.g., "7b", "13b")
                # Rule of thumb: 1B parameters ≈ 2GB in FP16, ≈ 1GB in Q4
                return int(size_value * 1500)  # Conservative estimate for Q4 quantization
    
    # Fallback: estimate based on model name patterns
    if any(size in name_lower for size in ['0.5b', '500m']):
        return 1024   # ~1GB
    elif any(size in name_lower for size in ['1b', '1.5b']):
        return 2048   # ~2GB
    elif any(size in name_lower for size in ['3b', '2.8b']):
        return 4096   # ~4GB
    elif any(size in name_lower for size in ['7b', '6.7b', '8b']):
        return 8192   # ~8GB
    elif any(size in name_lower for size in ['13b', '14b', '15b']):
        return 16384  # ~16GB
    elif any(size in name_lower for size in ['30b', '32b', '34b']):
        return 32768  # ~32GB
    elif any(size in name_lower for size in ['70b', '72b']):
        return 65536  # ~64GB
    elif any(size in name_lower for size in ['175b', '180b']):
        return 131072 # ~128GB
    
    # Embedding models are typically smaller
    if any(keyword in name_lower for keyword in ['embed', 'bge', 'minilm', 'e5', 'sentence']):
        if 'large' in name_lower:
            return 1024   # ~1GB for large embedding models
        else:
            return 512    # ~512MB for smaller embedding models
    
    # Default estimate for unknown models
    return 4096  # ~4GB default

def check_model_compatibility(model_name: str, model_size: str = None) -> Tuple[bool, str, Dict]:
    """Check if a model is compatible with current GPU memory"""
    gpu_info = get_gpu_memory_info()
    required_memory = estimate_model_memory_requirement(model_name, model_size)
    
    # Leave some buffer for system and other processes (20% of total or min 1GB)
    buffer_memory = max(1024, int(gpu_info['total'] * 0.2))
    usable_memory = gpu_info['available'] - buffer_memory
    
    is_compatible = required_memory <= usable_memory
    
    if is_compatible:
        message = f"✅ Model {model_name} is compatible (requires ~{required_memory}MB, {usable_memory}MB available)"
    else:
        shortage = required_memory - usable_memory
        message = f"❌ Model {model_name} requires ~{required_memory}MB but only {usable_memory}MB available (shortage: {shortage}MB)"
    
    return is_compatible, message, {
        'required_memory_mb': required_memory,
        'available_memory_mb': usable_memory,
        'gpu_total_mb': gpu_info['total'],
        'gpu_used_mb': gpu_info['used'],
        'gpu_free_mb': gpu_info['free'],
        'buffer_memory_mb': buffer_memory,
        'compatible': is_compatible,
        'shortage_mb': max(0, required_memory - usable_memory)
    }

# Add a singleton instance to be used throughout the application
_chatpdf_instance = None

def get_chatpdf_instance():
    """Get the singleton instance of ChatPDF"""
    global _chatpdf_instance
    if (_chatpdf_instance is None):
        logger.info("Creating new ChatPDF instance")
        _chatpdf_instance = ChatPDF()
        logger.info("ChatPDF instance created successfully")
    return _chatpdf_instance