"""
RAG Engine - Core RAG functionality
Handles LangChain, ChromaDB vector store, embeddings, and retrieval logic
Migrated from rag.py ChatPDF class - core functionality only
"""

import os
import gc
import logging
import json
import traceback
import chromadb
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document

# Try to import torch, with fallback if not available
try:
    import torch
except ImportError:
    torch = None

set_debug(True)
set_verbose(True)

# Set up some reasonable defaults for ChromaDB
os.environ["CHROMA_SERVER_NOFILE"] = "65536"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Core RAG engine handling vector store, embeddings, and retrieval
    This is the pure RAG logic without business orchestration
    """
    
    def __init__(self, llm_model: str = "deepseek-r1:latest", embedding_model: str = "mxbai-embed-large"):
        """Initialize RAG engine with models"""
        # Initialize default model parameters
        self.model_parameters = {
            'temperature': 0.7,
            'max_tokens': 2048,
            'top_p': 0.9,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0
        }
        
        # Initialize base attributes
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.embedding_provider = 'ollama'  # Default provider
        self.model = None
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        self.models_loaded = False
        
        # Text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Default prompts
        self.prompt = ChatPromptTemplate.from_template("""
            Answer the following question based on the provided context:
            Context: {context}
            Question: {question}
            Answer concisely and accurately.
        """)
        
        self.conversational_prompt = ChatPromptTemplate.from_template("""
            Answer the following question in a friendly, conversational tone:
            Context: {context}
            Question: {question}
            Be warm, personable, and engaging in your response.
        """)
        
        # ChromaDB configuration
        self.chroma_path = os.getenv('CHROMA_DB_PATH', '/app/data/chroma_db')
        self._ensure_chroma_dir()
        
        # Initialize models
        self.ensure_models_loaded()
        
        # Initialize vector store
        self._initialize_vector_store()

    def _get_collection_name(self) -> str:
        """Get collection name based on current embedding model"""
        safe_model_name = self.embedding_model.replace(":", "_").replace("-", "_").replace("/", "_")
        return f"embeddings_{safe_model_name}"

    def _ensure_chroma_dir(self):
        """Ensure Chroma directory exists with proper permissions"""
        try:
            path = Path(self.chroma_path)
            path.mkdir(parents=True, exist_ok=True)
            self._fix_permissions_recursive(path)
            logger.info(f"Ensured Chroma directory at {self.chroma_path} with proper permissions")
        except Exception as e:
            logger.error(f"Error creating Chroma directory: {str(e)}")
            raise

    def _fix_permissions_recursive(self, path: Path):
        """Fix permissions recursively for ChromaDB directories and files"""
        try:
            if path.is_dir():
                os.chmod(path, 0o777)
                for item in path.rglob("*"):
                    try:
                        if item.is_dir():
                            os.chmod(item, 0o777)
                        else:
                            os.chmod(item, 0o666)
                    except Exception as e:
                        logger.warning(f"Could not set permissions for {item}: {e}")
            else:
                os.chmod(path, 0o666)
        except Exception as e:
            logger.warning(f"Could not fix permissions for {path}: {e}")

    def _ensure_writable_before_operation(self):
        """Ensure ChromaDB directory is writable before any write operation"""
        try:
            if os.path.exists(self.chroma_path):
                self._fix_permissions_recursive(Path(self.chroma_path))
            
            # Test write access
            test_file = Path(self.chroma_path) / ".write_test"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            logger.error(f"ChromaDB directory not writable: {e}")
            raise RuntimeError(f"ChromaDB directory is not writable: {e}")

    def ensure_models_loaded(self, max_retries: int = 3):
        """Initialize models with retries and proper memory management"""
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"Retry {attempt}: Clearing memory before model load")
                    self.clear_gpu_memory()
                    import time
                    time.sleep(2)
                
                # Load embedding model
                if not self.embeddings:
                    logger.info(f"Loading embedding model: {self.embedding_model}")
                    
                    # Determine provider
                    embedding_provider = getattr(self, 'embedding_provider', 'ollama')
                    
                    # Apply compatibility override for BERT models
                    if self.embedding_model.startswith("mxbai-embed-large"):
                        embedding_provider = "huggingface"
                        logger.info(f"Using HuggingFace provider for BERT-based model: {self.embedding_model}")
                    
                    if embedding_provider == "huggingface":
                        self._load_huggingface_embeddings()
                    else:
                        self._load_ollama_embeddings()
                
                # Load LLM model
                if not self.model:
                    logger.info(f"Loading LLM model: {self.llm_model}")
                    self.model = ChatOllama(
                        model=self.llm_model,
                        temperature=self.model_parameters.get('temperature', 0.7),
                        base_url=os.getenv('OLLAMA_HOST', 'http://ollama:11434')
                    )
                
                self.models_loaded = True
                logger.info(f"Models loaded successfully (attempt {attempt + 1})")
                return
                
            except Exception as e:
                logger.error(f"Failed to load models (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error("Failed to load models after all retries")
                    raise

    def _load_huggingface_embeddings(self):
        """Load HuggingFace embedding model"""
        from langchain_huggingface import HuggingFaceEmbeddings
        import threading
        import time
        
        # Map short model name to full HuggingFace model name
        if self.embedding_model == "mxbai-embed-large":
            hf_model_name = "mixedbread-ai/mxbai-embed-large-v1"
        else:
            hf_model_name = self.embedding_model
        
        logger.info(f"Starting download of HuggingFace model: {hf_model_name}")
        
        # Progress indicator
        download_complete = threading.Event()
        start_time = time.time()
        
        def progress_logger():
            while not download_complete.is_set():
                elapsed = int(time.time() - start_time)
                logger.info(f"⏳ Still downloading {hf_model_name}... ({elapsed}s elapsed)")
                if download_complete.wait(10):
                    break
        
        progress_thread = threading.Thread(target=progress_logger, daemon=True)
        progress_thread.start()
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=hf_model_name,
                model_kwargs={'device': 'cuda' if torch and torch.cuda.is_available() else 'cpu'}
            )
            download_complete.set()
            elapsed = int(time.time() - start_time)
            logger.info(f"✅ Successfully loaded HuggingFace embedding model: {hf_model_name} ({elapsed}s)")
        except Exception as e:
            download_complete.set()
            raise e

    def _load_ollama_embeddings(self):
        """Load Ollama embedding model"""
        self.embeddings = OllamaEmbeddings(
            model=self.embedding_model,
            base_url=os.getenv('OLLAMA_HOST', 'http://ollama:11434')
        )
        logger.info(f"Loaded Ollama embedding model: {self.embedding_model}")

    def clear_gpu_memory(self):
        """Clear GPU memory if available"""
        try:
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("GPU memory cleared")
        except Exception as e:
            logger.warning(f"Could not clear GPU memory: {e}")

    def _initialize_vector_store(self):
        """Initialize ChromaDB vector store"""
        try:
            if not self.models_loaded:
                logger.warning("Models not loaded, initializing vector store may fail")
                return
            
            collection_name = self._get_collection_name()
            
            # Initialize ChromaDB client
            chroma_client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collection
            self.vector_store = Chroma(
                client=chroma_client,
                collection_name=collection_name,
                embedding_function=self.embeddings
            )
            
            logger.info(f"Vector store initialized with collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            logger.error(traceback.format_exc())

    def add_chunks_to_vector_store(self, chunks: List[Document], collection_name: str = None) -> bool:
        """Add document chunks to vector store"""
        try:
            if not self.vector_store:
                logger.error("Vector store not initialized")
                return False
            
            if not chunks:
                logger.warning("No chunks to add to vector store")
                return False
            
            self._ensure_writable_before_operation()
            
            # Filter complex metadata
            filtered_chunks = filter_complex_metadata(chunks)
            
            # Add to vector store
            self.vector_store.add_documents(filtered_chunks)
            
            logger.info(f"Added {len(chunks)} chunks to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add chunks to vector store: {e}")
            logger.error(traceback.format_exc())
            return False

    def remove_chunks_by_document_id(self, document_id: int, collection_name: str = None) -> int:
        """Remove all chunks for a document from vector store"""
        try:
            if not self.vector_store:
                logger.warning("Vector store not initialized")
                return 0
            
            # Get ChromaDB client
            chroma_client = self.vector_store._client
            
            # Use provided collection or get current one
            if collection_name:
                try:
                    collection = chroma_client.get_collection(collection_name)
                except Exception as e:
                    logger.warning(f"Collection {collection_name} not found: {e}")
                    return 0
            else:
                collection = self.vector_store._collection
            
            # Query for documents with this document_id
            results = collection.get(where={"document_id": str(document_id)})
            
            if results and results.get('ids'):
                # Delete all chunks
                collection.delete(ids=results['ids'])
                deleted_count = len(results['ids'])
                logger.info(f"Removed {deleted_count} chunks for document {document_id}")
                return deleted_count
            else:
                logger.info(f"No chunks found for document {document_id}")
                return 0
            
        except Exception as e:
            logger.error(f"Error removing chunks for document {document_id}: {e}")
            return 0

    def get_vector_store_stats(self) -> Dict:
        """Get statistics about the vector store"""
        try:
            if not self.vector_store:
                return {
                    "error": "No vector store initialized",
                    "collections": 0,
                    "total_documents": 0,
                    "models_loaded": self.models_loaded,
                    "embedding_model": self.embedding_model
                }
            
            # Get ChromaDB client
            chroma_client = self.vector_store._client
            collections = chroma_client.list_collections()
            
            stats = {
                "total_collections": len(collections),
                "current_collection": self._get_collection_name(),
                "current_embedding_model": self.embedding_model,
                "collections": [],
                "size_bytes": 0
            }
            
            total_docs = 0
            for collection in collections:
                try:
                    collection_count = collection.count()
                    total_docs += collection_count
                    
                    stats["collections"].append({
                        "name": collection.name,
                        "count": collection_count,
                        "embedding_model": collection.name.replace("embeddings_", "").replace("_", "-")
                    })
                except Exception as e:
                    logger.warning(f"Error getting stats for collection {collection.name}: {e}")
            
            stats["total_documents"] = total_docs
            
            # Calculate directory size
            try:
                chroma_path = Path(self.chroma_path)
                if chroma_path.exists():
                    total_size = sum(
                        f.stat().st_size 
                        for f in chroma_path.rglob("*") 
                        if f.is_file()
                    )
                    stats["size_bytes"] = total_size
            except Exception as e:
                logger.warning(f"Could not calculate vector store size: {e}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {
                "error": str(e),
                "collections": 0,
                "total_documents": 0
            }

    def clear_vector_store(self) -> bool:
        """Clear the entire vector store"""
        try:
            if self.vector_store and hasattr(self.vector_store, '_collection'):
                collection = self.vector_store._collection
                if collection:
                    collection.delete()
                    logger.info("Vector store cleared successfully")
                    return True
            logger.warning("Vector store not available for clearing")
            return False
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False

    def query(self, question: str, k: int = 4) -> Tuple[str, List[Document]]:
        """Query the vector store and generate answer"""
        try:
            if not self.vector_store:
                return "Vector store not initialized", []
            
            if not self.model:
                return "LLM model not initialized", []
            
            # Create retriever
            retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
            
            # Retrieve relevant documents
            docs = retriever.get_relevant_documents(question)
            
            # Format context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate answer
            chain = (
                {"context": lambda x: context, "question": RunnablePassthrough()}
                | self.prompt
                | self.model
                | StrOutputParser()
            )
            
            answer = chain.invoke(question)
            
            return answer, docs
            
        except Exception as e:
            logger.error(f"Error during query: {e}")
            return f"Error: {str(e)}", []

    def update_model_parameters(self, parameters: Dict):
        """Update LLM model parameters"""
        try:
            self.model_parameters.update(parameters)
            
            # Recreate model with new parameters
            if self.model:
                self.model = ChatOllama(
                    model=self.llm_model,
                    temperature=self.model_parameters.get('temperature', 0.7),
                    base_url=os.getenv('OLLAMA_HOST', 'http://ollama:11434')
                )
                logger.info(f"Updated model parameters: {self.model_parameters}")
            
        except Exception as e:
            logger.error(f"Error updating model parameters: {e}")

    def reload_models(self, llm_model: str = None, embedding_model: str = None):
        """Reload models with new model names"""
        try:
            if llm_model:
                self.llm_model = llm_model
                self.model = None
            
            if embedding_model:
                self.embedding_model = embedding_model
                self.embeddings = None
                self.vector_store = None
            
            self.models_loaded = False
            self.ensure_models_loaded()
            
            if embedding_model:
                self._initialize_vector_store()
            
            logger.info(f"Models reloaded: LLM={self.llm_model}, Embedding={self.embedding_model}")
            
        except Exception as e:
            logger.error(f"Error reloading models: {e}")
            raise


# Singleton instance
_rag_engine = None


def get_rag_engine() -> RAGEngine:
    """Get RAG engine singleton"""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
