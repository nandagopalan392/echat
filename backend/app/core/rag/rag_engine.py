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
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Import retrieval config and advanced retrieval components
from app.config.retrieval import get_retrieval_config_manager
from app.core.rag.reranker import get_reranker
from app.core.rag.retriever import AutoMergingRetriever

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
    
    def __init__(self, llm_model: str = "gemma2:2b", embedding_model: str = "mxbai-embed-large"):
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
        
        # HuggingFace model context limits (set when loading HF models)
        self._hf_tokenizer = None
        self._hf_max_position_embeddings = None
        self._hf_max_new_tokens = None
        self._hf_max_input_length = None
        
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
                    
                    # Check if this is a local finetuned HuggingFace model
                    is_local_finetuned = self.llm_model.startswith('/app/data/finetuned_models')
                    llm_provider = getattr(self, 'llm_provider', 'ollama')
                    
                    if is_local_finetuned or llm_provider == 'huggingface':
                        logger.info(f"Loading HuggingFace LLM model: {self.llm_model}")
                        self._load_huggingface_llm()
                    else:
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
        
        logger.info(f"📦 Loading HuggingFace model: {hf_model_name}")
        logger.info(f"💾 Cache location: /root/.cache/huggingface/hub/")
        logger.info(f"ℹ️  First load: downloads from internet (~20s). Subsequent loads: from cache (~10-20s)")
        
        # Progress indicator
        download_complete = threading.Event()
        start_time = time.time()
        
        def progress_logger():
            while not download_complete.is_set():
                elapsed = int(time.time() - start_time)
                logger.info(f"⏳ Loading {hf_model_name}... ({elapsed}s elapsed)")
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

    def _load_huggingface_llm(self):
        """Load a HuggingFace LLM model (including local finetuned PEFT/LoRA models)"""
        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import threading
        import time
        import os
        import json
        
        model_path = self.llm_model
        logger.info(f"📦 Loading HuggingFace LLM: {model_path}")
        
        # Progress indicator
        download_complete = threading.Event()
        start_time = time.time()
        
        def progress_logger():
            while not download_complete.is_set():
                elapsed = int(time.time() - start_time)
                logger.info(f"⏳ Loading HuggingFace LLM... ({elapsed}s elapsed)")
                if download_complete.wait(10):
                    break
        
        progress_thread = threading.Thread(target=progress_logger, daemon=True)
        progress_thread.start()
        
        try:
            # Determine device
            device = "cuda" if torch and torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Check if this is a PEFT/LoRA adapter (has adapter_config.json)
            is_peft_adapter = False
            adapter_config_path = os.path.join(model_path, "adapter_config.json")
            base_model_name = None
            
            if os.path.exists(adapter_config_path):
                is_peft_adapter = True
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                    base_model_name = adapter_config.get("base_model_name_or_path")
                logger.info(f"🔧 Detected PEFT/LoRA adapter, base model: {base_model_name}")
            
            if is_peft_adapter and base_model_name:
                # Load as PEFT adapter
                from peft import PeftModel
                
                # Load tokenizer from adapter path (it has the correct vocab size used during training)
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load base model
                logger.info(f"Loading base model: {base_model_name}")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    trust_remote_code=True,
                    device_map="auto" if device == "cuda" else None,
                    torch_dtype=torch.float16 if device == "cuda" and torch else None,
                    low_cpu_mem_usage=True
                )
                
                # CRITICAL: Resize base model embeddings to match the tokenizer used during training
                # This ensures adapter weights (which were trained with resized embeddings) can load correctly
                if len(tokenizer) != base_model.config.vocab_size:
                    logger.info(f"Resizing base model embeddings from {base_model.config.vocab_size} to {len(tokenizer)}")
                    base_model.resize_token_embeddings(len(tokenizer))
                
                # Load PEFT adapter
                logger.info(f"Loading PEFT adapter from: {model_path}")
                model = PeftModel.from_pretrained(base_model, model_path)
                
                # Merge adapter weights for faster inference
                logger.info("Merging adapter weights for inference...")
                model = model.merge_and_unload()
                
            else:
                # Standard model loading (not a PEFT adapter)
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    device_map="auto" if device == "cuda" else None,
                    torch_dtype=torch.float16 if device == "cuda" and torch else None,
                    low_cpu_mem_usage=True
                )
            
            # Get the model's max position embeddings (context window size)
            max_position_embeddings = getattr(model.config, 'max_position_embeddings', 2048)
            logger.info(f"Model max_position_embeddings: {max_position_embeddings}")
            
            # CRITICAL: Set tokenizer's model_max_length to match model's max position embeddings
            # This ensures proper truncation when inputs are too long
            tokenizer.model_max_length = max_position_embeddings
            logger.info(f"Set tokenizer.model_max_length to {max_position_embeddings}")
            
            # Limit max_new_tokens to avoid exceeding position embeddings
            # Leave room for input tokens (at least half for input, half for generation)
            requested_max_tokens = self.model_parameters.get('max_tokens', 512)
            # For small context models, cap at a reasonable value to leave room for input
            # We need to leave enough room for the input prompt
            safe_max_new_tokens = min(requested_max_tokens, max_position_embeddings // 4, 512)
            max_input_length = max_position_embeddings - safe_max_new_tokens
            logger.info(f"Setting max_new_tokens to {safe_max_new_tokens} (requested: {requested_max_tokens})")
            logger.info(f"Max input length: {max_input_length} tokens")
            
            # Store these for context truncation in queries
            self._hf_max_position_embeddings = max_position_embeddings
            self._hf_max_new_tokens = safe_max_new_tokens
            self._hf_max_input_length = max_input_length
            self._hf_tokenizer = tokenizer
            
            # Create text generation pipeline with truncation enabled
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=safe_max_new_tokens,
                temperature=self.model_parameters.get('temperature', 0.7),
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                truncation=True,  # Enable truncation for long inputs
            )
            
            # Wrap in LangChain
            self.model = HuggingFacePipeline(pipeline=pipe)
            
            download_complete.set()
            elapsed = int(time.time() - start_time)
            logger.info(f"✅ Successfully loaded HuggingFace LLM: {model_path} ({elapsed}s)")
            
        except Exception as e:
            download_complete.set()
            logger.error(f"Failed to load HuggingFace LLM: {e}")
            raise e

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

    def query(self, question: str, k: int = 4, user_id: Optional[str] = None) -> Tuple[str, List[Document]]:
        """Query the vector store and generate answer with retrieval config support"""
        try:
            if not self.vector_store:
                return "Vector store not initialized", []
            
            if not self.model:
                return "LLM model not initialized", []
            
            # Load retrieval config for the user
            retrieval_config = None
            if user_id:
                try:
                    config_manager = get_retrieval_config_manager()
                    retrieval_config = config_manager.get_config(user_id)
                    logger.info(f"[RETRIEVAL CONFIG] Loaded config for user {user_id}: reranker={retrieval_config.reranker_enabled}, auto_merge={retrieval_config.auto_merging_enabled}")
                except Exception as e:
                    logger.warning(f"Could not load retrieval config for user {user_id}: {e}")
            
            # Create retriever with max_chunks from config or default k
            max_chunks = retrieval_config.max_chunks if retrieval_config else k
            retriever = self.vector_store.as_retriever(search_kwargs={"k": max_chunks})
            
            # Retrieve relevant documents
            docs = retriever.invoke(question)
            logger.info(f"[RETRIEVAL] Retrieved {len(docs)} initial documents")
            
            # Apply auto-merging if enabled
            if retrieval_config and retrieval_config.auto_merging_enabled:
                try:
                    logger.info("[AUTO_MERGE] Auto-merging is enabled, applying hierarchical merging")
                    merger = AutoMergingRetriever(
                        
                        similarity_threshold=retrieval_config.auto_merging_similarity_threshold
                    )
                    docs = merger.merge_documents(docs)
                    logger.info(f"[AUTO_MERGE] After merging: {len(docs)} documents")
                except Exception as e:
                    logger.error(f"[AUTO_MERGE] Error during auto-merge: {e}")
            
            # Apply reranking if enabled
            if retrieval_config and retrieval_config.reranker_enabled and retrieval_config.reranker_model:
                try:
                    logger.info(f"[RERANKER] Reranking enabled with model {retrieval_config.reranker_model}")
                    reranker = get_reranker(
                        model_name=retrieval_config.reranker_model,
                        provider=retrieval_config.reranker_provider or "huggingface"
                    )
                    # Run async reranker in sync context
                    docs = asyncio.run(reranker.rerank(question, docs, top_k=max_chunks))
                    logger.info(f"[RERANKER] After reranking: {len(docs)} documents")
                except Exception as e:
                    logger.error(f"[RERANKER] Error during reranking: {e}")
            
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

    async def stream_response(
        self, 
        question: str, 
        k: int = 4, 
        style: str = "conversational",
        user_id: str = None
    ):
        """
        Stream response chunks for a question with retrieval config support
        
        Args:
            question: User question
            k: Number of documents to retrieve
            style: Response style ('conversational' or 'detailed')
            user_id: Optional user ID for loading retrieval config
            
        Yields:
            Response chunks as strings
        """
        try:
            if not self.vector_store:
                yield '{"error": "Vector store not initialized"}'
                return
            
            if not self.model:
                yield '{"error": "LLM model not initialized"}'
                return
            
            # Load retrieval config for the user
            retrieval_config = None
            if user_id:
                try:
                    config_manager = get_retrieval_config_manager()
                    retrieval_config = config_manager.get_config(user_id)
                    logger.info(f"[RETRIEVAL CONFIG] Loaded config for user {user_id}: reranker={retrieval_config.reranker_enabled}, auto_merge={retrieval_config.auto_merging_enabled}")
                except Exception as e:
                    logger.warning(f"Could not load retrieval config for user {user_id}: {e}")
            
            # Create retriever with max_chunks from config or default k
            max_chunks = retrieval_config.max_chunks if retrieval_config else k
            retriever = self.vector_store.as_retriever(search_kwargs={"k": max_chunks})
            
            # Retrieve relevant documents
            docs = retriever.invoke(question)
            logger.info(f"[RETRIEVAL] Retrieved {len(docs)} initial documents")
            
            # Apply auto-merging if enabled
            if retrieval_config and retrieval_config.auto_merging_enabled:
                try:
                    logger.info("[AUTO_MERGE] Auto-merging is enabled, applying hierarchical merging")
                    merger = AutoMergingRetriever(
                        
                        similarity_threshold=retrieval_config.auto_merging_similarity_threshold
                    )
                    docs = merger.merge_documents(docs)
                    logger.info(f"[AUTO_MERGE] After merging: {len(docs)} documents")
                except Exception as e:
                    logger.error(f"[AUTO_MERGE] Error during auto-merge: {e}")
            
            # Apply reranking if enabled
            if retrieval_config and retrieval_config.reranker_enabled and retrieval_config.reranker_model:
                try:
                    logger.info(f"[RERANKER] Reranking enabled with model {retrieval_config.reranker_model}")
                    reranker = get_reranker(
                        model_name=retrieval_config.reranker_model,
                        provider=retrieval_config.reranker_provider or "huggingface"
                    )
                    docs = await reranker.rerank(question, docs, top_k=max_chunks)
                    logger.info(f"[RERANKER] After reranking: {len(docs)} documents")
                except Exception as e:
                    logger.error(f"[RERANKER] Error during reranking: {e}")
            
            # Format context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Truncate context for HuggingFace models with limited context windows
            if hasattr(self, '_hf_tokenizer') and hasattr(self, '_hf_max_input_length'):
                try:
                    tokenizer = self._hf_tokenizer
                    max_input_length = self._hf_max_input_length
                    
                    # Estimate prompt overhead (template tokens + question)
                    question_tokens = len(tokenizer.encode(question, add_special_tokens=False))
                    prompt_overhead = 150  # Estimate for prompt template
                    
                    # Calculate available tokens for context
                    available_for_context = max_input_length - question_tokens - prompt_overhead
                    available_for_context = max(available_for_context, 256)  # Minimum context
                    
                    # Tokenize and truncate context if needed
                    context_tokens = tokenizer.encode(context, add_special_tokens=False)
                    if len(context_tokens) > available_for_context:
                        logger.warning(f"[HF CONTEXT] Context too long ({len(context_tokens)} tokens), truncating to {available_for_context} tokens")
                        truncated_tokens = context_tokens[:available_for_context]
                        context = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                        logger.info(f"[HF CONTEXT] Truncated context to {len(truncated_tokens)} tokens")
                    else:
                        logger.info(f"[HF CONTEXT] Context fits: {len(context_tokens)} tokens (max: {available_for_context})")
                except Exception as e:
                    logger.error(f"[HF CONTEXT] Error truncating context: {e}")
            
            # Select prompt based on style
            prompt = self.conversational_prompt if style == "conversational" else self.prompt
            
            # Create streaming chain
            chain = (
                {"context": lambda x: context, "question": RunnablePassthrough()}
                | prompt
                | self.model
                | StrOutputParser()
            )
            
            # Stream response
            async for chunk in chain.astream(question):
                yield chunk
            
        except Exception as e:
            logger.error(f"Error during streaming query: {e}")
            yield f'{{"error": "{str(e)}"}}'

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

    def reload_models(self, llm_model: str = None, embedding_model: str = None, 
                       llm_provider: str = None, embedding_provider: str = None):
        """Reload models with new model names and providers"""
        try:
            if llm_model:
                self.llm_model = llm_model
                self.model = None
            
            if llm_provider:
                self.llm_provider = llm_provider
                logger.info(f"Setting LLM provider: {llm_provider}")
            
            if embedding_model:
                self.embedding_model = embedding_model
                self.embeddings = None
                self.vector_store = None
            
            if embedding_provider:
                self.embedding_provider = embedding_provider
                logger.info(f"Setting embedding provider: {embedding_provider}")
            
            self.models_loaded = False
            self.ensure_models_loaded()
            
            if embedding_model:
                self._initialize_vector_store()
            
            logger.info(f"Models reloaded: LLM={self.llm_model} (provider={getattr(self, 'llm_provider', 'ollama')}), Embedding={self.embedding_model}")
            
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
