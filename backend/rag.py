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
from typing import Optional, List
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
            # Ensure write permissions
            path.chmod(0o777)
            logger.info(f"Ensured Chroma directory at {self.chroma_path}")
        except Exception as e:
            logger.error(f"Error creating Chroma directory: {str(e)}")
            raise

    def _initialize_vector_store(self):
        """Initialize vector store with existing data if available"""
        try:
            if not self.embeddings:
                logger.warning("Embeddings not loaded, skipping vector store initialization")
                return
                
            # Create settings for ChromaDB client
            settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
            
            # Create client and get collection name
            client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=settings
            )
            collection_name = self._get_collection_name()
            
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
            else:
                logger.error("Cannot create vector store: embeddings not loaded")
                
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
            
            # Update model names
            self.llm_model = llm_model
            self.embedding_model = embedding_model
            
            # Reset models to force reload
            self.models_loaded = False
            self.model = None
            self.embeddings = None
            
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
        """Initialize models with retries"""
        # DON'T check models_loaded here to prevent infinite recursion
        
        for attempt in range(max_retries):
            try:
                if not self.embeddings:
                    # Try to initialize embeddings
                    self.embeddings = OllamaEmbeddings(
                        model=self.embedding_model,
                        base_url=os.getenv('OLLAMA_HOST', 'http://ollama:11434')
                    )
                    # Test embeddings
                    self.embeddings.embed_query("test")
                
                if not self.model:
                    self.model = ChatOllama(
                        model=self.llm_model,
                        base_url=os.getenv('OLLAMA_HOST', 'http://ollama:11434'),
                        temperature=0.1,
                        num_ctx=2048,
                        timeout=120,  # Increase timeout to 2 minutes
                        streaming=True,  # Enable streaming
                        seed=42  # Add seed for consistent responses
                    )
                
                # Mark models as loaded after successful initialization
                self.models_loaded = True
                logger.info("Models loaded successfully")
                return True
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error("Failed to load models after maximum retries")
                    raise
                import asyncio
                asyncio.run(self.pull_models())
                time.sleep(5)  # Wait before retry

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
                    logger.info(f"Created new vector store with collection '{collection_name}'")
                
                # Add documents to vector store
                logger.debug("Adding documents to vector store")
                self.vector_store.add_documents(chunks)
                
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
                            self.vector_store.add_documents(chunks)
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

    @lru_cache(maxsize=100)  # Cache recent responses
    def _get_cached_response(self, query: str, context: str) -> str:
        """Cache responses for identical queries with same context"""
        formatted_prompt = self.prompt.format(context=context, question=query)
        return self.model.invoke(formatted_prompt)

    def ask(self, query: str) -> str:
        """
        Answer a query using the RAG pipeline.
        """
        self.ensure_models_loaded()
        try:
            # Ensure vector store is loaded
            if not self.vector_store:
                logger.error("No vector store available")
                raise ValueError("No vector store found. Please ingest a document first.")

            # Ensure retriever is initialized
            if not self.retriever:
                self.retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                )

            logger.info(f"Retrieving context for query: {query}")
            try:
                # Simplify retrieval process
                retrieved_docs = self.retriever.get_relevant_documents(query)

                if not retrieved_docs:
                    return "No relevant context found in the document to answer your question."

                context = "\n".join(doc.page_content for doc in retrieved_docs[:3])
                response = self._get_cached_response(query, context)
                
                # Extract string response if necessary
                if hasattr(response, 'content'):
                    return response.content
                return str(response)

            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                return f"Error: {str(e)}"

        except Exception as e:
            logger.error(f"Error in ask method: {str(e)}")
            raise

    # Fix the stream_response function to be either a generator function using yield or a regular async function
    async def stream_response(self, query: str, style: str = "standard"):
        """Get complete response as a proper async generator with metrics"""
        start_time = time.time()
        self.check_models()
        
        # Record query complexity
        estimate_query_complexity(query)
        
        try:
            # Get relevant documents for context - measure retrieval time
            retrieval_start = time.time()
            context_docs = self.retriever.get_relevant_documents(query) if self.retriever else []
            retrieval_end = time.time()
            logger.debug(f"Document retrieval took {retrieval_end - retrieval_start:.3f}s")
            
            print(f"[RAG] Retrieved {len(context_docs)} documents for query: {query[:50]}...")
            
            context = "\n".join(doc.page_content for doc in context_docs[:3]) if context_docs else ""
            
            # Create style-specific prompts with more distinctive differences
            if style == "conversational":
                # Create a prompt that encourages detailed, friendly explanations
                full_prompt = f"""
                You are a helpful and friendly assistant. Your goal is to explain things clearly and engagingly.

                Context: {context}
                Question: {query}

                Instructions for your response:
                - Use a warm, conversational tone as if talking to a friend
                - Include examples and analogies to make concepts clear
                - Use phrases like "Let me explain", "Think of it this way", "Here's what I think"
                - Break down complex ideas into simple steps
                - Be thorough and detailed in your explanations
                - Include practical tips or insights when relevant

                Please provide a comprehensive, friendly response that thoroughly addresses the question.
                """
            elif style == "detailed":
                # Create a prompt that encourages comprehensive, analytical responses
                full_prompt = f"""
                You are an expert analyst providing comprehensive technical analysis. Your goal is to provide thorough, detailed information.

                Context: {context}
                Question: {query}

                Instructions for your response:
                - Provide a comprehensive, technical analysis
                - Include multiple perspectives and considerations
                - Use precise technical terminology
                - Structure your response with clear sections or bullet points
                - Include background information and context
                - Discuss implications, pros/cons, or related concepts
                - Be thorough and leave no important detail unaddressed

                Please provide a detailed, analytical response that covers all aspects of the question.
                """
            else:  # standard style (concise)
                # Create a prompt that encourages brief, precise responses
                full_prompt = f"""
                You are a professional assistant focused on providing clear, concise answers. Your goal is efficiency and precision.

                Context: {context}
                Question: {query}

                Instructions for your response:
                - Be direct and to the point
                - Use professional, formal language
                - Focus on the most essential information
                - Avoid unnecessary elaboration
                - Structure your response clearly and logically
                - Use bullet points or numbered lists when appropriate
                - Get straight to the answer without lengthy introductions

                Please provide a concise, professional response that directly addresses the question.
                """
            
            print(f"[RAG] Will send prompt to DeepSeek ({style} style): {full_prompt[:200]}...")
            
            # Get the complete response using the single comprehensive prompt
            try:
                llm_start = time.time()
                # Create timeout configuration for longer AI responses
                timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes total timeout
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    # Get the complete response with DeepSeek's natural thinking + content
                    async with session.post(
                        f"{os.getenv('OLLAMA_HOST', 'http://ollama:11434')}/api/generate",
                        json={
                            "model": self.llm_model,
                            "prompt": full_prompt,
                            "stream": False,  # Get complete response
                        }
                    ) as response:
                        result = await response.json()
                        full_response = result.get("response", "")
                        
                        # Record generation time
                        generation_time = time.time() - llm_start
                        logger.debug(f"Response generation took {generation_time:.3f}s")
                        
                        # Try to split thinking and content if DeepSeek naturally provides it
                        thinking_content = ""
                        content = full_response
                        
                        # Look for thinking patterns that DeepSeek might use
                        thinking_markers = ["<think>", "<thinking>", "Let me think", "I need to think", "My thinking:"]
                        content_markers = ["</think>", "</thinking>", "Now,", "So,", "Therefore,", "In conclusion"]
                        
                        for marker in thinking_markers:
                            if marker.lower() in full_response.lower():
                                # Try to extract thinking section
                                parts = full_response.lower().split(marker.lower(), 1)
                                if len(parts) > 1:
                                    remaining = parts[1]
                                    # Look for content markers
                                    for content_marker in content_markers:
                                        if content_marker.lower() in remaining:
                                            thinking_part, content_part = remaining.split(content_marker.lower(), 1)
                                            thinking_content = thinking_part.strip()
                                            content = content_part.strip()
                                            break
                                break
                        
                        # If no thinking section was found, use the full response as content
                        if not thinking_content:
                            content = full_response
                    
                    # Evaluate hallucination rate
                    hallucination_score = estimate_hallucination(content, context_docs)
                    logger.debug(f"Hallucination score: {hallucination_score:.3f}")
                    
                    # Yield the complete response with both thinking and content
                    yield json.dumps({
                        "thinking": thinking_content,
                        "content": content,
                        "isComplete": True,
                        "style": style
                    })
                    
                # Record top-k accuracy based on context presence
                if context_docs:
                    # This is just a placeholder - in a real system you'd need ground truth
                    # Here we use simplified heuristics
                    k = len(context_docs)
                    if k > 0:
                        # Simple heuristic - decreases as k increases
                        accuracy_estimate = max(0.1, min(0.95, 1.0/(1.0 + 0.2*k)))
                        logger.debug(f"Top-{k} accuracy estimate: {accuracy_estimate:.3f}")
                        
            except Exception as e:
                logger.error(f"Error during ollama request: {str(e)}")
                yield json.dumps({"error": str(e)})

        except Exception as e:
            logger.error(f"Stream error: {str(e)}", exc_info=True)
            yield json.dumps({"error": str(e)})
        finally:
            # Record total response time
            total_time = time.time() - start_time
            logger.debug(f"LLM streaming response took {total_time:.2f}s")
                
    # Fix the generate_rlhf_options function to properly use yield instead of return
    async def generate_rlhf_options(self, query: str):
        """Generate multiple responses for RLHF ranking with metrics"""
        start_time = time.time()
        self.check_models()
        
        # Track query complexity
        complexity = estimate_query_complexity(query)
        logger.debug(f"Query complexity score: {complexity}")
        
        try:
            # Get relevant documents for context - measure retrieval time
            retrieval_start = time.time()
            context_docs = self.retriever.get_relevant_documents(query) if self.retriever else []
            retrieval_time = time.time() - retrieval_start
            logger.debug(f"Document retrieval took {retrieval_time:.3f}s")
                
            # Send initial status
            yield json.dumps({
                "content": "Generating responses for you..."
            })
            
            # Monitor reranker performance
            reranker_start = time.time()
            
            # Generate options based on context
            if not context_docs or len(context_docs) < 2:
                # Fallback approach with new metrics
                option1_prompt = f"""
                Answer the following question as well as you can:
                Question: {query}
                
                Be precise and detailed in your response."""
                
                option2_prompt = f"""
                Answer the following question in a conversational style:
                Question: {query}
                
                Be friendly and engaging in your response."""
                
                # Generate options with metrics
                llm_start = time.time()
                timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes total timeout
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{os.getenv('OLLAMA_HOST', 'http://ollama:11434')}/api/generate",
                        json={
                            "model": self.llm_model,
                            "prompt": option1_prompt,
                            "stream": False
                        }
                    ) as response1:
                        result1 = await response1.json()
                        option1 = result1.get("response", "")
                    
                    llm_time1 = time.time() - llm_start
                    logger.debug(f"Option 1 generation took {llm_time1:.3f}s")
                    
                    llm_start2 = time.time()
                    async with session.post(
                        f"{os.getenv('OLLAMA_HOST', 'http://ollama:11434')}/api/generate",
                        json={
                            "model": self.llm_model,
                            "prompt": option2_prompt,
                            "stream": False
                        }
                    ) as response2:
                        result2 = await response2.json()
                        option2 = result2.get("response", "")
                    
                    llm_time2 = time.time() - llm_start2
                    logger.debug(f"Option 2 generation took {llm_time2:.3f}s")
                    
                # Calculate hallucination rates
                hallucination1 = estimate_hallucination(option1, context_docs)
                hallucination2 = estimate_hallucination(option2, context_docs)
                logger.debug(f"Hallucination scores - Option1: {hallucination1:.3f}, Option2: {hallucination2:.3f}")
            else:
                # We have enough documents - use the reranker
                # Track reranking time
                split_start = time.time()
                
                # Split documents for options
                import random
                random.shuffle(context_docs)
                docs1 = context_docs[:len(context_docs)//2]
                docs2 = context_docs[len(context_docs)//2:]
                
                # Create prompts for each option
                context1 = "\n".join(doc.page_content for doc in docs1)
                context2 = "\n".join(doc.page_content for doc in docs2)
                
                prompt1 = f"""
                Answer based on this context:
                {context1}
                
                Question: {query}
                """
                
                prompt2 = f"""
                Answer based on this context:
                {context2}
                
                Question: {query}
                """
                
                # Generate responses with metrics
                llm_start = time.time()
                timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes total timeout
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{os.getenv('OLLAMA_HOST', 'http://ollama:11434')}/api/generate",
                        json={
                            "model": self.llm_model,
                            "prompt": prompt1,
                            "stream": False
                        }
                    ) as response1:
                        result1 = await response1.json()
                        option1 = result1.get("response", "")
                    
                    llm_time1 = time.time() - llm_start
                    logger.debug(f"Split option 1 generation took {llm_time1:.3f}s")
                    
                    llm_start2 = time.time()
                    async with session.post(
                        f"{os.getenv('OLLAMA_HOST', 'http://ollama:11434')}/api/generate",
                        json={
                            "model": self.llm_model,
                            "prompt": prompt2,
                            "stream": False
                        }
                    ) as response2:
                        result2 = await response2.json()
                        option2 = result2.get("response", "")
                    
                    llm_time2 = time.time() - llm_start2
                    logger.debug(f"Split option 2 generation took {llm_time2:.3f}s")
                
                # Record hallucination metrics
                hallucination1 = estimate_hallucination(option1, docs1)
                hallucination2 = estimate_hallucination(option2, docs2)
                logger.debug(f"Split hallucination scores - Option1: {hallucination1:.3f}, Option2: {hallucination2:.3f}")
                
                # Record reranking time
                rerank_time = time.time() - split_start
                logger.debug(f"Document split reranking took {rerank_time:.3f}s")
            
            # Format full prompt
            full_prompt = "I've generated two possible responses from different sources. Please select the one you prefer:"
            
            # Yield the final options
            yield json.dumps({
                "content": full_prompt,
                "full_response": full_prompt,
                "is_final": True,
                "response_options": [option1, option2],
                "rlhf_enabled": True
            })
            
        except Exception as e:
            logger.error(f"RLHF options error: {str(e)}", exc_info=True)
            yield json.dumps({"error": str(e)})
        finally:
            # Record total time
            total_time = time.time() - start_time
            logger.debug(f"RLHF response generation took {total_time:.2f}s")
    async def get_quick_response(self, query: str, style: str = "standard"):
        """Get a response quickly with simpler prompts and lower token counts"""
        self.check_models()
        try:
            # Get relevant documents for context - use smaller k
            context_docs = self.retriever.get_relevant_documents(query) if self.retriever else []
            context = "\n".join(doc.page_content for doc in context_docs[:2]) if context_docs else ""
            
            # Create simpler prompt
            prompt = f"""
            Answer briefly:
            Context: {context}
            Question: {query}
            """
            
            try:
                timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes total timeout
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    # Single request with lower parameters
                    async with session.post(
                        f"{os.getenv('OLLAMA_HOST', 'http://ollama:11434')}/api/generate",
                        json={
                            "model": self.llm_model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "num_ctx": 1024,  # Reduce context size
                                "num_predict": 256,  # Limit token generation
                                "temperature": 0.1  # Low temperature for faster, more deterministic responses
                            }
                        }
                    ) as response:
                        result = await response.json()
                        content = result.get("response", "")
                    
                    return json.dumps({
                        "content": content,
                        "isComplete": True
                    })
            except Exception as e:
                logger.error(f"Error during ollama request: {str(e)}")
                return json.dumps({"error": str(e)})
        except Exception as e:
            logger.error(f"Quick response error: {str(e)}", exc_info=True)
            return json.dumps({"error": str(e)})
            
    # Add a new method for faster responses with limited context
    async def get_simple_response(self, query: str):
        """Get a very simple response with minimal processing for quick answers"""
        self.check_models()
        try:
            # Use a direct prompt without context retrieval
            prompt = f"""
            Answer this question briefly and directly:
            Question: {query}
            
            Keep your answer concise and to the point.
            """
            
            try:
                # Single API call with low settings for speed
                timeout = aiohttp.ClientTimeout(total=60)  # 1 minute for simple responses
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{os.getenv('OLLAMA_HOST', 'http://ollama:11434')}/api/generate",
                        json={
                            "model": self.llm_model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.1,
                                "num_predict": 100,  # Very small to be quick
                                "num_ctx": 512,  # Minimal context window,
                                "top_p": 0.95
                            }
                        },
                        timeout=30  # Increase timeout to 30 seconds for simple responses
                    ) as response:
                        result = await response.json()
                        content = result.get("response", "")
                    
                    # Return minimal JSON
                    return json.dumps({
                        "content": content,
                        "isComplete": True
                    })
            except Exception as e:
                logger.error(f"Error during simple response: {str(e)}")
                return json.dumps({"content": "Sorry, I couldn't generate a response quickly enough."})
        except Exception as e:
            logger.error(f"Simple response error: {str(e)}")
            return json.dumps({"error": str(e)})
    def clear(self):
        """
        Reset the vector store and retriever.
        """
        logger.info("Clearing vector store and retriever.")
        self.vector_store = None
        self.retriever = None

    def get_retriever_stats(self):
        """Get statistics about the retriever for monitoring"""
        stats = {
            "has_vector_store": self.vector_store is not None,
            "has_retriever": self.retriever is not None,
            "vector_count": 0,
            "models_loaded": self.models_loaded,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model
        }
        
        # Get vector count if available
        if self.vector_store:
            try:
                # Try different ways to get collection size depending on Chroma version
                if hasattr(self.vector_store, '_collection') and hasattr(self.vector_store._collection, 'count'):
                    stats["vector_count"] = self.vector_store._collection.count()
                elif hasattr(self.vector_store, 'get') and callable(getattr(self.vector_store, 'get')):
                    result = self.vector_store.get()
                    stats["vector_count"] = len(result['ids']) if result and 'ids' in result else 0
                elif hasattr(self.vector_store, '_client') and hasattr(self.vector_store._client, 'count'):
                    stats["vector_count"] = self.vector_store._client.count()
            except Exception as e:
                logger.warning(f"Could not get vector count: {str(e)}")
        
        return stats

    def update_monitoring_metrics(self):
        """Update monitoring metrics with current state"""
        try:
            stats = self.get_retriever_stats()
            
            # Log vector store stats
            if "vector_count" in stats and stats["vector_count"] > 0:
                logger.debug(f"Vector store contains {stats['vector_count']} vectors")
            
            # Check for GPU metrics if applicable
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        memory_allocated = torch.cuda.memory_allocated(i)
                        logger.debug(f"GPU {i} memory allocated: {memory_allocated} bytes")
                        
                        # Basic utilization metric
                        memory_reserved = torch.cuda.memory_reserved(i)
                        if memory_reserved > 0:
                            utilization = (memory_allocated / memory_reserved) * 100
                            logger.debug(f"GPU {i} utilization: {utilization:.1f}%")
            except ImportError:
                # No torch available, try GPUtil
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    for i, gpu in enumerate(gpus):
                        logger.debug(f"GPU {i} load: {gpu.load * 100:.1f}%")
                        logger.debug(f"GPU {i} memory used: {gpu.memoryUsed} MB")
                except ImportError:
                    pass
                    
            # Log model state
            logger.debug(f"Models loaded - Embeddings: {bool(self.embeddings)}, LLM: {bool(self.model)}")
            
            logger.debug("Updated monitoring metrics")
        except Exception as e:
            logger.error(f"Error updating monitoring metrics: {str(e)}")
    
    def list_available_collections(self):
        """List all available collections in the vector database"""
        try:
            if not os.path.exists(self.chroma_path):
                return []
                
            settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
            client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=settings
            )
            
            collections = client.list_collections()
            collection_info = []
            
            for collection in collections:
                try:
                    count = collection.count()
                    collection_info.append({
                        'name': collection.name,
                        'count': count,
                        'is_current': collection.name == self._get_collection_name()
                    })
                except Exception as e:
                    logger.warning(f"Could not get info for collection {collection.name}: {str(e)}")
                    
            return collection_info
            
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            return []

    def get_vector_store_stats(self):
        """Get comprehensive statistics about the vector store"""
        try:
            stats = {
                "current_embedding_model": self.embedding_model,
                "current_collection": self._get_collection_name(),
                "vector_count": 0,
                "collections": self.list_available_collections(),
                "status": "ready" if self.vector_store else "not_initialized"
            }
            
            if self.vector_store:
                try:
                    # Try different ways to get collection size depending on Chroma version
                    if hasattr(self.vector_store, '_collection') and hasattr(self.vector_store._collection, 'count'):
                        stats["vector_count"] = self.vector_store._collection.count()
                    else:
                        # Fallback method
                        try:
                            # Try to get collection via client
                            settings = Settings(
                                anonymized_telemetry=False,
                                allow_reset=True
                            )
                            client = chromadb.PersistentClient(
                                path=self.chroma_path,
                                settings=settings
                            )
                            collection = client.get_collection(self._get_collection_name())
                            stats["vector_count"] = collection.count()
                        except Exception:
                            stats["vector_count"] = "unknown"
                except Exception as e:
                    logger.warning(f"Could not get vector count: {str(e)}")
                    stats["vector_count"] = "error"
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting vector store stats: {str(e)}")
            return {
                "current_embedding_model": self.embedding_model,
                "current_collection": self._get_collection_name(),
                "vector_count": "error",
                "collections": [],
                "status": "error",
                "error": str(e)
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