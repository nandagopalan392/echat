import logging
from typing import List, Dict, Any, Optional, Tuple
import os
import numpy as np
from langchain_core.documents import Document
import aiohttp
import json
import time
from functools import lru_cache
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """Reranker that uses cross-encoder models to improve retrieval relevance"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-large", provider: str = "ollama"):
        """
        Initialize the reranker.
        
        Args:
            model_name: Name of the reranker model to use.
            provider: Provider for the model ('ollama' or 'huggingface')
        """
        self.model_name = model_name
        self.provider = provider
        self.ollama_base_url = os.getenv('OLLAMA_HOST', 'http://ollama:11434')
        self._huggingface_model = None
        logger.info(f"Initialized CrossEncoderReranker with model {model_name} (provider: {provider})")
        
    def _load_huggingface_model(self):
        """Load HuggingFace model using sentence-transformers"""
        if self._huggingface_model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading HuggingFace model: {self.model_name}")
                self._huggingface_model = CrossEncoder(self.model_name)
                logger.info(f"Successfully loaded HuggingFace model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load HuggingFace model {self.model_name}: {e}")
                raise
        return self._huggingface_model
        
    async def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: User query
            documents: List of retrieved documents
            top_k: Number of top documents to return
            
        Returns:
            List of reranked documents, limited to top_k
        """
        print(f"[RERANKER] Reranking query: {query[:50]}... with {len(documents)} documents")
        
        if not documents:
            logger.warning("No documents to rerank")
            return []
            
        # Limit to a reasonable number of documents to rerank
        documents_to_rerank = documents[:min(10, len(documents))]
        
        # Prepare pairs for reranking
        pairs = [(query, doc.page_content) for doc in documents_to_rerank]
        
        try:
            # Get scores from the reranker
            scores = await self._compute_scores(pairs)
            
            if not scores or len(scores) != len(documents_to_rerank):
                logger.warning("Failed to get valid scores, returning original order")
                return documents[:top_k]
            
            # Print top scores for debugging
            sorted_scores = sorted(scores, reverse=True)
            print(f"[RERANKER] Top 3 relevance scores: {sorted_scores[:3]}")
            
            # Sort documents by score
            sorted_docs = [doc for _, doc in sorted(
                zip(scores, documents_to_rerank), 
                key=lambda pair: pair[0], 
                reverse=True
            )]
            
            # Print most relevant document preview
            if sorted_docs:
                print(f"[RERANKER] Most relevant doc preview: {sorted_docs[0].page_content[:100]}...")
            
            logger.info(f"Reranking complete. Top score: {max(scores) if scores else 'N/A'}")
            
            # Return top_k documents
            return sorted_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            # Fallback to original ranking
            return documents[:top_k]
    
    async def _compute_scores(self, pairs: List[tuple]) -> List[float]:
        """
        Compute relevance scores for query-document pairs.
        
        Args:
            pairs: List of (query, document) tuples
            
        Returns:
            List of relevance scores
        """
        try:
            # Use the configured reranker model
            print(f"[RERANKER] Using reranker model: {self.model_name} (provider: {self.provider})")
            
            # For HuggingFace models, use sentence-transformers directly
            if self.provider == "huggingface":
                return await self._compute_scores_huggingface(pairs)
            else:
                return await self._compute_scores_ollama(pairs)
                
        except Exception as e:
            logger.error(f"Error computing scores: {str(e)}")
            # Return neutral scores as fallback
            return [0.5] * len(pairs)
    
    async def _compute_scores_huggingface(self, pairs: List[tuple]) -> List[float]:
        """Compute scores using HuggingFace CrossEncoder"""
        try:
            model = self._load_huggingface_model()
            
            # Prepare input pairs for the cross-encoder
            input_pairs = [[query, doc] for query, doc in pairs]
            
            # Compute scores using the cross-encoder
            scores = model.predict(input_pairs)
            
            # Convert to list and ensure it's float
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            
            logger.info(f"HuggingFace reranker computed {len(scores)} scores")
            return [float(score) for score in scores]
            
        except Exception as e:
            logger.error(f"Error with HuggingFace reranker: {e}")
            return [0.5] * len(pairs)
    
    async def _compute_scores_ollama(self, pairs: List[tuple]) -> List[float]:
        """Compute scores using Ollama API (existing method)"""
        try:
            # For reranker models, we need to compute similarity scores between query-document pairs
            # First, get embeddings for queries and documents separately
            queries = [pair[0] for pair in pairs]
            documents = [pair[1] for pair in pairs]
            
            async with aiohttp.ClientSession() as session:
                # Try to use the reranker model first, fallback to main embedding model if not available
                model_to_use = self.model_name
                
                # Test if the model exists by trying a simple request
                try:
                    test_response = await session.post(
                        f"{self.ollama_base_url}/api/embed",
                        json={
                            "model": self.model_name,
                            "input": ["test"]
                        },
                        timeout=10
                    )
                    if test_response.status != 200:
                        error_text = await test_response.text()
                        if "not found" in error_text.lower():
                            logger.warning(f"Reranker model {self.model_name} not found, falling back to main embedding model")
                            # Get the main embedding model from database
                            try:
                                from chat_db import ChatDB
                                chat_db = ChatDB()
                                settings = chat_db.get_latest_model_settings()
                                if settings and settings.get('embedding'):
                                    model_to_use = settings['embedding']
                                    logger.info(f"Using main embedding model for reranking: {model_to_use}")
                                else:
                                    model_to_use = "bge-m3"  # fallback
                                    logger.info(f"Using fallback embedding model for reranking: {model_to_use}")
                            except Exception as e:
                                logger.warning(f"Could not get main embedding model: {e}, using fallback")
                                model_to_use = "bge-m3"
                except Exception as e:
                    logger.warning(f"Could not test reranker model {self.model_name}: {e}, using fallback")
                    model_to_use = "bge-m3"
                
                # Get embeddings for queries
                async with session.post(
                    f"{self.ollama_base_url}/api/embed",
                    json={
                        "model": model_to_use,
                        "input": queries
                    },
                    timeout=30
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Error from query embedding API: {error_text}")
                        return [0.5] * len(pairs)
                        
                    query_result = await response.json()
                    print(f"[RERANKER] Query embedding result keys: {query_result.keys() if isinstance(query_result, dict) else 'Not a dict'}")
                    
                    # Fix: Use 'embeddings' (plural) instead of 'embedding' (singular)
                    query_embeddings = query_result.get("embeddings", [])
                    
                    if not query_embeddings:
                        logger.error(f"No query embeddings received from API")
                        return [0.5] * len(pairs)
                
                # Get embeddings for documents
                async with session.post(
                    f"{self.ollama_base_url}/api/embed",
                    json={
                        "model": model_to_use,
                        "input": documents
                    },
                    timeout=30
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Error from document embedding API: {error_text}")
                        return [0.5] * len(pairs)
                        
                    doc_result = await response.json()
                    print(f"[RERANKER] Document embedding result keys: {doc_result.keys() if isinstance(doc_result, dict) else 'Not a dict'}")
                    
                    # Fix: Use 'embeddings' (plural) instead of 'embedding' (singular)
                    doc_embeddings = doc_result.get("embeddings", [])
                    
                    if not doc_embeddings:
                        logger.error(f"No document embeddings received from API")
                        return [0.5] * len(pairs)
                
                # Verify we have the right number of embeddings
                if len(query_embeddings) != len(pairs) or len(doc_embeddings) != len(pairs):
                    logger.error(f"Embedding count mismatch: {len(query_embeddings)} queries, {len(doc_embeddings)} docs, {len(pairs)} pairs")
                    return [0.5] * len(pairs)
                
                print(f"[RERANKER] Successfully got embeddings: {len(query_embeddings)} queries, {len(doc_embeddings)} documents")
                
                # Calculate similarity scores between corresponding query-document pairs
                scores = []
                for i, (query_emb, doc_emb) in enumerate(zip(query_embeddings, doc_embeddings)):
                    try:
                        query_vec = np.array(query_emb, dtype=np.float32)
                        doc_vec = np.array(doc_emb, dtype=np.float32)
                        
                        # Calculate cosine similarity
                        similarity = self._cosine_similarity(query_vec, doc_vec)
                        scores.append(similarity)
                        
                    except Exception as e:
                        logger.warning(f"Error calculating similarity for pair {i}: {e}")
                        scores.append(0.5)
                
                print(f"[RERANKER] Calculated {len(scores)} similarity scores")
                return scores
                    
        except Exception as e:
            logger.error(f"Error computing reranking scores: {str(e)}", exc_info=True)
            return [0.5] * len(pairs)  # Default middle scores on error
            
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors with better error handling"""
        try:
            # Convert to numpy arrays if they aren't already
            if not isinstance(a, np.ndarray):
                a = np.array(a, dtype=np.float32)
            if not isinstance(b, np.ndarray):
                b = np.array(b, dtype=np.float32)
                
            # Ensure arrays are 1D
            a = a.flatten()
            b = b.flatten()
            
            # Check for empty arrays
            if a.size == 0 or b.size == 0:
                logger.warning(f"Empty vectors in cosine similarity calculation")
                return 0.0
                
            # Check for zero vectors
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            
            if a_norm == 0 or b_norm == 0:
                logger.warning(f"Zero vector in cosine similarity calculation")
                return 0.0
                
            # Calculate normalized dot product
            similarity = np.dot(a, b) / (a_norm * b_norm)
            
            # Handle numerical issues
            if np.isnan(similarity) or np.isinf(similarity):
                logger.warning(f"Invalid similarity value: {similarity}")
                return 0.0
                
            # Constrain to valid range [-1, 1]
            similarity = max(-1.0, min(1.0, similarity))
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error in cosine similarity calculation: {str(e)}", exc_info=True)
            return 0.0  # Return neutral similarity on error
    
    async def generate_rlhf_responses(self, query: str, context: str, llm_model: str) -> Tuple[str, str]:
        """
        Generate two different responses for RLHF preference collection using different prompting strategies.
        
        Args:
            query: User query
            context: Context documents for RAG
            llm_model: LLM model to use for generation
            
        Returns:
            Tuple of (analytical_response, conversational_response)
        """
        # Create two different prompts with different instructions
        analytical_prompt = f"""
        Answer the following question based on the provided context using an analytical, precise style:
        
        Context: {context}
        
        Question: {query}
        
        Provide a detailed, factual answer that is comprehensive and accurate, with a focus on
        technical correctness and precision. Use specific terminology and structured reasoning.
        """
        
        conversational_prompt = f"""
        Answer the following question based on the provided context using a conversational, friendly style:
        
        Context: {context}
        
        Question: {query}
        
        Provide a warm, accessible answer that feels like a helpful conversation. Use simpler language,
        analogies where appropriate, and a more personal tone. Focus on being engaging and relatable.
        """
        
        try:
            # Run both generations concurrently for efficiency
            analytical_task = self._generate_response(analytical_prompt, llm_model)
            conversational_task = self._generate_response(conversational_prompt, llm_model)
            
            # Wait for both to complete
            analytical_response, conversational_response = await asyncio.gather(
                analytical_task, conversational_task
            )
            
            return analytical_response, conversational_response
            
        except Exception as e:
            logger.error(f"Error generating RLHF responses: {str(e)}")
            # Fallback responses
            analytical_fallback = "I couldn't generate an analytical response due to a technical issue."
            conversational_fallback = "Sorry, I couldn't create a conversational response right now."
            return analytical_fallback, conversational_fallback

    async def _generate_response(self, prompt: str, model: str) -> str:
        """Generate a response using the LLM."""
        print(f"[RERANKER] Generating response with model: {model}")
        print(f"[RERANKER] Prompt to DeepSeek (first 100 chars): {prompt[:100]}...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=30
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Error from generation API: {error_text}")
                        return "Error generating response."
                    
                    result = await response.json()
                    response_text = result.get("response", "")
                    print(f"[RERANKER] Response from DeepSeek (first 100 chars): {response_text[:100]}...")
                    return response_text
        except Exception as e:
            logger.error(f"Error in _generate_response: {str(e)}")
            return f"Error: {str(e)[:50]}..."

    async def generate_split_document_responses(self, query: str, documents: List[Document], llm_model: str) -> Tuple[str, str]:
        """
        Generate two responses using the top two ranked documents as separate contexts.
        Uses a single reranking operation for efficiency.
        
        Args:
            query: User query
            documents: List of documents to rerank and use
            llm_model: Name of the LLM model to use
            
        Returns:
            Tuple of (first_doc_response, second_doc_response)
        """
        print(f"[RERANKER] Generating split document responses for query: {query[:50]}...")
        
        if not documents or len(documents) < 2:
            print(f"[RERANKER] Insufficient documents ({len(documents)}) for split document responses")
            return "Insufficient context to answer the question.", "Insufficient context to answer the question."
            
        try:
            # SINGLE RERANKER CALL - this is the key optimization
            print(f"[RERANKER] Reranking {len(documents)} documents for split response generation")
            reranked_docs = await self.rerank(query, documents, top_k=5)  # Get top 5 to ensure we have at least 2
            
            if not reranked_docs or len(reranked_docs) < 2:
                print(f"[RERANKER] Not enough relevant documents after reranking, only got {len(reranked_docs)}")
                return "Insufficient relevant context found.", "Insufficient relevant context found."
                
            # Extract the top 2 documents
            doc1_content = reranked_docs[0].page_content
            doc2_content = reranked_docs[1].page_content
            
            print(f"[RERANKER] Document 1 relevance preview: {doc1_content[:100]}...")
            print(f"[RERANKER] Document 2 relevance preview: {doc2_content[:100]}...")
            
            # Create prompts for each document
            doc1_prompt = f"""
            Answer the following question using only the context provided below:
            
            Context: {doc1_content}
            
            Question: {query}
            
            Provide a detailed answer focusing exclusively on the information in this context.
            """
            
            doc2_prompt = f"""
            Answer the following question using only the context provided below:
            
            Context: {doc2_content}
            
            Question: {query}
            
            Provide a detailed answer focusing exclusively on the information in this context.
            """
            
            print(f"[RERANKER] Doc1 prompt to DeepSeek (first 100 chars): {doc1_prompt[:100]}...")
            print(f"[RERANKER] Doc2 prompt to DeepSeek (first 100 chars): {doc2_prompt[:100]}...")
            
            # Generate responses in parallel for efficiency
            doc1_task = self._generate_response(doc1_prompt, llm_model)
            doc2_task = self._generate_response(doc2_prompt, llm_model)
            
            # Wait for both to complete
            doc1_response, doc2_response = await asyncio.gather(doc1_task, doc2_task)
            
            print(f"[RERANKER] Doc1 response preview: {doc1_response[:100]}...")
            print(f"[RERANKER] Doc2 response preview: {doc2_response[:100]}...")
            
            return doc1_response, doc2_response
            
        except Exception as e:
            logger.error(f"Error generating split document responses: {str(e)}")
            print(f"[RERANKER] Error in generate_split_document_responses: {str(e)}")
            return (f"Error generating response from document: {str(e)[:50]}...", 
                   f"Error generating response from document: {str(e)[:50]}...")

class HybridReranker:
    """Combines multiple ranking strategies for better results"""
        
    def __init__(self, base_reranker=None):
        """Initialize the hybrid reranker"""
        self.cross_encoder = base_reranker or CrossEncoderReranker()
        self.is_finetuned = False
        self.finetuned_weights = {
            'semantic_score': 1.0,
            'exact_match_score': 0.5,
            'recency_score': 0.3
        }
        # Path for loading/saving weights
        self.weights_path = os.getenv('RERANKER_WEIGHTS_PATH', "/app/data/finetune/reranker")
        Path(self.weights_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized Hybrid Reranker (weights path: {self.weights_path})")
        
        # Try to load the latest weights if available
        self._try_load_latest_weights()
    
    def _try_load_latest_weights(self):
        """Attempt to load the most recent weights file"""
        try:
            weights_dir = Path(self.weights_path)
            if not weights_dir.exists():
                logger.info(f"Weights directory doesn't exist: {weights_dir}")
                return
                
            # Find the most recent weights file
            weight_files = list(weights_dir.glob("reranker_weights_*.json"))
            if not weight_files:
                logger.info("No weight files found")
                return
                
            # Sort by modification time, newest first
            latest_file = max(weight_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Found latest weights file: {latest_file}")
            
            # Load the weights
            with open(latest_file, 'r') as f:
                weights = json.load(f)
                self.load_finetuned_weights(weights)
                logger.info(f"Successfully loaded weights from {latest_file}")
                
        except Exception as e:
            logger.error(f"Error loading latest weights: {str(e)}")
            # Continue with default weights
            
    async def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """
        Apply hybrid reranking to improve document relevance
        
        Args:
            query: User query
            documents: Retrieved documents
            top_k: Number of documents to return
            
        Returns:
            Reranked documents
        """ 
        # First apply cross-encoder reranking
        reranked_docs = await self.cross_encoder.rerank(query, documents, top_k=top_k)
        
        # Apply additional strategies if needed:
        # 1. Prioritize documents that contain exact query terms
        # 2. Consider document recency if available
        
        return reranked_docs
        
    async def generate_rlhf_responses(self, query: str, context: str, llm_model: str) -> Tuple[str, str]:
        """ 
        Generate two different responses for RLHF using the cross_encoder.
        
        Args:
            query: User query
            context: Context documents for RAG
            llm_model: LLM model to use for generation
            
        Returns:
            Tuple of (analytical_response, conversational_response)
        """ 
        return await self.cross_encoder.generate_rlhf_responses(query, context, llm_model)
        
    async def generate_split_document_responses(self, query: str, documents: List[Document], llm_model: str) -> Tuple[str, str]:
        """
        Generate two responses using the top two ranked documents as separate contexts.
        
        Args:
            query: User query
            documents: List of documents to rerank and use
            llm_model: Name of the LLM model to use
            
        Returns:
            Tuple of (first_doc_response, second_doc_response)
        """
        return await self.cross_encoder.generate_split_document_responses(query, documents, llm_model)
        
    def score_exact_match(self, query: str, document: Document) -> float:
        """Score based on exact matches of query terms in document"""
        query_terms = set(query.lower().split())
        doc_terms = set(document.page_content.lower().split())
        
        if not query_terms:
            return 0.0
            
        # Calculate Jaccard similarity
        intersection = query_terms.intersection(doc_terms)
        union = query_terms.union(doc_terms)
        
        return len(intersection) / len(union) if union else 0.0
    
    def load_finetuned_weights(self, weights: Dict[str, float]) -> None:
        """
        Load fine-tuned weights for the reranker
        
        Args:
            weights: Dictionary of weight values for different scoring components
        """
        if not weights:
            return
            
        try:
            # Update weights with validation
            if 'semantic_score' in weights and isinstance(weights['semantic_score'], (int, float)):
                self.finetuned_weights['semantic_score'] = float(weights['semantic_score'])
                
            if 'exact_match_score' in weights and isinstance(weights['exact_match_score'], (int, float)):
                self.finetuned_weights['exact_match_score'] = float(weights['exact_match_score'])
                
            if 'recency_score' in weights and isinstance(weights['recency_score'], (int, float)):
                self.finetuned_weights['recency_score'] = float(weights['recency_score'])
                
            self.is_finetuned = True
            logger.info(f"Loaded fine-tuned weights: {self.finetuned_weights}")
        except Exception as e:
            logger.error(f"Error loading fine-tuned weights: {str(e)}")


# Global reranker instance to be used throughout the application
_reranker_instance = None
_current_reranker_model = "BAAI/bge-reranker-large"
_current_reranker_provider = "ollama"

def get_reranker(model_name: str = None, provider: str = None):
    """Get or create a singleton reranker instance"""
    global _reranker_instance, _current_reranker_model, _current_reranker_provider
    
    # Use provided parameters or fall back to globals
    effective_model = model_name or _current_reranker_model
    effective_provider = provider or _current_reranker_provider
    
    # Recreate instance if model/provider changed
    if (_reranker_instance is None or 
        effective_model != _current_reranker_model or 
        effective_provider != _current_reranker_provider):
        logger.info(f"Creating reranker instance with model: {effective_model} (provider: {effective_provider})")
        _current_reranker_model = effective_model
        _current_reranker_provider = effective_provider
        base_reranker = CrossEncoderReranker(model_name=effective_model, provider=effective_provider)
        _reranker_instance = HybridReranker(base_reranker=base_reranker)
    return _reranker_instance

def get_current_reranker_model():
    """Get the current reranker model name"""
    global _current_reranker_model
    return _current_reranker_model

def set_reranker_model(model_name: str, provider: str = "ollama"):
    """Set the reranker model and reset the instance"""
    global _reranker_instance, _current_reranker_model, _current_reranker_provider
    
    logger.info(f"Setting reranker model to: {model_name} (provider: {provider})")
    _current_reranker_model = model_name
    _current_reranker_provider = provider
    
    # Reset the reranker instance to force recreation with new model
    _reranker_instance = None
    
    # Create new instance with the specified model
    base_reranker = CrossEncoderReranker(model_name=model_name, provider=provider)
    _reranker_instance = HybridReranker(base_reranker=base_reranker)
    
    # Clear the RAG module's cached reranker instance
    try:
        from rag import clear_reranker_cache
        clear_reranker_cache()
    except ImportError:
        logger.warning("Could not import clear_reranker_cache from rag module")