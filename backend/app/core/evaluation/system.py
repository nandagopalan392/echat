"""
Comprehensive evaluation system for RAG applications.
This module provides evaluation metrics for chat responses including
groundedness, context relevance, and answer relevance using LLM-based evaluation.
"""

import json
import logging
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import asyncio
from dataclasses import dataclass
from enum import Enum
import requests
import time
from abc import ABC, abstractmethod
from app.core.evaluation.config import evaluation_config

logger = logging.getLogger(__name__)

class EvaluationMetric(Enum):
    """Supported evaluation metrics"""
    GROUNDEDNESS = "groundedness"
    CONTEXT_RELEVANCE = "context_relevance"
    ANSWER_RELEVANCE = "answer_relevance"
    COMPREHENSIVENESS = "comprehensiveness"
    COHERENCE = "coherence"
    CONCISENESS = "conciseness"

@dataclass
class EvaluationResult:
    """Result of a single evaluation"""
    metric: EvaluationMetric
    score: float  # Normalized 0-1 score
    raw_score: float  # Original score before normalization
    reasoning: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class RAGTriadResult:
    """Result of the RAG triad evaluation (groundedness, answer relevance, context relevance)"""
    groundedness: EvaluationResult
    answer_relevance: EvaluationResult
    context_relevance: EvaluationResult
    overall_score: float
    evaluation_time_seconds: float

class LLMProvider(ABC):
    """Abstract base class for LLM providers used in evaluation"""
    
    @abstractmethod
    def generate_response(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        """Generate a response from the LLM"""
        pass

class OllamaProvider(LLMProvider):
    """Ollama LLM provider for evaluation"""
    
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = (base_url or evaluation_config.OLLAMA_BASE_URL).rstrip('/')
        self.model = model or evaluation_config.EVALUATION_MODEL
        
    def generate_response(self, prompt: str, temperature: float = None, max_tokens: int = None) -> str:
        """Generate a response using Ollama API"""
        try:
            temp = temperature if temperature is not None else evaluation_config.EVALUATION_TEMPERATURE
            max_tok = max_tokens if max_tokens is not None else evaluation_config.EVALUATION_MAX_TOKENS
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {
                        "temperature": temp,
                        "num_predict": max_tok
                    }
                },
                timeout=evaluation_config.EVALUATION_TIMEOUT
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error when calling Ollama (timeout: {evaluation_config.EVALUATION_TIMEOUT}s): {e}")
            # Return a default low score for timeout cases instead of failing completely
            return "Error: Request timed out. The evaluation model is taking too long to respond."
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error when calling Ollama: {e}")
            return "Error: Could not connect to evaluation model."
        except Exception as e:
            logger.error(f"Error generating response from Ollama: {e}")
            raise

class EvaluationPrompts:
    """Prompts for different evaluation metrics"""
    
    GROUNDEDNESS_SYSTEM = """You are a GROUNDEDNESS grader; providing the extent to which an ANSWER is grounded in the provided CONTEXT. Respond only as a number from 0 to 10 where 0 is the lowest and 10 is the highest based on the Criteria below.

Criteria for GROUNDEDNESS:
- COMPLETENESS: The ANSWER completely answered the question with all information provided in the CONTEXT.
- ACCURACY: All information in the ANSWER is accurate and faithful to the CONTEXT.
- NO HALLUCINATION: The ANSWER does not include any information not present in the CONTEXT.

Score 10: The ANSWER is completely grounded in the CONTEXT with perfect accuracy and no hallucinations.
Score 7-9: The ANSWER is mostly grounded with minor gaps or slight inaccuracies.
Score 4-6: The ANSWER is partially grounded but has notable gaps or inaccuracies.
Score 1-3: The ANSWER has some grounding but significant hallucinations or inaccuracies.
Score 0: The ANSWER is not grounded in the CONTEXT at all."""

    GROUNDEDNESS_USER = """CONTEXT: {context}

ANSWER: {answer}

GROUNDEDNESS:"""

    CONTEXT_RELEVANCE_SYSTEM = """You are a CONTEXT RELEVANCE grader; providing the extent to which the CONTEXT is relevant to the QUESTION. Respond only as a number from 0 to 10 where 0 is the lowest and 10 is the highest based on the Criteria below.

Criteria for CONTEXT RELEVANCE:
- DIRECT RELEVANCE: The CONTEXT directly addresses the QUESTION.
- INFORMATION SUFFICIENCY: The CONTEXT contains sufficient information to answer the QUESTION.
- TOPIC ALIGNMENT: The CONTEXT is on the same topic as the QUESTION.

Score 10: The CONTEXT is perfectly relevant and sufficient to answer the QUESTION.
Score 7-9: The CONTEXT is highly relevant with minor gaps.
Score 4-6: The CONTEXT is moderately relevant but may lack some key information.
Score 1-3: The CONTEXT has limited relevance to the QUESTION.
Score 0: The CONTEXT is not relevant to the QUESTION at all."""

    CONTEXT_RELEVANCE_USER = """QUESTION: {question}

CONTEXT: {context}

RELEVANCE:"""

    ANSWER_RELEVANCE_SYSTEM = """You are an ANSWER RELEVANCE grader; providing the extent to which the ANSWER addresses the QUESTION. Respond only as a number from 0 to 10 where 0 is the lowest and 10 is the highest based on the Criteria below.

Criteria for ANSWER RELEVANCE:
- DIRECT RESPONSE: The ANSWER directly responds to what was asked in the QUESTION.
- COMPLETENESS: The ANSWER completely addresses all parts of the QUESTION.
- FOCUS: The ANSWER stays focused on the QUESTION without unnecessary tangents.

Score 10: The ANSWER perfectly addresses the QUESTION with complete focus and coverage.
Score 7-9: The ANSWER addresses the QUESTION well with minor gaps or slight tangents.
Score 4-6: The ANSWER partially addresses the QUESTION but may be incomplete or unfocused.
Score 1-3: The ANSWER has limited relevance to the QUESTION.
Score 0: The ANSWER does not address the QUESTION at all."""

    ANSWER_RELEVANCE_USER = """QUESTION: {question}

ANSWER: {answer}

RELEVANCE:"""

    COT_REASONS_TEMPLATE = """Please provide your reasoning step by step:

Criteria: [Explain the evaluation criteria you are applying]

Supporting Evidence: [Provide specific evidence from the text that supports your evaluation]

Score: [Provide your score from 0 to 10]"""

class RAGMetricsEvaluator:
    """Main evaluator class implementing RAG evaluation metrics"""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        
    def _extract_score_from_response(self, response: str, min_score: int = 0, max_score: int = 10) -> Tuple[float, str]:
        """Extract numeric score from LLM response"""
        try:
            # Look for patterns like "Score: 8" or just "8"
            score_patterns = [
                r"(?:Score|score):\s*(\d+(?:\.\d+)?)",
                r"\b(\d+(?:\.\d+)?)\s*(?:/\s*10|\s*out\s*of\s*10)?\b",
                r"(\d+(?:\.\d+)?)"
            ]
            
            for pattern in score_patterns:
                matches = re.findall(pattern, response)
                if matches:
                    score = float(matches[-1])  # Take the last match
                    if min_score <= score <= max_score:
                        normalized_score = (score - min_score) / (max_score - min_score)
                        return normalized_score, response
            
            # If no valid score found, return default
            logger.warning(f"Could not extract valid score from response: {response[:100]}...")
            return 0.5, response
            
        except Exception as e:
            logger.error(f"Error extracting score: {e}")
            return 0.5, response
    
    def _generate_score_and_reasons(
        self, 
        system_prompt: str, 
        user_prompt: str,
        min_score: int = 0,
        max_score: int = 10,
        temperature: float = 0.0
    ) -> Tuple[float, Dict[str, Any]]:
        """Generate score and reasoning using LLM"""
        try:
            # Combine prompts
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Generate response
            response = self.llm_provider.generate_response(
                prompt=full_prompt,
                temperature=temperature,
                max_tokens=512
            )
            
            # Extract score and normalize
            score, reasoning = self._extract_score_from_response(response, min_score, max_score)
            
            return score, {"reason": reasoning, "raw_response": response}
            
        except Exception as e:
            logger.error(f"Error generating score and reasons: {e}")
            return 0.5, {"reason": f"Error in evaluation: {str(e)}", "raw_response": ""}
    
    def groundedness_measure_with_cot_reasons(
        self,
        source: str,
        statement: str,
        min_score: int = 0,
        max_score: int = 10,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        Measure how well the statement is grounded in the source material.
        
        Args:
            source: The source context that should support the statement
            statement: The statement/answer to evaluate
            min_score: Minimum score value
            max_score: Maximum score value
            temperature: LLM temperature for generation
            
        Returns:
            Tuple of (normalized_score, reasons_dict)
        """
        user_prompt = EvaluationPrompts.GROUNDEDNESS_USER.format(
            context=source,
            answer=statement
        )
        
        # Add chain of thought template
        user_prompt = user_prompt.replace(
            "GROUNDEDNESS:", 
            EvaluationPrompts.COT_REASONS_TEMPLATE
        )
        
        return self._generate_score_and_reasons(
            system_prompt=EvaluationPrompts.GROUNDEDNESS_SYSTEM,
            user_prompt=user_prompt,
            min_score=min_score,
            max_score=max_score,
            temperature=temperature
        )
    
    def context_relevance_with_cot_reasons(
        self,
        question: str,
        context: str,
        min_score: int = 0,
        max_score: int = 10,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        Measure how relevant the context is to the question.
        
        Args:
            question: The user's question
            context: The retrieved context
            min_score: Minimum score value
            max_score: Maximum score value
            temperature: LLM temperature for generation
            
        Returns:
            Tuple of (normalized_score, reasons_dict)
        """
        user_prompt = EvaluationPrompts.CONTEXT_RELEVANCE_USER.format(
            question=question,
            context=context
        )
        
        # Add chain of thought template
        user_prompt = user_prompt.replace(
            "RELEVANCE:", 
            EvaluationPrompts.COT_REASONS_TEMPLATE
        )
        
        return self._generate_score_and_reasons(
            system_prompt=EvaluationPrompts.CONTEXT_RELEVANCE_SYSTEM,
            user_prompt=user_prompt,
            min_score=min_score,
            max_score=max_score,
            temperature=temperature
        )
    
    def relevance_with_cot_reasons(
        self,
        prompt: str,
        response: str,
        min_score: int = 0,
        max_score: int = 10,
        temperature: float = 0.0,
    ) -> Tuple[float, Dict]:
        """
        Measure how relevant the response is to the prompt/question.
        
        Args:
            prompt: The user's question/prompt
            response: The system's response
            min_score: Minimum score value
            max_score: Maximum score value
            temperature: LLM temperature for generation
            
        Returns:
            Tuple of (normalized_score, reasons_dict)
        """
        user_prompt = EvaluationPrompts.ANSWER_RELEVANCE_USER.format(
            question=prompt,
            answer=response
        )
        
        # Add chain of thought template
        user_prompt = user_prompt.replace(
            "RELEVANCE:", 
            EvaluationPrompts.COT_REASONS_TEMPLATE
        )
        
        return self._generate_score_and_reasons(
            system_prompt=EvaluationPrompts.ANSWER_RELEVANCE_SYSTEM,
            user_prompt=user_prompt,
            min_score=min_score,
            max_score=max_score,
            temperature=temperature
        )
    
    def rag_triad(
        self,
        question: str,
        answer: str,
        context: str,
        temperature: float = 0.0
    ) -> RAGTriadResult:
        """
        Evaluate a RAG interaction using the three core metrics: 
        groundedness, context relevance, and answer relevance.
        
        Args:
            question: The user's question
            answer: The system's answer
            context: The retrieved context
            temperature: LLM temperature for generation
            
        Returns:
            RAGTriadResult containing all three evaluations
        """
        print(f"🚀 RAGMetricsEvaluator: Starting rag_triad for question: {question[:50]}...")
        logger.info(f"🚀 RAGMetricsEvaluator: Starting rag_triad for question: {question[:50]}...")
        start_time = time.time()
        
        try:
            # Evaluate groundedness
            print(f"🚀 RAGMetricsEvaluator: Starting groundedness evaluation")
            logger.info(f"🚀 RAGMetricsEvaluator: Starting groundedness evaluation")
            groundedness_score, groundedness_reasons = self.groundedness_measure_with_cot_reasons(
                source=context,
                statement=answer,
                temperature=temperature
            )
            print(f"🚀 RAGMetricsEvaluator: Groundedness score: {groundedness_score}")
            logger.info(f"🚀 RAGMetricsEvaluator: Groundedness score: {groundedness_score}")
            
            groundedness_result = EvaluationResult(
                metric=EvaluationMetric.GROUNDEDNESS,
                score=groundedness_score,
                raw_score=groundedness_score * 10,  # Convert back to 0-10 scale
                reasoning=groundedness_reasons.get("reason", ""),
                timestamp=datetime.now(),
                metadata=groundedness_reasons
            )
            
            # Evaluate context relevance
            context_relevance_score, context_relevance_reasons = self.context_relevance_with_cot_reasons(
                question=question,
                context=context,
                temperature=temperature
            )
            
            context_relevance_result = EvaluationResult(
                metric=EvaluationMetric.CONTEXT_RELEVANCE,
                score=context_relevance_score,
                raw_score=context_relevance_score * 10,
                reasoning=context_relevance_reasons.get("reason", ""),
                timestamp=datetime.now(),
                metadata=context_relevance_reasons
            )
            
            # Evaluate answer relevance
            answer_relevance_score, answer_relevance_reasons = self.relevance_with_cot_reasons(
                prompt=question,
                response=answer,
                temperature=temperature
            )
            
            answer_relevance_result = EvaluationResult(
                metric=EvaluationMetric.ANSWER_RELEVANCE,
                score=answer_relevance_score,
                raw_score=answer_relevance_score * 10,
                reasoning=answer_relevance_reasons.get("reason", ""),
                timestamp=datetime.now(),
                metadata=answer_relevance_reasons
            )
            
            # Calculate overall score (weighted average)
            overall_score = (
                groundedness_score * evaluation_config.METRIC_WEIGHTS['groundedness'] +
                answer_relevance_score * evaluation_config.METRIC_WEIGHTS['answer_relevance'] +
                context_relevance_score * evaluation_config.METRIC_WEIGHTS['context_relevance']
            )
            
            evaluation_time = time.time() - start_time
            
            return RAGTriadResult(
                groundedness=groundedness_result,
                answer_relevance=answer_relevance_result,
                context_relevance=context_relevance_result,
                overall_score=overall_score,
                evaluation_time_seconds=evaluation_time
            )
            
        except Exception as e:
            logger.error(f"Error in RAG triad evaluation: {e}")
            # Return default/error results
            error_result = EvaluationResult(
                metric=EvaluationMetric.GROUNDEDNESS,
                score=0.0,
                raw_score=0.0,
                reasoning=f"Evaluation error: {str(e)}",
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )
            
            return RAGTriadResult(
                groundedness=error_result,
                answer_relevance=error_result,
                context_relevance=error_result,
                overall_score=0.0,
                evaluation_time_seconds=time.time() - start_time
            )

class EvaluationManager:
    """Manager class for handling evaluation workflows and storage"""
    
    def __init__(self, evaluator: RAGMetricsEvaluator):
        self.evaluator = evaluator
        self.evaluation_history: List[RAGTriadResult] = []
        self.max_history = evaluation_config.MAX_EVALUATION_HISTORY
    
    async def evaluate_rag_interaction(
        self,
        question: str,
        answer: str,
        context: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> RAGTriadResult:
        """
        Evaluate a complete RAG interaction asynchronously.
        
        Args:
            question: User's question
            answer: System's answer
            context: Retrieved context
            session_id: Optional session identifier
            user_id: Optional user identifier
            
        Returns:
            RAGTriadResult containing evaluation scores
        """
        try:
            # Run evaluation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.evaluator.rag_triad,
                question,
                answer,
                context
            )
            
            # Store in history (with size limit)
            self.evaluation_history.append(result)
            if len(self.evaluation_history) > self.max_history:
                self.evaluation_history = self.evaluation_history[-self.max_history:]
            
            # Log evaluation
            logger.info(
                f"RAG evaluation completed - Overall: {result.overall_score:.3f}, "
                f"Groundedness: {result.groundedness.score:.3f}, "
                f"Answer Relevance: {result.answer_relevance.score:.3f}, "
                f"Context Relevance: {result.context_relevance.score:.3f}, "
                f"Time: {result.evaluation_time_seconds:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in evaluation manager: {e}")
            raise
    
    def get_evaluation_statistics(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Get statistics from recent evaluations.
        
        Args:
            last_n: Number of recent evaluations to analyze (all if None)
            
        Returns:
            Dictionary containing evaluation statistics
        """
        if not self.evaluation_history:
            return {
                "total_evaluations": 0,
                "metrics": {},
                "summary": "No evaluations available"
            }
        
        # Get subset of evaluations
        evaluations = self.evaluation_history[-last_n:] if last_n else self.evaluation_history
        
        # Calculate statistics
        groundedness_scores = [e.groundedness.score for e in evaluations]
        answer_relevance_scores = [e.answer_relevance.score for e in evaluations]
        context_relevance_scores = [e.context_relevance.score for e in evaluations]
        overall_scores = [e.overall_score for e in evaluations]
        evaluation_times = [e.evaluation_time_seconds for e in evaluations]
        
        return {
            "total_evaluations": len(evaluations),
            "time_period": f"Last {len(evaluations)} evaluations",
            "metrics": {
                "groundedness": {
                    "mean": np.mean(groundedness_scores),
                    "std": np.std(groundedness_scores),
                    "min": np.min(groundedness_scores),
                    "max": np.max(groundedness_scores),
                    "median": np.median(groundedness_scores)
                },
                "answer_relevance": {
                    "mean": np.mean(answer_relevance_scores),
                    "std": np.std(answer_relevance_scores),
                    "min": np.min(answer_relevance_scores),
                    "max": np.max(answer_relevance_scores),
                    "median": np.median(answer_relevance_scores)
                },
                "context_relevance": {
                    "mean": np.mean(context_relevance_scores),
                    "std": np.std(context_relevance_scores),
                    "min": np.min(context_relevance_scores),
                    "max": np.max(context_relevance_scores),
                    "median": np.median(context_relevance_scores)
                },
                "overall_score": {
                    "mean": np.mean(overall_scores),
                    "std": np.std(overall_scores),
                    "min": np.min(overall_scores),
                    "max": np.max(overall_scores),
                    "median": np.median(overall_scores)
                }
            },
            "performance": {
                "avg_evaluation_time": np.mean(evaluation_times),
                "min_evaluation_time": np.min(evaluation_times),
                "max_evaluation_time": np.max(evaluation_times)
            },
            "thresholds": evaluation_config.QUALITY_THRESHOLDS
        }

# Global evaluation manager instance
_evaluation_manager: Optional[EvaluationManager] = None

def get_evaluation_manager() -> EvaluationManager:
    """Get or create the global evaluation manager instance"""
    global _evaluation_manager
    
    if _evaluation_manager is None:
        # Initialize with Ollama provider (can be configured)
        ollama_provider = OllamaProvider()
        evaluator = RAGMetricsEvaluator(ollama_provider)
        _evaluation_manager = EvaluationManager(evaluator)
    
    return _evaluation_manager

# Convenience functions for integration
async def evaluate_chat_response(
    question: str,
    answer: str,
    context: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> RAGTriadResult:
    """
    Convenience function to evaluate a chat response.
    
    Args:
        question: User's question
        answer: System's answer  
        context: Retrieved context
        session_id: Optional session ID
        user_id: Optional user ID
        
    Returns:
        RAGTriadResult with evaluation scores
    """
    manager = get_evaluation_manager()
    return await manager.evaluate_rag_interaction(
        question=question,
        answer=answer,
        context=context,
        session_id=session_id,
        user_id=user_id
    )

def get_recent_evaluation_stats(last_n: int = 100) -> Dict[str, Any]:
    """
    Get statistics from recent evaluations.
    
    Args:
        last_n: Number of recent evaluations to analyze
        
    Returns:
        Dictionary containing evaluation statistics
    """
    manager = get_evaluation_manager()
    return manager.get_evaluation_statistics(last_n=last_n)

# RAGEvaluator class for compatibility with Celery tasks
class RAGEvaluator:
    """
    Simplified RAG evaluator class for use in background tasks.
    
    This class provides a simplified interface that's compatible with
    the Celery task system while reusing the core RAGMetricsEvaluator logic.
    """
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        print("🚀 RAGEvaluator: Initializing RAGEvaluator")
        logger.info("🚀 RAGEvaluator: Initializing RAGEvaluator")
        if llm_provider is None:
            print("🚀 RAGEvaluator: Creating OllamaProvider")
            logger.info("🚀 RAGEvaluator: Creating OllamaProvider")
            llm_provider = OllamaProvider()
        print("🚀 RAGEvaluator: Creating RAGMetricsEvaluator")
        logger.info("🚀 RAGEvaluator: Creating RAGMetricsEvaluator")
        self.evaluator = RAGMetricsEvaluator(llm_provider)
        print("🚀 RAGEvaluator: RAGEvaluator initialization complete")
        logger.info("🚀 RAGEvaluator: RAGEvaluator initialization complete")
    
    def evaluate_rag_triad(
        self,
        query: str,
        response: str,
        context: str,
        temperature: float = 0.0
    ) -> RAGTriadResult:
        """
        Evaluate a RAG interaction using the three core metrics.
        
        Args:
            query: User's query/question
            response: System's response/answer
            context: Retrieved context
            temperature: LLM temperature for generation
            
        Returns:
            RAGTriadResult containing all three evaluations
        """
        print(f"🚀 RAGEvaluator: Starting rag_triad evaluation for query: {query[:50]}...")
        logger.info(f"🚀 RAGEvaluator: Starting rag_triad evaluation for query: {query[:50]}...")
        result = self.evaluator.rag_triad(
            question=query,
            answer=response,
            context=context,
            temperature=temperature
        )
        print(f"🚀 RAGEvaluator: Completed rag_triad evaluation with overall score: {result.overall_score}")
        logger.info(f"🚀 RAGEvaluator: Completed rag_triad evaluation with overall score: {result.overall_score}")
        return result
