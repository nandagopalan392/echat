"""
Celery tasks for background RAG evaluation processing
"""

import json
import logging
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from celery import current_task
from app.workers.celery_app import celery_app
from app.core.evaluation.system import RAGEvaluator, RAGTriadResult, EvaluationResult
from app.core.evaluation.config import evaluation_config
import redis
import os
import requests  # For synchronous HTTP calls to Ollama
import re  # For regex matching in response parsing

# Configure logging
logger = logging.getLogger(__name__)

# Redis client for pub/sub notifications - use same DB as Celery broker
redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))

def call_ollama_sync(model: str, prompt: str, temperature: float = 0.0) -> str:
    """Call Ollama API synchronously"""
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 1000
            }
        }
        
        response = requests.post(
            f"{os.getenv('OLLAMA_HOST', 'http://ollama:11434')}/api/chat",
            json=payload,
            timeout=60.0
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('message', {}).get('content', '').strip()
        else:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return ""
            
    except Exception as e:
        logger.error(f"Error calling Ollama API: {e}")
        return ""

def generate_single_qa_pair_sync(content_chunk: str, document_filename: str, difficulty: str, model_name: str, question_index: int, source_chunks: List[str] = None) -> dict:
    """Generate a single question-answer pair synchronously"""
    
    generation_prompt = f"""Based on the following document content, generate ONE high-quality question and answer pair.

Document Content:
{content_chunk}

Requirements:
- Generate a {difficulty} difficulty question
- The question should be specific and answerable from the given content
- Provide a comprehensive answer based on the text
- Ensure the question tests understanding of the content

Difficulty Guidelines:
- Easy: Direct questions about facts explicitly stated in the text
- Medium: Questions requiring some understanding and connection of concepts  
- Hard: Complex questions requiring analysis, synthesis, or deeper understanding

OUTPUT FORMAT (JSON):
{{
    "question": "Your generated question here",
    "answer": "Your comprehensive answer here based on the text", 
    "confidence": "high/medium/low - how confident you are in this Q&A pair",
    "reasoning": "Brief explanation of why this is a good {difficulty} question"
}}

Generate only the JSON output, nothing else."""

    try:
        response = call_ollama_sync(model_name, generation_prompt, temperature=0.7)
        if not response:
            return None
            
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                qa_data = json.loads(json_match.group())
                
                if qa_data.get('question') and qa_data.get('answer'):
                    # Create expected_chunks from the source chunks if provided, otherwise use content_chunk
                    expected_chunks = []
                    if source_chunks:
                        for chunk_idx, chunk in enumerate(source_chunks):
                            expected_chunks.append({
                                "text": chunk,
                                "title": document_filename,
                                "source": document_filename,
                                "chunk_index": chunk_idx,
                                "relevance_score": 1.0
                            })
                    else:
                        expected_chunks = [{
                            "text": content_chunk,
                            "title": document_filename,
                            "source": document_filename,
                            "chunk_index": 0,
                            "relevance_score": 1.0
                        }]
                    
                    return {
                        'question': qa_data['question'].strip(),
                        'answer': qa_data['answer'].strip(),
                        'difficulty': difficulty,
                        'source_file': document_filename,
                        'question_index': question_index,
                        'confidence': qa_data.get('confidence', 'medium'),
                        'reasoning': qa_data.get('reasoning', ''),
                        'expected_chunks': expected_chunks
                    }
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from LLM response: {response}")
                
        return None
        
    except Exception as e:
        logger.error(f"Error generating Q&A pair: {e}")
        return None

def chunk_content_sync(content: str, max_chunk_size: int = 2000) -> List[str]:
    """Split content into chunks synchronously"""
    if len(content) <= max_chunk_size:
        return [content]
    
    chunks = []
    sentences = content.split('. ')
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + sentence + '. ') <= max_chunk_size:
            current_chunk += sentence + '. '
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [content]

class ConcurrencyManager:
    """Manages task concurrency and rate limiting with priority support"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.dataset_creation_limit = 1  # SERIALIZED: Only 1 dataset creation at a time (Sentry pattern)
        self.evaluation_limit = 3        # Max 3 concurrent evaluations
        
        # Priority levels
        self.PRIORITY_HIGH = "high"      # Evaluations (user-facing)
        self.PRIORITY_LOW = "low"        # Dataset creation (background)
        
    def check_for_high_priority_tasks(self) -> bool:
        """Check if there are any high-priority tasks waiting"""
        evaluation_queue_key = "evaluation_queue"
        waiting_count = self.redis.llen(evaluation_queue_key)
        return waiting_count > 0
        
    def pause_low_priority_tasks(self):
        """Signal low-priority tasks to pause"""
        pause_key = "low_priority_pause"
        self.redis.setex(pause_key, 300, "1")  # Pause for 5 minutes max
        logger.info("🔄 PRIORITY: Signaled low-priority tasks to pause for high-priority task")
        
    def resume_low_priority_tasks(self):
        """Resume low-priority tasks"""
        pause_key = "low_priority_pause"
        self.redis.delete(pause_key)
        logger.info("▶️ PRIORITY: Resumed low-priority tasks")
        
    def should_pause_low_priority(self) -> bool:
        """Check if low-priority tasks should pause"""
        pause_key = "low_priority_pause"
        return self.redis.exists(pause_key)
    
    def acquire_dataset_creation_slot(self, task_id: str, timeout: int = 300) -> bool:
        """
        Acquire a slot for dataset creation with timeout and priority awareness
        Returns True if slot acquired, False if timeout
        """
        key = "dataset_creation_slots"
        acquired_key = f"dataset_creation_acquired:{task_id}"
        
        # Try to acquire slot for up to timeout seconds
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if high-priority tasks are waiting and we should pause
            if self.should_pause_low_priority():
                logger.info(f"📋 PRIORITY: Dataset creation task {task_id} pausing for high-priority task")
                time.sleep(5)  # Wait longer when paused
                continue
            
            # Get current count
            current_count = self.redis.scard(key)
            
            if current_count < self.dataset_creation_limit:
                # Try to acquire slot atomically
                if self.redis.sadd(key, task_id):
                    self.redis.expire(key, 3600)  # Expire in 1 hour
                    self.redis.setex(acquired_key, 3600, "1")
                    logger.info(f"📋 CONCURRENCY: Acquired dataset creation slot for task {task_id} ({current_count + 1}/{self.dataset_creation_limit})")
                    return True
            
            # Wait before retrying
            time.sleep(1)
            
        logger.warning(f"📋 CONCURRENCY: Failed to acquire dataset creation slot for task {task_id} after {timeout}s timeout")
        return False
    
    def release_dataset_creation_slot(self, task_id: str):
        """Release a dataset creation slot"""
        key = "dataset_creation_slots"
        acquired_key = f"dataset_creation_acquired:{task_id}"
        
        removed = self.redis.srem(key, task_id)
        self.redis.delete(acquired_key)
        
        if removed:
            current_count = self.redis.scard(key)
            logger.info(f"📋 CONCURRENCY: Released dataset creation slot for task {task_id} ({current_count}/{self.dataset_creation_limit})")
        
    def acquire_evaluation_slot(self, task_id: str, timeout: int = 60) -> bool:
        """
        Acquire a slot for evaluation with timeout and priority support
        Returns True if slot acquired, False if timeout
        """
        key = "evaluation_slots"
        acquired_key = f"evaluation_acquired:{task_id}"
        queue_key = "evaluation_queue"
        
        # Add to high-priority queue first
        self.redis.lpush(queue_key, task_id)
        
        # Signal low-priority tasks to pause if any are running
        if self.redis.scard("dataset_creation_slots") > 0:
            self.pause_low_priority_tasks()
        
        start_time = time.time()
        try:
            while time.time() - start_time < timeout:
                current_count = self.redis.scard(key)
                
                if current_count < self.evaluation_limit:
                    if self.redis.sadd(key, task_id):
                        self.redis.expire(key, 3600)
                        self.redis.setex(acquired_key, 3600, "1")
                        logger.info(f"⚡ CONCURRENCY: Acquired evaluation slot for task {task_id} ({current_count + 1}/{self.evaluation_limit})")
                        return True
                
                time.sleep(0.5)  # Shorter wait for evaluations
                
            logger.warning(f"⚡ CONCURRENCY: Failed to acquire evaluation slot for task {task_id} after {timeout}s timeout")
            return False
        finally:
            # Remove from queue whether successful or not
            self.redis.lrem(queue_key, 1, task_id)
    
    def release_evaluation_slot(self, task_id: str):
        """Release an evaluation slot and resume low-priority tasks if needed"""
        key = "evaluation_slots"
        acquired_key = f"evaluation_acquired:{task_id}"
        queue_key = "evaluation_queue"
        
        removed = self.redis.srem(key, task_id)
        self.redis.delete(acquired_key)
        
        if removed:
            current_count = self.redis.scard(key)
            logger.info(f"⚡ CONCURRENCY: Released evaluation slot for task {task_id} ({current_count}/{self.evaluation_limit})")
            
            # If no more evaluations in queue, resume low-priority tasks
            if self.redis.llen(queue_key) == 0:
                self.resume_low_priority_tasks()

# Global concurrency manager
concurrency_manager = ConcurrencyManager(redis_client)

class EvaluationTaskStatus:
    """Task status constants"""
    PENDING = "PENDING"
    STARTED = "STARTED"
    PROGRESS = "PROGRESS"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"

def publish_evaluation_update(task_id: str, status: str, data: Dict[str, Any]):
    """
    Publish evaluation update to Redis pub/sub channel and cache for polling
    
    This function both sends real-time updates via WebSocket and caches
    the latest status for clients using polling fallback.
    """
    try:
        update = {
            "task_id": task_id,
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        
        # Publish to WebSocket subscribers
        redis_client.publish(f"evaluation_updates:{task_id}", json.dumps(update))
        
        # Cache latest progress for polling clients (expire after 24 hours)
        redis_client.setex(f"task_progress:{task_id}", 86400, json.dumps(update))
        
        # For completed tasks, also cache in a separate key for longer retention
        if status in [EvaluationTaskStatus.SUCCESS, EvaluationTaskStatus.FAILURE]:
            redis_client.setex(f"task_final_status:{task_id}", 604800, json.dumps(update))  # 7 days
        
        logger.info(f"Published update for task {task_id}: {status}")
        
    except Exception as e:
        logger.error(f"Failed to publish evaluation update: {e}")

@celery_app.task(name="evaluation_tasks.cleanup_old_evaluation_results")
def cleanup_old_evaluation_results():
    """
    Periodic task to cleanup old evaluation results from Redis
    """
    try:
        # Find all evaluation result keys
        pattern = "evaluation_result:*"
        keys = redis_client.keys(pattern)
        
        cleaned_count = 0
        for key in keys:
            # Check if key exists and get TTL
            ttl = redis_client.ttl(key)
            if ttl == -1:  # Key exists but has no expiration
                # Set expiration to 1 hour for keys without TTL
                redis_client.expire(key, 3600)
                cleaned_count += 1
        
        # Also cleanup batch evaluation results
        batch_pattern = "batch_evaluation_result:*"
        batch_keys = redis_client.keys(batch_pattern)
        
        for key in batch_keys:
            ttl = redis_client.ttl(key)
            if ttl == -1:
                redis_client.expire(key, 3600)
                cleaned_count += 1
        
        logger.info(f"Cleanup task completed. Set expiration for {cleaned_count} keys.")
        return {"cleaned_keys": cleaned_count, "timestamp": datetime.utcnow().isoformat()}
        
    except Exception as e:
        logger.error(f"Cleanup task failed: {e}")
        raise

@celery_app.task(bind=True, name="evaluation_tasks.evaluate_dataset_with_rag")
def evaluate_dataset_with_rag(self, dataset_id: int, model_id: str, retrieval_config: Dict, user_id: str = "admin"):
    """
    Perform real RAG evaluation using dataset and actual RAG system with concurrency control
    
    Args:
        dataset_id: ID of the evaluation dataset to use
        model_id: Model to use for evaluation
        retrieval_config: Retrieval configuration
        user_id: User performing the evaluation
        
    Returns:
        Dict containing evaluation results
    """
    task_id = self.request.id
    start_time = time.time()
    
    logger.info(f"🚀 DATASET EVAL: Starting real dataset evaluation for task {task_id}")
    logger.info(f"🚀 DATASET EVAL: Dataset {dataset_id}, Model {model_id}, User {user_id}")
    
    # Try to acquire concurrency slot for evaluation
    if not concurrency_manager.acquire_evaluation_slot(task_id, timeout=60):
        error_msg = "Evaluation queue is full. Please try again later."
        logger.error(f"⚡ CONCURRENCY: {error_msg} Task: {task_id}")
        
        # Publish failure notification
        publish_evaluation_update(task_id, EvaluationTaskStatus.FAILURE, {
            "message": error_msg,
            "error": "evaluation_concurrency_limit_reached"
        })
        raise Exception(error_msg)
    
    try:
        # Get database connection
        from app.db import DatabaseConnection
        from app.db.repositories import EvaluationRepository
        
        db = DatabaseConnection()
        eval_repo = EvaluationRepository(db)
        
        # Get dataset from database first to get the name
        dataset = eval_repo.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        dataset_name = dataset.name
        
        # Update database to mark task as started
        try:
            # Create evaluation task record in database
            eval_repo.create_task(
                task_id=task_id,
                dataset_id=dataset_id,
                total_queries=0  # Will be updated later
            )
            
            eval_repo.update_task_progress(task_id, 0, "STARTED")
            logger.info(f"🚀 DATABASE: Created and started task {task_id}")
        except Exception as db_error:
            logger.error(f"🚀 DATABASE ERROR: Failed to create task: {db_error}")
        
        # Update task status to started
        publish_evaluation_update(task_id, EvaluationTaskStatus.STARTED, {
            "message": "Starting real dataset evaluation",
            "dataset_id": dataset_id,
            "model_id": model_id
        })
        logger.info(f"🚀 REDIS: Published STARTED status for task {task_id}")
        
        logger.info(f"🚀 DATASET: Loaded dataset '{dataset.name}'")
        
        # Load dataset content from file path
        questions = []
        file_path = dataset.file_path
        if file_path:
            try:
                import os
                # Try multiple possible paths
                possible_paths = [
                    os.path.join('/app/data/datasets', file_path),
                    os.path.join('/app/backend/data/chunking_configs', file_path),
                    os.path.join('/app/data/chunking_configs', file_path),
                    file_path  # In case it's already an absolute path
                ]
                
                dataset_content = None
                used_path = None
                
                for full_path in possible_paths:
                    try:
                        if os.path.exists(full_path):
                            logger.info(f"🚀 DATASET: Loading questions from {full_path}")
                            with open(full_path, 'r') as f:
                                dataset_content = json.load(f)
                            used_path = full_path
                            break
                    except Exception as path_error:
                        logger.debug(f"🚀 DATASET: Path {full_path} failed: {path_error}")
                        continue
                
                if not dataset_content:
                    raise FileNotFoundError(f"Could not find dataset file {file_path} in any expected location")
                    
                # Extract questions from dataset structure
                if 'items' in dataset_content:
                    # New format with items array
                    questions = dataset_content['items']
                    logger.info(f"🚀 DATASET: Found {len(questions)} items in dataset")
                elif 'questions' in dataset_content:
                    # Legacy format with questions array
                    questions = dataset_content['questions']
                    logger.info(f"🚀 DATASET: Found {len(questions)} questions in dataset")
                else:
                    logger.warning(f"🚀 DATASET: Unknown dataset format: {list(dataset_content.keys())}")
                    
            except Exception as file_error:
                logger.error(f"🚀 DATASET ERROR: Failed to load dataset file {file_path}: {file_error}")
        
        if not questions:
            logger.warning(f"🚀 DATASET: No questions found in dataset {dataset_id}, generating sample questions")
            # Generate sample questions based on documents
            questions = generate_sample_questions(dataset_id)
        
        logger.info(f"🚀 DATASET: Found {len(questions)} questions for evaluation")
        
        # Initialize RAG system with specified model and config
        publish_evaluation_update(task_id, EvaluationTaskStatus.PROGRESS, {
            "message": "Initializing RAG system",
            "progress": 0.1
        })
        
        # Use evaluation model from config, not from database
        from app.core.evaluation.config import EvaluationConfig
        evaluation_model = EvaluationConfig.EVALUATION_MODEL
        
        logger.info(f"🚀 RAG: Initializing RAG system with evaluation model {evaluation_model}")
        
        # Import RAG engine directly and create instance with evaluation model
        from app.core.rag import RAGEngine
        
        # Create RAG engine instance with evaluation model (NOT from database)
        rag_engine = RAGEngine(llm_model=evaluation_model)
        rag_engine.ensure_models_loaded()
        
        logger.info(f"🚀 RAG: RAG engine initialized with model {evaluation_model}")
        
        # Update retrieval configuration if needed
        if retrieval_config:
            logger.info(f"🚀 RAG: Applying retrieval config: {retrieval_config}")
            # The RAG engine uses default retrieval configuration
            # You can add specific config application here if needed
        
        # Initialize evaluation system
        logger.info(f"🚀 EVALUATOR: Initializing RAG evaluator for task {task_id}")
        evaluator = RAGEvaluator()
        logger.info(f"🚀 EVALUATOR: RAG evaluator initialized successfully")
        
        # Process questions and generate real Q&A pairs
        total_questions = min(len(questions), 5)  # Limit to 5 questions for demo
        evaluation_results = []
        
        for i, question_data in enumerate(questions[:total_questions]):
            current_progress = 0.2 + (i / total_questions) * 0.6
            
            publish_evaluation_update(task_id, EvaluationTaskStatus.PROGRESS, {
                "message": f"Processing question {i+1}/{total_questions}",
                "progress": current_progress
            })
            
            # Extract question from dataset
            if isinstance(question_data, dict):
                # New format: look for 'query' field first, then 'question'
                query = question_data.get('query') or question_data.get('question', str(question_data))
            else:
                query = str(question_data)
            
            logger.info(f"🚀 QUESTION {i+1}: {query[:100]}...")
            
            try:
                # Use RAG system to generate real answer and context
                logger.info(f"🚀 RAG: Generating answer for question {i+1}")
                
                # Get answer and context from RAG engine
                # The query method returns (answer, docs)
                response, context_docs = rag_engine.query(
                    question=query,
                    k=4  # Number of context chunks to retrieve
                )
                
                # Extract context from retrieved documents
                context_chunks = []
                context = ""
                
                if context_docs:
                    # Extract content from context documents
                    for doc in context_docs:
                        if hasattr(doc, 'page_content'):
                            content = doc.page_content
                            context_chunks.append({'content': content})
                    
                    context = "\n".join([chunk['content'] for chunk in context_chunks])
                else:
                    logger.warning(f"🚀 RAG: No context found for question {i+1}")
                    context_chunks = [{'content': 'No context available'}]
                    context = "No context available"
                
                logger.info(f"🚀 RAG: Generated {len(response)} char response with {len(context_chunks)} context chunks")
                
                # Perform evaluation on the real Q&A pair
                logger.info(f"🚀 EVALUATION: Evaluating real Q&A pair {i+1}")
                
                result = evaluator.evaluate_rag_triad(
                    query=query,
                    response=response,
                    context=context
                )
                
                evaluation_results.append({
                    "question": query,
                    "response": response,
                    "context_chunks": len(context_chunks),
                    "evaluation": {
                        "groundedness": {
                            "score": result.groundedness.score,
                            "raw_score": result.groundedness.raw_score,
                            "reasoning": result.groundedness.reasoning
                        },
                        "answer_relevance": {
                            "score": result.answer_relevance.score,
                            "raw_score": result.answer_relevance.raw_score,
                            "reasoning": result.answer_relevance.reasoning
                        },
                        "context_relevance": {
                            "score": result.context_relevance.score,
                            "raw_score": result.context_relevance.raw_score,
                            "reasoning": result.context_relevance.reasoning
                        },
                        "overall_score": result.overall_score,
                        "evaluation_time_seconds": result.evaluation_time_seconds
                    },
                    "question_index": i + 1
                })
                
                logger.info(f"🚀 EVALUATION: Completed question {i+1} - Overall score: {result.overall_score:.3f}")
                
            except Exception as q_error:
                logger.error(f"🚀 ERROR: Failed to process question {i+1}: {q_error}")
                # Add failed result
                evaluation_results.append({
                    "question": query,
                    "response": "",
                    "context_chunks": 0,
                    "evaluation": None,
                    "question_index": i + 1,
                    "error": str(q_error)
                })
        
        # Calculate aggregate results
        valid_results = [r for r in evaluation_results if r.get('evaluation')]
        
        if valid_results:
            avg_groundedness = sum(r['evaluation']['groundedness']['score'] for r in valid_results) / len(valid_results)
            avg_answer_relevance = sum(r['evaluation']['answer_relevance']['score'] for r in valid_results) / len(valid_results)
            avg_context_relevance = sum(r['evaluation']['context_relevance']['score'] for r in valid_results) / len(valid_results)
            avg_overall = sum(r['evaluation']['overall_score'] for r in valid_results) / len(valid_results)
        else:
            avg_groundedness = avg_answer_relevance = avg_context_relevance = avg_overall = 0.0
        
        # Create final result
        final_result = {
            "task_id": task_id,
            "evaluation_time": time.time() - start_time,
            "timestamp": datetime.utcnow().isoformat(),
            "dataset_id": dataset_id,
            "model_id": model_id,
            "user_id": user_id,
            "metadata": {
                "dataset_id": dataset_id,
                "model_id": model_id,
                "retrieval_config": retrieval_config,
                "test_type": "dataset_evaluation"
            },
            "results": {
                "groundedness": {"score": avg_groundedness},
                "answer_relevance": {"score": avg_answer_relevance},
                "context_relevance": {"score": avg_context_relevance},
                "overall_score": avg_overall,
                "total_questions": total_questions,
                "successful_evaluations": len(valid_results),
                "detailed_results": evaluation_results
            }
        }
        
        # Update database with results
        try:
            # Save each individual evaluation result to database
            for result_data in valid_results:
                eval_repo.save_evaluation_result(
                    task_id=task_id,
                    dataset_id=dataset_id,
                    query=result_data['question'],
                    expected_answer="",  # Not available in this context
                    actual_answer=result_data['response'],
                    context=str(result_data.get('context_chunks', 0)),
                    groundedness_score=result_data['evaluation']['groundedness']['score'],
                    relevance_score=result_data['evaluation']['answer_relevance']['score'],
                    quality_score=result_data['evaluation']['overall_score'],
                    latency_ms=int(result_data['evaluation']['evaluation_time_seconds'] * 1000),
                    model_used=model_id
                )
            
            # Complete the task with success status
            eval_repo.complete_task(task_id, status='completed')
            logger.info(f"🚀 DATABASE: Saved {len(valid_results)} evaluation results and completed task {task_id}")
        except Exception as db_error:
            logger.error(f"🚀 DATABASE ERROR: Failed to save results: {db_error}")
        
        # Final progress update
        publish_evaluation_update(task_id, EvaluationTaskStatus.SUCCESS, {
            "message": "Dataset evaluation completed",
            "progress": 1.0,
            "results": {
                "groundedness": {"score": avg_groundedness},
                "answer_relevance": {"score": avg_answer_relevance},
                "context_relevance": {"score": avg_context_relevance},
                "overall_score": avg_overall,
                "total_questions": total_questions,
                "successful_evaluations": len(valid_results)
            },
            "evaluation_time": time.time() - start_time
        })
        
        logger.info(f"🚀 COMPLETED: Dataset evaluation for task {task_id} in {time.time() - start_time:.2f}s")
        return final_result
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"🚀 ERROR: Dataset evaluation failed for task {task_id}: {error_msg}")
        
        # Update database with error
        try:
            eval_repo.complete_task(task_id, status='failed', error_message=error_msg)
            logger.info(f"🚀 DATABASE: Marked task {task_id} as failed")
        except Exception as db_error:
            logger.error(f"🚀 DATABASE ERROR: Failed to update task status: {db_error}")
        
        # Publish error update
        publish_evaluation_update(task_id, EvaluationTaskStatus.FAILURE, {
            "message": f"Evaluation failed: {error_msg}",
            "error": error_msg
        })
        
        raise
    
    finally:
        # Always release the evaluation slot
        concurrency_manager.release_evaluation_slot(task_id)

def generate_sample_questions(dataset_id: int) -> List[Dict]:
    """Generate sample questions for a dataset"""
    # For demo purposes, return some sample questions
    sample_questions = [
        {"question": "What is the main topic discussed in the documents?"},
        {"question": "What are the key benefits mentioned?"},
        {"question": "How does this relate to current technology?"},
        {"question": "What recommendations are provided?"},
        {"question": "What are the potential challenges identified?"}
    ]
    
    # Randomly select 3-5 questions
    num_questions = random.randint(3, 5)
    return random.sample(sample_questions, min(num_questions, len(sample_questions)))


@celery_app.task(bind=True, name="evaluation_tasks.create_dataset_background")
def create_dataset_background(
    self,
    name: str,
    description: str,
    document_ids: List[str],
    num_questions_per_doc: int = 3,
    model_name: str = "llama3",
    difficulty_levels: List[str] = None,
    user_id: str = "admin",
    dataset_id: int = None
) -> Dict[str, Any]:
    """
    Create evaluation dataset in background with progress tracking and concurrency control
    
    Args:
        name: Dataset name
        description: Dataset description
        document_ids: List of document IDs to use
        num_questions_per_doc: Number of questions per document
        model_name: Model to use for generation
        difficulty_levels: List of difficulty levels
        user_id: User creating the dataset
        
    Returns:
        Dict containing dataset creation results
    """
    task_id = self.request.id
    start_time = time.time()
    
    if difficulty_levels is None:
        difficulty_levels = ['easy', 'medium', 'hard']
    
    logger.info(f"🚀 DATASET CREATE: Starting async dataset creation for task {task_id}")
    logger.info(f"🚀 DATASET CREATE: Name '{name}', Documents: {len(document_ids)}, User: {user_id}")
    
    # Try to acquire concurrency slot
    if not concurrency_manager.acquire_dataset_creation_slot(task_id, timeout=300):
        error_msg = "Dataset creation queue is full. Please try again later."
        logger.error(f"📋 CONCURRENCY: {error_msg} Task: {task_id}")
        
        # Publish failure notification
        publish_evaluation_update(task_id, EvaluationTaskStatus.FAILURE, {
            "message": error_msg,
            "error": "concurrency_limit_reached"
        })
        raise Exception(error_msg)
    
    try:
        # Initialize progress update
        publish_evaluation_update(task_id, EvaluationTaskStatus.STARTED, {
            "message": "Starting dataset creation",
            "dataset_name": name,
            "total_documents": len(document_ids),
            "progress": 0.0
        })
        
        # Get database connection
        from app.db import DatabaseConnection
        from app.db.repositories import EvaluationRepository
        
        db = DatabaseConnection()
        eval_repo = EvaluationRepository(db)
        
        # Fetch documents (I/O bound - benefits from async)
        publish_evaluation_update(task_id, EvaluationTaskStatus.PROGRESS, {
            "message": "Fetching documents",
            "progress": 0.1
        })
        
        from document_storage import get_document_storage
        doc_storage = get_document_storage()
        
        documents = []
        for doc_id in document_ids:
            try:
                doc_info = doc_storage.get_document_info(doc_id)
                if doc_info:
                    documents.append(doc_info)
            except Exception as e:
                logger.warning(f"Could not fetch document {doc_id}: {e}")
                continue
        
        if not documents:
            raise ValueError("No valid documents found")
        
        # Create dataset record in database
        publish_evaluation_update(task_id, EvaluationTaskStatus.PROGRESS, {
            "message": "Using existing dataset record",
            "progress": 0.2
        })
        
        # Use the provided dataset_id (created by API endpoint before task was queued)
        if dataset_id:
            db_dataset_id = dataset_id
            logger.info(f"Using pre-created dataset record with ID: {db_dataset_id}")
        else:
            # Fallback: create dataset if not provided (backward compatibility)
            try:
                db_dataset_id = eval_repo.create_dataset(
                    name=name,
                    description=description,
                    created_by=user_id
                )
                logger.info(f"Created new dataset record in database with ID: {db_dataset_id}")
            except Exception as e:
                logger.error(f"Failed to create dataset in database: {e}")
                raise ValueError(f"Dataset creation failed: {str(e)}")
        
        # Initialize synchronous dataset generation (following Sentry pattern)
        publish_evaluation_update(task_id, EvaluationTaskStatus.PROGRESS, {
            "message": "Preparing for dataset generation",
            "progress": 0.3
        })
        
        # Note: No need for DatasetGenerator async initialization
        # We'll handle Ollama calls directly with requests (synchronous)
        
        publish_evaluation_update(task_id, EvaluationTaskStatus.PROGRESS, {
            "message": "Extracting document content",
            "progress": 0.4
        })
        
        # Prepare documents for the generator
        prepared_documents = []
        for document in documents:
            try:
                # Get document content asynchronously
                doc_content = doc_storage.get_document_content(document['id'])
                if doc_content:
                    prepared_doc = {
                        'id': document['id'],
                        'title': document.get('title', 'Unknown'),
                        'filename': document.get('filename', 'Unknown'),
                        'content': doc_content,
                        'metadata': document
                    }
                    prepared_documents.append(prepared_doc)
            except Exception as e:
                logger.warning(f"Could not get content for document {document['id']}: {e}")
                continue
        
        # Generate dataset using synchronous approach (Sentry pattern)
        all_items = []
        generation_stats = {
            "total_documents": len(prepared_documents),
            "questions_per_document": num_questions_per_doc,
            "model_used": model_name,
            "generation_start": datetime.utcnow().isoformat(),
            "success_count": 0,
            "error_count": 0
        }
        
        for doc_idx, document in enumerate(prepared_documents):
            # PRIORITY CHECK: Pause if high-priority evaluation tasks are waiting
            while concurrency_manager.should_pause_low_priority():
                logger.info(f"📋 PRIORITY: Dataset creation paused for high-priority task (document {doc_idx + 1}/{len(prepared_documents)})")
                publish_evaluation_update(task_id, EvaluationTaskStatus.PROGRESS, {
                    "message": f"Paused for high-priority task (document {doc_idx + 1}/{len(prepared_documents)})",
                    "progress": int(((doc_idx) / len(prepared_documents)) * 0.5 + 0.4),
                    "paused": True
                })
                time.sleep(10)  # Wait 10 seconds before checking again
                
            try:
                logger.info(f"Processing document {doc_idx + 1}/{len(prepared_documents)}: {document.get('filename', 'Unknown')}")
                
                content = document.get('content', '')
                if not content:
                    logger.warning(f"No content for document {document.get('filename')}")
                    continue
                
                # Chunk content
                chunks = chunk_content_sync(content, max_chunk_size=2000)
                
                # Generate questions for this document
                for i in range(num_questions_per_doc):
                    # PRIORITY CHECK: Also check between questions
                    if concurrency_manager.should_pause_low_priority():
                        logger.info(f"📋 PRIORITY: Dataset creation paused during question generation")
                        while concurrency_manager.should_pause_low_priority():
                            time.sleep(5)
                    
                    try:
                        # Select difficulty level and single chunk for focused evaluation
                        difficulty = random.choice(difficulty_levels)
                        selected_chunk = random.choice(chunks)
                        
                        # Generate Q&A pair with single chunk for expected_chunks
                        qa_pair = generate_single_qa_pair_sync(
                            content_chunk=selected_chunk,
                            document_filename=document.get('filename', 'Unknown'),
                            difficulty=difficulty,
                            model_name=model_name,
                            question_index=i + 1,
                            source_chunks=[selected_chunk]  # Pass single chunk as list for expected_chunks
                        )
                        
                        if qa_pair:
                            all_items.append(qa_pair)
                            generation_stats["success_count"] += 1
                        else:
                            generation_stats["error_count"] += 1
                            
                        # Small delay to avoid overwhelming Ollama
                        time.sleep(0.5)
                        
                    except Exception as e:
                        logger.error(f"Error generating question {i+1} for document: {e}")
                        generation_stats["error_count"] += 1
                        continue
                        
                # Update progress (ensure paused flag is removed)
                progress = int(((doc_idx + 1) / len(prepared_documents)) * 0.5 + 0.4)  # 40-90% range
                publish_evaluation_update(task_id, EvaluationTaskStatus.PROGRESS, {
                    "message": f"Generated questions for document {doc_idx + 1}/{len(prepared_documents)}",
                    "progress": progress / 100.0
                })
                        
            except Exception as e:
                logger.error(f"Error processing document {document.get('filename', 'Unknown')}: {e}")
                generation_stats["error_count"] += 1
                continue
        
        generation_stats["generation_end"] = datetime.utcnow().isoformat()
        generation_stats["total_questions_generated"] = len(all_items)
        
        # Create simple dataset structure
        generated_dataset = {
            'name': name,
            'description': description,
            'items': all_items,
            'generation_metadata': generation_stats
        }
        
        # Save dataset to file
        publish_evaluation_update(task_id, EvaluationTaskStatus.PROGRESS, {
            "message": "Saving dataset file",
            "progress": 0.9
        })
        
        import os
        import json
        
        # Create dataset filename
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        timestamp = int(datetime.utcnow().timestamp())
        filename = f"{safe_name}_{timestamp}.json"
        
        # Ensure dataset directory exists
        dataset_dir = "/app/data/datasets"
        os.makedirs(dataset_dir, exist_ok=True)
        file_path = os.path.join(dataset_dir, filename)
        
        # Save dataset synchronously  
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(generated_dataset, f, indent=2, ensure_ascii=False)
            logger.info(f"Dataset saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving dataset to file: {e}")
            raise
        
        # Get question count from items
        question_count = len(generated_dataset.get('items', []))
        
        # Update database record with file path and final stats
        try:
            eval_repo.update_dataset(
                dataset_id=db_dataset_id,
                file_path=file_path,
                question_count=question_count,
                document_count=len(documents),
                status='completed'
            )
            logger.info(f"📋 DATABASE: Updated dataset {db_dataset_id} with {question_count} questions from {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to update dataset record: {e}")
        
        # Final result
        result = {
            "task_id": task_id,
            "dataset_id": db_dataset_id,
            "name": name,
            "description": description,
            "file_path": file_path,
            "question_count": question_count,
            "document_count": len(documents),
            "generation_stats": getattr(generated_dataset, "generation_metadata", None),
            "creation_time": time.time() - start_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store result in Redis
        redis_client.setex(
            f"dataset_creation_result:{task_id}",
            3600,  # 1 hour expiration
            json.dumps(result)
        )
        
        # Publish success notification
        publish_evaluation_update(task_id, EvaluationTaskStatus.SUCCESS, {
            "message": "Dataset creation completed successfully",
            "progress": 1.0,
            "dataset_id": db_dataset_id,
            "dataset_name": name,
            "question_count": question_count,
            "document_count": len(documents),
            "creation_time": time.time() - start_time,
            "action": "dataset_created"  # Signal to UI to refresh datasets list
        })
        
        # Give time for WebSocket message to be sent before task completes
        time.sleep(0.5)
        
        logger.info(f"Completed async dataset creation for task {task_id} in {time.time() - start_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Async dataset creation task {task_id} failed: {e}")
        
        # Update dataset status to error if we have the dataset_id
        if 'db_dataset_id' in locals():
            try:
                eval_repo.update_dataset(
                    dataset_id=db_dataset_id,
                    status='failed'
                )
                logger.info(f"📋 DATABASE: Marked dataset {db_dataset_id} as failed")
            except Exception as update_error:
                logger.error(f"Failed to update dataset status to Error: {update_error}")
        
        error_result = {
            "task_id": task_id,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
            "creation_time": time.time() - start_time
        }
        
        # Publish failure notification
        publish_evaluation_update(task_id, EvaluationTaskStatus.FAILURE, {
            "message": f"Dataset creation failed: {str(e)}",
            "error": str(e)
        })
        
        raise Exception(f"Dataset creation failed: {e}")
    
    finally:
        # Always release the concurrency slot
        concurrency_manager.release_dataset_creation_slot(task_id)
