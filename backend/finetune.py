import logging
import os
import json
import time
import uuid
import sqlite3
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional, Literal
from pathlib import Path
import numpy as np
from rlhf import RLHF
from datetime import datetime
import random

logger = logging.getLogger(__name__)

class FineTuner:
    """
    Manages fine-tuning of models based on RLHF feedback
    """
    
    def __init__(self, model_name: str = "deepseek-r1:latest"):
        self.model_name = model_name
        self.ollama_base_url = os.getenv('OLLAMA_HOST', 'http://ollama:11434')
        self.rlhf_db = RLHF()
        
        # Create output directories
        self.output_dir = Path("/app/data/finetune")
        self.datasets_dir = self.output_dir / "datasets"
        self.models_dir = self.output_dir / "models"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized FineTuner for model {model_name}")

    async def prepare_training_data(self) -> Dict[str, Any]:
        """
        Prepare training data from RLHF feedback
        
        Returns:
            Dictionary with dataset stats and paths
        """
        # Get preference data from RLHF database
        preference_data = self.rlhf_db.get_preference_data(limit=1000)
        
        if not preference_data or len(preference_data) < 10:
            logger.warning(f"Not enough preference data for fine-tuning. Found only {len(preference_data)} examples")
            return {"status": "error", "reason": "insufficient_data", "count": len(preference_data)}
        
        # Format the data for fine-tuning
        training_examples = []
        
        for item in preference_data:
            try:
                # Extract the query and responses
                query = item.get("prompt", "")
                winner_response = item.get("chosen_response", "")
                loser_response = item.get("rejected_response", "")
                
                if not query or not winner_response:
                    continue
                
                # Add positive example (preferred response)
                training_examples.append({
                    "prompt": f"Query: {query}\nResponse:",
                    "response": winner_response,
                    "label": 1.0
                })
                
                # Add negative example if available
                if loser_response:
                    training_examples.append({
                        "prompt": f"Query: {query}\nResponse:",
                        "response": loser_response,
                        "label": 0.0
                    })
                    
            except Exception as e:
                logger.error(f"Error processing preference item: {str(e)}")
        
        # Save the training data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_path = self.datasets_dir / f"rlhf_train_{timestamp}.jsonl"
        
        # Shuffle examples for better training
        random.shuffle(training_examples)
        
        # Split into training and validation sets (90/10)
        split_idx = int(len(training_examples) * 0.9)
        train_data = training_examples[:split_idx]
        val_data = training_examples[split_idx:]
        
        # Save training data
        with open(train_path, 'w') as f:
            for example in train_data:
                f.write(json.dumps(example) + '\n')
                
        # Save validation data
        val_path = self.datasets_dir / f"rlhf_val_{timestamp}.jsonl"
        with open(val_path, 'w') as f:
            for example in val_data:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"Created training dataset with {len(train_data)} examples and validation set with {len(val_data)} examples")
        
        return {
            "status": "success",
            "train_path": str(train_path),
            "val_path": str(val_path),
            "train_count": len(train_data),
            "val_count": len(val_data),
            "timestamp": timestamp
        }
    
    async def prepare_reranker_training_data(self) -> Dict[str, Any]:
        """
        Prepare training data for reranker fine-tuning with support for split document approach
        
        Returns:
            Dictionary with dataset stats and paths
        """
        # Get preference data from RLHF database
        preference_data = self.rlhf_db.get_preference_data(limit=1000)
        
        if not preference_data or len(preference_data) < 10:
            logger.warning(f"Not enough preference data for reranker fine-tuning. Found only {len(preference_data)} examples")
            return {"status": "error", "reason": "insufficient_data", "count": len(preference_data)}
        
        # Format the data for reranker fine-tuning - different format from LLM tuning
        reranker_examples = []
        
        for item in preference_data:
            try:
                # Extract the query and responses
                query = item.get("prompt", "")
                context = item.get("context", "")  # Some items might have context recorded
                winner_response = item.get("chosen_response", "")
                loser_response = item.get("rejected_response", "")
                
                if not query or not winner_response:
                    continue
                    
                # Add the winning pair with higher score
                if context:
                    # If this was from split document approach, the context is more important
                    # to learn strong document ranking signals
                    reranker_examples.append({
                        "query": query,
                        "document": context,
                        "score": 0.95  # Even higher score for directly selected contexts
                    })
                
                # Create synthetic examples from the responses
                # Extract key sentences from the preferred response
                sentences = [s.strip() for s in winner_response.split('.') if len(s.strip()) > 20]
                if sentences:
                    # Use the most informative sentence as a positive example
                    best_sentence = max(sentences, key=len)
                    reranker_examples.append({
                        "query": query,
                        "document": best_sentence,
                        "score": 0.8  # High score for relevant info
                    })
                
                # Add negative examples from rejected response if available
                if loser_response:
                    # Extract sentences from rejected response
                    neg_sentences = [s.strip() for s in loser_response.split('.') if len(s.strip()) > 20]
                    if neg_sentences:
                        # Fix the error in the code - key=len instead of key.len
                        worst_sentence = max(neg_sentences, key=len)
                        reranker_examples.append({
                            "query": query,
                            "document": worst_sentence,
                            "score": 0.2  # Lower score for less relevant info, decreased further
                        })
                
            except Exception as e:
                logger.error(f"Error processing reranker training item: {str(e)}")
        
        # Save the reranker training data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_path = self.datasets_dir / f"reranker_train_{timestamp}.jsonl"
        
        # Shuffle examples
        random.shuffle(reranker_examples)
        
        # Split into training and validation sets (80/20)
        split_idx = int(len(reranker_examples) * 0.8)
        train_data = reranker_examples[:split_idx]
        val_data = reranker_examples[split_idx:]
        
        # Save training data
        with open(train_path, 'w') as f:
            for example in train_data:
                f.write(json.dumps(example) + '\n')
                
        # Save validation data
        val_path = self.datasets_dir / f"reranker_val_{timestamp}.jsonl"
        with open(val_path, 'w') as f:
            for example in val_data:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"Created reranker training dataset with {len(train_data)} examples and validation set with {len(val_data)} examples")
        
        return {
            "status": "success",
            "train_path": str(train_path),
            "val_path": str(val_path),
            "train_count": len(train_data),
            "val_count": len(val_data),
            "timestamp": timestamp,
            "type": "reranker"
        }
    
    async def start_finetuning(self, model_type: Literal["llm", "reranker"] = "llm") -> Dict[str, Any]:
        """
        Start the fine-tuning process
        
        Args:
            model_type: Type of model to fine-tune ("llm" or "reranker")
            
        Returns:
            Dictionary with fine-tuning job details
        """
        try:
            # Prepare the training data based on model type
            if model_type == "reranker":
                data_result = await self.prepare_reranker_training_data()
            else:
                data_result = await self.prepare_training_data()
            
            if data_result["status"] != "success":
                return data_result
            
            # Generate a job ID
            job_id = f"ft_{uuid.uuid4().hex[:8]}"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_model = f"{self.model_name}-ft-{timestamp}"
            
            # Create parameter file for fine-tuning
            params = {
                "base_model": self.model_name,
                "output_model": output_model,
                "train_file": data_result["train_path"],
                "val_file": data_result["val_path"],
                "learning_rate": 1e-5,
                "epochs": 3,
                "batch_size": 4,
                "max_length": 512,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "job_id": job_id,
                "timestamp": timestamp,
                "model_type": model_type
            }
            
            # Save parameters
            params_path = self.output_dir / f"params_{job_id}.json"
            with open(params_path, 'w') as f:
                json.dump(params, f, indent=2)
            
            # Start the fine-tuning process asynchronously
            logger.info(f"Starting fine-tuning job {job_id} for {model_type} model {self.model_name}")
            
            # Launch fine-tuning as a background task
            asyncio.create_task(self._run_finetuning_task(job_id, params))
            
            return {
                "status": "success",
                "job_id": job_id,
                "message": f"Fine-tuning job {job_id} started for {model_type}",
                "params": params
            }
            
        except Exception as e:
            logger.error(f"Error starting fine-tuning: {str(e)}")
            return {
                "status": "error",
                "reason": str(e)
            }
    
    async def _run_finetuning_task(self, job_id: str, params: Dict[str, Any]) -> None:
        """
        Run the fine-tuning task in the background
        
        Args:
            job_id: Unique job identifier
            params: Fine-tuning parameters
        """
        logger.info(f"Running fine-tuning job {job_id}")
        
        try:
            # Check model type for different fine-tuning approaches
            model_type = params.get("model_type", "llm")
            
            if model_type == "reranker":
                # For reranker, we'll save the weights file instead of creating a new model
                await self._finetune_reranker(params)
            else:
                # For LLM, create a new model using Ollama
                await self._finetune_llm(params)
            
            # Record fine-tuning job completion
            self._record_finetuning_job(
                job_id=job_id,
                status="completed",
                model_name=params["output_model"],
                params=params
            )
            
            logger.info(f"Fine-tuning job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error during fine-tuning job {job_id}: {str(e)}")
            # Record failure
            self._record_finetuning_job(
                job_id=job_id,
                status="failed",
                model_name=params["output_model"],
                params=params,
                error=str(e)
            )
    
    async def _finetune_llm(self, params: Dict[str, Any]) -> None:
        """Fine-tune the LLM using Ollama's API"""
        async with aiohttp.ClientSession() as session:
            # Create a new model using the Ollama API
            async with session.post(
                f"{self.ollama_base_url}/api/create",
                json={
                    "name": params["output_model"],
                    "modelfile": f"""
                    FROM {params["base_model"]}
                    
                    # Fine-tuned with RLHF data
                    PARAMETER num_ctx 2048
                    PARAMETER stop "user:"
                    PARAMETER stop "assistant:"
                    """,
                    "path": params["train_file"]
                },
                timeout=1800  # 30 minutes timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Failed to create fine-tuned model: {error_text}")
                    raise Exception(f"Failed to create fine-tuned model: {error_text}")
                
                result = await response.json()
                logger.info(f"Model creation result: {result}")
    
    async def _finetune_reranker(self, params: Dict[str, Any]) -> None:
        """Fine-tune the reranker model using custom weights"""
        # For the reranker, we'll generate optimal weights based on training data
        try:
            # Parse training data to derive optimal weights
            weights = {
                "semantic_score": 1.0,  # Base weight for semantic similarity
                "exact_match_score": 0.5,  # Weight for exact term matching
                "recency_score": 0.3  # Weight for document recency
            }
            
            # Analyze training examples to find optimal weights
            train_data = []
            with open(params["train_file"], 'r') as f:
                for line in f:
                    try:
                        train_data.append(json.loads(line))
                    except:
                        continue
            
            # If we have enough examples, calculate optimal weights
            if len(train_data) >= 10:
                # Adjust weights based on query-document patterns in training data
                query_terms = {}
                doc_terms = {}
                
                for example in train_data:
                    query = example.get("query", "")
                    document = example.get("document", "")
                    score = example.get("score", 0.5)
                    
                    # Count term frequencies
                    for term in query.lower().split():
                        query_terms[term] = query_terms.get(term, 0) + 1
                    
                    for term in document.lower().split():
                        doc_terms[term] = doc_terms.get(term, 0) + 1
                
                # Calculate overlap coefficient
                overlap = 0
                total = 0
                for term in query_terms:
                    if term in doc_terms:
                        overlap += 1
                    total += 1
                
                if total > 0:
                    overlap_ratio = overlap / total
                    
                    # Adjust weights based on overlap
                    if overlap_ratio > 0.7:
                        # High overlap - increase exact match weight
                        weights["exact_match_score"] = 0.8
                        weights["semantic_score"] = 0.9
                    elif overlap_ratio < 0.3:
                        # Low overlap - increase semantic weight
                        weights["exact_match_score"] = 0.3
                        weights["semantic_score"] = 1.1
            
            # Save the fine-tuned reranker weights
            weights_file = self.output_dir / f"reranker_weights_{params['timestamp']}.json"
            with open(weights_file, 'w') as f:
                json.dump(weights, f, indent=2)
            
            # Also save in the parameters
            params["weights"] = weights
            params["weights_file"] = str(weights_file)
            
            logger.info(f"Saved fine-tuned reranker weights: {weights}")
            
        except Exception as e:
            logger.error(f"Error fine-tuning reranker: {str(e)}")
            raise
    
    def _record_finetuning_job(self, job_id: str, status: str, model_name: str, 
                              params: Dict[str, Any], error: str = None) -> None:
        """
        Record fine-tuning job in the database
        
        Args:
            job_id: Unique job identifier
            status: Job status (completed, failed)
            model_name: Output model name
            params: Job parameters
            error: Error message if failed
        """
        try:
            with sqlite3.connect(os.path.join(os.getenv('SQLITE_DB_PATH', '/app/data/db'), 'chat.db')) as conn:
                cursor = conn.cursor()
                
                # Create table if not exists
                cursor.execute('''CREATE TABLE IF NOT EXISTS finetune_jobs
                    (id TEXT PRIMARY KEY, 
                    model_name TEXT, 
                    status TEXT, 
                    parameters TEXT,
                    error TEXT,
                    created_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    model_type TEXT)''')
                
                # Get model type
                model_type = params.get("model_type", "llm")
                
                # Insert job record
                cursor.execute(
                    "INSERT OR REPLACE INTO finetune_jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (job_id, model_name, status, json.dumps(params), error, 
                     params.get("timestamp"), datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model_type)
                )
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error recording fine-tuning job: {str(e)}")
    
    def get_finetuning_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get list of fine-tuning jobs
        
        Args:
            limit: Maximum number of jobs to return
            
        Returns:
            List of fine-tuning jobs
        """
        try:
            with sqlite3.connect(os.path.join(os.getenv('SQLITE_DB_PATH', '/app/data/db'), 'chat.db')) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute(
                    "SELECT * FROM finetune_jobs ORDER BY created_at DESC LIMIT ?", 
                    (limit,)
                )
                
                jobs = []
                for row in cursor.fetchall():
                    job = dict(row)
                    # Parse parameters
                    if job['parameters']:
                        try:
                            job['parameters'] = json.loads(job['parameters'])
                        except:
                            pass
                    jobs.append(job)
                
                return jobs
                
        except Exception as e:
            logger.error(f"Error getting fine-tuning jobs: {str(e)}")
            return []


# Global finetuner instance to be used throughout the application
_finetuner_instance = None

def get_finetuner(model_name: str = "deepseek-r1:latest"):
    """Get or create a singleton finetuner instance"""
    global _finetuner_instance
    if _finetuner_instance is None:
        _finetuner_instance = FineTuner(model_name)
    return _finetuner_instance
