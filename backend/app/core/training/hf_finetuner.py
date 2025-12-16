import os
import json
import logging
import asyncio
import torch
import time

# Disable DeepSpeed and other CUDA compilation requiring libraries before any other imports
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"
os.environ["WANDB_DISABLED"] = "true"
os.environ["DEEPSPEED_DISABLE"] = "true"
os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
os.environ["ACCELERATE_USE_FSDP"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    EarlyStoppingCallback, get_linear_schedule_with_warmup, TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from app.db.repositories.experiment_repository import experiment_repository, ExperimentStatus
from app.core.training.metrics import create_metrics_collector, cleanup_metrics_collector, get_metrics_collector

logger = logging.getLogger(__name__)

class HuggingFaceFineTuner:
    """HuggingFace model finetuning engine with LoRA support"""
    
    def __init__(self):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use the HuggingFace cache volume mount to avoid triggering file watcher restarts
        self.cache_dir = os.environ.get("HF_HOME", "/root/.cache/huggingface")
        self.models_dir = "/app/data/finetuned_models"
        self.datasets_dir = "/app/data/datasets"
        
        # Create directories
        for dir_path in [self.cache_dir, self.models_dir, self.datasets_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"HuggingFace FineTuner initialized on device: {self.device}")
    
    async def start_training(self, experiment_id: str, config: Dict[str, Any]) -> bool:
        """Start training process for an experiment"""
        try:
            # Update experiment status
            experiment_repository.update_experiment_status(experiment_id, ExperimentStatus.RUNNING)
            
            # Run training in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._train_model, experiment_id, config)
            
            if result:
                experiment_repository.update_experiment_status(experiment_id, ExperimentStatus.COMPLETED)
                logger.info(f"Training completed successfully for experiment {experiment_id}")
                return True
            else:
                experiment_repository.update_experiment_status(
                    experiment_id, ExperimentStatus.FAILED, "Training failed"
                )
                return False
                
        except Exception as e:
            logger.error(f"Training failed for experiment {experiment_id}: {e}")
            experiment_repository.update_experiment_status(
                experiment_id, ExperimentStatus.FAILED, str(e)
            )
            return False
    
    def _train_model(self, experiment_id: str, config: Dict[str, Any]) -> bool:
        """Core training logic (runs in thread executor)"""
        try:
            # Create metrics collector for this training run
            metrics_collector = create_metrics_collector(experiment_id)
            
            # Load and prepare dataset
            dataset_id = config.get('dataset_id')
            train_dataset, eval_dataset = self._prepare_dataset(config['dataset_path'], dataset_id)
            
            # Load model and tokenizer
            model, tokenizer = self._load_model_and_tokenizer(config['base_model'])
            
            # Setup LoRA if enabled
            if config.get('use_lora', True):
                model = self._setup_lora(model, config)
            
            # Prepare training arguments
            training_args = self._create_training_arguments(experiment_id, config)
            
            # Create trainer
            trainer = self._create_trainer(
                model, tokenizer, train_dataset, eval_dataset, training_args, experiment_id, config.get('max_length', 512), metrics_collector
            )
            
            # Start training
            logger.info(f"Starting training for experiment {experiment_id}")
            logger.info(f"Training will run for {config.get('epochs', 5)} epochs with batch size 1")
            logger.info(f"Total training steps expected: {len(train_dataset) * config.get('epochs', 5)}")
            
            trainer.train()
            
            logger.info(f"Training completed for experiment {experiment_id}")
            
            # Save model with dataset suffix
            dataset_name = config.get('dataset_id', 'unknown')
            base_model_name = config.get('base_model', 'model').replace('/', '-')
            model_save_path = os.path.join(self.models_dir, f"{base_model_name}-{dataset_name}-{experiment_id}")
            trainer.save_model(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            
            logger.info(f"Model saved to: {model_save_path}")
            
            # Update experiment with model path
            experiment_repository.update_experiment(experiment_id, {"model_path": model_save_path})
            
            # Update experiment with final metrics
            eval_results = trainer.evaluate()
            
            # Save complete training metrics history before cleanup
            if metrics_collector:
                complete_metrics = metrics_collector.get_metrics_summary()
                # Combine evaluation results with training history
                final_metrics = {
                    **eval_results,
                    'training_history': complete_metrics.get('metrics', {}),
                    'final_progress': complete_metrics.get('progress', {}),
                    'training_completed': True,
                    'completion_time': time.time()
                }
                experiment_repository.update_experiment_metrics(experiment_id, final_metrics)
            else:
                experiment_repository.update_experiment_metrics(experiment_id, eval_results)
            
            logger.info(f"Training completed successfully for experiment {experiment_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False
        finally:
            # Always cleanup metrics collector
            cleanup_metrics_collector(experiment_id)
    
    def _train_model_with_progress(
        self, 
        experiment_id: str, 
        config: Dict[str, Any],
        progress_callback: callable = None
    ) -> bool:
        """
        Core training logic with progress callback support for Celery tasks.
        
        This method is used by the Celery background task to run training
        with progress updates published to Redis/WebSocket.
        
        Args:
            experiment_id: The experiment ID
            config: Training configuration
            progress_callback: Callback function(progress, message, epoch, total_epochs)
            
        Returns:
            bool: True if training succeeded, False otherwise
        """
        try:
            # Create metrics collector for this training run
            metrics_collector = create_metrics_collector(experiment_id)
            
            total_epochs = config.get('epochs', 3)
            
            # Report progress: Loading dataset
            if progress_callback:
                progress_callback(0.1, "Loading and preparing dataset", 0, total_epochs)
            
            # Load and prepare dataset
            dataset_id = config.get('dataset_id')
            train_dataset, eval_dataset = self._prepare_dataset(config['dataset_path'], dataset_id)
            
            # Report progress: Loading model
            if progress_callback:
                progress_callback(0.15, "Loading model and tokenizer", 0, total_epochs)
            
            # Load model and tokenizer
            model, tokenizer = self._load_model_and_tokenizer(config['base_model'])
            
            # Report progress: Setting up LoRA
            if progress_callback:
                progress_callback(0.2, "Setting up LoRA adapter", 0, total_epochs)
            
            # Setup LoRA if enabled
            if config.get('use_lora', True):
                model = self._setup_lora(model, config)
            
            # Prepare training arguments
            training_args = self._create_training_arguments(experiment_id, config)
            
            # Create a custom callback for progress reporting with metrics
            class ProgressCallback(TrainerCallback):
                def __init__(self, callback, total_epochs, metrics_collector_ref):
                    self.callback = callback
                    self.total_epochs = total_epochs
                    self.metrics_collector = metrics_collector_ref
                    self.last_loss = None
                    
                def on_epoch_begin(self, args, state, control, **kwargs):
                    if self.callback:
                        epoch = state.epoch or 0
                        # Progress from 0.25 to 0.9 during training epochs
                        progress = 0.25 + (epoch / self.total_epochs) * 0.65
                        self.callback(
                            progress, 
                            f"Training epoch {int(epoch) + 1}/{self.total_epochs}", 
                            int(epoch), 
                            self.total_epochs,
                            {"loss": self.last_loss, "step": state.global_step}
                        )
                
                def on_epoch_end(self, args, state, control, **kwargs):
                    if self.callback:
                        epoch = (state.epoch or 0)
                        progress = 0.25 + (epoch / self.total_epochs) * 0.65
                        self.callback(
                            progress, 
                            f"Completed epoch {int(epoch)}/{self.total_epochs}", 
                            int(epoch), 
                            self.total_epochs,
                            {"loss": self.last_loss, "step": state.global_step}
                        )
                        
                def on_log(self, args, state, control, logs=None, **kwargs):
                    """Capture training logs (loss values)"""
                    if logs:
                        self.last_loss = logs.get('loss')
                        
                def on_step_end(self, args, state, control, **kwargs):
                    # Report progress every 10 steps with loss information
                    if state.global_step % 10 == 0 and self.callback:
                        epoch = state.epoch or 0
                        progress = 0.25 + (epoch / self.total_epochs) * 0.65
                        self.callback(
                            progress, 
                            f"Training step {state.global_step}", 
                            int(epoch), 
                            self.total_epochs,
                            {"loss": self.last_loss, "step": state.global_step}
                        )
            
            # Create trainer with progress callback
            trainer = self._create_trainer(
                model, tokenizer, train_dataset, eval_dataset, training_args, 
                experiment_id, config.get('max_length', 512), metrics_collector
            )
            
            # Add progress callback if provided
            if progress_callback:
                trainer.add_callback(ProgressCallback(progress_callback, total_epochs, metrics_collector))
            
            # Report progress: Starting training
            if progress_callback:
                progress_callback(0.25, "Starting training", 0, total_epochs)
            
            # Start training
            logger.info(f"Starting training for experiment {experiment_id}")
            trainer.train()
            
            # Report progress: Saving model
            if progress_callback:
                progress_callback(0.92, "Saving model", total_epochs, total_epochs)
            
            # Save model
            dataset_name = config.get('dataset_id', 'unknown')
            base_model_name = config.get('base_model', 'model').replace('/', '-')
            model_save_path = os.path.join(self.models_dir, f"{base_model_name}-{dataset_name}-{experiment_id}")
            trainer.save_model(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            
            logger.info(f"Model saved to: {model_save_path}")
            
            # Update experiment with model path
            experiment_repository.update_experiment(experiment_id, {"model_path": model_save_path})
            
            # Report progress: Evaluating
            if progress_callback:
                progress_callback(0.95, "Running final evaluation", total_epochs, total_epochs)
            
            # Update experiment with final metrics
            eval_results = trainer.evaluate()
            
            # Save complete training metrics history
            if metrics_collector:
                complete_metrics = metrics_collector.get_metrics_summary()
                final_metrics = {
                    **eval_results,
                    'training_history': complete_metrics.get('metrics', {}),
                    'final_progress': complete_metrics.get('progress', {}),
                    'training_completed': True,
                    'completion_time': time.time()
                }
                experiment_repository.update_experiment_metrics(experiment_id, final_metrics)
            else:
                experiment_repository.update_experiment_metrics(experiment_id, eval_results)
            
            # Report progress: Complete
            if progress_callback:
                progress_callback(1.0, "Training completed successfully", total_epochs, total_epochs)
            
            logger.info(f"Training completed successfully for experiment {experiment_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            if progress_callback:
                progress_callback(0.0, f"Training failed: {str(e)}", 0, config.get('epochs', 3))
            return False
        finally:
            # Always cleanup metrics collector
            cleanup_metrics_collector(experiment_id)
    
    def _prepare_dataset(self, dataset_path: str, dataset_id: str = None) -> tuple:
        """Prepare training and evaluation datasets"""
        temp_path = None  # Initialize temp_path
        try:
            # Check if this is a virtual dataset (from database) or real file
            is_virtual_dataset = dataset_path.startswith('converted_') or (
                dataset_path.startswith('/app/converted_') and not os.path.exists(dataset_path)
            )
            
            if is_virtual_dataset and dataset_id:
                # This is a generated dataset stored in database
                logger.info(f"Loading dataset samples from database for dataset_id: {dataset_id}")
                samples = experiment_repository.get_dataset_samples(dataset_id)
                
                if not samples:
                    raise ValueError(f"No samples found for dataset_id: {dataset_id}")
                
                logger.info(f"Retrieved {len(samples)} samples from database")
                
                # Create temporary JSONL file
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
                temp_path = temp_file.name
                
                for sample in samples:
                    # Convert DatasetSample objects to training format dict
                    sample_dict = sample.to_training_format() if hasattr(sample, 'to_training_format') else sample
                    temp_file.write(json.dumps(sample_dict) + '\n')
                temp_file.close()
                
                logger.info(f"Created temporary dataset file: {temp_path} with {len(samples)} samples")
                dataset_path = temp_path
            else:
                # This is a regular file-based dataset
                logger.info(f"Loading dataset from file: {dataset_path}")
                
                # Check if file exists
                if not os.path.exists(dataset_path):
                    raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            
            # Load dataset based on file format
            if dataset_path.endswith('.jsonl'):
                dataset = load_dataset('json', data_files=dataset_path, split='train')
            elif dataset_path.endswith('.csv'):
                dataset = load_dataset('csv', data_files=dataset_path, split='train')
            else:
                raise ValueError(f"Unsupported dataset format: {dataset_path}")
            
            # Clean up temporary file if created
            if temp_path and temp_path != dataset_path:
                import os
                try:
                    os.unlink(temp_path)
                    logger.info(f"Cleaned up temporary file: {temp_path}")
                except:
                    pass
            
            # Split into train/eval
            dataset = dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = dataset['train']
            eval_dataset = dataset['test']
            
            # Validate dataset structure
            logger.info(f"Dataset loaded: {len(train_dataset)} train, {len(eval_dataset)} eval samples")
            
            # Check first few samples for data quality
            if len(train_dataset) > 0:
                sample = train_dataset[0]
                logger.info(f"Dataset columns: {list(sample.keys())}")
                for key, value in sample.items():
                    logger.info(f"Sample {key}: {type(value)} - {str(value)[:100]}...")
            
            return train_dataset, eval_dataset
            
        except Exception as e:
            # Clean up temp file on error
            if temp_path:
                try:
                    os.unlink(temp_path)
                except:
                    pass
            logger.error(f"Dataset preparation failed: {e}")
            raise
    
    def _load_model_and_tokenizer(self, model_name: str):
        """Load model and tokenizer from HuggingFace"""
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                padding_side="right"
            )
            
            # Add pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with conservative settings to avoid CUDA compilation
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                dtype=torch.float32,  # Use dtype instead of torch_dtype
                device_map=None,  # Disable auto device mapping to avoid DeepSpeed triggers
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Move to device manually after loading
            model = model.to(self.device)
            
            # Resize token embeddings if needed
            model.resize_token_embeddings(len(tokenizer))
            
            logger.info(f"Loaded model {model_name}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def _setup_lora(self, model, config: Dict[str, Any]):
        """Setup LoRA (Low-Rank Adaptation) for efficient finetuning"""
        try:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config.get('lora_r', 8),
                lora_alpha=config.get('lora_alpha', 32),
                lora_dropout=config.get('lora_dropout', 0.1),
                target_modules=config.get('lora_target_modules', ["q_proj", "v_proj"])
            )
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
            logger.info("LoRA configuration applied")
            return model
            
        except Exception as e:
            logger.error(f"LoRA setup failed: {e}")
            raise
    
    def _create_training_arguments(self, experiment_id: str, config: Dict[str, Any]) -> TrainingArguments:
        """Create training arguments"""
        output_dir = os.path.join(self.models_dir, experiment_id, "checkpoints")
        
        # Disable DeepSpeed to avoid CUDA compilation issues in containers
        os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"
        os.environ["WANDB_DISABLED"] = "true"
        
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.get('epochs', 5),  # Increased epochs for small dataset
            per_device_train_batch_size=1,  # Smaller batch size for small dataset
            per_device_eval_batch_size=1,
            warmup_steps=config.get('warmup_steps', 50),  # Reduced warmup steps
            learning_rate=config.get('learning_rate', 5e-5),  # Reduced learning rate
            fp16=False,  # Disable FP16 to avoid CUDA compilation issues
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=1,  # Log every step for small dataset
            save_steps=config.get('save_steps', 100),
            eval_steps=5,  # Evaluate more frequently
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            remove_unused_columns=False,
            dataloader_num_workers=0,  # Avoid multiprocessing issues in containers
            gradient_accumulation_steps=4,  # Increased to simulate larger batch
            weight_decay=config.get('weight_decay', 0.01),
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            report_to=[],  # Disable wandb/tensorboard for now
            deepspeed=None,  # Explicitly disable DeepSpeed
            disable_tqdm=False,
            prediction_loss_only=True,
            ddp_find_unused_parameters=False,
            use_cpu=not torch.cuda.is_available(),  # Force CPU if no CUDA dev tools
            no_cuda=False  # Keep CUDA enabled for model inference
        )
    
    def _create_trainer(self, model, tokenizer, train_dataset, eval_dataset, 
                       training_args, experiment_id: str, max_length: int = 512, metrics_collector=None) -> Trainer:
        """Create Trainer instance"""
        
        def tokenize_function(examples):
            """Tokenize examples for causal language modeling"""
            
            # Helper function to clean and validate text
            def clean_text(text):
                if text is None:
                    return ""
                if isinstance(text, list):
                    # If it's a list, join with space
                    return " ".join(str(item) for item in text if item is not None)
                return str(text)
            
            # Determine text format and clean data
            if 'text' in examples:
                texts = [clean_text(text) for text in examples['text']]
            elif 'prompt' in examples and 'completion' in examples:
                # For instruction-following datasets
                prompts = [clean_text(prompt) for prompt in examples['prompt']]
                completions = [clean_text(completion) for completion in examples['completion']]
                texts = [f"{prompt}\n{completion}" for prompt, completion in zip(prompts, completions)]
            elif 'instruction' in examples and 'output' in examples:
                # For Q-C-A instruction datasets (Question-Context-Answer format)
                instructions = [clean_text(instruction) for instruction in examples['instruction']]
                outputs = [clean_text(output) for output in examples['output']]
                
                if 'input' in examples:
                    inputs = [clean_text(inp) for inp in examples['input']]
                    # Check if any inputs are non-empty
                    if any(inp.strip() for inp in inputs):
                        texts = [f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}" 
                                for instruction, input_text, output in zip(instructions, inputs, outputs)]
                    else:
                        texts = [f"Instruction: {instruction}\nOutput: {output}" 
                                for instruction, output in zip(instructions, outputs)]
                else:
                    texts = [f"Instruction: {instruction}\nOutput: {output}" 
                            for instruction, output in zip(instructions, outputs)]
            elif 'input' in examples and 'output' in examples:
                # For Q-C-A datasets with input/output format (from DatasetSample.to_training_format())
                # The input field contains the context/question, output contains the answer
                inputs = [clean_text(inp) for inp in examples['input']]
                outputs = [clean_text(output) for output in examples['output']]
                texts = [f"Input: {input_text}\nOutput: {output}" 
                        for input_text, output in zip(inputs, outputs)]
            else:
                raise ValueError("Dataset must have 'text' field, 'prompt'+'completion' fields, 'instruction'+'output' fields, or 'input'+'output' fields")
            
            # Filter out empty texts
            texts = [text for text in texts if text.strip()]
            
            if not texts:
                raise ValueError("No valid texts found after cleaning")
            
            # Tokenize with proper padding
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_attention_mask=True
            )
            
            # Don't set labels here - DataCollatorForLanguageModeling will handle it
            return tokenized
        
        # Tokenize datasets with error handling
        try:
            logger.info("Starting dataset tokenization...")
            train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
            eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)
            logger.info("Dataset tokenization completed successfully")
        except Exception as e:
            logger.error(f"Dataset tokenization failed: {e}")
            # Try to get more details about the problematic data
            if len(train_dataset) > 0:
                sample = train_dataset[0]
                logger.error(f"First sample data: {sample}")
            raise
        
        # Validate tokenized data
        logger.info(f"Train dataset size after tokenization: {len(train_dataset)}")
        logger.info(f"Eval dataset size after tokenization: {len(eval_dataset)}")
        
        # Check a sample to ensure tokenization worked
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            logger.info(f"Sample tokenized data keys: {sample.keys()}")
            logger.info(f"Sample input_ids length: {len(sample['input_ids'])}")
        
        # Data collator for causal language modeling - handles labels automatically
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Not masked language modeling
            pad_to_multiple_of=None  # No padding to multiple
        )
        
        # Custom callback for logging to database and metrics
        class ExperimentCallback(TrainerCallback):
            def __init__(self, experiment_id: str, metrics_collector):
                self.experiment_id = experiment_id
                self.metrics_collector = metrics_collector
                self.step_count = 0
            
            def on_train_begin(self, args, state, control, **kwargs):
                # Initialize progress tracking
                total_steps = state.max_steps if state.max_steps > 0 else args.num_train_epochs * len(train_dataset) // args.per_device_train_batch_size
                self.metrics_collector.update_training_progress(
                    epoch=0,
                    total_epochs=int(args.num_train_epochs),
                    step=0,
                    total_steps=total_steps,
                    batches_per_epoch=len(train_dataset) // args.per_device_train_batch_size
                )
            
            def on_epoch_begin(self, args, state, control, **kwargs):
                # Update epoch progress
                self.metrics_collector.update_training_progress(
                    epoch=int(state.epoch),
                    total_epochs=int(args.num_train_epochs),
                    step=state.global_step,
                    total_steps=state.max_steps if state.max_steps > 0 else int(args.num_train_epochs) * len(train_dataset) // args.per_device_train_batch_size
                )
            
            def on_log(self, args, state, control, model, logs=None, **kwargs):
                if logs:
                    self.step_count += 1
                    
                    # Update metrics collector
                    self.metrics_collector.log_training_step(
                        logs=logs,
                        epoch=int(state.epoch) if state.epoch else 0,
                        step=state.global_step
                    )
                    
                    # Update progress
                    self.metrics_collector.update_training_progress(
                        epoch=int(state.epoch) if state.epoch else 0,
                        total_epochs=int(args.num_train_epochs),
                        step=state.global_step,
                        total_steps=state.max_steps if state.max_steps > 0 else int(args.num_train_epochs) * len(train_dataset) // args.per_device_train_batch_size
                    )
                    
                    # Log to database
                    experiment_repository.log_training_step(
                        self.experiment_id,
                        epoch=int(state.epoch) if state.epoch else 0,
                        step=state.global_step,
                        loss=logs.get('train_loss', logs.get('loss', 0)),
                        eval_loss=logs.get('eval_loss'),
                        learning_rate=logs.get('learning_rate', 0),
                        accuracy=logs.get('eval_accuracy')
                    )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[
                ExperimentCallback(experiment_id, metrics_collector),
                EarlyStoppingCallback(early_stopping_patience=3)
            ]
        )
        
        return trainer
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available HuggingFace models for finetuning"""
        # Comprehensive list of HuggingFace models suitable for finetuning
        return [
            # GPT Models
            {
                "name": "gpt2",
                "description": "OpenAI GPT-2 base model",
                "size": "small",
                "type": "text-generation"
            },
            {
                "name": "gpt2-medium",
                "description": "OpenAI GPT-2 medium model",
                "size": "medium",
                "type": "text-generation"
            },
            {
                "name": "gpt2-large",
                "description": "OpenAI GPT-2 large model", 
                "size": "large",
                "type": "text-generation"
            },
            {
                "name": "distilgpt2",
                "description": "Lightweight GPT-2 model",
                "size": "small",
                "type": "text-generation"
            },
            
            # GPT-Neo/GPT-J Models
            {
                "name": "EleutherAI/gpt-neo-125M",
                "description": "GPT-Neo 125M parameter model",
                "size": "small",
                "type": "text-generation"
            },
            {
                "name": "EleutherAI/gpt-neo-1.3B",
                "description": "GPT-Neo 1.3B parameter model",
                "size": "large",
                "type": "text-generation"
            },
            {
                "name": "EleutherAI/gpt-neo-2.7B",
                "description": "GPT-Neo 2.7B parameter model",
                "size": "large",
                "type": "text-generation"
            },
            {
                "name": "EleutherAI/gpt-j-6B",
                "description": "GPT-J 6B parameter model",
                "size": "large",
                "type": "text-generation"
            },
            
            # OPT Models
            {
                "name": "facebook/opt-125m",
                "description": "OPT 125M parameter model",
                "size": "small",
                "type": "text-generation"
            },
            {
                "name": "facebook/opt-350m",
                "description": "OPT 350M parameter model",
                "size": "small",
                "type": "text-generation"
            },
            {
                "name": "facebook/opt-1.3b",
                "description": "OPT 1.3B parameter model",
                "size": "medium",
                "type": "text-generation"
            },
            {
                "name": "facebook/opt-2.7b",
                "description": "OPT 2.7B parameter model",
                "size": "large",
                "type": "text-generation"
            },
            {
                "name": "facebook/opt-6.7b",
                "description": "OPT 6.7B parameter model",
                "size": "large",
                "type": "text-generation"
            },
            
            # T5 Models
            {
                "name": "t5-small",
                "description": "T5 small encoder-decoder model",
                "size": "small",
                "type": "text-to-text"
            },
            {
                "name": "t5-base",
                "description": "T5 base encoder-decoder model",
                "size": "medium",
                "type": "text-to-text"
            },
            {
                "name": "t5-large",
                "description": "T5 large encoder-decoder model",
                "size": "large",
                "type": "text-to-text"
            },
            {
                "name": "google/flan-t5-small",
                "description": "Instruction-tuned T5 small model",
                "size": "small",
                "type": "instruction-following"
            },
            {
                "name": "google/flan-t5-base",
                "description": "Instruction-tuned T5 base model",
                "size": "medium",
                "type": "instruction-following"
            },
            {
                "name": "google/flan-t5-large",
                "description": "Instruction-tuned T5 large model",
                "size": "large",
                "type": "instruction-following"
            },
            
            # BERT Models
            {
                "name": "bert-base-uncased",
                "description": "BERT base uncased model",
                "size": "medium",
                "type": "encoder-only"
            },
            {
                "name": "bert-large-uncased",
                "description": "BERT large uncased model",
                "size": "large",
                "type": "encoder-only"
            },
            {
                "name": "distilbert-base-uncased",
                "description": "DistilBERT base uncased model",
                "size": "small",
                "type": "encoder-only"
            },
            
            # RoBERTa Models
            {
                "name": "roberta-base",
                "description": "RoBERTa base model",
                "size": "medium",
                "type": "encoder-only"
            },
            {
                "name": "roberta-large",
                "description": "RoBERTa large model",
                "size": "large",
                "type": "encoder-only"
            },
            {
                "name": "distilroberta-base",
                "description": "DistilRoBERTa base model",
                "size": "small",
                "type": "encoder-only"
            },
            
            # Dialog/Conversational Models
            {
                "name": "microsoft/DialoGPT-small",
                "description": "DialoGPT small conversational model",
                "size": "small",
                "type": "conversational"
            },
            {
                "name": "microsoft/DialoGPT-medium",
                "description": "DialoGPT medium conversational model",
                "size": "medium",
                "type": "conversational"
            },
            {
                "name": "microsoft/DialoGPT-large",
                "description": "DialoGPT large conversational model",
                "size": "large",
                "type": "conversational"
            },
            
            # Code Models
            {
                "name": "microsoft/CodeGPT-small-py",
                "description": "CodeGPT small Python model",
                "size": "small",
                "type": "code-generation"
            },
            {
                "name": "Salesforce/codegen-350M-mono",
                "description": "CodeGen 350M monolingual model",
                "size": "small",
                "type": "code-generation"
            },
            {
                "name": "Salesforce/codegen-2B-mono",
                "description": "CodeGen 2B monolingual model",
                "size": "large",
                "type": "code-generation"
            },
            
            # Llama Models (Meta)
            {
                "name": "huggyllama/llama-7b",
                "description": "Llama 7B base model",
                "size": "large",
                "type": "text-generation"
            },
            {
                "name": "huggyllama/llama-13b",
                "description": "Llama 13B base model",
                "size": "large",
                "type": "text-generation"
            },
            
            # Mistral Models
            {
                "name": "mistralai/Mistral-7B-v0.1",
                "description": "Mistral 7B base model",
                "size": "large",
                "type": "text-generation"
            },
            {
                "name": "mistralai/Mistral-7B-Instruct-v0.1",
                "description": "Mistral 7B instruction model",
                "size": "large",
                "type": "instruction-following"
            },
            
            # Falcon Models
            {
                "name": "tiiuae/falcon-7b",
                "description": "Falcon 7B base model",
                "size": "large",
                "type": "text-generation"
            },
            {
                "name": "tiiuae/falcon-7b-instruct",
                "description": "Falcon 7B instruction model",
                "size": "large",
                "type": "instruction-following"
            },
            
            # Bloom Models
            {
                "name": "bigscience/bloom-560m",
                "description": "BLOOM 560M parameter model",
                "size": "small",
                "type": "text-generation"
            },
            {
                "name": "bigscience/bloom-1b1",
                "description": "BLOOM 1.1B parameter model",
                "size": "medium",
                "type": "text-generation"
            },
            {
                "name": "bigscience/bloom-3b",
                "description": "BLOOM 3B parameter model",
                "size": "large",
                "type": "text-generation"
            },
            
            # MPT Models
            {
                "name": "mosaicml/mpt-7b",
                "description": "MPT 7B base model",
                "size": "large",
                "type": "text-generation"
            },
            {
                "name": "mosaicml/mpt-7b-instruct",
                "description": "MPT 7B instruction model",
                "size": "large",
                "type": "instruction-following"
            },
            
            # Other Popular Models
            {
                "name": "allenai/longformer-base-4096",
                "description": "Longformer base model (4096 tokens)",
                "size": "medium",
                "type": "encoder-only"
            },
            {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "description": "Sentence transformer model",
                "size": "small",
                "type": "sentence-embedding"
            },
            {
                "name": "microsoft/deberta-base",
                "description": "DeBERTa base model",
                "size": "medium",
                "type": "encoder-only"
            },
            {
                "name": "microsoft/deberta-large",
                "description": "DeBERTa large model",
                "size": "large",
                "type": "encoder-only"
            },
            
            # Qwen Models (Alibaba)
            {
                "name": "Qwen/Qwen2.5-0.5B",
                "description": "Qwen 2.5 0.5B base model",
                "size": "small",
                "type": "text-generation"
            },
            {
                "name": "Qwen/Qwen2.5-0.5B-Instruct",
                "description": "Qwen 2.5 0.5B instruction model",
                "size": "small",
                "type": "instruction-following"
            },
            {
                "name": "Qwen/Qwen2.5-1.5B",
                "description": "Qwen 2.5 1.5B base model",
                "size": "medium",
                "type": "text-generation"
            },
            {
                "name": "Qwen/Qwen2.5-1.5B-Instruct",
                "description": "Qwen 2.5 1.5B instruction model",
                "size": "medium",
                "type": "instruction-following"
            },
            {
                "name": "Qwen/Qwen2.5-3B",
                "description": "Qwen 2.5 3B base model",
                "size": "medium",
                "type": "text-generation"
            },
            {
                "name": "Qwen/Qwen2.5-3B-Instruct",
                "description": "Qwen 2.5 3B instruction model",
                "size": "medium",
                "type": "instruction-following"
            },
            {
                "name": "Qwen/Qwen2.5-7B",
                "description": "Qwen 2.5 7B base model",
                "size": "large",
                "type": "text-generation"
            },
            {
                "name": "Qwen/Qwen2.5-7B-Instruct",
                "description": "Qwen 2.5 7B instruction model",
                "size": "large",
                "type": "instruction-following"
            },
            
            # TinyLlama Models
            {
                "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "description": "TinyLlama 1.1B chat model",
                "size": "small",
                "type": "conversational"
            },
            {
                "name": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                "description": "TinyLlama 1.1B base model",
                "size": "small",
                "type": "text-generation"
            },
            
            # Phi Models (Microsoft)
            {
                "name": "microsoft/phi-1_5",
                "description": "Phi-1.5 1.3B parameter model",
                "size": "small",
                "type": "text-generation"
            },
            {
                "name": "microsoft/phi-2",
                "description": "Phi-2 2.7B parameter model",
                "size": "medium",
                "type": "text-generation"
            },
            
            # Gemma Models (Google)
            {
                "name": "google/gemma-2b",
                "description": "Gemma 2B base model",
                "size": "medium",
                "type": "text-generation"
            },
            {
                "name": "google/gemma-2b-it",
                "description": "Gemma 2B instruction model",
                "size": "medium",
                "type": "instruction-following"
            },
            {
                "name": "google/gemma-7b",
                "description": "Gemma 7B base model",
                "size": "large",
                "type": "text-generation"
            },
            {
                "name": "google/gemma-7b-it",
                "description": "Gemma 7B instruction model",
                "size": "large",
                "type": "instruction-following"
            }
        ]
    
    def validate_dataset(self, file_path: str) -> Dict[str, Any]:
        """Validate uploaded dataset"""
        try:
            if not os.path.exists(file_path):
                return {"valid": False, "error": "File not found"}
            
            file_size = os.path.getsize(file_path)
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                return {"valid": False, "error": "File too large (max 100MB)"}
            
            # Check format and load sample
            if file_path.endswith('.jsonl'):
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) < 10:
                        return {"valid": False, "error": "Dataset too small (minimum 10 samples)"}
                    
                    # Validate JSON format
                    try:
                        sample = json.loads(lines[0])
                        if not ('text' in sample or ('prompt' in sample and 'completion' in sample)):
                            return {
                                "valid": False, 
                                "error": "JSONL must have 'text' field or 'prompt'+'completion' fields"
                            }
                    except json.JSONDecodeError:
                        return {"valid": False, "error": "Invalid JSONL format"}
                    
                    return {
                        "valid": True,
                        "num_samples": len(lines),
                        "format": "jsonl",
                        "file_size": file_size
                    }
            
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                if len(df) < 10:
                    return {"valid": False, "error": "Dataset too small (minimum 10 samples)"}
                
                if not ('text' in df.columns or ('prompt' in df.columns and 'completion' in df.columns)):
                    return {
                        "valid": False,
                        "error": "CSV must have 'text' column or 'prompt'+'completion' columns"
                    }
                
                return {
                    "valid": True,
                    "num_samples": len(df),
                    "format": "csv",
                    "file_size": file_size
                }
            
            else:
                return {"valid": False, "error": "Unsupported format (use .jsonl or .csv)"}
        
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}

# Global instance
hf_finetuner = HuggingFaceFineTuner()
