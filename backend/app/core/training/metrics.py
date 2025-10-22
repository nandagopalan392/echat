import time
import psutil
import json
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)

class TrainingMetricsCollector:
    """Collects and stores training metrics for real-time monitoring"""
    
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.start_time = time.time()
        self.metrics_history = []
        self.system_metrics = []
        self.is_running = False
        self.monitoring_thread = None
        
        # Training state
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_step = 0
        self.total_steps = 0
        self.current_batch = 0
        self.batches_per_epoch = 0
        
        # Metrics
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []
        self.accuracies = []
        
    def start_monitoring(self):
        """Start system resource monitoring in background thread"""
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitor_resources)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        logger.info(f"Started monitoring for experiment {self.experiment_id}")
    
    def stop_monitoring(self):
        """Stop system resource monitoring"""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
        logger.info(f"Stopped monitoring for experiment {self.experiment_id}")
    
    def _monitor_resources(self):
        """Monitor system resources every 5 seconds"""
        while self.is_running:
            try:
                timestamp = time.time()
                
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # GPU metrics
                gpu_metrics = []
                if GPU_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        for gpu in gpus:
                            gpu_metrics.append({
                                'id': gpu.id,
                                'name': gpu.name,
                                'load': gpu.load * 100,
                                'memory_used': gpu.memoryUsed,
                                'memory_total': gpu.memoryTotal,
                                'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                                'temperature': gpu.temperature
                            })
                    except Exception as e:
                        logger.warning(f"GPU monitoring error: {e}")
                
                system_metric = {
                    'timestamp': timestamp,
                    'cpu_percent': cpu_percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_total_gb': memory.total / (1024**3),
                    'memory_percent': memory.percent,
                    'gpu_metrics': gpu_metrics
                }
                
                self.system_metrics.append(system_metric)
                
                # Keep only last 100 system metrics to prevent memory issues
                if len(self.system_metrics) > 100:
                    self.system_metrics = self.system_metrics[-100:]
                    
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
                time.sleep(5)
    
    def update_training_progress(self, epoch: int, total_epochs: int, 
                               step: int, total_steps: int, 
                               batch: int = 0, batches_per_epoch: int = 0):
        """Update training progress metrics"""
        self.current_epoch = epoch
        self.total_epochs = total_epochs
        self.current_step = step
        self.total_steps = total_steps
        self.current_batch = batch
        self.batches_per_epoch = batches_per_epoch
    
    def log_training_step(self, logs: Dict[str, Any], epoch: int, step: int):
        """Log training step metrics"""
        timestamp = time.time()
        elapsed_time = timestamp - self.start_time
        
        # Extract metrics from logs
        train_loss = logs.get('loss', logs.get('train_loss'))
        eval_loss = logs.get('eval_loss')
        learning_rate = logs.get('learning_rate')
        accuracy = logs.get('accuracy', logs.get('eval_accuracy'))
        
        # Store in history
        if train_loss is not None:
            self.train_losses.append({'step': step, 'epoch': epoch, 'value': train_loss, 'timestamp': timestamp})
        if eval_loss is not None:
            self.eval_losses.append({'step': step, 'epoch': epoch, 'value': eval_loss, 'timestamp': timestamp})
        if learning_rate is not None:
            self.learning_rates.append({'step': step, 'epoch': epoch, 'value': learning_rate, 'timestamp': timestamp})
        if accuracy is not None:
            self.accuracies.append({'step': step, 'epoch': epoch, 'value': accuracy, 'timestamp': timestamp})
        
        metric_entry = {
            'timestamp': timestamp,
            'elapsed_time': elapsed_time,
            'epoch': epoch,
            'step': step,
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'learning_rate': learning_rate,
            'accuracy': accuracy,
            **logs  # Include all other logs
        }
        
        self.metrics_history.append(metric_entry)
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current training progress information"""
        elapsed_time = time.time() - self.start_time
        
        # Calculate ETA
        eta = None
        if self.current_epoch > 0 and self.total_epochs > 0:
            time_per_epoch = elapsed_time / self.current_epoch
            remaining_epochs = self.total_epochs - self.current_epoch
            eta = remaining_epochs * time_per_epoch
        
        # Calculate throughput
        samples_per_sec = 0
        if elapsed_time > 0 and self.current_step > 0:
            samples_per_sec = self.current_step / elapsed_time
        
        return {
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'current_batch': self.current_batch,
            'batches_per_epoch': self.batches_per_epoch,
            'elapsed_time': elapsed_time,
            'eta': eta,
            'samples_per_sec': samples_per_sec,
            'epoch_progress': (self.current_epoch / max(self.total_epochs, 1)) * 100,
            'overall_progress': (self.current_step / max(self.total_steps, 1)) * 100
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary for dashboard"""
        progress = self.get_progress_info()
        
        # Latest system metrics
        latest_system = self.system_metrics[-1] if self.system_metrics else {}
        
        return {
            'experiment_id': self.experiment_id,
            'progress': progress,
            'metrics': {
                'train_losses': self.train_losses[-50:],  # Last 50 entries
                'eval_losses': self.eval_losses[-50:],
                'learning_rates': self.learning_rates[-50:],
                'accuracies': self.accuracies[-50:]
            },
            'system': {
                'current': latest_system,
                'history': self.system_metrics[-20:]  # Last 20 entries (100 seconds)
            },
            'training_logs': self.metrics_history[-20:]  # Last 20 training logs
        }

# Global registry for active metric collectors
_active_collectors: Dict[str, TrainingMetricsCollector] = {}

def get_metrics_collector(experiment_id: str) -> Optional[TrainingMetricsCollector]:
    """Get metrics collector for experiment"""
    return _active_collectors.get(experiment_id)

def create_metrics_collector(experiment_id: str) -> TrainingMetricsCollector:
    """Create and start a new metrics collector"""
    collector = TrainingMetricsCollector(experiment_id)
    collector.start_monitoring()
    _active_collectors[experiment_id] = collector
    return collector

def cleanup_metrics_collector(experiment_id: str):
    """Stop and cleanup metrics collector"""
    collector = _active_collectors.pop(experiment_id, None)
    if collector:
        collector.stop_monitoring()