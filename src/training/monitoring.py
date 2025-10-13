# src/training/monitoring.py
"""
Weights & Biases integration for comprehensive experiment tracking.

W&B provides:
1. Real-time metric visualization (loss curves, learning rate, etc.)
2. Model versioning and artifact management
3. Hyperparameter tracking and comparison
4. System monitoring (memory, GPU usage)
5. Reproducibility (exact config for every run)

Why W&B?
- Industry standard (used by OpenAI, Anthropic, etc.)
- Free for personal use
- Beautiful dashboards
- Easy collaboration and sharing
- API for programmatic access
"""

import wandb
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict
import psutil


@dataclass
class WandBConfig:
    """Configuration for W&B logging."""
    project: str = "gpt-training-weekend"
    entity: Optional[str] = None  # Your W&B username
    name: Optional[str] = None    # Run name (auto-generated if None)
    tags: List[str] = None        # Tags for organizing runs
    notes: str = ""               # Description of this experiment
    
    # Logging intervals
    log_interval: int = 100       # Log metrics every N steps
    checkpoint_interval: int = 1000  # Save checkpoints every N steps
    
    # What to log
    log_gradients: bool = True    # Log gradient statistics
    log_model: bool = True        # Save model checkpoints
    log_code: bool = True         # Save code for reproducibility
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class WandBLogger:
    """
    Comprehensive W&B logger for training monitoring.
    
    This class handles all W&B interactions including:
    - Initializing runs
    - Logging metrics, images, and artifacts
    - Managing checkpoints
    - Creating reports
    
    Example:
        >>> config = {'learning_rate': 6e-4, 'batch_size': 8}
        >>> logger = WandBLogger('my-project', config)
        >>> 
        >>> for step in range(1000):
        >>>     loss = train_step()
        >>>     logger.log_training_step(step, loss, lr, grad_norm)
        >>> 
        >>> logger.finish()
    """
    
    def __init__(
        self,
        project_name: str,
        config: Dict[str, Any],
        wandb_config: Optional[WandBConfig] = None,
        resume: bool = False
    ):
        """
        Initialize W&B logger.
        
        Args:
            project_name: W&B project name
            config: Training configuration dict
            wandb_config: W&B-specific configuration
            resume: Resume previous run if it exists
        """
        self.project_name = project_name
        self.config = config
        self.wandb_config = wandb_config or WandBConfig()
        
        # Initialize W&B run
        self.run = wandb.init(
            project=project_name,
            entity=self.wandb_config.entity,
            name=self.wandb_config.name,
            tags=self.wandb_config.tags,
            notes=self.wandb_config.notes,
            config=config,
            resume="allow" if resume else False,
            save_code=self.wandb_config.log_code
        )
        
        self.step = 0
        self.start_time = time.time()
        
        # Track best metrics
        self.best_metrics = {
            'train_loss': float('inf'),
            'eval_loss': float('inf'),
            'eval_perplexity': float('inf')
        }
        
        print(f"🎨 W&B Logger initialized:")
        print(f"   Project: {project_name}")
        print(f"   Run: {self.run.name}")
        print(f"   URL: {self.run.url}")
        print(f"   Run ID: {self.run.id}")
    
    def log_training_step(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        grad_norm: float,
        tokens_per_sec: Optional[float] = None,
        memory_gb: Optional[float] = None
    ):
        """
        Log metrics for a training step.
        
        This is called EVERY training step (or every N steps).
        
        Args:
            step: Current training step
            loss: Training loss
            learning_rate: Current learning rate
            grad_norm: Gradient norm (for monitoring stability)
            tokens_per_sec: Training throughput
            memory_gb: Memory usage in GB
        """
        self.step = step
        
        # Core training metrics
        metrics = {
            "train/loss": loss,
            "train/learning_rate": learning_rate,
            "train/grad_norm": grad_norm,
            "step": step,
        }
        
        # Optional metrics
        if tokens_per_sec is not None:
            metrics["train/tokens_per_sec"] = tokens_per_sec
        
        if memory_gb is not None:
            metrics["system/memory_gb"] = memory_gb
            metrics["system/memory_percent"] = (memory_gb / psutil.virtual_memory().total) * 100
        
        # Training progress
        elapsed_time = time.time() - self.start_time
        metrics["system/elapsed_hours"] = elapsed_time / 3600
        
        # Update best loss
        if loss < self.best_metrics['train_loss']:
            self.best_metrics['train_loss'] = loss
            metrics["train/best_loss"] = loss
        
        # Log to W&B
        wandb.log(metrics, step=step)
    
    def log_evaluation(
        self,
        step: int,
        eval_loss: float,
        eval_perplexity: float,
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log evaluation metrics.
        
        Called after evaluation on validation set.
        
        Args:
            step: Current training step
            eval_loss: Validation loss
            eval_perplexity: Validation perplexity
            additional_metrics: Other metrics (accuracy, etc.)
        """
        metrics = {
            "eval/loss": eval_loss,
            "eval/perplexity": eval_perplexity,
            "step": step
        }
        
        # Add additional metrics
        if additional_metrics:
            for key, value in additional_metrics.items():
                metrics[f"eval/{key}"] = value
        
        # Track improvements
        if eval_loss < self.best_metrics['eval_loss']:
            self.best_metrics['eval_loss'] = eval_loss
            metrics["eval/best_loss"] = eval_loss
        
        if eval_perplexity < self.best_metrics['eval_perplexity']:
            self.best_metrics['eval_perplexity'] = eval_perplexity
            metrics["eval/best_perplexity"] = eval_perplexity
        
        wandb.log(metrics, step=step)
    
    def log_gradients(
        self,
        model: nn.Module,
        step: int
    ):
        """
        Log gradient statistics for monitoring training stability.
        
        This helps detect:
        - Exploding gradients (norm too high)
        - Vanishing gradients (norm too low)
        - Dead neurons (no gradient flow)
        
        Args:
            model: Model to analyze
            step: Current training step
        """
        if not self.wandb_config.log_gradients:
            return
        
        metrics = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Gradient statistics
                grad_data = param.grad.data
                
                metrics[f"gradients/{name}/norm"] = grad_data.norm().item()
                metrics[f"gradients/{name}/mean"] = grad_data.mean().item()
                metrics[f"gradients/{name}/std"] = grad_data.std().item()
                metrics[f"gradients/{name}/max"] = grad_data.max().item()
                metrics[f"gradients/{name}/min"] = grad_data.min().item()
                
                # Parameter statistics (for comparison)
                param_data = param.data
                metrics[f"parameters/{name}/norm"] = param_data.norm().item()
                metrics[f"parameters/{name}/mean"] = param_data.mean().item()
                metrics[f"parameters/{name}/std"] = param_data.std().item()
                
                # Ratio (important for monitoring)
                grad_norm = grad_data.norm().item()
                param_norm = param_data.norm().item()
                if param_norm > 0:
                    metrics[f"ratios/{name}/grad_to_param"] = grad_norm / param_norm
        
        wandb.log(metrics, step=step)
    
    def log_text_samples(
        self,
        step: int,
        prompts: List[str],
        generations: List[str],
        n_samples: int = 5
    ):
        """
        Log text generation samples.
        
        This creates a nice table in W&B showing your model's generations.
        
        Args:
            step: Current training step
            prompts: Input prompts
            generations: Generated text
            n_samples: Number of samples to log
        """
        # Create table
        columns = ["Prompt", "Generation"]
        data = []
        
        for prompt, generation in zip(prompts[:n_samples], generations[:n_samples]):
            data.append([prompt, generation])
        
        table = wandb.Table(columns=columns, data=data)
        wandb.log({"samples/generations": table}, step=step)
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        loss: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save model checkpoint as W&B artifact.
        
        W&B artifacts provide:
        - Versioning (automatically tracks all checkpoints)
        - Metadata (save training state with checkpoint)
        - Easy downloading (restore checkpoints from any run)
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            step: Current training step
            loss: Current loss (for metadata)
            metadata: Additional metadata
        
        Returns:
            str: Artifact name
        """
        if not self.wandb_config.log_model:
            return ""
        
        # Create checkpoint
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
        }
        
        if metadata:
            checkpoint['metadata'] = metadata
        
        # Save locally first
        checkpoint_path = f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Create W&B artifact
        artifact = wandb.Artifact(
            name=f"model-{self.run.id}",
            type="model",
            description=f"Model checkpoint at step {step}",
            metadata={
                'step': step,
                'loss': loss,
                'learning_rate': self.config.get('learning_rate'),
                **(metadata or {})
            }
        )
        
        # Add checkpoint file
        artifact.add_file(checkpoint_path)
        
        # Log artifact
        self.run.log_artifact(artifact)
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        print(f"   Artifact: {artifact.name}:v{artifact.version}")
        
        return artifact.name
    
    def create_summary_report(self):
        """
        Create a summary report at end of training.
        
        This creates a nice summary table with all important metrics.
        """
        summary = {
            "final/train_loss": self.best_metrics['train_loss'],
            "final/eval_loss": self.best_metrics['eval_loss'],
            "final/eval_perplexity": self.best_metrics['eval_perplexity'],
            "final/total_steps": self.step,
            "final/total_hours": (time.time() - self.start_time) / 3600
        }
        
        # Log summary
        for key, value in summary.items():
            wandb.run.summary[key] = value
        
        print(f"\n{'='*60}")
        print("Training Summary:")
        print(f"{'='*60}")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")
    
    def finish(self):
        """
        Finish the W&B run.
        
        Always call this at the end of training!
        """
        self.create_summary_report()
        wandb.finish()
        print("✅ W&B run finished successfully!")


class MetricsCollector:
    """
    Collect and aggregate metrics during training.
    
    This class handles:
    - Running averages (smooth loss curves)
    - Min/max tracking (for monitoring ranges)
    - Histogram data (for analyzing distributions)
    
    Example:
        >>> collector = MetricsCollector()
        >>> 
        >>> for step in range(1000):
        >>>     loss = train_step()
        >>>     collector.add('train_loss', loss)
        >>>     
        >>>     if step % 100 == 0:
        >>>         avg_loss = collector.get_average('train_loss')
        >>>         print(f"Average loss: {avg_loss:.4f}")
        >>>         collector.reset()
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics collector.
        
        Args:
            window_size: Number of values to keep for running average
        """
        self.window_size = window_size
        self.metrics = {}
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
    
    def add(self, name: str, value: float):
        """
        Add a metric value.
        
        Args:
            name: Metric name
            value: Metric value
        """
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(value)
        
        # Keep only last window_size values
        if len(self.metrics[name]) > self.window_size:
            self.metrics[name] = self.metrics[name][-self.window_size:]
    
    def get_average(self, name: str) -> float:
        """Get average of metric."""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        return sum(self.metrics[name]) / len(self.metrics[name])
    
    def get_latest(self, name: str) -> float:
        """Get latest value of metric."""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        return self.metrics[name][-1]
    
    def get_min(self, name: str) -> float:
        """Get minimum value of metric."""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        return min(self.metrics[name])
    
    def get_max(self, name: str) -> float:
        """Get maximum value of metric."""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        return max(self.metrics[name])
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        summary = {}
        
        for name in self.metrics:
            summary[name] = {
                'average': self.get_average(name),
                'latest': self.get_latest(name),
                'min': self.get_min(name),
                'max': self.get_max(name),
                'count': len(self.metrics[name])
            }
        
        return summary


class ExperimentManager:
    """
    High-level experiment management.
    
    This combines W&B logging with local checkpointing and provides
    a unified interface for experiment tracking.
    
    Example:
        >>> manager = ExperimentManager('my-experiment', config)
        >>> 
        >>> for step, batch in enumerate(dataloader):
        >>>     loss = train_step(batch)
        >>>     
        >>>     # Log metrics
        >>>     manager.log_step(step, loss, lr, grad_norm)
        >>>     
        >>>     # Save checkpoint
        >>>     if step % 1000 == 0:
        >>>         manager.save_checkpoint(model, optimizer, step)
        >>> 
        >>> manager.finish()
    """
    
    def __init__(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        output_dir: str = "outputs",
        use_wandb: bool = True
    ):
        """
        Initialize experiment manager.
        
        Args:
            experiment_name: Name for this experiment
            config: Training configuration
            output_dir: Directory for local checkpoints
            use_wandb: Whether to use W&B
        """
        self.experiment_name = experiment_name
        self.config = config
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        
        # Initialize W&B logger
        if use_wandb:
            self.wandb_logger = WandBLogger(experiment_name, config)
        else:
            self.wandb_logger = None
        
        # Initialize metrics collector
        self.metrics_collector = MetricsCollector()
        
        # Save config locally
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"📁 Experiment Manager initialized:")
        print(f"   Name: {experiment_name}")
        print(f"   Output dir: {self.output_dir}")
        print(f"   W&B enabled: {use_wandb}")
    
    def log_step(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        grad_norm: float,
        **kwargs
    ):
        """
        Log a training step.
        
        Args:
            step: Training step
            loss: Loss value
            learning_rate: Current LR
            grad_norm: Gradient norm
            **kwargs: Additional metrics
        """
        # Add to collector
        self.metrics_collector.add('loss', loss)
        self.metrics_collector.add('learning_rate', learning_rate)
        self.metrics_collector.add('grad_norm', grad_norm)
        
        # Log to W&B
        if self.wandb_logger:
            self.wandb_logger.log_training_step(
                step, loss, learning_rate, grad_norm, **kwargs
            )
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        is_best: bool = False
    ):
        """
        Save checkpoint locally and to W&B.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            step: Current step
            is_best: Whether this is the best model so far
        """
        # Save locally
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config,
            'metrics': self.metrics_collector.get_summary()
        }
        
        # Regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Best checkpoint
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"💎 New best model saved: {best_path}")
        
        # Save to W&B
        if self.wandb_logger:
            loss = self.metrics_collector.get_latest('loss')
            self.wandb_logger.save_checkpoint(
                model, optimizer, step, loss
            )
    
    def finish(self):
        """Finish experiment and cleanup."""
        if self.wandb_logger:
            self.wandb_logger.finish()
        
        # Save final metrics summary
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.metrics_collector.get_summary(), f, indent=2)
        
        print(f"✅ Experiment finished!")
        print(f"   Results saved to: {self.output_dir}")


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("W&B Integration Demo")
    print("=" * 60)
    
    # Configuration
    config = {
        'model': 'small_gpt',
        'learning_rate': 6e-4,
        'batch_size': 8,
        'max_steps': 1000,
        'block_size': 512
    }
    
    # Initialize experiment manager
    manager = ExperimentManager(
        experiment_name="demo-run",
        config=config,
        use_wandb=False  # Set to True to actually use W&B
    )
    
    # Simulate training
    print("\n📊 Simulating training...")
    
    for step in range(100):
        # Simulate training step
        loss = 5.0 * (0.99 ** step) + 0.1  # Decreasing loss
        lr = 6e-4 * (1 - step / 100)       # Decreasing LR
        grad_norm = 1.0 + 0.1 * (step % 10)  # Varying gradient norm
        
        # Log metrics
        manager.log_step(
            step=step,
            loss=loss,
            learning_rate=lr,
            grad_norm=grad_norm,
            tokens_per_sec=2000.0
        )
        
        # Print progress
        if step % 20 == 0:
            avg_loss = manager.metrics_collector.get_average('loss')
            print(f"Step {step:3d}: Loss={loss:.4f}, Avg Loss={avg_loss:.4f}, LR={lr:.2e}")
    
    # Finish experiment
    manager.finish()
    
    print("\n" + "=" * 60)
    print("✓ Demo completed!")
    print("=" * 60)