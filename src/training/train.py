"""
Complete GPT Training Script with Hydra Configuration

This is a production-ready training script that integrates:
- Modern GPT architecture (RoPE, RMSNorm, SwiGLU)
- AdamW optimization with SGDR scheduling
- MPS-optimized training pipeline
- Comprehensive evaluation and logging
- Checkpoint management and recovery
- Multi-run hyperparameter sweeps

Usage:
    # Basic training
    python scripts/train.py

    # Override config
    python scripts/train.py training.max_steps=10000 model.n_layer=8

    # Hyperparameter sweep
    python scripts/train.py --multirun training.lr=6e-4,3e-4,1e-4
"""

import os
import sys
import time
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.gpt import GPTConfig, SmallGPT
from src.training.mps_optimizer import MPSDeviceManager
from src.evaluation.metrics import ModelEvaluator
from src.data.dataset import MemoryMappedDataset, GPTTokenizer


@dataclass
class TrainingState:
    """Tracks training state for checkpointing."""
    step: int = 0
    epoch: int = 0
    best_val_loss: float = float('inf')
    best_val_ppl: float = float('inf')
    tokens_seen: int = 0
    total_train_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'step': self.step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'best_val_ppl': self.best_val_ppl,
            'tokens_seen': self.tokens_seen,
            'total_train_time': self.total_train_time
        }

    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> 'TrainingState':
        return cls(**state_dict)


class HydraTrainingScript:
    """
    Main training script with Hydra configuration management.

    Handles:
    - Configuration loading and validation
    - Component initialization (model, optimizer, data)
    - Error handling and recovery
    - Logging and monitoring setup
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize training script with Hydra config.

        Args:
            cfg: Hydra configuration object
        """
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save full config
        config_path = self.output_dir / "config.yaml"
        with open(config_path, 'w') as f:
            OmegaConf.save(cfg, f)

        print("=" * 70)
        print("INITIALIZING TRAINING SCRIPT")
        print("=" * 70)
        print(f"\n📁 Output directory: {self.output_dir}")
        print(f"⚙️  Config saved to: {config_path}")

        # Initialize components
        self.device = self._setup_device()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.evaluator = None
        self.state = TrainingState()

    def _setup_device(self) -> torch.device:
        """Setup and validate compute device."""
        if torch.backends.mps.is_available() and self.cfg.system.device == 'mps':
            device = torch.device('mps')
            print(f"\n✅ Using Apple Silicon MPS")
        elif torch.cuda.is_available() and self.cfg.system.device == 'cuda':
            device = torch.device('cuda')
            print(f"\n✅ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print(f"\n⚠️  Using CPU (slower)")

        return device

    def initialize_model(self) -> nn.Module:
        """Initialize GPT model with config."""
        print("\n🤖 Initializing model...")

        model_cfg = GPTConfig(
            vocab_size=self.cfg.model.vocab_size,
            n_layer=self.cfg.model.n_layer,
            n_head=self.cfg.model.n_head,
            n_embd=self.cfg.model.n_embd,
            dropout=self.cfg.model.dropout,
            block_size=self.cfg.model.block_size,
            bias=self.cfg.model.bias,
            use_gradient_checkpointing=self.cfg.training.gradient_checkpointing
        )

        model = SmallGPT(model_cfg)
        model = model.to(self.device)

        # Print model statistics
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"   Total parameters: {n_params:,}")
        print(f"   Trainable parameters: {n_trainable:,}")
        print(f"   Model size: {n_params * 4 / 1024 ** 2:.2f} MB")

        self.model = model
        return model

    def initialize_optimizer(self) -> torch.optim.Optimizer:
        """Initialize AdamW optimizer with parameter groups."""
        print("\n⚙️  Initializing optimizer...")

        # Separate parameters into decay and no-decay groups
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # No weight decay for biases and layer norms
            if param.dim() < 2 or 'ln' in name or 'bias' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {'params': decay_params, 'weight_decay': self.cfg.training.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.cfg.training.lr,
            betas=(self.cfg.training.beta1, self.cfg.training.beta2),
            eps=self.cfg.training.eps
        )

        print(f"   Learning rate: {self.cfg.training.lr}")
        print(f"   Weight decay: {self.cfg.training.weight_decay}")
        print(f"   Betas: ({self.cfg.training.beta1}, {self.cfg.training.beta2})")
        print(f"   Decay params: {len(decay_params)}")
        print(f"   No-decay params: {len(no_decay_params)}")

        self.optimizer = optimizer
        return optimizer

    def initialize_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Initialize SGDR cosine annealing scheduler with warmup."""
        print("\n📈 Initializing learning rate scheduler...")

        def get_lr(step):
            """Calculate learning rate with warmup and cosine annealing."""
            # Warmup phase
            if step < self.cfg.training.warmup_steps:
                return step / self.cfg.training.warmup_steps

            # Cosine annealing after warmup
            progress = (step - self.cfg.training.warmup_steps) / \
                       (self.cfg.training.max_steps - self.cfg.training.warmup_steps)

            return self.cfg.training.min_lr_ratio + \
                (1 - self.cfg.training.min_lr_ratio) * \
                0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, get_lr)

        print(f"   Warmup steps: {self.cfg.training.warmup_steps}")
        print(f"   Max steps: {self.cfg.training.max_steps}")
        print(f"   Min LR ratio: {self.cfg.training.min_lr_ratio}")

        self.scheduler = scheduler
        return scheduler

    def initialize_data(self) -> Tuple[DataLoader, DataLoader]:
        """Initialize training and validation dataloaders."""
        print("\n📊 Initializing data loaders...")

        # Initialize tokenizer
        tokenizer = GPTTokenizer(
            vocab_size=self.cfg.model.vocab_size
        )

        # Resolve paths relative to project root (not hydra output dir)
        # Get the project root (where train.py is located, go up 2 levels)
        project_root = Path(__file__).parent.parent.parent
        train_path = project_root / self.cfg.data.train_path
        val_path = project_root / self.cfg.data.val_path

        print(f"   Current working directory: {os.getcwd()}")
        print(f"   Project root: {project_root}")
        print(f"   Train path (config): {self.cfg.data.train_path}")
        print(f"   Train path (resolved): {train_path}")
        print(f"   Train path exists: {train_path.exists()}")

        # Training dataset
        if train_path.exists():
            train_dataset = MemoryMappedDataset(
                str(train_path),
                block_size=self.cfg.model.block_size,
                stride=self.cfg.data.stride
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.cfg.training.batch_size,
                shuffle=True,
                num_workers=0,  # MPS limitation
                pin_memory=False
            )

            print(f"   Train dataset: {len(train_dataset)} samples")
        else:
            print(f"   ⚠️  Train data not found at {train_path}")
            print(f"   Creating dummy dataset for testing...")
            train_loader = self._create_dummy_dataloader('train')

        # Validation dataset
        if val_path.exists():
            val_dataset = MemoryMappedDataset(
                str(val_path),
                block_size=self.cfg.model.block_size,
                stride=self.cfg.model.block_size  # No overlap for validation
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.cfg.training.eval_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )

            print(f"   Val dataset: {len(val_dataset)} samples")
        else:
            print(f"   ⚠️  Val data not found at {val_path}")
            print(f"   Creating dummy dataset for testing...")
            val_loader = self._create_dummy_dataloader('val')

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Initialize evaluator
        self.evaluator = ModelEvaluator(self.model, tokenizer, self.device)

        return train_loader, val_loader

    def _create_dummy_dataloader(self, split: str) -> DataLoader:
        """Create dummy dataloader for testing."""

        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size, seq_len, vocab_size):
                self.size = size
                self.seq_len = seq_len
                self.vocab_size = vocab_size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                x = torch.randint(0, self.vocab_size, (self.seq_len,))
                y = torch.randint(0, self.vocab_size, (self.seq_len,))
                return {'input_ids': x, 'targets': y}

        size = 1000 if split == 'train' else 100
        dataset = DummyDataset(size, self.cfg.model.block_size, self.cfg.model.vocab_size)

        return DataLoader(
            dataset,
            batch_size=self.cfg.training.batch_size if split == 'train' else self.cfg.training.eval_batch_size,
            shuffle=(split == 'train'),
            num_workers=0
        )

    def initialize_wandb(self):
        """Initialize Weights & Biases logging."""
        if not self.cfg.wandb.enabled:
            print("\n⚠️  W&B logging disabled")
            return

        print("\n📊 Initializing Weights & Biases...")

        wandb.init(
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            config=OmegaConf.to_container(self.cfg, resolve=True),
            dir=str(self.output_dir),
            resume='allow',
            id=self.cfg.wandb.run_id if self.cfg.wandb.run_id else None
        )

        # Watch model
        if self.cfg.wandb.watch_model:
            wandb.watch(self.model, log='all', log_freq=100)

        print(f"   Project: {self.cfg.wandb.project}")
        print(f"   Run: {wandb.run.name}")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load checkpoint for resuming training.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            True if checkpoint loaded successfully
        """
        if not os.path.exists(checkpoint_path):
            return False

        print(f"\n🔄 Loading checkpoint from {checkpoint_path}...")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load model
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load scheduler
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Load training state
            self.state = TrainingState.from_dict(checkpoint['training_state'])

            print(f"   ✅ Resumed from step {self.state.step}")
            print(f"   Best validation loss: {self.state.best_val_loss:.4f}")

            return True

        except Exception as e:
            print(f"   ❌ Error loading checkpoint: {e}")
            return False


class TrainingLoop:
    """
    Main training loop with step-by-step execution.

    Handles:
    - Training steps with gradient accumulation
    - Checkpoint management
    - Evaluation scheduling
    - Progress tracking
    """

    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler._LRScheduler,
            train_loader: DataLoader,
            val_loader: DataLoader,
            evaluator: ModelEvaluator,
            cfg: DictConfig,
            device: torch.device,
            output_dir: Path,
            state: TrainingState
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.evaluator = evaluator
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        self.state = state

        # Create checkpoint directory
        self.checkpoint_dir = output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Training metrics
        self.train_losses = []
        self.val_losses = []

    def train(self):
        """Execute complete training loop."""
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"\nMax steps: {self.cfg.training.max_steps}")
        print(f"Gradient accumulation: {self.cfg.training.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.cfg.training.batch_size * self.cfg.training.gradient_accumulation_steps}")
        print(f"Eval interval: {self.cfg.training.eval_interval}")
        print(f"Save interval: {self.cfg.training.save_interval}")

        self.model.train()
        train_iter = iter(self.train_loader)

        start_time = time.time()
        step_start_time = time.time()

        # Progress bar
        pbar = tqdm(
            initial=self.state.step,
            total=self.cfg.training.max_steps,
            desc="Training"
        )

        try:
            while self.state.step < self.cfg.training.max_steps:
                # Training step
                loss, metrics = self._training_step(train_iter)

                # Update progress
                self.state.step += 1
                self.state.tokens_seen += self.cfg.training.batch_size * self.cfg.model.block_size

                # Calculate throughput
                step_time = time.time() - step_start_time
                tokens_per_sec = (self.cfg.training.batch_size * self.cfg.model.block_size) / step_time

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                    'tok/s': f'{tokens_per_sec:.0f}'
                })

                # Logging
                if self.state.step % self.cfg.training.log_interval == 0:
                    self._log_metrics(loss, metrics, step_time, tokens_per_sec)

                # Evaluation
                if self.state.step % self.cfg.training.eval_interval == 0:
                    val_metrics = self._evaluate()
                    self._log_val_metrics(val_metrics)

                    # Check if best model
                    if val_metrics['loss'] < self.state.best_val_loss:
                        self.state.best_val_loss = val_metrics['loss']
                        self.state.best_val_ppl = val_metrics['perplexity']
                        self._save_checkpoint('best')

                    self.model.train()

                # Save checkpoint
                if self.state.step % self.cfg.training.save_interval == 0:
                    self._save_checkpoint(f'step_{self.state.step}')

                step_start_time = time.time()

        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user")
            print("Saving checkpoint...")
            self._save_checkpoint('interrupted')

        except Exception as e:
            print(f"\n\n❌ Training failed with error: {e}")
            print("Saving checkpoint...")
            self._save_checkpoint('error')
            raise

        finally:
            pbar.close()

            # Final evaluation
            print("\n" + "=" * 70)
            print("FINAL EVALUATION")
            print("=" * 70)

            final_metrics = self._evaluate()
            self._log_val_metrics(final_metrics, prefix="final")

            # Save final checkpoint
            self._save_checkpoint('final')

            # Training summary
            total_time = time.time() - start_time
            self.state.total_train_time += total_time

            self._print_training_summary(total_time, final_metrics)

    def _training_step(self, train_iter) -> Tuple[float, Dict[str, float]]:
        """
        Execute single training step with gradient accumulation.

        Returns:
            Tuple of (loss, metrics_dict)
        """
        total_loss = 0.0

        for micro_step in range(self.cfg.training.gradient_accumulation_steps):
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Move to device
            input_ids = batch[0].to(self.device)
            targets = batch[1].to(self.device)

            # Forward pass
            with torch.amp.autocast(device_type=str(self.device), dtype=torch.float16,
                                    enabled=self.cfg.training.mixed_precision):
                logits, _ = self.model(input_ids)

                # Calculate loss
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1)
                )

                # Scale loss for gradient accumulation
                loss = loss / self.cfg.training.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            total_loss += loss.item()

        # Gradient clipping
        if self.cfg.training.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.training.grad_clip
            )
        else:
            grad_norm = 0.0

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        # Metrics
        metrics = {
            'grad_norm': grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
            'lr': self.scheduler.get_last_lr()[0]
        }

        return total_loss, metrics

    def _evaluate(self) -> Dict[str, float]:
        """Run evaluation on validation set."""
        print(f"\n📊 Evaluating at step {self.state.step}...")

        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
                input_ids = batch[0].to(self.device)
                targets = batch[1].to(self.device)

                logits, _ = self.model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction='sum'
                )

                total_loss += loss.item()
                total_tokens += targets.numel()

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        metrics = {
            'loss': avg_loss,
            'perplexity': perplexity,
            'tokens': total_tokens
        }

        print(f"   Val Loss: {avg_loss:.4f}")
        print(f"   Val PPL: {perplexity:.2f}")

        return metrics

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{name}.pt"

        checkpoint = {
            'step': self.state.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_state': self.state.to_dict(),
            'config': OmegaConf.to_container(self.cfg, resolve=True)
        }

        torch.save(checkpoint, checkpoint_path)

        # Also save as latest
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)

        print(f"   💾 Saved checkpoint: {checkpoint_path.name}")

    def _log_metrics(self, loss: float, metrics: Dict[str, float], step_time: float, tokens_per_sec: float):
        """Log training metrics."""
        if self.cfg.wandb.enabled:
            wandb.log({
                'train/loss': loss,
                'train/grad_norm': metrics['grad_norm'],
                'train/learning_rate': metrics['lr'],
                'train/tokens_per_sec': tokens_per_sec,
                'train/step_time': step_time,
                'step': self.state.step
            })

    def _log_val_metrics(self, metrics: Dict[str, float], prefix: str = "val"):
        """Log validation metrics."""
        if self.cfg.wandb.enabled:
            wandb.log({
                f'{prefix}/loss': metrics['loss'],
                f'{prefix}/perplexity': metrics['perplexity'],
                'step': self.state.step
            })

    def _print_training_summary(self, total_time: float, final_metrics: Dict[str, float]):
        """Print final training summary."""
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        print(f"\n✅ Training completed!")
        print(f"\n📊 Statistics:")
        print(f"   Total steps: {self.state.step}")
        print(f"   Total tokens: {self.state.tokens_seen:,}")
        print(f"   Training time: {total_time / 3600:.2f} hours")
        print(f"   Average throughput: {self.state.tokens_seen / total_time:.0f} tokens/sec")
        print(f"\n🎯 Final Metrics:")
        print(f"   Final val loss: {final_metrics['loss']:.4f}")
        print(f"   Final val PPL: {final_metrics['perplexity']:.2f}")
        print(f"   Best val loss: {self.state.best_val_loss:.4f}")
        print(f"   Best val PPL: {self.state.best_val_ppl:.2f}")
        print(f"\n💾 Checkpoints saved to: {self.checkpoint_dir}")


class ExperimentRunner:
    """
    Experiment runner for multi-run support and hyperparameter sweeps.

    Handles:
    - Multiple training runs with different configs
    - Hyperparameter sweep orchestration
    - Results aggregation and comparison
    """

    def __init__(self, base_cfg: DictConfig):
        self.base_cfg = base_cfg
        self.results = []

    def run_experiment(self, cfg: DictConfig) -> Dict[str, Any]:
        """
        Run single experiment with given config.

        Args:
            cfg: Experiment configuration

        Returns:
            Experiment results dictionary
        """
        print("\n" + "=" * 70)
        print(f"RUNNING EXPERIMENT: {cfg.experiment_name}")
        print("=" * 70)

        try:
            # Initialize training script
            script = HydraTrainingScript(cfg)

            # Initialize all components
            script.initialize_model()
            script.initialize_optimizer()
            script.initialize_scheduler()
            script.initialize_data()
            script.initialize_wandb()

            # Resume from checkpoint if specified
            if cfg.resume_from_checkpoint:
                script.load_checkpoint(cfg.resume_from_checkpoint)

            # Create training loop
            training_loop = TrainingLoop(
                model=script.model,
                optimizer=script.optimizer,
                scheduler=script.scheduler,
                train_loader=script.train_loader,
                val_loader=script.val_loader,
                evaluator=script.evaluator,
                cfg=cfg,
                device=script.device,
                output_dir=script.output_dir,
                state=script.state
            )

            # Run training
            training_loop.train()

            # Collect results
            results = {
                'experiment_name': cfg.experiment_name,
                'config': OmegaConf.to_container(cfg, resolve=True),
                'final_step': training_loop.state.step,
                'best_val_loss': training_loop.state.best_val_loss,
                'best_val_ppl': training_loop.state.best_val_ppl,
                'total_tokens': training_loop.state.tokens_seen,
                'training_time': training_loop.state.total_train_time,
                'output_dir': str(script.output_dir)
            }

            # Close W&B
            if cfg.wandb.enabled:
                wandb.finish()

            return results

        except Exception as e:
            print(f"\n❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()

            return {
                'experiment_name': cfg.experiment_name,
                'error': str(e),
                'status': 'failed'
            }

    def run_sweep(self, sweep_configs: list) -> list:
        """
        Run multiple experiments with different configurations.

        Args:
            sweep_configs: List of configuration dictionaries

        Returns:
            List of results from all experiments
        """
        print("\n" + "=" * 70)
        print(f"STARTING HYPERPARAMETER SWEEP")
        print(f"Total experiments: {len(sweep_configs)}")
        print("=" * 70)

        all_results = []

        for i, sweep_cfg in enumerate(sweep_configs, 1):
            print(f"\n{'=' * 70}")
            print(f"EXPERIMENT {i}/{len(sweep_configs)}")
            print(f"{'=' * 70}")

            # Merge with base config
            cfg = OmegaConf.merge(self.base_cfg, sweep_cfg)

            # Run experiment
            result = self.run_experiment(cfg)
            all_results.append(result)

            # Save intermediate results
            self._save_sweep_results(all_results)

        # Final summary
        self._print_sweep_summary(all_results)

        return all_results

    def _save_sweep_results(self, results: list):
        """Save sweep results to JSON."""
        results_path = Path(self.base_cfg.output_dir) / "sweep_results.json"

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Sweep results saved to: {results_path}")

    def _print_sweep_summary(self, results: list):
        """Print summary of all experiments."""
        print("\n" + "=" * 70)
        print("SWEEP SUMMARY")
        print("=" * 70)

        successful = [r for r in results if 'error' not in r]
        failed = [r for r in results if 'error' in r]

        print(f"\n✅ Successful experiments: {len(successful)}")
        print(f"❌ Failed experiments: {len(failed)}")

        if successful:
            print("\n🏆 Best Experiments:")
            print("-" * 70)

            # Sort by best validation loss
            sorted_results = sorted(successful, key=lambda x: x['best_val_loss'])

            for i, result in enumerate(sorted_results[:3], 1):
                print(f"\n{i}. {result['experiment_name']}")
                print(f"   Val Loss: {result['best_val_loss']:.4f}")
                print(f"   Val PPL: {result['best_val_ppl']:.2f}")
                print(f"   Output: {result['output_dir']}")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for training.

    Args:
        cfg: Hydra configuration
    """
    # Print configuration
    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg))

    # Run experiment
    runner = ExperimentRunner(cfg)
    result = runner.run_experiment(cfg)

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)

    return result


if __name__ == "__main__":
    main()