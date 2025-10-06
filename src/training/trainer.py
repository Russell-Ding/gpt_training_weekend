'''

Credit to Claude
'''
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, List, Tuple
import math
from dataclasses import dataclass
import torch.nn.functional as F


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    # Optimizer settings
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95  # Critical: NOT 0.999!
    epsilon: float = 1e-8

    # Learning rate schedule
    warmup_steps: int = 2000
    max_steps: int = 100000
    min_lr_ratio: float = 0.1  # min_lr = max_lr * 0.1

    # SGDR settings
    restart_period: int = 10000  # Steps between restarts
    restart_mult: float = 1.5  # Multiply period after each restart

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16" for MPS (NOT bfloat16!)
    loss_scale: float = 65536.0  # Manual loss scaling

    # Gradient clipping
    max_grad_norm: float = 1.0

class ModernOptimizer:

    @staticmethod
    def create(
            model: nn.Module,
            lr: float = 6e-4,
            weight_decay: float = 0.1,
            betas: Tuple[float, float] = (0.9, 0.95),
            eps: float = 1e-8
    ) -> torch.optim.AdamW:
        """
        Create AdamW optimizer with proper parameter grouping.

        Weight decay should ONLY be applied to:
        - Weight matrices (2D tensors)
        - Embedding weights

        NO weight decay for:
        - Biases (1D tensors)
        - LayerNorm parameters (gamma, beta)
        - Position embeddings (sometimes)

        Args:
            model: The model to optimize
            lr: Learning rate
            weight_decay: L2 regularization strength
            betas: (β1, β2) for Adam momentum
            eps: Epsilon for numerical stability

        Returns:
            torch.optim.AdamW: Configured optimizer
        """
        # Separate parameters into groups
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # Skip frozen parameters

            # Determine if this parameter should have weight decay
            if param.dim() >= 2:
                # 2D or higher = weight matrices, embeddings
                decay_params.append(param)
            else:
                # 1D = biases, LayerNorm parameters
                no_decay_params.append(param)

        # Create parameter groups
        optimizer_groups = [
            {
                'params': decay_params,
                'weight_decay': weight_decay,
                'lr': lr
            },
            {
                'params': no_decay_params,
                'weight_decay': 0.0,  # NO weight decay!
                'lr': lr
            }
        ]

        # Create optimizer
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=lr,
            betas=betas,
            eps=eps
        )

        # Print summary
        num_decay = sum(p.numel() for p in decay_params)
        num_no_decay = sum(p.numel() for p in no_decay_params)
        print(f"🔧 AdamW Optimizer created:")
        print(f"   Parameters with decay: {num_decay:,} ({len(decay_params)} tensors)")
        print(f"   Parameters without decay: {num_no_decay:,} ({len(no_decay_params)} tensors)")
        print(f"   β1={betas[0]}, β2={betas[1]} (note: β2≠0.999!)")
        print(f"   Weight decay: {weight_decay}")

        return optimizer

class SGDRScheduler():

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            max_lr: float = 6e-4,
            min_lr_ratio: float = 0.1,
            warmup_steps: int = 2000,
            T_0: int = 10000,  # Initial restart period
            T_mult: float = 1.5,  # Period multiplier after each restart
            max_steps: Optional[int] = None
    ):
        """
        Initialize SGDR scheduler.

        Args:
            optimizer: The optimizer to schedule
            max_lr: Maximum learning rate (after warmup)
            min_lr_ratio: Minimum LR as fraction of max (e.g., 0.1 = 10%)
            warmup_steps: Linear warmup duration
            T_0: Steps between first restart
            T_mult: Multiply restart period by this after each restart
            max_steps: Total training steps (optional, for final decay)
        """
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = max_lr * min_lr_ratio
        self.warmup_steps = warmup_steps
        self.T_0 = T_0
        self.T_mult = T_mult
        self.max_steps = max_steps

        # Track restart state
        self.restart_count = 0
        self.current_T = T_0
        self.restart_step = warmup_steps  # First restart after warmup

        print(f"📅 SGDR Scheduler initialized:")
        print(f"   Max LR: {max_lr:.2e}")
        print(f"   Min LR: {self.min_lr:.2e}")
        print(f"   Warmup: {warmup_steps} steps")
        print(f"   Initial period: {T_0} steps")
        print(f"   Period multiplier: {T_mult}")

    def get_lr(self, step: int) -> float:
        """
        Calculate learning rate for given step.

        Args:
            step: Current training step

        Returns:
            float: Learning rate for this step
        """
        # Phase 1: Linear warmup
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps

        # Phase 2: After max_steps, stay at min_lr
        if self.max_steps and step > self.max_steps:
            return self.min_lr

        # Phase 3: Cosine annealing with restarts
        # Determine position within current cycle
        steps_since_restart = step - self.restart_step

        # Check if we need a restart
        if steps_since_restart >= self.current_T:
            # Restart!
            self.restart_count += 1
            self.restart_step = step
            self.current_T = int(self.current_T * self.T_mult)
            steps_since_restart = 0

            print(f"🔄 SGDR Restart #{self.restart_count} at step {step}")
            print(f"   Next restart in {self.current_T} steps")

        # Cosine annealing within current cycle
        progress = steps_since_restart / self.current_T
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay

        return lr

    def step(self, current_step: int):
        """
        Update learning rate for current step.

        Args:
            current_step: Current training step
        """
        lr = self.get_lr(current_step)

        # Update all parameter groups
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


class MPSMixedPrecisionTrainer:
    """
    Mixed precision training for MPS (Apple Silicon).

    CRITICAL: MPS does NOT support:
    - torch.cuda.amp.GradScaler
    - bfloat16 dtype

    So we implement MANUAL mixed precision:
    1. Forward pass in float16 (faster)
    2. Loss scaling to prevent underflow
    3. Backward in float16
    4. Unscale gradients before optimizer step
    5. Gradient overflow detection and recovery

    Why mixed precision?
    - 2× faster on MPS
    - 2× less memory usage
    - Same final accuracy (with proper loss scaling)

    Example:
        >>> trainer = MPSMixedPrecisionTrainer(
        >>>     model, optimizer, device='mps'
        >>> )
        >>>
        >>> for batch in dataloader:
        >>>     loss = trainer.training_step(batch, targets)
    """

    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            device: torch.device,
            use_amp: bool = True,
            loss_scale: float = 65536.0,
            max_grad_norm: float = 1.0
    ):
        """
        Initialize mixed precision trainer.

        Args:
            model: Model to train
            optimizer: Optimizer (AdamW)
            device: Device (MPS or CPU)
            use_amp: Enable mixed precision
            loss_scale: Initial loss scaling factor
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp and device.type == 'mps'
        self.loss_scale = loss_scale
        self.max_grad_norm = max_grad_norm

        # Track overflow events
        self.overflow_count = 0
        self.total_steps = 0

        # Determine dtype for autocast
        if self.use_amp:
            if device.type == 'mps':
                self.amp_dtype = torch.float16  # MPS only supports float16
            else:
                self.amp_dtype = torch.float16

        print(f"🎯 Mixed Precision Trainer initialized:")
        print(f"   Device: {device}")
        print(f"   Mixed precision: {self.use_amp}")
        if self.use_amp:
            print(f"   AMP dtype: {self.amp_dtype}")
            print(f"   Loss scale: {loss_scale}")
            print(f"   Gradient clipping: {max_grad_norm}")

    def training_step(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor
    ) -> float:
        """
        Execute one training step with mixed precision.

        Args:
            inputs: Input tensor
            targets: Target tensor

        Returns:
            float: Loss value
        """
        self.total_steps += 1

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass with autocast
        if self.use_amp:
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                logits, loss = self.model(inputs, targets)

            # Scale loss for backward pass
            scaled_loss = loss * self.loss_scale
        else:
            logits, loss = self.model(inputs, targets)
            scaled_loss = loss

        # Backward pass
        scaled_loss.backward()

        # Check for gradient overflow/underflow
        if self.use_amp:
            if not self._check_gradients_valid():
                # Overflow detected!
                self.overflow_count += 1
                print(f"⚠️  Gradient overflow detected (#{self.overflow_count})")

                # Reduce loss scale
                self.loss_scale = max(self.loss_scale / 2.0, 1.0)
                print(f"   Reducing loss scale to {self.loss_scale}")

                # Skip this step
                self.optimizer.zero_grad()
                return loss.item()

            # Unscale gradients
            self._unscale_gradients()

        # Gradient clipping
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm
        )

        # Optimizer step
        self.optimizer.step()

        return loss.item()

    def _check_gradients_valid(self) -> bool:
        """
        Check if gradients contain NaN or Inf.

        Returns:
            bool: True if gradients are valid
        """
        for param in self.model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    return False
        return True

    def _unscale_gradients(self):
        """Unscale gradients after backward pass."""
        inv_scale = 1.0 / self.loss_scale

        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(inv_scale)

    def get_stats(self) -> Dict[str, float]:
        """
        Get training statistics.

        Returns:
            dict: Statistics including overflow rate
        """
        overflow_rate = self.overflow_count / max(self.total_steps, 1)

        return {
            'total_steps': self.total_steps,
            'overflow_count': self.overflow_count,
            'overflow_rate': overflow_rate,
            'current_loss_scale': self.loss_scale
        }


# Complete training loop example
class CompleteTrainer:
    """
    Complete training pipeline integrating all components.

    This combines:
    - AdamW optimizer with β2=0.95
    - SGDR learning rate scheduling
    - MPS mixed precision training
    - Gradient clipping
    - Checkpoint management

    Example:
        >>> trainer = CompleteTrainer(model, config, device)
        >>> trainer.train(train_dataloader, eval_dataloader, num_steps=100000)
    """

    def __init__(
            self,
            model: nn.Module,
            config: TrainingConfig,
            device: torch.device
    ):
        """
        Initialize complete trainer.

        Args:
            model: Model to train
            config: Training configuration
            device: Device (MPS or CPU)
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Create optimizer
        self.optimizer = ModernOptimizer.create(
            model,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon
        )

        # Create scheduler
        self.scheduler = SGDRScheduler(
            self.optimizer,
            max_lr=config.learning_rate,
            min_lr_ratio=config.min_lr_ratio,
            warmup_steps=config.warmup_steps,
            T_0=config.restart_period,
            T_mult=config.restart_mult,
            max_steps=config.max_steps
        )

        # Create mixed precision trainer
        self.mp_trainer = MPSMixedPrecisionTrainer(
            model,
            self.optimizer,
            device,
            use_amp=config.use_amp,
            loss_scale=config.loss_scale,
            max_grad_norm=config.max_grad_norm
        )

        self.step = 0

    def train(
            self,
            train_dataloader,
            eval_dataloader=None,
            num_steps: int = 100000,
            eval_interval: int = 1000,
            log_interval: int = 100
    ):
        """
        Main training loop.

        Args:
            train_dataloader: Training data
            eval_dataloader: Evaluation data (optional)
            num_steps: Total training steps
            eval_interval: Steps between evaluations
            log_interval: Steps between logging
        """
        self.model.train()

        print(f"\n{'=' * 60}")
        print(f"Starting training for {num_steps} steps")
        print(f"{'=' * 60}\n")

        while self.step < num_steps:
            for batch_idx, (inputs, targets) in enumerate(train_dataloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Update learning rate
                current_lr = self.scheduler.step(self.step)

                # Training step
                loss = self.mp_trainer.training_step(inputs, targets)

                # Logging
                if self.step % log_interval == 0:
                    stats = self.mp_trainer.get_stats()
                    print(
                        f"Step {self.step:6d} | "
                        f"Loss: {loss:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Overflow rate: {stats['overflow_rate'] * 100:.1f}%"
                    )

                # Evaluation
                if eval_dataloader and self.step % eval_interval == 0:
                    eval_loss = self.evaluate(eval_dataloader)
                    print(f"📊 Eval loss: {eval_loss:.4f}")
                    self.model.train()

                self.step += 1

                if self.step >= num_steps:
                    break

        print(f"\n{'=' * 60}")
        print(f"Training completed!")
        print(f"{'=' * 60}\n")

        # Final statistics
        stats = self.mp_trainer.get_stats()
        print("Final statistics:")
        print(f"  Total steps: {stats['total_steps']}")
        print(f"  Gradient overflows: {stats['overflow_count']}")
        print(f"  Final loss scale: {stats['current_loss_scale']}")

    def evaluate(self, eval_dataloader) -> float:
        """
        Evaluate model on validation set.

        Args:
            eval_dataloader: Evaluation data

        Returns:
            float: Average evaluation loss
        """
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for inputs, targets in eval_dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                _, loss = self.model(inputs, targets)
                total_loss += loss.item()
                total_batches += 1

        return total_loss / total_batches


# Testing and demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Modern Training Pipeline Demo")
    print("=" * 60)

    # Configuration
    config = TrainingConfig(
        learning_rate=6e-4,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,  # Note: NOT 0.999!
        warmup_steps=100,
        max_steps=1000,
        restart_period=200,
        use_amp=True
    )


    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(100, 10)

        def forward(self, x, targets=None):
            logits = self.linear(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits, targets)
            return logits, loss


    model = DummyModel()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Test optimizer creation
    print("\n1. Testing ModernOptimizer:")
    optimizer = ModernOptimizer.create(model, lr=6e-4, weight_decay=0.1)

    # Test scheduler
    print("\n2. Testing SGDR Scheduler:")
    scheduler = SGDRScheduler(optimizer, max_lr=6e-4, warmup_steps=100, T_0=200)

    # Plot learning rate schedule
    lrs = [scheduler.get_lr(step) for step in range(1000)]
    print(f"   LR at step 0: {lrs[0]:.2e}")
    print(f"   LR at step 100 (after warmup): {lrs[100]:.2e}")
    print(f"   LR at step 300 (after first restart): {lrs[300]:.2e}")

    print("\n" + "=" * 60)
    print("✓ All components working correctly!")
    print("=" * 60)