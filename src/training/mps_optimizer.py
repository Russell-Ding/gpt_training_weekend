# src/training/mps_optimizer.py
"""

All credit to Claude

MPS (Metal Performance Shaders) optimization layer for Apple Silicon.

This module provides utilities to maximize training performance and stability
on M1/M2/M3 MacBook Pros by handling:
- Memory management and OOM prevention
- Optimal batch size detection
- Attention chunking for long sequences
- Real-time performance monitoring
"""

import torch
import psutil
import time
import gc
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import warnings


@dataclass
class MPSMemoryStats:
    """Statistics about MPS memory usage."""
    total_gb: float
    allocated_gb: float
    reserved_gb: float
    free_gb: float
    utilization: float  # 0.0 to 1.0


class MPSDeviceManager:
    """
    Manages MPS device and memory for optimal training on Apple Silicon.

    Key features:
    - Auto-detects available memory
    - Finds optimal batch size through binary search
    - Provides safe forward pass with OOM recovery
    - Monitors memory usage in real-time

    Example:
        >>> manager = MPSDeviceManager(target_memory_ratio=0.7)
        >>> device = manager.device
        >>> model = model.to(device)
        >>> optimal_batch_size = manager.find_optimal_batch_size(model, sample_input)
    """

    def __init__(self, target_memory_ratio: float = 0.7):
        """
        Initialize MPS device manager.

        Args:
            target_memory_ratio: Fraction of total memory to use (0.0-1.0).
                                Default 0.7 leaves headroom for OS and other apps.
        """
        self.target_memory_ratio = target_memory_ratio
        self.device = self._setup_device()
        self.total_memory_gb = self._get_total_memory()
        self.max_memory_gb = self.total_memory_gb * target_memory_ratio

        print(f"🍎 MPS Device Manager initialized")
        print(f"   Total memory: {self.total_memory_gb:.1f} GB")
        print(f"   Target usage: {self.max_memory_gb:.1f} GB ({target_memory_ratio * 100:.0f}%)")

    def _setup_device(self) -> torch.device:
        """
        Set up the compute device with MPS fallback to CPU.

        Returns:
            torch.device: MPS device if available, else CPU
        """
        if torch.backends.mps.is_available():
            try:
                # Test MPS is actually working
                test = torch.zeros(1, device='mps')
                del test
                return torch.device('mps')
            except Exception as e:
                warnings.warn(f"MPS available but failed test: {e}. Falling back to CPU.")
                return torch.device('cpu')
        else:
            warnings.warn("MPS not available. Using CPU (will be slow!).")
            return torch.device('cpu')

    def _get_total_memory(self) -> float:
        """
        Get total system memory in GB.

        Returns:
            float: Total RAM in gigabytes
        """
        # On macOS, MPS uses unified memory
        # So we check system RAM
        total_bytes = psutil.virtual_memory().total
        return total_bytes / (1024 ** 3)  # Convert to GB

    def get_memory_stats(self) -> MPSMemoryStats:
        """
        Get current memory usage statistics.

        Returns:
            MPSMemoryStats: Current memory usage details
        """
        if self.device.type != 'mps':
            # Return dummy stats for CPU
            vm = psutil.virtual_memory()
            return MPSMemoryStats(
                total_gb=vm.total / 1e9,
                allocated_gb=vm.used / 1e9,
                reserved_gb=0.0,
                free_gb=vm.available / 1e9,
                utilization=vm.percent / 100.0
            )

        # For MPS, we track system memory since it's unified
        vm = psutil.virtual_memory()
        total = vm.total / 1e9
        used = vm.used / 1e9
        available = vm.available / 1e9

        return MPSMemoryStats(
            total_gb=total,
            allocated_gb=used,
            reserved_gb=0.0,  # MPS doesn't expose this
            free_gb=available,
            utilization=vm.percent / 100.0
        )

    def cleanup_memory(self):
        """
        Aggressive memory cleanup for MPS.

        Call this after OOM errors or between training phases.
        """
        if self.device.type == 'mps':
            # MPS-specific cleanup
            torch.mps.empty_cache()

        # General PyTorch cleanup
        gc.collect()

        # Give OS time to reclaim memory
        time.sleep(0.1)

    def find_optimal_batch_size(
            self,
            model: torch.nn.Module,
            sample_input: torch.Tensor,
            min_batch_size: int = 1,
            max_batch_size: int = 128,
            safety_margin: float = 0.9
    ) -> int:
        """
        Find the largest batch size that fits in memory using binary search.

        This is crucial for MPS because:
        1. Memory limits are strict (no virtual memory like CUDA)
        2. OOM crashes are hard to recover from
        3. Optimal batch size varies by model architecture

        Args:
            model: The model to test
            sample_input: Example input tensor (batch_size=1)
            min_batch_size: Minimum batch size to try
            max_batch_size: Maximum batch size to try
            safety_margin: Use this fraction of max working batch size (0.9 = 90%)

        Returns:
            int: Optimal batch size that fits in memory with safety margin

        Example:
            >>> sample = torch.randint(0, 1000, (1, 512))  # Single example
            >>> optimal_bs = manager.find_optimal_batch_size(model, sample)
            >>> print(f"Use batch_size={optimal_bs}")
        """
        print(f"\n🔍 Finding optimal batch size (range: {min_batch_size}-{max_batch_size})...")

        model.eval()  # Evaluation mode for testing
        original_device = next(model.parameters()).device
        model = model.to(self.device)
        sample_input = sample_input.to(self.device)

        # Binary search for max batch size
        left, right = min_batch_size, max_batch_size
        best_batch_size = min_batch_size

        while left <= right:
            mid = (left + right) // 2

            # Create batch of size 'mid'
            batch = sample_input.repeat(mid, *([1] * (sample_input.dim() - 1)))

            # Try forward pass
            success = self._test_batch_size(model, batch)

            if success:
                best_batch_size = mid
                left = mid + 1
                print(f"   ✓ Batch size {mid} works")
            else:
                right = mid - 1
                print(f"   ✗ Batch size {mid} failed (OOM)")

            # Cleanup after each test
            del batch
            self.cleanup_memory()

        # Apply safety margin
        optimal_batch_size = int(best_batch_size * safety_margin)

        model = model.to(original_device)
        model.train()  # Back to training mode

        print(f"✅ Optimal batch size: {optimal_batch_size} (max: {best_batch_size}, safety: {safety_margin})")
        return optimal_batch_size

    def _test_batch_size(self, model: torch.nn.Module, batch: torch.Tensor) -> bool:
        """
        Test if a batch size works without OOM.

        Args:
            model: Model to test
            batch: Input batch

        Returns:
            bool: True if batch size works, False if OOM
        """
        try:
            with torch.no_grad():
                # Forward pass
                _ = model(batch)

            # If we got here, it worked!
            return True

        except RuntimeError as e:
            if "MPS backend out of memory" in str(e) or "out of memory" in str(e).lower():
                return False
            else:
                # Some other error - re-raise
                raise e

    def safe_forward_pass(
            self,
            model: torch.nn.Module,
            inputs: torch.Tensor,
            max_retries: int = 3,
            fallback_chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Execute forward pass with automatic OOM recovery.

        If OOM occurs:
        1. Clean up memory
        2. Try chunking the input
        3. Fall back to CPU if needed

        Args:
            model: Model to run
            inputs: Input tensor
            max_retries: Maximum retry attempts
            fallback_chunk_size: If OOM, split into chunks of this size

        Returns:
            torch.Tensor: Model output

        Example:
            >>> inputs = torch.randint(0, 1000, (32, 512)).to(device)
            >>> outputs = manager.safe_forward_pass(model, inputs)
        """
        for attempt in range(max_retries):
            try:
                # Try normal forward pass
                return model(inputs)

            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    # Not an OOM error, re-raise
                    raise e

                print(f"⚠️  OOM on attempt {attempt + 1}/{max_retries}")

                # Cleanup
                self.cleanup_memory()

                # Try chunking if we have more retries
                if attempt < max_retries - 1 and fallback_chunk_size:
                    print(f"   Trying chunked forward pass (chunk_size={fallback_chunk_size})")
                    return self._chunked_forward_pass(model, inputs, fallback_chunk_size)

                # Last resort: fall back to CPU
                if attempt == max_retries - 1:
                    print("   ⚠️  Falling back to CPU (slow!)")
                    return self._cpu_forward_pass(model, inputs)

        raise RuntimeError(f"Failed after {max_retries} attempts")

    def _chunked_forward_pass(
            self,
            model: torch.nn.Module,
            inputs: torch.Tensor,
            chunk_size: int
    ) -> torch.Tensor:
        """
        Process input in chunks to reduce memory usage.

        Args:
            model: Model to run
            inputs: Input tensor (batch_size, seq_len)
            chunk_size: Process this many examples at a time

        Returns:
            torch.Tensor: Concatenated outputs
        """
        batch_size = inputs.size(0)
        outputs = []

        for i in range(0, batch_size, chunk_size):
            chunk = inputs[i:i + chunk_size]
            chunk_out = model(chunk)
            outputs.append(chunk_out)

            # Cleanup between chunks
            if i + chunk_size < batch_size:
                self.cleanup_memory()

        return torch.cat(outputs, dim=0)

    def _cpu_forward_pass(
            self,
            model: torch.nn.Module,
            inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Fall back to CPU for forward pass.

        Args:
            model: Model to run
            inputs: Input tensor

        Returns:
            torch.Tensor: Model output (moved back to original device)
        """
        original_device = inputs.device

        # Move to CPU
        model_cpu = model.cpu()
        inputs_cpu = inputs.cpu()

        # Forward pass
        with torch.no_grad():
            outputs_cpu = model_cpu(inputs_cpu)

        # Move back
        model.to(original_device)
        outputs = outputs_cpu.to(original_device)

        return outputs


class MPSMemoryProfiler:
    """
    Real-time memory profiling and OOM prediction for MPS training.

    Tracks memory usage over time and warns before OOM occurs.

    Example:
        >>> profiler = MPSMemoryProfiler(manager)
        >>> profiler.start_monitoring()
        >>>
        >>> for batch in dataloader:
        >>>     profiler.log_step()
        >>>     outputs = model(batch)
        >>>
        >>>     if profiler.is_oom_likely():
        >>>         print("Warning: OOM likely!")
        >>>         profiler.suggest_actions()
    """

    def __init__(self, device_manager: MPSDeviceManager):
        """
        Initialize memory profiler.

        Args:
            device_manager: MPS device manager instance
        """
        self.manager = device_manager
        self.history = []
        self.step = 0
        self.monitoring = False

    def start_monitoring(self):
        """Start memory monitoring."""
        self.monitoring = True
        self.history = []
        self.step = 0
        print("📊 Memory monitoring started")

    def log_step(self):
        """Log memory usage for current step."""
        if not self.monitoring:
            return

        stats = self.manager.get_memory_stats()
        self.history.append({
            'step': self.step,
            'allocated_gb': stats.allocated_gb,
            'utilization': stats.utilization,
            'timestamp': time.time()
        })
        self.step += 1

    def is_oom_likely(self, threshold: float = 0.9) -> bool:
        """
        Predict if OOM is likely based on current usage.

        Args:
            threshold: Memory utilization threshold (0.0-1.0)

        Returns:
            bool: True if OOM likely
        """
        if not self.history:
            return False

        current = self.history[-1]
        return current['utilization'] > threshold

    def suggest_actions(self) -> Dict[str, Any]:
        """
        Suggest actions to reduce memory usage.

        Returns:
            dict: Suggested actions and their potential impact
        """
        current_stats = self.manager.get_memory_stats()

        suggestions = {
            'reduce_batch_size': {
                'action': 'Reduce batch size by 50%',
                'impact': 'High - will immediately reduce memory usage'
            },
            'enable_gradient_checkpointing': {
                'action': 'Enable gradient checkpointing in model config',
                'impact': 'Medium - trades compute for memory'
            },
            'reduce_sequence_length': {
                'action': 'Reduce sequence length / block_size',
                'impact': 'High - quadratic memory savings for attention'
            },
            'cleanup_memory': {
                'action': 'Call torch.mps.empty_cache()',
                'impact': 'Low - only helps if memory is fragmented'
            }
        }

        return suggestions

    def get_summary(self) -> str:
        """
        Get summary of memory usage during monitoring.

        Returns:
            str: Formatted summary
        """
        if not self.history:
            return "No monitoring data available"

        allocations = [h['allocated_gb'] for h in self.history]
        utilizations = [h['utilization'] for h in self.history]

        summary = f"""
Memory Profiling Summary:
  Steps monitored: {len(self.history)}
  Avg allocation: {sum(allocations) / len(allocations):.2f} GB
  Max allocation: {max(allocations):.2f} GB
  Avg utilization: {sum(utilizations) / len(utilizations) * 100:.1f}%
  Max utilization: {max(utilizations) * 100:.1f}%
  OOM warnings: {sum(1 for u in utilizations if u > 0.9)}
        """.strip()

        return summary


# src/training/mps_optimizer.py (continued)
"""
MPSAttentionOptimizer: Specialized attention handling for Apple Silicon.

Key Problem: MPS has strict buffer size limits that cause crashes for long sequences.
Solution: Chunked attention that splits computation into smaller pieces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import warnings


class MPSAttentionOptimizer:
    """
    Optimizes attention computation for MPS (Metal Performance Shaders).

    CRITICAL MPS LIMITATIONS:
    1. Buffer size limit: ~2GB per tensor
    2. Attention matrices can exceed this for long sequences
    3. No automatic fallback - will just crash!

    Example problem:
        batch=8, heads=12, seq=4096, head_dim=64
        Attention matrix: 8 * 12 * 4096 * 4096 * 4 bytes = 6.4GB
        → EXCEEDS MPS LIMIT → CRASH! 💥

    Solution:
        Split into chunks that each fit within limits
        Process sequentially, concatenate results

    Example:
        >>> optimizer = MPSAttentionOptimizer(
        >>>     max_sequence_length=4096,
        >>>     chunk_size=1024,
        >>>     enable_validation=True
        >>> )
        >>>
        >>> # In your attention layer:
        >>> attn_output = optimizer.compute_attention(q, k, v, causal_mask)
    """

    # MPS buffer size limits (conservative estimates)
    MAX_BUFFER_SIZE_BYTES = 2 * 1024 ** 3  # 2GB
    MAX_ATTENTION_ELEMENTS = 200_000_000  # ~200M elements in attention matrix

    def __init__(
            self,
            max_sequence_length: int = 8192,
            chunk_size: int = 2048,
            enable_validation: bool = True,
            enable_warnings: bool = True
    ):
        """
        Initialize MPS attention optimizer.

        Args:
            max_sequence_length: Maximum sequence length to support
            chunk_size: Size of chunks for processing long sequences
            enable_validation: Validate buffer sizes before operations
            enable_warnings: Print warnings for potential issues
        """
        self.max_sequence_length = max_sequence_length
        self.chunk_size = chunk_size
        self.enable_validation = enable_validation
        self.enable_warnings = enable_warnings

        if self.enable_warnings:
            print(f"🔧 MPSAttentionOptimizer initialized:")
            print(f"   Max sequence: {max_sequence_length}")
            print(f"   Chunk size: {chunk_size}")

    def validate_buffer_size(
            self,
            batch_size: int,
            n_heads: int,
            seq_len: int,
            dtype: torch.dtype = torch.float32
    ) -> Tuple[bool, str]:
        """
        Check if attention computation will exceed MPS buffer limits.

        The attention matrix has shape: (batch, n_heads, seq_len, seq_len)
        For MPS, this cannot exceed ~2GB or ~200M elements.

        Args:
            batch_size: Batch size
            n_heads: Number of attention heads
            seq_len: Sequence length
            dtype: Data type (affects bytes per element)

        Returns:
            Tuple[bool, str]: (is_valid, reason_if_invalid)

        Example:
            >>> optimizer = MPSAttentionOptimizer()
            >>> is_valid, msg = optimizer.validate_buffer_size(8, 12, 4096)
            >>> if not is_valid:
            >>>     print(f"Cannot run: {msg}")
        """
        # Calculate attention matrix size
        total_elements = batch_size * n_heads * seq_len * seq_len

        # Bytes per element
        bytes_per_element = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2  # Note: bfloat16 NOT supported on MPS!
        }.get(dtype, 4)

        total_bytes = total_elements * bytes_per_element

        # Check element limit
        if total_elements > self.MAX_ATTENTION_ELEMENTS:
            return False, (
                f"Attention matrix too large: {total_elements:,} elements "
                f"(max: {self.MAX_ATTENTION_ELEMENTS:,})"
            )

        # Check byte limit
        if total_bytes > self.MAX_BUFFER_SIZE_BYTES:
            return False, (
                f"Attention matrix too large: {total_bytes / 1e9:.2f}GB "
                f"(max: {self.MAX_BUFFER_SIZE_BYTES / 1e9:.2f}GB)"
            )

        return True, "Valid"

    def should_use_chunking(
            self,
            batch_size: int,
            n_heads: int,
            seq_len: int
    ) -> bool:
        """
        Determine if chunking is needed for this configuration.

        Args:
            batch_size: Batch size
            n_heads: Number of attention heads
            seq_len: Sequence length

        Returns:
            bool: True if chunking should be used
        """
        if seq_len <= self.chunk_size:
            return False  # Small enough, no chunking needed

        # Check if full attention would exceed limits
        is_valid, _ = self.validate_buffer_size(batch_size, n_heads, seq_len)
        return not is_valid

    def compute_attention(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            causal_mask: Optional[torch.Tensor] = None,
            dropout_p: float = 0.0,
            scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute attention with automatic chunking if needed.

        This is the MAIN method you'll use in your attention layer.
        It automatically decides whether to use chunking based on input size.

        Args:
            query: Query tensor (batch, n_heads, seq_len, head_dim)
            key: Key tensor (batch, n_heads, seq_len, head_dim)
            value: Value tensor (batch, n_heads, seq_len, head_dim)
            causal_mask: Optional causal mask (1, 1, seq_len, seq_len)
            dropout_p: Dropout probability
            scale: Attention scaling factor (default: 1/sqrt(head_dim))

        Returns:
            torch.Tensor: Attention output (batch, n_heads, seq_len, head_dim)

        Example:
            >>> # In your MultiHeadAttention forward():
            >>> q, k, v = self.qkv_proj(x).split(...)
            >>> attn_out = optimizer.compute_attention(q, k, v, self.causal_mask)
        """
        B, n_heads, T, head_dim = query.shape

        # Validate configuration
        if self.enable_validation:
            is_valid, msg = self.validate_buffer_size(B, n_heads, T, query.dtype)
            if not is_valid and T <= self.chunk_size:
                # Even a single chunk exceeds limits!
                raise RuntimeError(
                    f"Sequence too long even for chunking: {msg}\n"
                    f"Suggestions:\n"
                    f"  - Reduce batch_size (current: {B})\n"
                    f"  - Reduce sequence length (current: {T})\n"
                    f"  - Reduce n_heads (current: {n_heads})"
                )

        # Decide whether to use chunking
        use_chunking = self.should_use_chunking(B, n_heads, T)

        if use_chunking:
            if self.enable_warnings:
                print(f"⚠️  Using chunked attention for seq_len={T}")
            return self._chunked_attention(
                query, key, value, causal_mask, dropout_p, scale
            )
        else:
            return self._standard_attention(
                query, key, value, causal_mask, dropout_p, scale
            )

    def _standard_attention(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            causal_mask: Optional[torch.Tensor],
            dropout_p: float,
            scale: Optional[float]
    ) -> torch.Tensor:
        """
        Standard scaled dot-product attention (no chunking).

        This is the normal attention computation when sequences are short enough.
        """
        B, n_heads, T, head_dim = query.shape

        # Scale factor
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        # Compute attention scores: Q @ K^T
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        # Shape: (B, n_heads, T, T)

        # Apply causal mask if provided
        if causal_mask is not None:
            mask = causal_mask[:, :, :T, :T]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply dropout
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        # Apply attention to values: attn_weights @ V
        attn_output = torch.matmul(attn_weights, value)
        # Shape: (B, n_heads, T, head_dim)

        return attn_output

    def _chunked_attention(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            causal_mask: Optional[torch.Tensor],
            dropout_p: float,
            scale: Optional[float]
    ) -> torch.Tensor:
        """
        Memory-efficient chunked attention for long sequences.

        KEY INSIGHT:
        Instead of computing the full (T x T) attention matrix at once,
        we process the queries in chunks, where each chunk attends to
        all previous keys/values (for causal attention).

        Memory savings:
        - Full attention: B * heads * T * T elements
        - Chunked: B * heads * chunk_size * T elements (much smaller!)

        Example:
            T=4096, chunk_size=1024
            Full: 4096 * 4096 = 16.8M elements per head
            Chunked (max): 1024 * 4096 = 4.2M elements per head
            Savings: 75% memory reduction!
        """
        B, n_heads, T, head_dim = query.shape

        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        # Split into chunks
        num_chunks = (T + self.chunk_size - 1) // self.chunk_size
        outputs = []

        for chunk_idx in range(num_chunks):
            # Determine chunk boundaries
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, T)
            chunk_len = end_idx - start_idx

            # Extract query chunk (only process these queries)
            q_chunk = query[:, :, start_idx:end_idx, :]  # (B, heads, chunk_len, head_dim)

            # For causal attention: keys/values include ALL previous tokens
            # This is why we include [:end_idx] not [start_idx:end_idx]
            k_chunk = key[:, :, :end_idx, :]  # (B, heads, end_idx, head_dim)
            v_chunk = value[:, :, :end_idx, :]  # (B, heads, end_idx, head_dim)

            # Compute attention for this chunk
            # Attention matrix: (B, heads, chunk_len, end_idx)
            # This is much smaller than full (T, T) matrix!
            attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * scale

            # Apply causal mask
            if causal_mask is not None:
                # Extract relevant portion of mask
                mask_chunk = causal_mask[:, :, start_idx:end_idx, :end_idx]
                attn_scores = attn_scores.masked_fill(mask_chunk == 0, float('-inf'))

            # Softmax
            attn_weights = F.softmax(attn_scores, dim=-1)

            # Dropout
            if dropout_p > 0.0:
                attn_weights = F.dropout(attn_weights, p=dropout_p)

            # Apply to values
            chunk_output = torch.matmul(attn_weights, v_chunk)
            # Shape: (B, heads, chunk_len, head_dim)

            outputs.append(chunk_output)

            # Cleanup between chunks (important for MPS!)
            if chunk_idx < num_chunks - 1:
                del attn_scores, attn_weights, chunk_output
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        # Concatenate all chunks
        final_output = torch.cat(outputs, dim=2)
        # Shape: (B, heads, T, head_dim)

        return final_output

    def get_recommended_chunk_size(
            self,
            batch_size: int,
            n_heads: int,
            target_memory_gb: float = 1.0
    ) -> int:
        """
        Calculate optimal chunk size for given configuration.

        Args:
            batch_size: Batch size
            n_heads: Number of attention heads
            target_memory_gb: Target memory per chunk (GB)

        Returns:
            int: Recommended chunk size

        Example:
            >>> optimizer = MPSAttentionOptimizer()
            >>> chunk_size = optimizer.get_recommended_chunk_size(
            >>>     batch_size=8, n_heads=12, target_memory_gb=0.5
            >>> )
            >>> print(f"Use chunk_size={chunk_size}")
        """
        # Each attention chunk needs: batch * heads * chunk_size * seq_len * 4 bytes
        # We want this to be <= target_memory_gb

        # Solve for chunk_size:
        # chunk_size * seq_len = target_memory_gb * 1e9 / (batch * heads * 4)
        # For worst case, assume seq_len = max_sequence_length

        bytes_available = target_memory_gb * 1e9
        bytes_per_element = 4  # float32

        max_elements = bytes_available / (batch_size * n_heads * bytes_per_element)

        # chunk_size * seq_len <= max_elements
        # To be safe, use sqrt(max_elements) for both dimensions
        recommended_chunk = int(math.sqrt(max_elements))

        # Round down to nearest power of 2 for efficiency
        recommended_chunk = 2 ** int(math.log2(recommended_chunk))

        # Clamp to reasonable range
        recommended_chunk = max(256, min(recommended_chunk, 4096))

        return recommended_chunk


# Integration example with MultiHeadAttention
class MPSOptimizedMultiHeadAttention(nn.Module):
    """
    Multi-head attention with MPS optimizations built-in.

    This is how you'd integrate MPSAttentionOptimizer into your
    actual attention layer.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Q, K, V projections
        self.qkv_proj = nn.Linear(self.n_embd, 3 * self.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)

        # Dropout
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

        # MPS optimizer
        self.mps_optimizer = MPSAttentionOptimizer(
            max_sequence_length=config.block_size,
            chunk_size=config.attention_chunk_size,
            enable_validation=True,
            enable_warnings=False  # Disable in training loop
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with MPS-optimized attention.

        Args:
            x: Input tensor (batch, seq_len, n_embd)

        Returns:
            torch.Tensor: Output tensor (batch, seq_len, n_embd)
        """
        B, T, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape to multi-head format
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # MPS-optimized attention computation
        # This automatically handles chunking if needed!
        attn_output = self.mps_optimizer.compute_attention(
            query=q,
            key=k,
            value=v,
            causal_mask=self.causal_mask,
            dropout_p=self.dropout if self.training else 0.0
        )

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)

        return output


# Testing and validation utilities
def test_mps_attention_limits():
    """
    Test to find actual MPS limits on your MacBook Pro.

    Run this to discover the exact limits of your hardware.
    """
    print("=" * 60)
    print("Testing MPS Attention Limits")
    print("=" * 60)

    optimizer = MPSAttentionOptimizer()

    # Test different configurations
    configs = [
        (8, 12, 1024),  # Small
        (8, 12, 2048),  # Medium
        (8, 12, 4096),  # Large
        (8, 12, 8192),  # Very large
        (4, 12, 8192),  # Reduced batch
        (2, 12, 16384),  # Maximum sequence
    ]

    for batch, heads, seq_len in configs:
        is_valid, msg = optimizer.validate_buffer_size(batch, heads, seq_len)
        needs_chunk = optimizer.should_use_chunking(batch, heads, seq_len)

        status = "✓" if is_valid else "✗"
        chunk_status = "CHUNKING REQUIRED" if needs_chunk else "DIRECT OK"

        print(f"{status} B={batch:2d} H={heads:2d} T={seq_len:5d} | {chunk_status:20s} | {msg}")

    print("=" * 60)


if __name__ == "__main__":
    # Run tests
    test_mps_attention_limits()

    # Example: Create optimized attention layer
    print("\n" + "=" * 60)
    print("Example: MPS-Optimized Attention Layer")
    print("=" * 60)

    from dataclasses import dataclass


    @dataclass
    class TestConfig:
        n_embd: int = 768
        n_head: int = 12
        dropout: float = 0.1
        block_size: int = 2048
        attention_chunk_size: int = 1024
        bias: bool = False


    config = TestConfig()
    attn_layer = MPSOptimizedMultiHeadAttention(config)

    # Test forward pass
    x = torch.randn(4, 512, 768)  # (batch=4, seq=512, embd=768)
    output = attn_layer(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Forward pass successful!")


    print("=" * 60)
    print("MPS Optimization Layer - Demo")
    print("=" * 60)

    # Initialize device manager
    manager = MPSDeviceManager(target_memory_ratio=0.7)


    # Create a simple test model
    class SimpleGPT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(1000, 256)
            self.linear1 = torch.nn.Linear(256, 512)
            self.linear2 = torch.nn.Linear(512, 1000)

        def forward(self, x):
            x = self.embed(x)
            x = x.mean(dim=1)  # Pool sequence
            x = self.linear1(x)
            x = torch.relu(x)
            x = self.linear2(x)
            return x


    model = SimpleGPT().to(manager.device)

    # Find optimal batch size
    sample_input = torch.randint(0, 1000, (1, 100))
    optimal_bs = manager.find_optimal_batch_size(
        model,
        sample_input,
        max_batch_size=64
    )

    # Test memory profiling
    profiler = MPSMemoryProfiler(manager)
    profiler.start_monitoring()

    print("\n" + "=" * 60)
    print("Running test training steps...")
    print("=" * 60)

    for step in range(5):
        profiler.log_step()

        # Simulate training
        inputs = torch.randint(0, 1000, (optimal_bs, 100)).to(manager.device)
        outputs = manager.safe_forward_pass(model, inputs)

        stats = manager.get_memory_stats()
        print(f"Step {step}: Memory {stats.allocated_gb:.2f}GB ({stats.utilization * 100:.1f}%)")

        if profiler.is_oom_likely(threshold=0.85):
            print("⚠️  High memory usage detected!")
            suggestions = profiler.suggest_actions()
            print("Suggestions:")
            for key, value in suggestions.items():
                print(f"  - {value['action']}: {value['impact']}")

    print("\n" + "=" * 60)
    print(profiler.get_summary())
    print("=" * 60)