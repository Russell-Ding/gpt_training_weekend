# src/data/dataset.py
"""
Efficient data pipeline for GPT training on Apple Silicon.

Key components:
1. MemoryMappedDataset: Handle large files without loading into RAM
2. GPTTokenizer: Fast tokenization with TikToken
3. MPS_DataLoader: MPS-compatible data loading (no multiprocessing)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tiktoken
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import json


class MemoryMappedDataset(Dataset):
    """
    Memory-mapped dataset for efficient large file handling.

    KEY PROBLEM: Training datasets can be huge (100GB+)
    - Loading entire dataset into RAM: Impossible!
    - Reading from disk each time: Slow!

    SOLUTION: Memory mapping
    - OS maps file to virtual memory
    - Only loads needed chunks into RAM
    - Fast random access (like it's in RAM)
    - Minimal memory footprint

    Example:
        >>> # Prepare data (once)
        >>> tokens = tokenize_corpus(text)
        >>> np.array(tokens, dtype=np.uint16).tofile('train.bin')
        >>>
        >>> # Load for training (fast!)
        >>> dataset = MemoryMappedDataset('train.bin', block_size=1024)
        >>> loader = DataLoader(dataset, batch_size=8)
    """

    def __init__(
            self,
            data_path: str,
            block_size: int = 1024,
            stride: Optional[int] = None,
            vocab_size: int = 50257,
            dtype: Optional[np.dtype] = None
    ):
        """
        Initialize memory-mapped dataset.

        Args:
            data_path: Path to binary file
            block_size: Sequence length for training
            stride: Step size for sliding window (default: block_size)
            vocab_size: Vocabulary size (for validation)
            dtype: Data type to use (auto-detected if None)
        """
        self.data_path = Path(data_path)
        self.block_size = block_size
        self.stride = stride or block_size
        self.vocab_size = vocab_size

        # Auto-detect dtype based on vocab_size if not specified
        if dtype is None:
            if vocab_size < 256:
                dtype = np.uint8
            elif vocab_size < 65536:
                dtype = np.uint16
            else:
                dtype = np.uint32

        self.dtype = dtype

        # Memory map the data file
        # dtype selection based on vocab size for memory efficiency:
        # - uint8: vocab < 256 (character-level)
        # - uint16: vocab < 65,536 (most BPE tokenizers)
        # - uint32: vocab >= 65,536 (very large vocabularies)
        self.data = np.memmap(
            str(self.data_path),
            dtype=self.dtype,
            mode='r'  # Read-only
        )

        # Calculate number of sequences
        # We need block_size + 1 tokens (input + target)
        self.num_sequences = (len(self.data) - block_size) // self.stride

        print(f"📊 MemoryMappedDataset loaded:")
        print(f"   File: {self.data_path}")
        print(f"   Total tokens: {len(self.data):,}")
        print(f"   Data type: {self.dtype}")
        print(f"   Block size: {block_size}")
        print(f"   Stride: {self.stride}")
        print(f"   Sequences: {self.num_sequences:,}")
        print(f"   Memory footprint: ~{self.data.nbytes / 1e6:.1f}MB (virtual)")

    def __len__(self) -> int:
        """Number of sequences in dataset."""
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get one training sequence.

        Args:
            idx: Sequence index

        Returns:
            Tuple of (input_tokens, target_tokens)

        How sliding window works:
            stride=block_size (no overlap):
            [0:1024], [1024:2048], [2048:3072], ...

            stride=block_size//2 (50% overlap):
            [0:1024], [512:1536], [1024:2048], ...
        """
        # Calculate start position
        start_idx = idx * self.stride

        # Extract block_size + 1 tokens
        # +1 because we need targets shifted by 1
        chunk = self.data[start_idx:start_idx + self.block_size + 1]

        # Handle end of file (might be shorter than block_size + 1)
        if len(chunk) < self.block_size + 1:
            # Pad with zeros (will be masked in loss)
            chunk = np.pad(
                chunk,
                (0, self.block_size + 1 - len(chunk)),
                mode='constant',
                constant_values=0
            )

        # Input: first block_size tokens
        # Target: next block_size tokens (shifted by 1)
        input_tokens = torch.from_numpy(chunk[:-1].astype(np.int64))
        target_tokens = torch.from_numpy(chunk[1:].astype(np.int64))

        return input_tokens, target_tokens

    @staticmethod
    def prepare_data(
            text_file: str,
            output_file: str,
            tokenizer,
            max_tokens: Optional[int] = None
    ):
        """
        Prepare training data from text file.

        This is a one-time preprocessing step.

        Args:
            text_file: Input text file
            output_file: Output binary file
            tokenizer: Tokenizer instance
            max_tokens: Maximum tokens to process (for testing)
        """
        print(f"📝 Preparing data from {text_file}...")

        # Read text
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # Tokenize
        print("   Tokenizing...")
        tokens = tokenizer.encode(text)

        if max_tokens:
            tokens = tokens[:max_tokens]

        # Convert to numpy array (uint16 for space efficiency)
        tokens_array = np.array(tokens, dtype=np.uint16)

        # Save to binary file
        print(f"   Saving {len(tokens):,} tokens to {output_file}...")
        tokens_array.tofile(output_file)

        print(f"✅ Data prepared!")
        print(f"   Tokens: {len(tokens):,}")
        print(f"   File size: {tokens_array.nbytes / 1e6:.1f}MB")


class GPTTokenizer:
    """
    Fast tokenizer using TikToken (OpenAI's tokenizer).

    Why TikToken?
    - 10× faster than HuggingFace tokenizers
    - Same tokenization as GPT-2/3/4
    - Minimal dependencies
    - Handles edge cases well

    Vocabulary optimization:
    - GPT-2: 50,257 tokens (large, slow for small models)
    - Optimized: 16,384 tokens (small, fast, good enough)

    Example:
        >>> tokenizer = GPTTokenizer(vocab_size=16384)
        >>> tokens = tokenizer.encode("Hello, world!")
        >>> text = tokenizer.decode(tokens)
    """

    def __init__(
            self,
            vocab_size: int = 50257,
            model_name: str = "gpt2"
    ):
        """
        Initialize tokenizer.

        Args:
            vocab_size: Vocabulary size (50257 for GPT-2, or smaller)
            model_name: Base tokenizer model
        """
        self.vocab_size = vocab_size
        self.model_name = model_name

        # Load base tokenizer
        self.tokenizer = tiktoken.get_encoding(model_name)

        # Special tokens
        self.eos_token = "<|endoftext|>"
        # The .encode() method is for general text and raises an error on special tokens by default.
        # To get a special token's ID, access the tokenizer's special_tokens dictionary.
        # This is the idiomatic and correct way to do it with tiktoken.
        self.eos_token_id = self.tokenizer.eot_token

        print(f"🔤 GPTTokenizer initialized:")
        print(f"   Base model: {model_name}")
        print(f"   Vocab size: {vocab_size:,}")
        print(f"   EOS token ID: {self.eos_token_id}")

    def encode(
            self,
            text: str,
            add_eos: bool = False
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_eos: Whether to add EOS token at end

        Returns:
            List of token IDs
        """
        tokens = self.tokenizer.encode(text)

        # Clip to vocabulary size if needed
        if self.vocab_size < 50257:
            tokens = [min(t, self.vocab_size - 1) for t in tokens]

        if add_eos:
            tokens.append(self.eos_token_id)

        return tokens

    def decode(
            self,
            tokens: List[int],
            skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            tokens: List of token IDs
            skip_special_tokens: Remove special tokens from output

        Returns:
            Decoded text
        """
        # Filter special tokens if requested
        if skip_special_tokens:
            tokens = [t for t in tokens if t != self.eos_token_id]

        # Decode
        text = self.tokenizer.decode(tokens)

        return text

    def encode_batch(
            self,
            texts: List[str],
            max_length: int = 1024,
            padding: bool = True,
            truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Batch encode multiple texts.

        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            padding: Pad to max_length
            truncation: Truncate to max_length

        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors
        """
        all_input_ids = []
        all_attention_masks = []

        for text in texts:
            # Encode
            tokens = self.encode(text)

            # Truncate if needed
            if truncation and len(tokens) > max_length:
                tokens = tokens[:max_length]

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * len(tokens)

            # Pad if needed
            if padding and len(tokens) < max_length:
                padding_length = max_length - len(tokens)
                tokens.extend([0] * padding_length)
                attention_mask.extend([0] * padding_length)

            all_input_ids.append(tokens)
            all_attention_masks.append(attention_mask)

        return {
            'input_ids': torch.tensor(all_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(all_attention_masks, dtype=torch.long)
        }


class MPS_DataLoader:
    """
    Device-agnostic DataLoader that auto-configures for MPS, CUDA, or CPU.

    DEVICE-SPECIFIC OPTIMIZATIONS:

    MPS (Apple Silicon):
    - num_workers=0 (multiprocessing conflicts with MPS context)
    - pin_memory=False (unified memory, not needed)

    CUDA (NVIDIA GPUs):
    - num_workers=4 (parallel data loading)
    - pin_memory=True (faster CPU->GPU transfers)

    CPU:
    - num_workers=4 (parallel processing)
    - pin_memory=False (no GPU)

    Example:
        >>> # Auto-detects device type and optimizes accordingly
        >>> loader = MPS_DataLoader(
        >>>     dataset,
        >>>     batch_size=8,
        >>>     device='cuda'  # Will use CUDA-optimized settings
        >>> )
        >>>
        >>> for batch in loader:
        >>>     # batch is already on specified device
        >>>     outputs = model(batch)
    """

    def __init__(
            self,
            dataset: Dataset,
            batch_size: int = 8,
            shuffle: bool = True,
            device: Optional[torch.device] = None,
            drop_last: bool = True,
            num_workers: Optional[int] = None
    ):
        """
        Initialize device-optimized DataLoader.

        Args:
            dataset: Dataset to load from
            batch_size: Batch size
            shuffle: Shuffle data each epoch
            device: Target device (auto-detected if None)
            drop_last: Drop last incomplete batch
            num_workers: Number of workers (auto-configured if None)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        elif isinstance(device, str):
            device = torch.device(device)

        self.device = device

        # Auto-configure num_workers and pin_memory based on device
        if num_workers is None:
            if self.device.type == 'mps':
                num_workers = 0  # MPS requires 0 workers
                pin_memory = False
                optimization = "MPS (unified memory)"
            elif self.device.type == 'cuda':
                num_workers = 4  # CUDA benefits from parallel loading
                pin_memory = True  # Speed up CPU->GPU transfers
                optimization = "CUDA (pinned memory + workers)"
            else:  # CPU
                num_workers = 4
                pin_memory = False
                optimization = "CPU (parallel workers)"
        else:
            # User specified num_workers - respect it
            pin_memory = (self.device.type == 'cuda')
            optimization = "Manual configuration"

        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Create underlying DataLoader with device-optimized settings
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=(num_workers > 0)  # Only if using workers
        )

        print(f"🔄 DataLoader initialized ({optimization}):")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {batch_size}")
        print(f"   Num workers: {num_workers}")
        print(f"   Pin memory: {pin_memory}")
        print(f"   Dataset size: {len(dataset):,}")
        print(f"   Batches per epoch: {len(self.dataloader):,}")

    def __iter__(self):
        """Iterate over batches."""
        for batch in self.dataloader:
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = tuple(b.to(self.device, non_blocking=True) for b in batch)
            else:
                batch = batch.to(self.device, non_blocking=True)

            yield batch

    def __len__(self):
        """Number of batches."""
        return len(self.dataloader)


class DynamicBatchDataLoader:
    """
    Advanced: Dynamic batching based on sequence length.

    Problem: Fixed batch size wastes computation
    - Short sequences: Could fit more in memory
    - Long sequences: Might OOM with fixed batch size

    Solution: Adjust batch size based on sequence length
    - Target: constant number of TOKENS per batch, not sequences

    Example:
        >>> loader = DynamicBatchDataLoader(
        >>>     dataset,
        >>>     target_tokens=8192,  # 8K tokens per batch
        >>>     device='mps'
        >>> )
        >>>
        >>> # Short sequences (512 tokens): batch_size = 16
        >>> # Long sequences (2048 tokens): batch_size = 4
    """

    def __init__(
            self,
            dataset: Dataset,
            target_tokens: int = 8192,
            max_batch_size: int = 32,
            device: Optional[torch.device] = None
    ):
        """
        Initialize dynamic batch loader.

        Args:
            dataset: Dataset to load from
            target_tokens: Target tokens per batch
            max_batch_size: Maximum batch size
            device: Target device
        """
        self.dataset = dataset
        self.target_tokens = target_tokens
        self.max_batch_size = max_batch_size
        self.device = device or torch.device('cpu')

        print(f"🔄 DynamicBatchDataLoader initialized:")
        print(f"   Target tokens/batch: {target_tokens:,}")
        print(f"   Max batch size: {max_batch_size}")

    def __iter__(self):
        """Iterate with dynamic batching."""
        indices = list(range(len(self.dataset)))

        # Group indices by sequence length for efficiency
        # (In practice, you'd sort by actual sequence lengths)
        i = 0
        while i < len(indices):
            # Get one example to determine sequence length
            example_input, _ = self.dataset[indices[i]]
            seq_len = len(example_input)

            # Calculate optimal batch size
            batch_size = min(
                self.target_tokens // seq_len,
                self.max_batch_size
            )
            batch_size = max(1, batch_size)  # At least 1

            # Collect batch
            batch_inputs = []
            batch_targets = []

            for j in range(batch_size):
                if i + j >= len(indices):
                    break

                input_tokens, target_tokens = self.dataset[indices[i + j]]
                batch_inputs.append(input_tokens)
                batch_targets.append(target_tokens)

            # Stack into tensors
            batch_inputs = torch.stack(batch_inputs).to(self.device)
            batch_targets = torch.stack(batch_targets).to(self.device)

            yield batch_inputs, batch_targets

            i += len(batch_inputs)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Data Pipeline Demo")
    print("=" * 60)

    # 1. Test tokenizer
    print("\n1. Testing GPTTokenizer:")
    tokenizer = GPTTokenizer(vocab_size=50257)

    test_text = "Hello, world! This is a test of the GPT tokenizer."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)

    print(f"   Original: {test_text}")
    print(f"   Tokens: {tokens[:10]}... ({len(tokens)} total)")
    print(f"   Decoded: {decoded}")

    # Test batch encoding
    batch_texts = [
        "First example text.",
        "Second example, longer text with more tokens.",
        "Third example."
    ]
    batch_encoded = tokenizer.encode_batch(batch_texts, max_length=20, padding=True)
    print(f"\n   Batch encoding:")
    print(f"   Input IDs shape: {batch_encoded['input_ids'].shape}")
    print(f"   Attention mask shape: {batch_encoded['attention_mask'].shape}")

    # 2. Prepare sample data
    print("\n2. Preparing sample dataset:")
    sample_text = "This is a test. " * 1000  # Repeat for sufficient data

    # Encode and save
    tokens = tokenizer.encode(sample_text)
    tokens_array = np.array(tokens, dtype=np.uint16)
    tokens_array.tofile('test_data.bin')

    print(f"   Created test_data.bin with {len(tokens):,} tokens")

    # 3. Test memory-mapped dataset
    print("\n3. Testing MemoryMappedDataset:")
    dataset = MemoryMappedDataset(
        'test_data.bin',
        block_size=128,
        stride=64  # 50% overlap
    )

    # Get a sample
    inputs, targets = dataset[0]
    print(f"   Sample input shape: {inputs.shape}")
    print(f"   Sample target shape: {targets.shape}")
    print(f"   Input tokens: {inputs[:10]}...")
    print(f"   Target tokens: {targets[:10]}...")

    # 4. Test MPS DataLoader
    print("\n4. Testing MPS_DataLoader:")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    loader = MPS_DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        device=device
    )

    # Get first batch
    for batch_inputs, batch_targets in loader:
        print(f"   Batch inputs shape: {batch_inputs.shape}")
        print(f"   Batch targets shape: {batch_targets.shape}")
        print(f"   Device: {batch_inputs.device}")
        break  # Just show first batch

    # 5. Test dynamic batching
    print("\n5. Testing DynamicBatchDataLoader:")
    dynamic_loader = DynamicBatchDataLoader(
        dataset,
        target_tokens=512,  # Small for demo
        max_batch_size=8,
        device=device
    )

    batch_sizes = []
    for i, (batch_inputs, batch_targets) in enumerate(dynamic_loader):
        batch_sizes.append(batch_inputs.shape[0])
        if i >= 5:  # Show first few batches
            break

    print(f"   First few batch sizes: {batch_sizes}")
    print(f"   Average batch size: {sum(batch_sizes) / len(batch_sizes):.1f}")

    # Cleanup
    import os

    os.remove('test_data.bin')

    print("\n" + "=" * 60)
    print("✓ All data pipeline components working!")
    print("=" * 60)