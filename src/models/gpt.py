from dataclasses import dataclass
import torch
import torch.nn as nn
import math


@dataclass
class GPTConfig:
    vocab_size: int = 16384  # Vocabulary size (reduced for efficiency)
    n_layerL: int =  6  # Number of transformer layers
    n_head: int =6  # Number of attention heads
    n_embd: int = 384  # Embedding dimension
    dropout: float = 0.1  # Dropout rate
    block_size: int = 512  # Maximum sequence length
    bias: bool = False

    # Advanced architecture features
    use_flash_attention: bool =  False  # Flash attention (not available on MPS)
    use_gradient_checkpointing: bool =  True  # Save memory at cost of compute
    weight_tying: bool = True

    # MPS optimizations
    use_gradient_checkpointing: bool = True
    attention_chunk_size: int = 2048
    max_batch_size: int = 8

    def __post_init__(self):
        # Validation
        assert self.n_embd % self.n_head == 0, f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"

        # Computed properties
        self.head_dim = self.n_embd // self.n_head


class MultiHeadAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.embedding_d = config.n_embd
        self.n_heads = config.n_head
        self.head_dim = config.head_dim
        self.key_layers = nn.Linear(self.embedding_d, self.embedding_d, bias = config.bias)
        self.query_layers = nn.Linear(self.embedding_d, self.embedding_d, bias = config.bias)
        self.value_layers = nn.Linear(self.embedding_d, self.embedding_d, bias = config.bias)

        self.attention_dropout = nn.Dropout(config.dropout)

        # MPS - specific settings
        self.chunk_size = config.attention_chunk_size
        self.memory_efficient = config.memory_efficient_attention

        # Causal mask (register as buffer so it moves with model to device) credit to claude
        self.register_buffer("causal_mask",
                             torch.tril(torch.ones(self.block_size, self.block_size))
                             .view(1, 1, self.block_size, self.block_size))

    def forward(self, input: torch.Tensor):
        k_heads = self.key_layers(input).view(input.shape[0], self.n_heads, input.shape[1], self.head_dim)
        q_heads = self.query_layers(input).view(input.shape[0], self.n_heads, input.shape[1], self.head_dim)
        v_heads = self.value_layers(input).view(input.shape[0], self.n_heads, input.shape[1], self.head_dim)

        if input.shape[1] <= self.chunk_size:
            #### use standard_attention
            self._standard_attention(k_heads, q_heads, v_heads)


    def _standard_attention(self, k_heads: torch.Tensor, q_heads: torch.Tensor, v_heads: torch.Tensor) -> torch.Tensor:
        '''
        This is the normal way to develop attention mechanism
        :param input: (batch_size, seq_len, embedding_dim)
        :return:
        '''

        seq_length = q_heads.shape[2]

        mask = self.causal_mask[:,:,:seq_length,:seq_length] #### get the mask


        raw_score = torch.matmul(k_heads, q_heads.transpose(-1, -2))

        raw_score.masked_fill(mask == 0, float('-inf'))

        #### get the matrix multiplication results
        #### get batch-size, n_heads, seq_len, seq_len
        kq = nn.functional.softmax(raw_score/math.sqrt(self.head_dim), dim=-1)

        h_atten = torch.matmul(kq, v_heads) #### batch-size, n_heads, seq_len, self.head_dim


        return h_atten.view(input.shape[0], input.shape[1], -1)

    def _chunked_attention(self, input: torch.Tensor):
        '''

        :param input:(batch_size, seq_len, embedding_dim)
        :return:
        '''

        # Process in chunks to avoid MPS memory limits
        chunk_size = min(self.chunk_size, input.size[1]) #### seq_length is longer than possible trunks
        num_chunks = (input.size[1] + chunk_size - 1) // chunk_size









