from dataclasses import dataclass
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional



@dataclass
class GPTConfig:
    vocab_size: int = 16384  # Vocabulary size (reduced for efficiency)
    n_layer: int =  6  # Number of transformer layers
    n_head: int =6  # Number of attention heads
    n_embd: int = 384  # Embedding dimension
    dropout: float = 0.1  # Dropout rate
    block_size: int = 512  # Maximum sequence length
    bias: bool = False

    # Advanced architecture features
    use_flash_attention: bool =  False  # Flash attention (not available on MPS)
    use_gradient_checkpointing: bool =  True  # Save memory at cost of compute
    weight_tying: bool = True
    memory_efficient_attention: bool = True  # Memory-efficient attention

    # MPS optimizations
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
        self.dropout = config.dropout

        self.attention_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # MPS - specific settings
        self.chunk_size = config.attention_chunk_size
        self.memory_efficient = config.memory_efficient_attention
        self.block_size = config.block_size

        # Causal mask (register as buffer so it moves with model to device) credit to claude
        self.register_buffer("causal_mask",
                             torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

        # Output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, input: torch.Tensor):
        ### each of them is converted into (batch_size, n_heads, seq_len, head_dim)
        k_heads = self.key_layers(input).view(input.shape[0], self.n_heads, input.shape[1], self.head_dim)
        q_heads = self.query_layers(input).view(input.shape[0], self.n_heads, input.shape[1], self.head_dim)
        v_heads = self.value_layers(input).view(input.shape[0], self.n_heads, input.shape[1], self.head_dim)

        if input.shape[1] <= self.chunk_size:
            #### use standard_attention
            attn_output = self._standard_attention(k_heads, q_heads, v_heads)
        else:
            attn_output = self._chunked_attention(k_heads, q_heads, v_heads)

        # Output projection and dropout
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)

        return output

    def _standard_attention(self, k_heads: torch.Tensor, q_heads: torch.Tensor, v_heads: torch.Tensor) -> torch.Tensor:
        '''
        This is the normal way to develop attention mechanism
        :param input: (batch_size, seq_len, embedding_dim)
        :return:
        '''

        mask = self.causal_mask[:,:,:q_heads.shape[2],:k_heads.shape[2]] #### get the mask


        raw_score = torch.matmul(q_heads,k_heads.transpose(-1, -2)) ### (query_length, key_length)

        raw_score.masked_fill(mask == 0, float('-inf'))

        #### get the matrix multiplication results
        #### get batch-size, n_heads, seq_len, seq_len
        kq = nn.functional.softmax(raw_score/math.sqrt(self.head_dim), dim=-1)

        h_atten = torch.matmul(self.attention_dropout(kq), v_heads) #### batch-size, n_heads, query_length, self.head_dim


        return h_atten.view(k_heads.shape[0], q_heads.shape[2], -1)

    def _chunked_attention(self, k_heads: torch.Tensor, q_heads: torch.Tensor, v_heads: torch.Tensor):
        '''

        :param input:(batch_size, seq_len, embedding_dim)
        :return:
        '''

        # Process in chunks to avoid MPS memory limits
        chunk_size = min(self.chunk_size, q_heads.shape[1]) #### seq_length is longer than possible trunks
        num_chunks = (q_heads.shape[1] + chunk_size - 1) // chunk_size

        outputs = []

        for idx in range(num_chunks):
            #### fetch current trunk index
            chunk_start = chunk_size*idx
            chunk_ends = min(chunk_size*idx + chunk_size, q_heads.shape[1] )

            q_chunk = q_heads[:,:, chunk_start:chunk_ends,:]
            k_chunk = k_heads[:,:, :chunk_ends,:]
            v_chunk = v_heads[:,:, :chunk_ends,:]

            # Compute attention for this chunk
            chunk_output = self._standard_attention(q_chunk, k_chunk, v_chunk)
            outputs.append(chunk_output) ### each is (batch_szie, chunk_size, num_heads*d_dimension)

        return torch.cat(outputs, axis=1)


class FeedForward(nn.Module):
    '''
    Feed forward structure with GELU activation
    '''

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        self.embedding_d = config.n_embd

        self.dropout = config.dropout

        self.nn_dropout = nn.Dropout(self.dropout)

        self.hidden_d = 4*self.embedding_d

        self.linear1 = nn.Linear(self.embedding_d, self.hidden_d, bias = config.bias)
        self.linear2 = nn.Linear(self.hidden_d,self.embedding_d, bias=config.bias)



    def forward(self, inputs):

        inputs = F.gelu(self.linear1(inputs))
        inputs = self.nn_dropout(self.linear2(inputs))

        return inputs

class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Layer normalization (pre-norm style, like GPT-2)
        self.normal_layer1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.normal_layer2 = nn.LayerNorm(config.n_embd, bias=config.bias)

        # Multi-head attention
        self.attn = MultiHeadAttention(config)

        # Feed-forward network
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor):
        residual = x
        x_norm1 = self.normal_layer1(x)
        atten_output = self.attn(x_norm1)

        #### add and normalize
        x = atten_output + residual

        residual = x
        x_norm2 = self.normal_layer1(x)

        ### forward layer
        ffn_output = self.ffn(x_norm2)

        #### add and normalize
        x = ffn_output + residual

        return x

class SmallGPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embed = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks - each gets the same config
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])


        self.final_normal = nn.LayerNorm(config.n_embd, bias=config.bias)

        # Output projection (language modeling head)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using GPT-2 style initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None):
        # Ensure inputs are long type (required for embeddings, not affected by autocast)
        inputs = inputs.long()

        batch_szie, token_length =  inputs.shape

        position_ids = torch.arange(0, token_length, dtype=torch.long,
                                    device=inputs.device
                                    )
        embedded_tokens = self.token_embed(inputs)
        position_tokens = self.pos_embed(position_ids)

        x = self.dropout(embedded_tokens+position_tokens)

        for layer in self.blocks:
            x = layer(x)

        x = self.final_normal(x)

        # Language modeling head
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            # Ensure targets are long type (required for cross_entropy)
            targets = targets.long()
            # Flatten for cross-entropy calculation
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-1)

        return logits, loss

    def get_num_params(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def configure_optimizers(self, weight_decay: float, learning_rate: float):
        """Configure optimizer with proper weight decay groups from config."""
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                # Apply weight decay to weights but not biases or embeddings
                if param.dim() >= 2:  # Weights (2D or higher)
                    decay_params.append(param)
                else:  # Biases, embeddings (1D)
                    no_decay_params.append(param)

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))
        return optimizer


# Example usage showing config pattern
def create_model_from_config():
    #### this one credit to claude sonnet
    """Example of how config flows through the architecture."""

    # Create configuration
    config = GPTConfig(
        vocab_size=16384,
        n_layer=6,  # TransformerBlock will create 6 blocks
        n_head=6,  # MultiHeadAttention will use 6 heads
        n_embd=384,  # All components will use 384d embeddings
        block_size=512,  # All components respect 512 token limit
        dropout=0.1,  # All dropout layers use 0.1
        bias=False,  # All linear layers have no bias

        # MPS optimizations flow to attention
        attention_chunk_size=1024,
        memory_efficient_attention=True
    )

    # Config is passed down through the hierarchy:
    # SmallGPT(config)
    #   ├── TransformerBlock(config) [x6]
    #   │   ├── MultiHeadAttention(config)
    #   │   └── FeedForward(config)
    #   └── ... other components

    model = SmallGPT(config)

    print(f"Model created with {model.get_num_params():,} parameters")
    print(f"Using {config.n_layer} layers, {config.n_head} heads, {config.n_embd}d embeddings")

    return model, config








