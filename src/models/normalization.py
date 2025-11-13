import torch
import torch.nn.functional as F
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps= 1e-6):
        super(RMSNorm, self).__init__()
        self.weights = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.sum(torch.pow(x,2), axis=-1, keepdim=True)/x.shape[-1] + self.eps)

        return x/rms*self.weights

    def extra_repr(self) -> str:
        return f'dim={self.weight.shape[0]}, eps={self.eps}'


class MPSRMSNorm(RMSNorm):
    """MPS-optimized RMSNorm with improved memory efficiency."""


    def __init__(self, dim: int, eps: float = 1e-6, use_float32_stats: bool = True):
        """
        Args:
            use_float32_stats: Compute RMS in float32 for stability
        """
        super().__init__(dim, eps)
        self.use_float32_stats = use_float32_stats

    def _compute_rms(self, x):
        if self.use_float32_stats and x.dtype != torch.float32:
            # Cast to float32 for statistics computation
            x_float32 = x.float()
            # Compute mean of squares
            # Use keepdim=True to maintain dimensions for broadcasting
            mean_sq = torch.mean(x_float32 ** 2, dim=-1, keepdim=True)
            # Compute RMS
            rms = torch.sqrt(mean_sq + self.eps)
            # Cast back to original dtype
            return rms.to(x.dtype)
        else:
            # Standard computation (if already float32 or not using mixed precision)
            mean_sq = torch.mean(x ** 2, dim=-1, keepdim=True)
            return torch.sqrt(mean_sq + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Even if x is float16, compute RMS in float32 for stability

        rms = self._compute_rms(x)

        # Normalize: x / rms (broadcasting rms over the feature dimension)
        normalized = x / rms

        if self.weights is not None:
            # Fused multiply - weight is shape (dim,), broadcasts correctly
            normalized = normalized * self.weights

        return normalized