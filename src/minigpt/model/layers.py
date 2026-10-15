import torch
import torch.nn.functional as F
from torch import nn

from minigpt.model.attention import CausalSelfAttention


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(n_embd, hidden_dim, bias=False)
        self.up = nn.Linear(n_embd, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(
        self, n_embd: int, n_head: int, hidden_dim: int, max_seq_len: int, dropout: float = 0.0
    ):
        super().__init__()
        self.attn_norm = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, max_seq_len, dropout)
        self.mlp_norm = RMSNorm(n_embd)
        self.mlp = SwiGLU(n_embd, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        x = x + self.dropout(self.mlp(self.mlp_norm(x)))
        return x
