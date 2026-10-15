import torch
import torch.nn.functional as F
from torch import nn

from minigpt.model.rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.dropout = dropout
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, n_embd = x.shape
        q, k, v = self.qkv(x).split(n_embd, dim=2)
        q, k, v = (
            t.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v)
        )
        q, k = self.rope(q), self.rope(k)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, n_embd)
        return self.proj(y)
