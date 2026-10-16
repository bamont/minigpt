import torch
import torch.nn.functional as F
from torch import nn

from minigpt.model.rope import RotaryEmbedding

KVCache = tuple[torch.Tensor, torch.Tensor]


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

    def _project(self, x: torch.Tensor, start_pos: int) -> tuple[torch.Tensor, ...]:
        batch, seq_len, n_embd = x.shape
        q, k, v = self.qkv(x).split(n_embd, dim=2)
        q, k, v = (
            t.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v)
        )
        return self.rope(q, start_pos), self.rope(k, start_pos), v

    def _merge(self, y: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = y.shape
        return self.proj(y.transpose(1, 2).contiguous().view(batch, seq_len, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._project(x, 0)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        return self._merge(y)

    def forward_cached(
        self, x: torch.Tensor, past_kv: KVCache | None = None
    ) -> tuple[torch.Tensor, KVCache]:
        start_pos = 0 if past_kv is None else past_kv[0].size(2)
        q, k, v = self._project(x, start_pos)
        if past_kv is not None:
            if x.size(1) != 1:
                raise ValueError("Cached decoding processes one token at a time")
            k = torch.cat((past_kv[0], k), dim=2)
            v = torch.cat((past_kv[1], v), dim=2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=past_kv is None)
        return self._merge(y), (k, v)
