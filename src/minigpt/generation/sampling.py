from collections.abc import Iterator

import torch
import torch.nn.functional as F

from minigpt.model import GPT


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        remove = probs.cumsum(dim=-1) - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


@torch.no_grad()
def generate_stream(
    model: GPT,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_id: int | None = None,
) -> Iterator[int]:
    if not prompt_ids:
        raise ValueError("prompt_ids must contain at least one token")
    budget = min(max_new_tokens, model.config.block_size - len(prompt_ids))
    if budget <= 0:
        raise ValueError(
            f"Prompt of {len(prompt_ids)} tokens leaves no room within block_size {model.config.block_size}"
        )

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        logits, past = model.forward_cached(idx)
        for step in range(budget):
            next_id = sample_next_token(logits[:, -1, :], temperature, top_k, top_p)
            token = next_id.item()
            yield token
            if token == eos_id or step == budget - 1:
                return
            logits, past = model.forward_cached(next_id, past)
    finally:
        model.train(was_training)


def generate(model: GPT, prompt_ids: list[int], max_new_tokens: int, **kwargs) -> list[int]:
    return list(generate_stream(model, prompt_ids, max_new_tokens, **kwargs))
