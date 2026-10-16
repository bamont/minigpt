from pathlib import Path

import torch

from minigpt.model.gpt import GPT, GPTConfig


def load_model(path: str | Path, device: torch.device) -> GPT:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()
