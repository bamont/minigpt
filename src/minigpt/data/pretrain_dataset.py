import numpy as np
import torch


class PretrainDataset:
    def __init__(self, path, block_size: int):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        if len(self.tokens) <= block_size + 1:
            raise ValueError(
                f"Token file has {len(self.tokens)} tokens, need more than block_size + 1 ({block_size + 1})"
            )

    def __len__(self) -> int:
        return len(self.tokens)

    def get_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(len(self.tokens) - self.block_size - 1, (batch_size,)).tolist()
        x = np.stack([self.tokens[s : s + self.block_size] for s in starts])
        y = np.stack([self.tokens[s + 1 : s + 1 + self.block_size] for s in starts])
        x = torch.from_numpy(x.astype(np.int64))
        y = torch.from_numpy(y.astype(np.int64))
        if device.type == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)
