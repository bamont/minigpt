import numpy as np
import torch

IGNORE_INDEX = -100


def build_batch(
    sequences: list[np.ndarray], prompt_lengths: list[int], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(seq) for seq in sequences) - 1
    inputs = np.full((len(sequences), width), pad_id, dtype=np.int64)
    targets = np.full((len(sequences), width), IGNORE_INDEX, dtype=np.int64)
    for row, (seq, prompt_length) in enumerate(zip(sequences, prompt_lengths)):
        length = len(seq) - 1
        row_targets = seq[1:].copy()
        row_targets[: prompt_length - 1] = IGNORE_INDEX
        inputs[row, :length] = seq[:-1]
        targets[row, :length] = row_targets
    return torch.from_numpy(inputs), torch.from_numpy(targets)


class SFTDataset:
    def __init__(self, path, pad_id: int):
        with np.load(path) as data:
            self.tokens = data["tokens"]
            self.starts = data["starts"]
            self.prompt_lengths = data["prompt_lengths"]
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.prompt_lengths)

    def collate(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = [
            self.tokens[self.starts[i] : self.starts[i + 1]].astype(np.int64) for i in indices
        ]
        prompt_lengths = [int(self.prompt_lengths[i]) for i in indices]
        return build_batch(sequences, prompt_lengths, self.pad_id)

    def get_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(len(self), (batch_size,)).tolist()
        x, y = self.collate(indices)
        return x.to(device), y.to(device)
