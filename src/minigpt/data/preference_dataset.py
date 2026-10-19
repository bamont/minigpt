import numpy as np
import torch

from minigpt.data.sft_dataset import build_batch


class PreferenceDataset:
    def __init__(self, path, pad_id: int):
        with np.load(path) as data:
            self.tokens = data["tokens"]
            self.starts = data["starts"]
            self.prompt_lengths = data["prompt_lengths"]
            self.chosen_lengths = data["chosen_lengths"]
            self.rejected_lengths = data["rejected_lengths"]
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.prompt_lengths)

    def _split(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        start = int(self.starts[index])
        prompt_end = start + int(self.prompt_lengths[index])
        chosen_end = prompt_end + int(self.chosen_lengths[index])
        stop = chosen_end + int(self.rejected_lengths[index])
        tokens = self.tokens[start:stop].astype(np.int64)
        return tokens[: prompt_end - start], tokens[prompt_end - start : chosen_end - start], tokens[chosen_end - start :]

    def collate(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        chosen_sequences, rejected_sequences, prompt_lengths = [], [], []
        for index in indices:
            prompt, chosen, rejected = self._split(index)
            chosen_sequences.append(np.concatenate([prompt, chosen]))
            rejected_sequences.append(np.concatenate([prompt, rejected]))
            prompt_lengths.append(len(prompt))
        return build_batch(
            chosen_sequences + rejected_sequences, prompt_lengths + prompt_lengths, self.pad_id
        )

    def get_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(len(self), (batch_size,)).tolist()
        x, y = self.collate(indices)
        return x.to(device), y.to(device)
