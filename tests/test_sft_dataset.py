import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from minigpt.data.sft_dataset import IGNORE_INDEX, SFTDataset

PAD_ID = 99


class TestSFTDataset(unittest.TestCase):
    def setUp(self):
        sequences = [[10, 11, 12, 13, 14], [20, 21, 22]]
        prompt_lengths = [3, 2]
        starts = np.cumsum([0] + [len(seq) for seq in sequences])
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "sft.npz"
            np.savez(
                path,
                tokens=np.array([t for seq in sequences for t in seq], dtype=np.uint16),
                starts=starts.astype(np.int64),
                prompt_lengths=np.array(prompt_lengths, dtype=np.int32),
            )
            self.dataset = SFTDataset(path, PAD_ID)

    def test_length(self):
        self.assertEqual(len(self.dataset), 2)

    def test_inputs_are_shifted_and_padded(self):
        x, _ = self.dataset.collate([0, 1])
        self.assertEqual(x.tolist(), [[10, 11, 12, 13], [20, 21, PAD_ID, PAD_ID]])

    def test_prompt_and_padding_are_masked_in_targets(self):
        _, y = self.dataset.collate([0, 1])
        self.assertEqual(
            y.tolist(),
            [
                [IGNORE_INDEX, IGNORE_INDEX, 13, 14],
                [IGNORE_INDEX, 22, IGNORE_INDEX, IGNORE_INDEX],
            ],
        )

    def test_get_batch_shapes(self):
        x, y = self.dataset.get_batch(4, torch.device("cpu"))
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(x.size(0), 4)
        self.assertEqual(x.dtype, torch.int64)


if __name__ == "__main__":
    unittest.main()
