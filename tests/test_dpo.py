import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from minigpt.data.preference_dataset import PreferenceDataset
from minigpt.data.sft_dataset import IGNORE_INDEX
from minigpt.model import GPT, GPTConfig
from minigpt.training.dpo import DPOTrainer, dpo_loss, sequence_logprobs
from minigpt.training.trainer import TrainConfig

SMALL_CONFIG = GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=64)
PAD_ID = 63


def write_pairs(path: Path, pairs: list[tuple[list[int], list[int], list[int]]]) -> None:
    starts = [0]
    for prompt, chosen, rejected in pairs:
        starts.append(starts[-1] + len(prompt) + len(chosen) + len(rejected))
    np.savez(
        path,
        tokens=np.array([t for p, c, r in pairs for t in p + c + r], dtype=np.uint16),
        starts=np.array(starts, dtype=np.int64),
        prompt_lengths=np.array([len(p) for p, _, _ in pairs], dtype=np.int32),
        chosen_lengths=np.array([len(c) for _, c, _ in pairs], dtype=np.int32),
        rejected_lengths=np.array([len(r) for _, _, r in pairs], dtype=np.int32),
    )


class TestDPOLoss(unittest.TestCase):
    def test_identical_policy_and_reference_gives_log_two(self):
        zeros = torch.zeros(4)
        loss, accuracy, margin = dpo_loss(zeros, zeros, zeros, zeros, beta=0.1)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        self.assertEqual(accuracy.item(), 0.0)
        self.assertEqual(margin.item(), 0.0)

    def test_preferring_chosen_lowers_loss(self):
        zeros = torch.zeros(4)
        loss, accuracy, margin = dpo_loss(torch.full((4,), 5.0), zeros, zeros, zeros, beta=0.1)
        self.assertLess(loss.item(), math.log(2))
        self.assertEqual(accuracy.item(), 1.0)
        self.assertGreater(margin.item(), 0.0)

    def test_preferring_rejected_raises_loss(self):
        zeros = torch.zeros(4)
        loss, accuracy, _ = dpo_loss(zeros, torch.full((4,), 5.0), zeros, zeros, beta=0.1)
        self.assertGreater(loss.item(), math.log(2))
        self.assertEqual(accuracy.item(), 0.0)

    def test_reference_offsets_are_subtracted(self):
        policy_chosen, policy_rejected = torch.tensor([2.0]), torch.tensor([0.0])
        loss_same, _, _ = dpo_loss(policy_chosen, policy_rejected, policy_chosen, policy_rejected, beta=0.1)
        self.assertAlmostEqual(loss_same.item(), math.log(2), places=6)


class TestSequenceLogprobs(unittest.TestCase):
    def test_matches_masked_cross_entropy(self):
        torch.manual_seed(0)
        model = GPT(SMALL_CONFIG).eval()
        inputs = torch.randint(0, 62, (3, 10))
        targets = torch.randint(0, 62, (3, 10))
        targets[:, :4] = IGNORE_INDEX
        targets[2, 8:] = IGNORE_INDEX
        with torch.no_grad():
            logits, _ = model(inputs)
            expected = -F.cross_entropy(
                logits.transpose(1, 2), targets, ignore_index=IGNORE_INDEX, reduction="none"
            ).sum(dim=-1)
            result = sequence_logprobs(model, inputs, targets)
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))


class TestPreferenceDataset(unittest.TestCase):
    def test_collate_puts_chosen_rows_before_rejected_rows(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "pairs.npz"
            write_pairs(path, [([1, 2, 3], [4, 5], [6, 7, 8])])
            dataset = PreferenceDataset(path, PAD_ID)
        self.assertEqual(len(dataset), 1)
        x, y = dataset.collate([0])
        self.assertEqual(x.tolist(), [[1, 2, 3, 4, PAD_ID], [1, 2, 3, 6, 7]])
        self.assertEqual(
            y.tolist(),
            [
                [IGNORE_INDEX, IGNORE_INDEX, 4, 5, IGNORE_INDEX],
                [IGNORE_INDEX, IGNORE_INDEX, 6, 7, 8],
            ],
        )

    def test_get_batch_returns_chosen_and_rejected_rows(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "pairs.npz"
            write_pairs(path, [([1, 2], [3, 4], [5, 6]), ([7, 8], [9, 10], [11, 12])])
            dataset = PreferenceDataset(path, PAD_ID)
        x, y = dataset.get_batch(3, torch.device("cpu"))
        self.assertEqual(x.size(0), 6)
        self.assertEqual(x.shape, y.shape)


class TestDPOTrainer(unittest.TestCase):
    def test_training_reduces_the_loss(self):
        rng = np.random.default_rng(0)
        pairs = [
            (
                rng.integers(0, 60, 4).tolist(),
                rng.integers(0, 60, 6).tolist(),
                rng.integers(0, 60, 6).tolist(),
            )
            for _ in range(16)
        ]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "pairs.npz"
            write_pairs(path, pairs)
            dataset = PreferenceDataset(path, PAD_ID)
            torch.manual_seed(0)
            policy = GPT(SMALL_CONFIG)
            reference = GPT(SMALL_CONFIG)
            reference.load_state_dict(policy.state_dict())
            config = TrainConfig(
                out_dir=str(Path(tmp) / "run"),
                max_steps=40,
                batch_size=4,
                grad_accum_steps=1,
                learning_rate=1e-3,
                warmup_steps=2,
                eval_interval=20,
                eval_iters=5,
                log_interval=10,
                checkpoint_interval=20,
                device="cpu",
                dtype="float32",
            )
            DPOTrainer(policy, reference, config, 0.1, dataset.get_batch, dataset.get_batch).fit()
            run_dir = Path(tmp) / "run"
            lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            evals = [r for r in map(json.loads, lines) if r["type"] == "eval"]
            self.assertTrue((run_dir / "best.pt").exists())
        self.assertAlmostEqual(evals[0]["val_loss"], math.log(2), places=4)
        self.assertLess(evals[-1]["val_loss"], evals[0]["val_loss"])


if __name__ == "__main__":
    unittest.main()
