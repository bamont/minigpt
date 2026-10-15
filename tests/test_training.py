import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from minigpt.data.pretrain_dataset import PretrainDataset
from minigpt.model import GPT, GPTConfig
from minigpt.training.trainer import TrainConfig, Trainer, get_lr

MODEL_CONFIG = GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=64)


class TestSchedule(unittest.TestCase):
    def setUp(self):
        self.config = TrainConfig(
            max_steps=100, warmup_steps=10, learning_rate=1e-3, min_lr_ratio=0.1
        )

    def test_warmup_is_linear(self):
        self.assertAlmostEqual(get_lr(0, self.config), 1e-4)
        self.assertAlmostEqual(get_lr(9, self.config), 1e-3)

    def test_peak_after_warmup(self):
        self.assertAlmostEqual(get_lr(10, self.config), 1e-3)

    def test_decays_monotonically_to_min(self):
        lrs = [get_lr(step, self.config) for step in range(10, 101)]
        self.assertTrue(all(a >= b for a, b in zip(lrs, lrs[1:])))
        self.assertAlmostEqual(lrs[-1], 1e-4)

    def test_stays_at_min_after_max_steps(self):
        self.assertAlmostEqual(get_lr(1000, self.config), 1e-4)


class TestPretrainDataset(unittest.TestCase):
    def test_batch_shapes_and_shift(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "tokens.bin"
            np.arange(1000, dtype=np.uint16).tofile(path)
            dataset = PretrainDataset(path, block_size=8)
            x, y = dataset.get_batch(4, torch.device("cpu"))
        self.assertEqual(x.shape, (4, 8))
        self.assertEqual(x.dtype, torch.int64)
        self.assertTrue(torch.equal(y, x + 1))

    def test_too_few_tokens_raises(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "tokens.bin"
            np.arange(5, dtype=np.uint16).tofile(path)
            with self.assertRaises(ValueError):
                PretrainDataset(path, block_size=8)


class TestTrainer(unittest.TestCase):
    def make_trainer(self, tmp: str, max_steps: int) -> Trainer:
        path = Path(tmp) / "tokens.bin"
        if not path.exists():
            (np.arange(4000) % 64).astype(np.uint16).tofile(path)
        dataset = PretrainDataset(path, MODEL_CONFIG.block_size)
        config = TrainConfig(
            out_dir=str(Path(tmp) / "run"),
            max_steps=max_steps,
            batch_size=16,
            grad_accum_steps=2,
            learning_rate=3e-3,
            warmup_steps=5,
            eval_interval=20,
            eval_iters=5,
            log_interval=10,
            checkpoint_interval=10,
            device="cpu",
            dtype="float32",
        )
        torch.manual_seed(0)
        return Trainer(GPT(MODEL_CONFIG), config, dataset.get_batch, dataset.get_batch)

    @staticmethod
    def read_evals(run_dir: Path) -> list[dict]:
        lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines]
        return [r for r in records if r["type"] == "eval"]

    def test_training_reduces_loss_and_writes_artifacts(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            self.make_trainer(tmp, 60).fit()
            run_dir = Path(tmp) / "run"
            self.assertTrue((run_dir / "last.pt").exists())
            self.assertTrue((run_dir / "best.pt").exists())
            evals = self.read_evals(run_dir)
        self.assertLess(evals[-1]["val_loss"], evals[0]["val_loss"])

    def test_resume_continues_from_checkpoint(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            self.make_trainer(tmp, 20).fit()
            self.make_trainer(tmp, 40).fit(resume=True)
            run_dir = Path(tmp) / "run"
            checkpoint = torch.load(run_dir / "last.pt", weights_only=True)
            steps = [e["step"] for e in self.read_evals(run_dir)]
        self.assertEqual(checkpoint["step"], 40)
        self.assertEqual(steps.count(20), 1)


if __name__ == "__main__":
    unittest.main()
