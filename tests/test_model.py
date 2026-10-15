import math
import unittest

import torch
import torch.nn.functional as F

from minigpt.model import GPT, GPTConfig
from minigpt.model.rope import RotaryEmbedding

SMALL_CONFIG = GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=64)


class TestRotaryEmbedding(unittest.TestCase):
    def test_preserves_norm(self):
        rope = RotaryEmbedding(head_dim=16, max_seq_len=32)
        x = torch.randn(2, 4, 10, 16)
        self.assertTrue(torch.allclose(rope(x).norm(dim=-1), x.norm(dim=-1), atol=1e-5))

    def test_scores_depend_on_relative_position_only(self):
        rope = RotaryEmbedding(head_dim=16, max_seq_len=32)
        q = torch.randn(1, 1, 1, 16).expand(1, 1, 10, 16)
        k = torch.randn(1, 1, 1, 16).expand(1, 1, 10, 16)
        scores = rope(q) @ rope(k).transpose(-1, -2)
        self.assertTrue(torch.allclose(scores[..., :-1, :-1], scores[..., 1:, 1:], atol=1e-5))

    def test_odd_head_dim_raises(self):
        with self.assertRaises(ValueError):
            RotaryEmbedding(head_dim=15, max_seq_len=8)


class TestGPT(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_output_shapes(self):
        model = GPT(SMALL_CONFIG)
        idx = torch.randint(0, 64, (3, 16))
        logits, loss = model(idx, idx)
        self.assertEqual(logits.shape, (3, 16, 64))
        self.assertEqual(loss.shape, ())

    def test_no_loss_without_targets(self):
        model = GPT(SMALL_CONFIG)
        _, loss = model(torch.randint(0, 64, (1, 8)))
        self.assertIsNone(loss)

    def test_initial_loss_is_close_to_uniform(self):
        model = GPT(SMALL_CONFIG)
        idx = torch.randint(0, 64, (8, 32))
        targets = torch.randint(0, 64, (8, 32))
        _, loss = model(idx, targets)
        self.assertAlmostEqual(loss.item(), math.log(64), delta=0.5)

    def test_ignore_index_masks_targets(self):
        model = GPT(SMALL_CONFIG)
        idx = torch.randint(0, 64, (2, 16))
        targets = idx.clone()
        targets[:, :8] = -100
        logits, masked_loss = model(idx, targets)
        expected = F.cross_entropy(logits[:, 8:].reshape(-1, 64), idx[:, 8:].reshape(-1))
        self.assertAlmostEqual(masked_loss.item(), expected.item(), places=5)

    def test_is_causal(self):
        model = GPT(SMALL_CONFIG).eval()
        idx = torch.randint(0, 64, (1, 16))
        changed = idx.clone()
        changed[0, 10:] = (changed[0, 10:] + 1) % 64
        with torch.no_grad():
            logits, _ = model(idx)
            changed_logits, _ = model(changed)
        self.assertTrue(torch.allclose(logits[:, :10], changed_logits[:, :10], atol=1e-5))

    def test_weights_are_tied(self):
        model = GPT(SMALL_CONFIG)
        self.assertIs(model.lm_head.weight, model.tok_emb.weight)

    def test_too_long_sequence_raises(self):
        model = GPT(SMALL_CONFIG)
        with self.assertRaises(ValueError):
            model(torch.randint(0, 64, (1, SMALL_CONFIG.block_size + 1)))

    def test_default_config_parameter_count(self):
        count = GPT(GPTConfig()).num_parameters()
        self.assertGreater(count, 20_000_000)
        self.assertLess(count, 50_000_000)

    def test_overfits_a_tiny_batch(self):
        model = GPT(SMALL_CONFIG)
        data = torch.randint(0, 64, (4, 33))
        inputs, targets = data[:, :-1], data[:, 1:]
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        _, initial_loss = model(inputs, targets)
        for _ in range(300):
            _, loss = model(inputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), 1.0)
        self.assertLess(loss.item(), initial_loss.item() * 0.25)


if __name__ == "__main__":
    unittest.main()
