import unittest

import torch

from minigpt.generation import generate, generate_batch, sample_next_token
from minigpt.model import GPT, GPTConfig
from minigpt.model.rope import RotaryEmbedding

SMALL_CONFIG = GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=64)


class TestRotaryOffset(unittest.TestCase):
    def test_start_pos_matches_full_sequence(self):
        rope = RotaryEmbedding(head_dim=16, max_seq_len=32)
        x = torch.randn(1, 2, 10, 16)
        full = rope(x)
        part = rope(x[..., 4:, :], start_pos=4)
        self.assertTrue(torch.allclose(full[..., 4:, :], part, atol=1e-6))


class TestKVCache(unittest.TestCase):
    def test_cached_logits_match_full_forward(self):
        torch.manual_seed(0)
        model = GPT(SMALL_CONFIG).eval()
        idx = torch.randint(0, 64, (1, 12))
        with torch.no_grad():
            full_logits, _ = model(idx)
            logits, past = model.forward_cached(idx[:, :8])
            self.assertTrue(torch.allclose(logits, full_logits[:, :8], atol=1e-4))
            for position in range(8, 12):
                logits, past = model.forward_cached(idx[:, position : position + 1], past)
                self.assertTrue(
                    torch.allclose(logits[:, 0], full_logits[:, position], atol=1e-4)
                )

    def test_exceeding_block_size_raises(self):
        model = GPT(SMALL_CONFIG).eval()
        with self.assertRaises(ValueError):
            model.forward_cached(torch.randint(0, 64, (1, SMALL_CONFIG.block_size + 1)))


class TestSampling(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_zero_temperature_is_argmax(self):
        logits = torch.tensor([[0.1, 2.0, 0.5], [3.0, 0.0, 1.0]])
        result = sample_next_token(logits, temperature=0)
        self.assertEqual(result.tolist(), [[1], [0]])

    def test_output_shape(self):
        result = sample_next_token(torch.randn(4, 10))
        self.assertEqual(result.shape, (4, 1))

    def test_top_k_restricts_choices(self):
        logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]])
        draws = {sample_next_token(logits, top_k=3).item() for _ in range(300)}
        self.assertTrue(draws <= {0, 1, 2})

    def test_top_k_one_is_argmax(self):
        logits = torch.tensor([[1.0, 3.0, 2.0]])
        draws = {sample_next_token(logits, temperature=2.0, top_k=1).item() for _ in range(50)}
        self.assertEqual(draws, {1})

    def test_top_p_keeps_the_dominant_token(self):
        logits = torch.tensor([[10.0, 0.0, 0.0]])
        draws = {sample_next_token(logits, top_p=0.5).item() for _ in range(100)}
        self.assertEqual(draws, {0})


class TestGenerate(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = GPT(SMALL_CONFIG)

    def test_respects_max_new_tokens(self):
        out = generate(self.model, [1, 2, 3], max_new_tokens=7, temperature=1.0)
        self.assertEqual(len(out), 7)

    def test_stops_at_eos(self):
        first = generate(self.model, [1, 2, 3], max_new_tokens=5, temperature=0)[0]
        out = generate(self.model, [1, 2, 3], max_new_tokens=5, temperature=0, eos_id=first)
        self.assertEqual(out, [first])

    def test_limited_by_block_size(self):
        prompt = list(range(SMALL_CONFIG.block_size - 2))
        out = generate(self.model, prompt, max_new_tokens=10, temperature=1.0)
        self.assertEqual(len(out), 2)

    def test_prompt_too_long_raises(self):
        with self.assertRaises(ValueError):
            generate(self.model, list(range(SMALL_CONFIG.block_size)), max_new_tokens=5)

    def test_empty_prompt_raises(self):
        with self.assertRaises(ValueError):
            generate(self.model, [], max_new_tokens=5)

    def test_same_seed_gives_same_output(self):
        torch.manual_seed(42)
        first = generate(self.model, [1, 2], max_new_tokens=10, temperature=1.0)
        torch.manual_seed(42)
        second = generate(self.model, [1, 2], max_new_tokens=10, temperature=1.0)
        self.assertEqual(first, second)

    def test_restores_training_mode(self):
        self.model.train()
        generate(self.model, [1, 2], max_new_tokens=3)
        self.assertTrue(self.model.training)


class TestGenerateBatch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = GPT(SMALL_CONFIG)

    def test_greedy_batch_matches_single_generation(self):
        prompts = torch.randint(0, 64, (3, 5))
        batch = generate_batch(self.model, prompts, max_new_tokens=8, temperature=0)
        for row, prompt in zip(batch, prompts.tolist()):
            self.assertEqual(row, generate(self.model, prompt, max_new_tokens=8, temperature=0))

    def test_returns_one_sequence_per_prompt(self):
        prompts = torch.randint(0, 64, (4, 6))
        batch = generate_batch(self.model, prompts, max_new_tokens=5, temperature=1.0)
        self.assertEqual(len(batch), 4)
        self.assertTrue(all(len(row) == 5 for row in batch))

    def test_truncates_each_row_at_eos(self):
        prompts = torch.randint(0, 64, (3, 5))
        reference = generate_batch(self.model, prompts, max_new_tokens=6, temperature=0)
        eos_id = reference[0][0]
        batch = generate_batch(self.model, prompts, max_new_tokens=6, temperature=0, eos_id=eos_id)
        self.assertEqual(batch[0], [eos_id])
        for row in batch:
            self.assertTrue(row[-1] == eos_id or len(row) == 6)
            self.assertEqual(row.count(eos_id), 1 if row[-1] == eos_id else 0)

    def test_restores_training_mode(self):
        self.model.train()
        generate_batch(self.model, torch.randint(0, 64, (2, 4)), max_new_tokens=3)
        self.assertTrue(self.model.training)

    def test_prompt_too_long_raises(self):
        prompts = torch.randint(0, 64, (2, SMALL_CONFIG.block_size))
        with self.assertRaises(ValueError):
            generate_batch(self.model, prompts, max_new_tokens=5)


if __name__ == "__main__":
    unittest.main()
