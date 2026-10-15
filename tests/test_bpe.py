import tempfile
import unittest
from pathlib import Path

from minigpt.tokenizer import BPETokenizer

CORPUS = [
    "Once upon a time, there was a little girl named Lily. She loved to play in the garden.",
    "One day, Lily found a shiny red ball under a big tree. She was so happy!",
    "Her friend Tom came over and they played together until the sun went down.\n\nThe end.",
    "The dog barked at the cat, and the cat ran up the tallest tree it could find.",
    "Mom said, \"It's time to go home.\" They didn't want to leave, but they were tired.",
] * 20

VOCAB_SIZE = 300


class TestBPETokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = BPETokenizer.train(CORPUS, VOCAB_SIZE)

    def test_vocab_size(self):
        self.assertEqual(self.tokenizer.vocab_size, VOCAB_SIZE)

    def test_roundtrip(self):
        samples = [
            "",
            "Hello, world!",
            "Once upon a time, there was a little girl.",
            "  leading and trailing spaces  ",
            "line one\nline two\n\n\nline three",
            "caf\u00e9 \u00e9t\u00e9 na\u00efve",
            "\u4f60\u597d\uff0c\u4e16\u754c",
            "emoji \U0001f642 test",
            "numbers 12345 and symbols @#$%",
        ]
        for text in samples:
            with self.subTest(text=text):
                self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_compression(self):
        text = CORPUS[0]
        self.assertLess(len(self.tokenizer.encode(text)), len(text.encode("utf-8")))

    def test_special_tokens(self):
        eos_id = self.tokenizer.special_tokens["<|endoftext|>"]
        text = "The end.<|endoftext|>Next story."
        with_special = self.tokenizer.encode(text, allow_special=True)
        without_special = self.tokenizer.encode(text)
        self.assertEqual(with_special.count(eos_id), 1)
        self.assertNotIn(eos_id, without_special)
        self.assertEqual(self.tokenizer.decode(with_special), text)
        self.assertEqual(self.tokenizer.decode(without_special), text)

    def test_special_token_ids_follow_merges(self):
        self.assertEqual(self.tokenizer.special_tokens["<|endoftext|>"], VOCAB_SIZE - 1)

    def test_training_is_deterministic(self):
        other = BPETokenizer.train(CORPUS, VOCAB_SIZE)
        self.assertEqual(self.tokenizer.merges, other.merges)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            self.tokenizer.save(path)
            loaded = BPETokenizer.load(path)
        text = "Lily and Tom played in the garden.<|endoftext|>"
        self.assertEqual(loaded.merges, self.tokenizer.merges)
        self.assertEqual(loaded.special_tokens, self.tokenizer.special_tokens)
        self.assertEqual(loaded.encode(text, allow_special=True), self.tokenizer.encode(text, allow_special=True))

    def test_invalid_utf8_is_replaced(self):
        self.assertEqual(self.tokenizer.decode([0xFF]), "\ufffd")

    def test_unknown_id_raises(self):
        with self.assertRaises(ValueError):
            self.tokenizer.decode([VOCAB_SIZE + 10])

    def test_vocab_size_too_small_raises(self):
        with self.assertRaises(ValueError):
            BPETokenizer.train(CORPUS, vocab_size=100)

    def test_special_tokens_are_not_merged_during_training(self):
        corpus = ["story one<|endoftext|>story two<|endoftext|>"] * 50
        tokenizer = BPETokenizer.train(corpus, vocab_size=280)
        for token_bytes in tokenizer.vocab.values():
            self.assertNotIn(b"<|", token_bytes)


if __name__ == "__main__":
    unittest.main()
