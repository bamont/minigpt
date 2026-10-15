import heapq
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import regex

SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
DEFAULT_SPECIAL_TOKENS = ("<|endoftext|>",)
NUM_BYTE_TOKENS = 256

Pair = tuple[int, int]


def merge_pair(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    merged = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


def _special_regex(tokens: Iterable[str], capture: bool) -> regex.Pattern | None:
    escaped = "|".join(regex.escape(token) for token in tokens)
    if not escaped:
        return None
    return regex.compile(f"({escaped})" if capture else escaped)


class BPETokenizer:
    def __init__(
        self,
        merges: Iterable[Pair] = (),
        special_tokens: Iterable[str] = DEFAULT_SPECIAL_TOKENS,
        pattern: str = SPLIT_PATTERN,
    ):
        self.pattern = pattern
        self._split = regex.compile(pattern)

        self.merges: dict[Pair, int] = {}
        for pair in merges:
            self.merges[tuple(pair)] = NUM_BYTE_TOKENS + len(self.merges)

        self.special_tokens: dict[str, int] = {}
        for token in dict.fromkeys(special_tokens):
            self.special_tokens[token] = NUM_BYTE_TOKENS + len(self.merges) + len(self.special_tokens)

        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(NUM_BYTE_TOKENS)}
        for (left, right), idx in self.merges.items():
            self.vocab[idx] = self.vocab[left] + self.vocab[right]

        self._id_to_special = {idx: token for token, idx in self.special_tokens.items()}
        self._special_split = _special_regex(self.special_tokens, capture=True)
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return NUM_BYTE_TOKENS + len(self.merges) + len(self.special_tokens)

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        special_tokens: Iterable[str] = DEFAULT_SPECIAL_TOKENS,
        pattern: str = SPLIT_PATTERN,
    ) -> "BPETokenizer":
        special_tokens = tuple(dict.fromkeys(special_tokens))
        num_merges = vocab_size - NUM_BYTE_TOKENS - len(special_tokens)
        if num_merges < 0:
            raise ValueError(
                f"vocab_size must be at least {NUM_BYTE_TOKENS + len(special_tokens)}, got {vocab_size}"
            )

        splitter = regex.compile(pattern)
        special_re = _special_regex(special_tokens, capture=False)

        chunk_counts: Counter[str] = Counter()
        for text in texts:
            segments = special_re.split(text) if special_re else [text]
            for segment in segments:
                chunk_counts.update(splitter.findall(segment))

        words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        freqs = list(chunk_counts.values())

        pair_counts: dict[Pair, int] = defaultdict(int)
        pair_to_words: dict[Pair, set[int]] = defaultdict(set)
        for i, word in enumerate(words):
            for pair in zip(word, word[1:]):
                pair_counts[pair] += freqs[i]
                pair_to_words[pair].add(i)

        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        merges: list[Pair] = []
        while len(merges) < num_merges and heap:
            neg_count, best = heapq.heappop(heap)
            if pair_counts.get(best, 0) != -neg_count:
                continue

            new_id = NUM_BYTE_TOKENS + len(merges)
            merges.append(best)

            touched: set[Pair] = set()
            for i in list(pair_to_words[best]):
                word, freq = words[i], freqs[i]
                for pair in zip(word, word[1:]):
                    pair_counts[pair] -= freq
                    pair_to_words[pair].discard(i)
                    touched.add(pair)
                word = merge_pair(word, best, new_id)
                words[i] = word
                for pair in zip(word, word[1:]):
                    pair_counts[pair] += freq
                    pair_to_words[pair].add(i)
                    touched.add(pair)

            for pair in touched:
                if pair_counts[pair] > 0:
                    heapq.heappush(heap, (-pair_counts[pair], pair))
                else:
                    del pair_counts[pair]
                    pair_to_words.pop(pair, None)

        return cls(merges, special_tokens, pattern)

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge_pair(ids, pair, self.merges[pair])
        self._cache[chunk] = ids
        return ids

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        if allow_special and self._special_split:
            segments = self._special_split.split(text)
        else:
            segments = [text]

        ids: list[int] = []
        for segment in segments:
            if allow_special and segment in self.special_tokens:
                ids.append(self.special_tokens[segment])
                continue
            for chunk in self._split.findall(segment):
                ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        parts = []
        for idx in ids:
            if idx in self.vocab:
                parts.append(self.vocab[idx])
            elif idx in self._id_to_special:
                parts.append(self._id_to_special[idx].encode("utf-8"))
            else:
                raise ValueError(f"Unknown token id: {idx}")
        return b"".join(parts).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        data = {
            "pattern": self.pattern,
            "merges": [list(pair) for pair in self.merges],
            "special_tokens": list(self.special_tokens),
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([tuple(pair) for pair in data["merges"]], data["special_tokens"], data["pattern"])
