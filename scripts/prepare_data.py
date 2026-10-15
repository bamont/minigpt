import argparse
import os
from collections import deque
from itertools import islice
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from minigpt.tokenizer import BPETokenizer
from minigpt.tokenizer.train_tokenizer import iter_documents

EOS_TOKEN = "<|endoftext|>"

_tokenizer = None
_eos_id = None


def init_worker(tokenizer_path: str) -> None:
    global _tokenizer, _eos_id
    _tokenizer = BPETokenizer.load(tokenizer_path)
    _eos_id = _tokenizer.special_tokens[EOS_TOKEN]


def encode_batch(documents: list[str]) -> np.ndarray:
    ids = []
    for document in documents:
        document = document.strip()
        if document:
            ids.extend(_tokenizer.encode(document))
            ids.append(_eos_id)
    return np.array(ids, dtype=np.uint16)


def batched(iterable, size: int):
    iterator = iter(iterable)
    while batch := list(islice(iterator, size)):
        yield batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode a text corpus into a flat uint16 token file.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--batch-docs", type=int, default=2000)
    parser.add_argument("--max-chars", type=int, default=None)
    args = parser.parse_args()

    tokenizer = BPETokenizer.load(args.tokenizer)
    if tokenizer.vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError("Vocabulary is too large for uint16 storage")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    documents = iter_documents([args.input], EOS_TOKEN, args.max_chars)

    total_tokens = 0
    pending = deque()
    max_pending = args.workers * 4

    with open(args.output, "wb") as out, Pool(
        args.workers, init_worker, (str(args.tokenizer),)
    ) as pool:

        def write_next() -> int:
            array = pending.popleft().get()
            array.tofile(out)
            return array.size

        for batch in batched(documents, args.batch_docs):
            pending.append(pool.apply_async(encode_batch, (batch,)))
            if len(pending) >= max_pending:
                total_tokens += write_next()
        while pending:
            total_tokens += write_next()

    print(f"Wrote {total_tokens:,} tokens to {args.output}")


if __name__ == "__main__":
    main()
