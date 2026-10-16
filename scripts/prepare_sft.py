import argparse
from collections import Counter
from itertools import islice
from pathlib import Path

import numpy as np

from minigpt.data.sft_format import build_prompt, parse_example
from minigpt.tokenizer import BPETokenizer
from minigpt.tokenizer.train_tokenizer import iter_documents

EOS_TOKEN = "<|endoftext|>"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize an instruction corpus for SFT.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=200000)
    parser.add_argument("--block-size", type=int, default=512)
    args = parser.parse_args()

    tokenizer = BPETokenizer.load(args.tokenizer)
    eos_id = tokenizer.special_tokens[EOS_TOKEN]

    chunks: list[np.ndarray] = []
    starts = [0]
    prompt_lengths: list[int] = []
    skipped: Counter[str] = Counter()

    documents = islice(iter_documents([args.input], EOS_TOKEN, None), args.max_examples)
    for document in documents:
        parsed = parse_example(document)
        if parsed is None:
            skipped["malformed"] += 1
            continue
        fields, story = parsed
        prompt_ids = tokenizer.encode(build_prompt(fields))
        response_ids = tokenizer.encode(story) + [eos_id]
        if len(prompt_ids) + len(response_ids) > args.block_size:
            skipped["too_long"] += 1
            continue
        chunks.append(np.array(prompt_ids + response_ids, dtype=np.uint16))
        starts.append(starts[-1] + len(prompt_ids) + len(response_ids))
        prompt_lengths.append(len(prompt_ids))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        tokens=np.concatenate(chunks),
        starts=np.array(starts, dtype=np.int64),
        prompt_lengths=np.array(prompt_lengths, dtype=np.int32),
    )

    total_tokens = starts[-1]
    print(f"Kept {len(chunks):,} examples ({total_tokens:,} tokens, {total_tokens / len(chunks):.0f} per example)")
    print(f"Skipped: {dict(skipped)}")


if __name__ == "__main__":
    main()
