import argparse
import json
from itertools import islice
from pathlib import Path

from minigpt.data.sft_format import parse_example
from minigpt.tokenizer.train_tokenizer import iter_documents

EOS_TOKEN = "<|endoftext|>"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract instruction prompts that contain required words.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/dpo_prompts.jsonl"))
    parser.add_argument("--skip", type=int, default=200000)
    parser.add_argument("--count", type=int, default=20000)
    args = parser.parse_args()

    documents = islice(iter_documents([args.input], EOS_TOKEN, None), args.skip, None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for document in documents:
            parsed = parse_example(document)
            if parsed is None or "Words" not in parsed[0]:
                continue
            f.write(json.dumps(parsed[0]) + "\n")
            written += 1
            if written == args.count:
                break
    print(f"Wrote {written:,} prompts to {args.output}")


if __name__ == "__main__":
    main()
