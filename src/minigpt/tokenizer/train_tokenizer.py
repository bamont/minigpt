import argparse
from collections.abc import Iterator
from pathlib import Path

from minigpt.tokenizer.bpe import DEFAULT_SPECIAL_TOKENS, BPETokenizer

READ_BLOCK_SIZE = 1 << 20


def iter_documents(paths: list[Path], delimiter: str, max_chars: int | None) -> Iterator[str]:
    seen = 0
    for path in paths:
        carry = ""
        with open(path, encoding="utf-8") as f:
            while block := f.read(READ_BLOCK_SIZE):
                *documents, carry = (carry + block).split(delimiter)
                for document in documents:
                    yield document
                    seen += len(document)
                    if max_chars and seen >= max_chars:
                        return
        if carry:
            yield carry
            seen += len(carry)
            if max_chars and seen >= max_chars:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tokenizer.json"))
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--special-tokens", nargs="+", default=list(DEFAULT_SPECIAL_TOKENS))
    args = parser.parse_args()

    documents = iter_documents(args.input, args.special_tokens[0], args.max_chars)
    tokenizer = BPETokenizer.train(documents, args.vocab_size, args.special_tokens)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.output)
    print(f"Saved tokenizer with {tokenizer.vocab_size} tokens to {args.output}")


if __name__ == "__main__":
    main()
