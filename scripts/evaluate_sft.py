import argparse
import json
import re
import sys
from pathlib import Path

import torch

from minigpt.data.sft_format import build_prompt, parse_example
from minigpt.generation import generate
from minigpt.model.checkpoint import load_model
from minigpt.tokenizer import BPETokenizer
from minigpt.tokenizer.train_tokenizer import iter_documents

EOS_TOKEN = "<|endoftext|>"
WORD_PATTERN = re.compile(r"[a-z']+")


def load_prompts(path: Path, count: int) -> list[dict[str, str]]:
    prompts = []
    for document in iter_documents([path], EOS_TOKEN, None):
        parsed = parse_example(document)
        if parsed is not None and "Words" in parsed[0]:
            prompts.append(parsed[0])
            if len(prompts) == count:
                break
    return prompts


def score_story(story: str, words: list[str]) -> tuple[int, int]:
    vocabulary = set(WORD_PATTERN.findall(story.lower()))
    exact = sum(word in vocabulary for word in words)
    prefix = sum(any(token.startswith(word) for token in vocabulary) for word in words)
    return exact, prefix


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Measure how well generated stories follow the requested words.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer.json"))
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    torch.manual_seed(args.seed)

    tokenizer = BPETokenizer.load(args.tokenizer)
    eos_id = tokenizer.special_tokens[EOS_TOKEN]
    model = load_model(args.checkpoint, device)
    prompts = load_prompts(args.input, args.num_examples)

    all_exact = all_prefix = finished = 0
    recall_sum = 0.0
    token_sum = 0
    for count, fields in enumerate(prompts, start=1):
        words = [word.strip().lower() for word in fields["Words"].split(",") if word.strip()]
        prompt_ids = tokenizer.encode(build_prompt(fields))
        ids = generate(
            model,
            prompt_ids,
            args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_id=eos_id,
        )
        ended = bool(ids) and ids[-1] == eos_id
        story = tokenizer.decode(ids[:-1] if ended else ids)
        exact, prefix = score_story(story, words)
        all_exact += exact == len(words)
        all_prefix += prefix == len(words)
        recall_sum += exact / len(words)
        finished += ended
        token_sum += len(ids)
        if count % 10 == 0:
            print(f"{count}/{len(prompts)} done", flush=True)

    total = len(prompts)
    results = {
        "checkpoint": str(args.checkpoint),
        "examples": total,
        "all_words_exact": all_exact / total,
        "all_words_prefix": all_prefix / total,
        "word_recall": recall_sum / total,
        "finished_with_eos": finished / total,
        "mean_generated_tokens": token_sum / total,
    }
    print(json.dumps(results, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
