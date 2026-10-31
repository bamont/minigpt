import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from minigpt.data.scoring import required_words, score_story
from minigpt.data.sft_dataset import IGNORE_INDEX, build_batch
from minigpt.data.sft_format import build_prompt, parse_example
from minigpt.generation import generate
from minigpt.model.checkpoint import load_model
from minigpt.tokenizer import BPETokenizer
from minigpt.tokenizer.train_tokenizer import iter_documents
from minigpt.training.dpo import sequence_logprobs

EOS_TOKEN = "<|endoftext|>"
LENGTH_BUCKETS = ((0, 150), (150, 250), (250, 10**6))


def load_prompts(path: Path, count: int) -> list[dict[str, str]]:
    prompts = []
    for document in iter_documents([path], EOS_TOKEN, None):
        parsed = parse_example(document)
        if parsed is not None and "Words" in parsed[0]:
            prompts.append(parsed[0])
            if len(prompts) == count:
                break
    return prompts


@torch.no_grad()
def judge_nll(
    judge: torch.nn.Module, prompt_ids: list[int], ids: list[int], pad_id: int, device: torch.device
) -> float:
    sequence = np.array(prompt_ids + ids, dtype=np.int64)
    inputs, targets = build_batch([sequence], [len(prompt_ids)], pad_id)
    inputs, targets = inputs.to(device), targets.to(device)
    logprob = sequence_logprobs(judge, inputs, targets).item()
    return -logprob / max(1, int((targets != IGNORE_INDEX).sum()))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Measure how well generated stories follow the requested words.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--judge-checkpoint", type=Path, default=Path("runs/sft/best.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer.json"))
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--details", type=Path, default=None)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    torch.manual_seed(args.seed)

    tokenizer = BPETokenizer.load(args.tokenizer)
    eos_id = tokenizer.special_tokens[EOS_TOKEN]
    model = load_model(args.checkpoint, device)
    judge = model if args.judge_checkpoint == args.checkpoint else load_model(args.judge_checkpoint, device)
    prompts = load_prompts(args.input, args.num_examples)

    records = []
    for count, fields in enumerate(prompts, start=1):
        words = required_words(fields["Words"])
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
        records.append(
            {
                "words": words,
                "exact": exact,
                "prefix": prefix,
                "tokens": len(ids),
                "finished": ended,
                "judge_nll": judge_nll(judge, prompt_ids, ids, eos_id, device),
                "story": story,
            }
        )
        if count % 10 == 0:
            print(f"{count}/{len(prompts)} done", flush=True)

    total = len(records)

    def rate(items: list[dict], key: str) -> float | None:
        return sum(r[key] == len(r["words"]) for r in items) / len(items) if items else None

    by_length = {}
    for low, high in LENGTH_BUCKETS:
        bucket = [r for r in records if low <= r["tokens"] < high]
        by_length[f"{low}-{high if high < 10**6 else 'inf'}"] = {
            "examples": len(bucket),
            "all_words_exact": rate(bucket, "exact"),
        }

    results = {
        "checkpoint": str(args.checkpoint),
        "judge_checkpoint": str(args.judge_checkpoint),
        "examples": total,
        "all_words_exact": rate(records, "exact"),
        "all_words_prefix": rate(records, "prefix"),
        "word_recall": sum(r["exact"] / len(r["words"]) for r in records) / total,
        "finished_with_eos": sum(r["finished"] for r in records) / total,
        "mean_generated_tokens": sum(r["tokens"] for r in records) / total,
        "mean_judge_nll": sum(r["judge_nll"] for r in records) / total,
        "by_length": by_length,
    }
    print(json.dumps(results, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.details:
        args.details.parent.mkdir(parents=True, exist_ok=True)
        with open(args.details, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
