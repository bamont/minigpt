import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from minigpt.data.preference_pairs import Sample, select_pair
from minigpt.data.scoring import required_words, score_story
from minigpt.data.sft_format import build_prompt
from minigpt.generation import generate_batch
from minigpt.model.checkpoint import load_model
from minigpt.tokenizer import BPETokenizer

EOS_TOKEN = "<|endoftext|>"


def save_pairs(path: Path, pairs: list[tuple[list[int], list[int], list[int]]]) -> None:
    starts = [0]
    for prompt, chosen, rejected in pairs:
        starts.append(starts[-1] + len(prompt) + len(chosen) + len(rejected))
    np.savez(
        path,
        tokens=np.array([t for prompt, chosen, rejected in pairs for t in prompt + chosen + rejected], dtype=np.uint16),
        starts=np.array(starts, dtype=np.int64),
        prompt_lengths=np.array([len(p) for p, _, _ in pairs], dtype=np.int32),
        chosen_lengths=np.array([len(c) for _, c, _ in pairs], dtype=np.int32),
        rejected_lengths=np.array([len(r) for _, _, r in pairs], dtype=np.int32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample stories and build preference pairs from word coverage.")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/sft/best.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-length-ratio", type=float, default=2.0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    tokenizer = BPETokenizer.load(args.tokenizer)
    eos_id = tokenizer.special_tokens[EOS_TOKEN]
    model = load_model(args.checkpoint, device)

    with open(args.prompts, encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f]
    encoded = [tokenizer.encode(build_prompt(fields)) for fields in prompts]
    buckets: dict[int, list[int]] = defaultdict(list)
    for index, ids in enumerate(encoded):
        buckets[len(ids)].append(index)

    prompts_per_batch = max(1, args.batch_size // args.samples_per_prompt)
    pairs: list[tuple[list[int], list[int], list[int]]] = []
    chosen_scores, rejected_scores, chosen_lengths, rejected_lengths = [], [], [], []
    all_words_samples = total_samples = no_pair = processed = batches = 0

    for length in sorted(buckets):
        indices = buckets[length]
        for start in range(0, len(indices), prompts_per_batch):
            chunk = indices[start : start + prompts_per_batch]
            batch = torch.tensor(
                [encoded[i] for i in chunk for _ in range(args.samples_per_prompt)], device=device
            )
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                outputs = generate_batch(
                    model,
                    batch,
                    args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    eos_id=eos_id,
                )

            for position, index in enumerate(chunk):
                words = required_words(prompts[index]["Words"])
                samples = []
                for ids in outputs[position * args.samples_per_prompt : (position + 1) * args.samples_per_prompt]:
                    finished = ids[-1] == eos_id
                    story = tokenizer.decode(ids[:-1] if finished else ids)
                    samples.append(Sample(ids, finished, score_story(story, words)[0]))
                all_words_samples += sum(s.score == len(words) for s in samples)
                total_samples += len(samples)

                pair = select_pair(samples, rng, args.max_length_ratio)
                if pair is None:
                    no_pair += 1
                    continue
                chosen, rejected = pair
                pairs.append((encoded[index], chosen.ids, rejected.ids))
                chosen_scores.append(chosen.score)
                rejected_scores.append(rejected.score)
                chosen_lengths.append(len(chosen.ids))
                rejected_lengths.append(len(rejected.ids))

            processed += len(chunk)
            batches += 1
            if batches % 20 == 0:
                print(f"{processed}/{len(prompts)} prompts, {len(pairs)} pairs", flush=True)

    if not pairs:
        raise RuntimeError("No preference pairs were produced")

    rng.shuffle(pairs)
    num_val = max(1, int(len(pairs) * args.val_fraction))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_pairs(args.output_dir / "dpo_val.npz", pairs[:num_val])
    save_pairs(args.output_dir / "dpo_train.npz", pairs[num_val:])

    mean = lambda values: sum(values) / len(values)
    print(f"Prompts: {len(prompts):,}, pairs: {len(pairs):,} ({num_val:,} for validation), without pair: {no_pair:,}")
    print(f"Samples containing all words: {all_words_samples / total_samples:.1%}")
    print(f"Mean words covered, chosen vs rejected: {mean(chosen_scores):.2f} vs {mean(rejected_scores):.2f}")
    print(f"Mean tokens, chosen vs rejected: {mean(chosen_lengths):.0f} vs {mean(rejected_lengths):.0f}")


if __name__ == "__main__":
    main()
