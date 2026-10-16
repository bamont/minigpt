import argparse
import sys
from pathlib import Path

import torch

from minigpt.data.sft_format import build_prompt
from minigpt.generation import generate_stream
from minigpt.model.checkpoint import load_model
from minigpt.tokenizer import BPETokenizer

EOS_TOKEN = "<|endoftext|>"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Generate text with a trained mini-GPT.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/pretrain/best.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer.json"))
    parser.add_argument("--prompt", default="")
    parser.add_argument("--words", default=None)
    parser.add_argument("--features", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--sentence", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    tokenizer = BPETokenizer.load(args.tokenizer)
    eos_id = tokenizer.special_tokens[EOS_TOKEN]
    model = load_model(args.checkpoint, device)

    fields = {
        "Features": args.features,
        "Words": args.words,
        "Summary": args.summary,
        "Random sentence": args.sentence,
    }
    fields = {name: value for name, value in fields.items() if value}
    prompt = build_prompt(fields) if fields else args.prompt
    prompt_ids = tokenizer.encode(prompt) or [eos_id]

    for sample in range(args.num_samples):
        if sample > 0:
            print("\n" + "-" * 40 + "\n")
        print(prompt, end="", flush=True)
        ids: list[int] = []
        printed = 0
        stream = generate_stream(
            model,
            prompt_ids,
            args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_id=eos_id,
        )
        for token in stream:
            if token == eos_id:
                break
            ids.append(token)
            text = tokenizer.decode(ids)
            if text.endswith("\ufffd"):
                continue
            print(text[printed:], end="", flush=True)
            printed = len(text)
        print()


if __name__ == "__main__":
    main()
