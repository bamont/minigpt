import argparse
from pathlib import Path

import torch
import yaml

from minigpt.data.sft_dataset import SFTDataset
from minigpt.model.checkpoint import load_model
from minigpt.tokenizer import BPETokenizer
from minigpt.training.pretrain import apply_overrides
from minigpt.training.trainer import TrainConfig, Trainer

EOS_TOKEN = "<|endoftext|>"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the pre-trained GPT on instruction data.")
    parser.add_argument("--config", type=Path, default=Path("configs/sft.yaml"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--override", nargs="*", default=[], metavar="SECTION.KEY=VALUE")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = apply_overrides(config, args.override)

    model = load_model(config["init"]["checkpoint"], torch.device("cpu"))
    tokenizer = BPETokenizer.load(config["data"]["tokenizer"])
    pad_id = tokenizer.special_tokens[EOS_TOKEN]
    train_data = SFTDataset(config["data"]["train"], pad_id)
    val_data = SFTDataset(config["data"]["val"], pad_id)

    print(f"Parameters: {model.num_parameters():,}")
    print(f"Train examples: {len(train_data):,}, validation examples: {len(val_data):,}")

    trainer = Trainer(model, TrainConfig(**config["train"]), train_data.get_batch, val_data.get_batch)
    trainer.fit(resume=args.resume)


if __name__ == "__main__":
    main()
