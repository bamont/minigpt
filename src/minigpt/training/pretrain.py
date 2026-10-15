import argparse
from pathlib import Path

import yaml

from minigpt.data.pretrain_dataset import PretrainDataset
from minigpt.model import GPT, GPTConfig
from minigpt.training.trainer import TrainConfig, Trainer


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    for item in overrides:
        key, value = item.split("=", 1)
        section, name = key.split(".", 1)
        config[section][name] = yaml.safe_load(value)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train the GPT model.")
    parser.add_argument("--config", type=Path, default=Path("configs/pretrain.yaml"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--override", nargs="*", default=[], metavar="SECTION.KEY=VALUE")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = apply_overrides(config, args.override)

    model_config = GPTConfig(**config["model"])
    train_config = TrainConfig(**config["train"])
    train_data = PretrainDataset(config["data"]["train"], model_config.block_size)
    val_data = PretrainDataset(config["data"]["val"], model_config.block_size)

    model = GPT(model_config)
    print(f"Parameters: {model.num_parameters():,}")
    print(f"Train tokens: {len(train_data):,}, validation tokens: {len(val_data):,}")

    trainer = Trainer(model, train_config, train_data.get_batch, val_data.get_batch)
    trainer.fit(resume=args.resume)


if __name__ == "__main__":
    main()
