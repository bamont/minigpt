import argparse
import math
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import nn

from minigpt.data.preference_dataset import PreferenceDataset
from minigpt.data.sft_dataset import IGNORE_INDEX
from minigpt.model.checkpoint import load_model
from minigpt.tokenizer import BPETokenizer
from minigpt.training.pretrain import apply_overrides
from minigpt.training.trainer import BatchFn, TrainConfig, Trainer

EOS_TOKEN = "<|endoftext|>"


def sequence_logprobs(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits, _ = model(inputs)
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    token_logprobs = logprobs.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (token_logprobs * (targets != IGNORE_INDEX)).sum(dim=-1)


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    margins = beta * ((policy_chosen - policy_rejected) - (reference_chosen - reference_rejected))
    loss = -F.logsigmoid(margins).mean()
    return loss, (margins > 0).float().mean(), margins.mean()


class DPOTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        reference: nn.Module,
        config: TrainConfig,
        beta: float,
        get_train_batch: BatchFn,
        get_val_batch: BatchFn,
    ):
        super().__init__(model, config, get_train_batch, get_val_batch)
        self.reference = reference.to(self.device).eval()
        for param in self.reference.parameters():
            param.requires_grad_(False)
        self.beta = beta

    def _compute_loss(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad(), self._autocast():
            reference = sequence_logprobs(self.reference, inputs, targets)
        with self._autocast():
            policy = sequence_logprobs(self.model, inputs, targets)
        half = inputs.size(0) // 2
        return dpo_loss(policy[:half], policy[half:], reference[:half], reference[half:], self.beta)

    def _train_step(self) -> tuple[float, float, int]:
        config = self.config
        total_loss = 0.0
        tokens = 0
        for _ in range(config.grad_accum_steps):
            inputs, targets = self.get_train_batch(config.batch_size, self.device)
            loss, _, _ = self._compute_loss(inputs, targets)
            loss = loss / config.grad_accum_steps
            self.scaler.scale(loss).backward()
            total_loss += loss.item()
            tokens += inputs.numel()
        return total_loss, self._apply_gradients(total_loss), tokens

    @torch.no_grad()
    def _evaluate_split(self, get_batch: BatchFn) -> tuple[float, float, float]:
        losses, accuracies, margins = [], [], []
        for _ in range(self.config.eval_iters):
            inputs, targets = get_batch(self.config.batch_size, self.device)
            loss, accuracy, margin = self._compute_loss(inputs, targets)
            losses.append(loss.item())
            accuracies.append(accuracy.item())
            margins.append(margin.item())
        return statistics.fmean(losses), statistics.fmean(accuracies), statistics.fmean(margins)

    def _evaluate_and_log(self) -> None:
        self.model.eval()
        train_loss, train_accuracy, _ = self._evaluate_split(self.get_train_batch)
        val_loss, val_accuracy, val_margin = self._evaluate_split(self.get_val_batch)
        self.model.train()
        print(
            f"step {self.step}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, "
            f"val accuracy {val_accuracy:.3f}, val margin {val_margin:.3f}"
        )
        self._log(
            {
                "type": "eval",
                "step": self.step,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_margin": val_margin,
            }
        )
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save_best()


def main() -> None:
    parser = argparse.ArgumentParser(description="Align the SFT model with Direct Preference Optimization.")
    parser.add_argument("--config", type=Path, default=Path("configs/dpo.yaml"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--override", nargs="*", default=[], metavar="SECTION.KEY=VALUE")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = apply_overrides(config, args.override)

    cpu = torch.device("cpu")
    policy = load_model(config["init"]["checkpoint"], cpu)
    reference = load_model(config["init"]["checkpoint"], cpu)
    tokenizer = BPETokenizer.load(config["data"]["tokenizer"])
    pad_id = tokenizer.special_tokens[EOS_TOKEN]
    train_data = PreferenceDataset(config["data"]["train"], pad_id)
    val_data = PreferenceDataset(config["data"]["val"], pad_id)

    print(f"Parameters: {policy.num_parameters():,}")
    print(f"Train pairs: {len(train_data):,}, validation pairs: {len(val_data):,}")

    trainer = DPOTrainer(
        policy,
        reference,
        TrainConfig(**config["train"]),
        config["dpo"]["beta"],
        train_data.get_batch,
        val_data.get_batch,
    )
    trainer.fit(resume=args.resume)


if __name__ == "__main__":
    main()
