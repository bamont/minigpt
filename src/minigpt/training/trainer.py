import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

Batch = tuple[torch.Tensor, torch.Tensor]
BatchFn = Callable[[int, torch.device], Batch]

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


@dataclass
class TrainConfig:
    out_dir: str = "runs/pretrain"
    max_steps: int = 20000
    batch_size: int = 32
    grad_accum_steps: int = 2
    learning_rate: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dtype: str = "float16"
    eval_interval: int = 500
    eval_iters: int = 50
    log_interval: int = 20
    checkpoint_interval: int = 500
    time_limit_minutes: float | None = None
    seed: int = 1337
    device: str = "auto"


def get_lr(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    decay_steps = max(1, config.max_steps - config.warmup_steps)
    progress = min((step - config.warmup_steps) / decay_steps, 1.0)
    min_lr = config.learning_rate * config.min_lr_ratio
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (config.learning_rate - min_lr)


def build_optimizer(model: nn.Module, config: TrainConfig, device_type: str) -> torch.optim.AdamW:
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": config.weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        fused=device_type == "cuda",
    )


class Trainer:
    def __init__(
        self, model: nn.Module, config: TrainConfig, get_train_batch: BatchFn, get_val_batch: BatchFn
    ):
        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.config = config
        self.get_train_batch = get_train_batch
        self.get_val_batch = get_val_batch
        self.out_dir = Path(config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.amp_dtype = DTYPES[config.dtype]
        self.amp_enabled = self.device.type == "cuda" and self.amp_dtype != torch.float32
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_enabled and self.amp_dtype == torch.float16
        )
        self.optimizer = build_optimizer(self.model, config, self.device.type)
        self.step = 0
        self.best_val_loss = float("inf")

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled
        )

    def _log(self, record: dict) -> None:
        with open(self.out_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _save(self, path: Path, payload: dict) -> None:
        tmp_path = path.with_suffix(".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

    def _save_last(self) -> None:
        self._save(
            self.out_dir / "last.pt",
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "step": self.step,
                "best_val_loss": self.best_val_loss,
                "model_config": asdict(self.model.config),
                "train_config": asdict(self.config),
            },
        )

    def _save_best(self) -> None:
        self._save(
            self.out_dir / "best.pt",
            {
                "model": self.model.state_dict(),
                "step": self.step,
                "val_loss": self.best_val_loss,
                "model_config": asdict(self.model.config),
            },
        )

    def _load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.step = checkpoint["step"]
        self.best_val_loss = checkpoint["best_val_loss"]

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        self.model.eval()
        results = {}
        for split, get_batch in (("train", self.get_train_batch), ("val", self.get_val_batch)):
            losses = torch.zeros(self.config.eval_iters)
            for i in range(self.config.eval_iters):
                x, y = get_batch(self.config.batch_size, self.device)
                with self._autocast():
                    _, loss = self.model(x, y)
                losses[i] = loss.item()
            results[split] = losses.mean().item()
        self.model.train()
        return results

    def _evaluate_and_log(self) -> None:
        losses = self.estimate_loss()
        perplexity = math.exp(min(losses["val"], 20.0))
        print(
            f"step {self.step}: train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}, val perplexity {perplexity:.2f}"
        )
        self._log(
            {
                "type": "eval",
                "step": self.step,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "val_perplexity": perplexity,
            }
        )
        if losses["val"] < self.best_val_loss:
            self.best_val_loss = losses["val"]
            self._save_best()

    def _train_step(self) -> tuple[float, float, int]:
        config = self.config
        total_loss = 0.0
        tokens = 0
        for _ in range(config.grad_accum_steps):
            x, y = self.get_train_batch(config.batch_size, self.device)
            with self._autocast():
                _, loss = self.model(x, y)
            loss = loss / config.grad_accum_steps
            self.scaler.scale(loss).backward()
            total_loss += loss.item()
            tokens += x.numel()
        return total_loss, self._apply_gradients(total_loss), tokens

    def _apply_gradients(self, total_loss: float) -> float:
        if not math.isfinite(total_loss):
            raise RuntimeError(f"Loss is {total_loss} at step {self.step}")
        self.scaler.unscale_(self.optimizer)
        max_norm = self.config.grad_clip if self.config.grad_clip > 0 else float("inf")
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        return float(grad_norm)

    def fit(self, resume: bool = False) -> None:
        config = self.config
        last_path = self.out_dir / "last.pt"
        if resume and last_path.exists():
            self._load(last_path)
            print(f"Resumed from step {self.step}")
        torch.manual_seed(config.seed + self.step)

        tokens_per_step = (
            config.batch_size * config.grad_accum_steps * self.model.config.block_size
        )
        print(
            f"Device: {self.device}, dtype: {config.dtype if self.amp_enabled else 'float32'}, "
            f"tokens per step: {tokens_per_step:,}"
        )

        start_step = self.step
        started_at = time.time()
        window_start = time.time()
        window_loss, window_steps, window_tokens = 0.0, 0, 0

        self.model.train()
        while self.step < config.max_steps:
            if self.step % config.eval_interval == 0 and not (self.step == start_step and self.step > 0):
                self._evaluate_and_log()

            lr = get_lr(self.step, config)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            loss, grad_norm, tokens = self._train_step()
            self.step += 1
            window_loss += loss
            window_steps += 1
            window_tokens += tokens

            if self.step % config.log_interval == 0:
                elapsed = time.time() - window_start
                record = {
                    "type": "train",
                    "step": self.step,
                    "loss": window_loss / window_steps,
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "tokens_per_sec": window_tokens / elapsed,
                }
                print(
                    f"step {self.step}: loss {record['loss']:.4f}, lr {lr:.2e}, "
                    f"grad norm {grad_norm:.2f}, {record['tokens_per_sec']:,.0f} tokens/s"
                )
                self._log(record)
                window_start = time.time()
                window_loss, window_steps, window_tokens = 0.0, 0, 0

            if self.step % config.checkpoint_interval == 0:
                self._save_last()

            if config.time_limit_minutes and (time.time() - started_at) / 60 >= config.time_limit_minutes:
                print(f"Time limit reached at step {self.step}; saving checkpoint")
                self._save_last()
                return

        self._evaluate_and_log()
        self._save_last()
