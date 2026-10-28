import math
from typing import Iterable, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


class WarmupThenPlateauScheduler:
    """
    Linear warmup by step, then ReduceLROnPlateau by monitored metric.

    Usage:
        - During training step:
              scheduler.step_step()
        - After validation:
              scheduler.step_metric(val_auc)

    In PyTorch Lightning, easiest is:
        1. return optimizer + plateau scheduler config from configure_optimizers
        2. manually do warmup in optimizer_step
    But if you want a single returned object, this wrapper is convenient.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        peak_lr: float,
        plateau_scheduler: ReduceLROnPlateau,
    ):
        self.optimizer = optimizer
        self.warmup_steps = max(0, int(warmup_steps))
        self.peak_lr = float(peak_lr)
        self.plateau_scheduler = plateau_scheduler

        self.global_step = 0
        self.finished_warmup = self.warmup_steps == 0

        # init lr to 0 if warmup is used
        if self.warmup_steps > 0:
            for pg in self.optimizer.param_groups:
                pg["lr"] = 0.0
        else:
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.peak_lr

    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def step_step(self):
        """
        Call this after each optimizer step.
        Only active during warmup.
        """
        if self.finished_warmup:
            self.global_step += 1
            return

        self.global_step += 1

        warmup_progress = min(self.global_step / self.warmup_steps, 1.0)
        lr = self.peak_lr * warmup_progress
        self._set_lr(lr)

        if self.global_step >= self.warmup_steps:
            self.finished_warmup = True
            self._set_lr(self.peak_lr)

    def step_metric(self, metric: float):
        """
        Call this after validation, e.g. with val_auc_crf.
        """
        if not self.finished_warmup:
            return
        self.plateau_scheduler.step(metric)

    def state_dict(self):
        return {
            "global_step": self.global_step,
            "finished_warmup": self.finished_warmup,
            "plateau_scheduler": self.plateau_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.global_step = state_dict["global_step"]
        self.finished_warmup = state_dict["finished_warmup"]
        self.plateau_scheduler.load_state_dict(state_dict["plateau_scheduler"])


def build_optimizer_scheduler_plateau(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    peak_lr: float,
    weight_decay: float,
    estimated_stepping_batches: int,
    warmup_ratio: float = 0.05,
    end_lr: float = 0.0,
):
    """
    Build AdamW optimizer + linear warmup + ReduceLROnPlateau scheduler.

    Args:
        named_parameters:
            model.named_parameters()
        peak_lr:
            target lr after warmup
        weight_decay:
            weight decay for decayed params
        estimated_stepping_batches:
            used only to estimate warmup_steps
        warmup_ratio:
            warmup_steps = estimated_stepping_batches * warmup_ratio
        end_lr:
            min lr for plateau scheduler

    Returns:
        optimizer, scheduler

    Note:
        `scheduler` is a wrapper with:
            - scheduler.step_step()   # call every optimizer step
            - scheduler.step_metric(metric)  # call after validation
    """
    if peak_lr <= 0:
        raise ValueError(f"peak_lr must be > 0, got {peak_lr}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
    if estimated_stepping_batches <= 0:
        raise ValueError(
            f"estimated_stepping_batches must be > 0, got {estimated_stepping_batches}"
        )
    if not (0.0 <= warmup_ratio < 1.0):
        raise ValueError(f"warmup_ratio must be in [0, 1), got {warmup_ratio}")
    if end_lr < 0:
        raise ValueError(f"end_lr must be >= 0, got {end_lr}")
    if end_lr > peak_lr:
        raise ValueError(f"end_lr must be <= peak_lr, got end_lr={end_lr}, peak_lr={peak_lr}")

    named_parameters = list(named_parameters)

    decay_params = []
    no_decay_params = []

    no_decay_keywords = (
        "bias",
        "norm",
        "bn",
        "ln",
        "layernorm",
        "layer_norm",
        "batchnorm",
        "batch_norm",
    )

    for name, param in named_parameters:
        if not param.requires_grad:
            continue

        name_lower = name.lower()

        # Skip decay for 1D and norm-like params.
        if param.ndim == 1 or any(k in name_lower for k in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=peak_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    warmup_steps = int(estimated_stepping_batches * warmup_ratio)

    plateau = ReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",
        factor=0.6,
        patience=3,
        threshold=1e-3,
        threshold_mode="abs",
        cooldown=1,
        min_lr=end_lr,
    )

    scheduler = WarmupThenPlateauScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        peak_lr=peak_lr,
        plateau_scheduler=plateau,
    )

    return optimizer, scheduler