import math
import torch
from torch.optim.lr_scheduler import LambdaLR
from typing import Iterable, Tuple


def build_adamw_warmup_cosine(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    peak_lr: float,
    weight_decay: float,
    estimated_stepping_batches: int,
    warmup_ratio: float = 0.05,
    end_lr: float = 0.0,
):
    """
    AdamW + Warmup + Cosine (step-based).

    Args:
        named_parameters: model.named_parameters()
        peak_lr: peak/base lr used by AdamW
        weight_decay: weight decay applied to decay params
        estimated_stepping_batches: total optimizer steps (e.g. trainer.estimated_stepping_batches)
        warmup_ratio: warmup steps ratio in [0, 1). e.g. 0.05 means 5% steps warmup
        end_lr: minimum lr at the end of cosine (absolute lr)

    Returns:
        optimizer, scheduler
    """
    total_steps = int(estimated_stepping_batches)

    # Guard against invalid values.
    if total_steps <= 0:
        raise ValueError(f"estimated_stepping_batches must be > 0, got {estimated_stepping_batches}")
    if not (0.0 <= warmup_ratio < 1.0):
        raise ValueError(f"warmup_ratio must be in [0, 1), got {warmup_ratio}")

    warmup_steps = int(round(total_steps * warmup_ratio))
    warmup_steps = max(1, warmup_steps) if warmup_ratio > 0 else 0

    decay, no_decay = [], []
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if param.dim() == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=peak_lr,
    )

    # Precompute constants.
    end_over_peak = (end_lr / peak_lr) if peak_lr > 0 else 0.0

    def lr_lambda(step: int):
        # Warmup.
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)

        # Cosine decay.
        denom = max(1, total_steps - warmup_steps)
        progress = (step - warmup_steps) / denom
        # Clamp progress to [0, 1].
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return end_over_peak + (1.0 - end_over_peak) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler
