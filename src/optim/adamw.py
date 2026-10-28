import torch
from typing import Iterable, Tuple

def build_adamw(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    peak_lr: float,
    weight_decay: float,
    estimated_stepping_batches: None,
    warmup_ratio: None,
    end_lr: None,
):
    """
    Build AdamW optimizer with parameter groups:
      - decay: parameters with weight decay
      - no_decay: bias and norm parameters without weight decay

    Args:
        named_parameters: model.named_parameters()
        peak_lr: peak learning rate
        weight_decay: weight decay for decay group

    Returns:
        torch.optim.AdamW
    """
    decay, no_decay = [], []

    for name, param in named_parameters:
        if not param.requires_grad:
            continue

        if param.dim() == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer_grouped_parameters = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=peak_lr,
    )

    return optimizer, None
