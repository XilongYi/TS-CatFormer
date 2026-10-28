from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class DataSplit:
    train: Any
    val: Any
    test: Any
    train_coord: Any = None
    val_coord: Any = None
    test_coord: Any = None


class Standardizer:
    def __init__(self, mean: float, std: float) -> None:
        self.mean = float(mean)
        self.std = float(std) if float(std) != 0.0 else 1.0

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def destandardize(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.std) + self.mean


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def check_early_stopping(
    current_loss: float,
    best_loss: float,
    counter: int,
    patience: int,
    min_delta: float,
) -> tuple[int, bool]:
    if current_loss < best_loss - min_delta:
        return 0, False
    counter += 1
    return counter, counter >= patience


def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    best_val_loss: float,
    model_dir: str | Path,
) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_val_loss": best_val_loss,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, model_dir / "model.pt")


def load_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    model_dir: str | Path,
    strict: bool = False,
    verbose: bool = True,
) -> tuple[torch.nn.Module, torch.optim.Optimizer | None, int, float]:
    model_dir = Path(model_dir)
    model_path = model_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"[load_model] checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(
        checkpoint["model_state_dict"], strict=strict
    )
    if verbose:
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print("[load_model] state dict loaded with mismatches")
            if incompatible.missing_keys:
                print(f"  missing: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                print(f"  unexpected: {incompatible.unexpected_keys}")
        else:
            print("[load_model] all parameters matched")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as exc:  # pragma: no cover
            print(f"[load_model] optimizer state not loaded: {exc}")

    return (
        model,
        optimizer,
        int(checkpoint.get("epoch", 0)),
        float(checkpoint.get("best_val_loss", float("inf"))),
    )


def save_standardizer(mean: float, std: float, model_dir: str | Path) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": mean, "std": std}, model_dir / "standardizer.pt")


def load_standardizer(model_dir: str | Path) -> tuple[float | None, float | None]:
    path = Path(model_dir) / "standardizer.pt"
    if not path.exists():
        return None, None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return float(payload["mean"]), float(payload["std"])


def cosine_with_warmup_lambda(
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
):
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )

    return lr_lambda


def resolve_device(device_name: str | None, use_cuda: bool = True) -> torch.device:
    if device_name is not None:
        if device_name == "gpu":
            device_name = "cuda"
        if device_name == "cpu":
            return torch.device("cpu")
        if device_name == "cuda" and torch.cuda.is_available() and use_cuda:
            return torch.device("cuda")
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def rmse(labels: list[float], preds: list[float]) -> float:
    arr_l = np.asarray(labels, dtype=float)
    arr_p = np.asarray(preds, dtype=float)
    return float(np.sqrt(np.mean((arr_l - arr_p) ** 2)))


def mae(labels: list[float], preds: list[float]) -> float:
    arr_l = np.asarray(labels, dtype=float)
    arr_p = np.asarray(preds, dtype=float)
    return float(np.mean(np.abs(arr_l - arr_p)))


def classification_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    labels = labels.astype(int)
    preds = preds.astype(int)
    accuracy = float((labels == preds).mean()) if labels.size else 0.0
    f1_per_class = []
    for cls in range(num_classes):
        tp = int(((preds == cls) & (labels == cls)).sum())
        fp = int(((preds == cls) & (labels != cls)).sum())
        fn = int(((preds != cls) & (labels == cls)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_per_class.append(f1)

    tp_micro = int((preds == labels).sum())
    total = int(labels.size)
    precision_micro = tp_micro / total if total else 0.0
    recall_micro = tp_micro / total if total else 0.0
    if precision_micro + recall_micro == 0:
        f1_micro = 0.0
    else:
        f1_micro = 2 * precision_micro * recall_micro / (
            precision_micro + recall_micro
        )

    return {
        "accuracy": accuracy,
        "f1_macro": float(sum(f1_per_class) / len(f1_per_class)) if f1_per_class else 0.0,
        "f1_micro": float(f1_micro),
    }

