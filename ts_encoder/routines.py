from __future__ import annotations

import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from .utils import (
    Standardizer,
    check_early_stopping,
    classification_metrics,
    load_model,
    mae,
    rmse,
    save_model,
    save_standardizer,
)


class BondMaskCELoss(torch.nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        bond_mask: torch.Tensor,
        bond_change_label: torch.Tensor,
    ) -> torch.Tensor:
        bond_mask = bond_mask.bool()
        if bond_mask.sum().item() == 0:
            return logits.sum() * 0.0
        return torch.nn.functional.cross_entropy(
            logits[bond_mask],
            bond_change_label[bond_mask].long(),
            reduction=self.reduction,
        )


def build_optimizer(config: dict, model: torch.nn.Module):
    opt_cfg = config["optimizer"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 1e-3)),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-2)),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        eps=float(opt_cfg.get("eps", 1e-8)),
        amsgrad=bool(opt_cfg.get("amsgrad", False)),
    )


def build_scheduler(config: dict, optimizer: torch.optim.Optimizer):
    sched_cfg = config["scheduler"]
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(
            1e-6,
            config["scheduler_lambda"](step),
        ),
    )


def predict_with_labels(model, loader, stdzer: Standardizer, device):
    model.eval()
    preds_list, labels_list = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y_ts = y["ts_label"].to(device).float()
            out = model(X).squeeze(-1)
            preds = stdzer.destandardize(out)
            preds_list.extend(preds.detach().cpu().tolist())
            labels_list.extend(y_ts.detach().cpu().tolist())
    return preds_list, labels_list


def train_regression(config: dict, loaders, model, device) -> None:
    routine_cfg = config["task"]
    stdzer = Standardizer(loaders.train.dataset.mean, loaders.train.dataset.std)
    optimizer = build_optimizer(config, model)
    config["scheduler_lambda"] = config["scheduler_factory"](
        int(config["scheduler"].get("num_warmup_steps", 10)),
        int(routine_cfg["epochs"]),
        float(config["scheduler"].get("num_cycles", 0.5)),
    )
    scheduler = build_scheduler(config, optimizer)
    loss_fn = torch.nn.MSELoss(reduction=str(config["loss"].get("reduction", "sum")))

    start_epoch = 0
    best_val_rmse = float("inf")
    early_stop_counter = 0

    if routine_cfg.get("use_pretrained_model", False):
        model, optimizer, start_epoch, _ = load_model(
            model,
            optimizer,
            routine_cfg["load_model_path"],
        )
        print(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, int(routine_cfg["epochs"])):
        start_time = time.time()
        model.train()
        se_sum = 0.0
        n = 0

        for X, y in loaders.train:
            X = X.to(device)
            y_ts = y["ts_label"].to(device).float()

            optimizer.zero_grad()
            out = model(X).squeeze(-1)
            loss = loss_fn(out, stdzer.standardize(y_ts))
            loss.backward()

            if routine_cfg.get("clip_grad_norm", True):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(routine_cfg.get("clip_grad_norm_value", 1.0)),
                )

            optimizer.step()

            preds = stdzer.destandardize(out)
            se_sum += ((preds - y_ts) ** 2).sum().item()
            n += y_ts.numel()

        train_rmse = (se_sum / max(n, 1)) ** 0.5
        val_preds, val_labels = predict_with_labels(model, loaders.val, stdzer, device)
        val_rmse = rmse(val_labels, val_preds)
        scheduler.step()

        early_stop_counter, should_stop = check_early_stopping(
            val_rmse,
            best_val_rmse,
            early_stop_counter,
            int(routine_cfg["patience"]),
            float(routine_cfg["min_delta"]),
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            if routine_cfg.get("save_model_parameters", False):
                save_model(model, optimizer, epoch, best_val_rmse, routine_cfg["checkpoint_dir"])
                save_standardizer(loaders.train.dataset.mean, loaders.train.dataset.std, routine_cfg["checkpoint_dir"])

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}, Train RMSE: {train_rmse:.6f}, "
            f"Val RMSE: {val_rmse:.6f}, LR: {current_lr:.3e}, "
            f"Time: {time.time() - start_time:.2f}s"
        )
        if should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    if routine_cfg.get("save_model_parameters", False):
        model, _, _, _ = load_model(model, optimizer, routine_cfg["checkpoint_dir"])
    test_preds, test_labels = predict_with_labels(model, loaders.test, stdzer, device)
    print(f"Test RMSE: {rmse(test_labels, test_preds):.6f}")
    print(f"Test MAE:  {mae(test_labels, test_preds):.6f}")


@torch.no_grad()
def eval_bond_change(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_masked = 0
    for X, _ in loader:
        X = X.to(device)
        logits, bond_mask, bond_change_label = model(X)
        bond_mask = bond_mask.bool()
        masked = int(bond_mask.sum().item())
        if masked == 0:
            continue
        loss = loss_fn(logits, bond_mask, bond_change_label)
        total_loss += float(loss.item()) * masked
        total_masked += masked
    return total_loss / total_masked if total_masked > 0 else 0.0


@torch.no_grad()
def predict_bond_change(model, loader, device, num_classes: int):
    model.eval()
    all_preds = []
    all_labels = []
    for X, _ in loader:
        X = X.to(device)
        logits, bond_mask, bond_change_label = model(X)
        bond_mask = bond_mask.bool()
        if bond_mask.sum().item() == 0:
            continue
        all_preds.append(logits[bond_mask].argmax(dim=-1).cpu())
        all_labels.append(bond_change_label[bond_mask].long().cpu())
    if not all_preds:
        return None
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    metrics = classification_metrics(labels, preds, num_classes=num_classes)
    metrics["preds"] = preds
    metrics["labels"] = labels
    return metrics


def train_bond_change(config: dict, loaders, model, device) -> None:
    routine_cfg = config["task"]
    optimizer = build_optimizer(config, model)
    config["scheduler_lambda"] = config["scheduler_factory"](
        int(config["scheduler"].get("num_warmup_steps", 10)),
        int(routine_cfg["epochs"]),
        float(config["scheduler"].get("num_cycles", 0.5)),
    )
    scheduler = build_scheduler(config, optimizer)
    loss_fn = BondMaskCELoss(reduction=str(config["loss"].get("reduction", "mean")))

    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(int(routine_cfg["epochs"])):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        total_masked = 0

        for X, _ in loaders.train:
            X = X.to(device)
            optimizer.zero_grad()
            logits, bond_mask, bond_change_label = model(X)
            bond_mask = bond_mask.bool()
            masked = int(bond_mask.sum().item())
            if masked == 0:
                continue
            loss = loss_fn(logits, bond_mask, bond_change_label)
            loss.backward()
            if routine_cfg.get("clip_grad_norm", True):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(routine_cfg.get("clip_grad_norm_value", 1.0)),
                )
            optimizer.step()
            total_loss += float(loss.item()) * masked
            total_masked += masked

        train_loss = total_loss / total_masked if total_masked > 0 else 0.0
        val_loss = eval_bond_change(model, loaders.val, loss_fn, device)
        val_metrics = predict_bond_change(
            model,
            loaders.val,
            device,
            int(config["model"].get("bond_mask_num_classes", 31)),
        )
        scheduler.step()

        early_stop_counter, should_stop = check_early_stopping(
            val_loss,
            best_val_loss,
            early_stop_counter,
            int(routine_cfg["patience"]),
            float(routine_cfg["min_delta"]),
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if routine_cfg.get("save_model_parameters", False):
                save_model(model, optimizer, epoch, best_val_loss, routine_cfg["checkpoint_dir"])

        print(
            f"Epoch {epoch}, Train CE: {train_loss:.6f}, "
            f"Val CE: {val_loss:.6f}, Time: {time.time() - start_time:.2f}s"
        )
        if val_metrics is not None:
            print(
                f"  Val Accuracy: {val_metrics['accuracy']:.4f}, "
                f"F1-macro: {val_metrics['f1_macro']:.4f}, "
                f"F1-micro: {val_metrics['f1_micro']:.4f}"
            )
        if should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    if routine_cfg.get("save_model_parameters", False):
        model, _, _, _ = load_model(model, optimizer, routine_cfg["checkpoint_dir"])

    test_metrics = predict_bond_change(
        model,
        loaders.test,
        device,
        int(config["model"].get("bond_mask_num_classes", 31)),
    )
    if test_metrics is None:
        print("[WARN] No masked bonds found in test set.")
        return
    print("===== Test Metrics (masked bonds) =====")
    print(f"Test Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Test F1-macro : {test_metrics['f1_macro']:.4f}")
    print(f"Test F1-micro : {test_metrics['f1_micro']:.4f}")


def _save_tensor_artifact(tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(tensor, path)


def _collect_embedding_store(model, loaders, device, routine_cfg: dict):
    split_order = {
        "train": loaders.train,
        "val": loaders.val,
        "test": loaders.test,
    }
    store = defaultdict(list)

    for split in routine_cfg.get("splits", ["train", "val", "test"]):
        loader = split_order[split]
        for batch in loader:
            X = batch[0] if isinstance(batch, (tuple, list)) else batch
            X = X.to(device, non_blocking=True)
            out = model(X, return_batch_embedding=True)
            x_cpu = out.x.detach().cpu()
            smiles_list = list(out.smiles)
            if routine_cfg.get("use_ptr_if_available", True) and getattr(out, "ptr", None) is not None:
                ptr = out.ptr.detach().cpu().tolist()
                for idx in range(len(ptr) - 1):
                    store[str(smiles_list[idx])].append(x_cpu[ptr[idx]:ptr[idx + 1]])
            else:
                batch_vec = out.batch.detach().cpu()
                num_graphs = int(batch_vec.max().item()) + 1 if batch_vec.numel() else 0
                for idx in range(num_graphs):
                    store[str(smiles_list[idx])].append(x_cpu[batch_vec == idx])

    if routine_cfg.get("keep_duplicates_as_list", True):
        return dict(store)
    return {key: value[0] if len(value) == 1 else value for key, value in store.items()}


def _get_embedding_tensor(candidate):
    if isinstance(candidate, torch.Tensor):
        return candidate
    if isinstance(candidate, (list, tuple)) and candidate:
        first = candidate[0]
        if isinstance(first, torch.Tensor):
            return first
    return None


def _build_indexed_embedding_tensors(
    reaction_df: pd.DataFrame,
    ts_reaction_dict: dict,
    reaction_column: str,
    index_column: str,
    dtype=torch.float16,
    mask_dtype=torch.uint8,
    strict_rest_shape=True,
    verbose=True,
):
    if reaction_column not in reaction_df.columns:
        raise KeyError(f"missing reaction column: {reaction_column}")
    if index_column not in reaction_df.columns:
        raise KeyError(f"missing index column: {index_column}")

    rows = []
    skipped_idx = 0
    for reaction, r_idx in reaction_df[[reaction_column, index_column]].itertuples(index=False, name=None):
        if pd.isna(r_idx):
            skipped_idx += 1
            continue
        rows.append((str(reaction), int(r_idx)))

    if not rows:
        raise ValueError('no valid reaction/index rows found for indexed TS export')

    size = max(idx for _, idx in rows) + 1
    rest_shape = None
    lmax = 0
    found = 0
    miss_key = 0
    bad_type = 0
    bad_rest = 0

    for reaction, _ in rows:
        present = reaction in ts_reaction_dict
        tensor = _get_embedding_tensor(ts_reaction_dict.get(reaction))
        if tensor is None or tensor.dim() < 2:
            if present:
                bad_type += 1
            else:
                miss_key += 1
            continue

        this_rest = tuple(tensor.shape[1:])
        if rest_shape is None:
            rest_shape = this_rest
        elif this_rest != rest_shape:
            bad_rest += 1
            if strict_rest_shape:
                continue
            continue

        lmax = max(lmax, int(tensor.shape[0]))
        found += 1

    if rest_shape is None or lmax == 0 or found == 0:
        raise ValueError(
            'no valid TS tensors found while building indexed export: '
            f'miss_key={miss_key}, bad_type={bad_type}, bad_rest={bad_rest}'
        )

    if verbose:
        print(
            f"[Indexed TS] size={size}, found={found}, Lmax={lmax}, rest_shape={rest_shape}, "
            f"miss_key={miss_key}, bad_type={bad_type}, bad_rest={bad_rest}, skipped_idx={skipped_idx}"
        )

    emb = torch.zeros((size, lmax, *rest_shape), dtype=dtype)
    mask = torch.zeros((size, lmax), dtype=mask_dtype)

    filled = 0
    overwritten = 0
    seen = set()
    for reaction, r_idx in rows:
        tensor = _get_embedding_tensor(ts_reaction_dict.get(reaction))
        if tensor is None or tensor.dim() < 2 or tuple(tensor.shape[1:]) != rest_shape:
            continue
        tensor = tensor.detach().cpu().to(dtype=dtype).contiguous()
        li = min(int(tensor.shape[0]), lmax)
        if r_idx in seen:
            overwritten += 1
        seen.add(r_idx)
        emb[r_idx].zero_()
        mask[r_idx].zero_()
        emb[r_idx, :li] = tensor[:li]
        mask[r_idx, :li] = 1
        filled += 1

    if verbose:
        print(f"[Indexed TS] filled={filled}, overwritten_idx={overwritten}")

    return emb, mask


@torch.no_grad()
def export_embeddings(config: dict, loaders, model, device) -> None:
    routine_cfg = config["task"]
    model, _, epoch, best_val = load_model(model, None, routine_cfg["checkpoint_dir"])
    model = model.to(device)
    model.eval()

    final = _collect_embedding_store(model, loaders, device, routine_cfg)
    save_path = routine_cfg["save_path"]
    _save_tensor_artifact(final, save_path)
    print(f"[Embeddings] saved {len(final)} keys to: {save_path} (epoch={epoch}, best_val={best_val})")


@torch.no_grad()
def export_indexed_embeddings(config: dict, loaders, model, device) -> None:
    routine_cfg = config["task"]
    model, _, epoch, best_val = load_model(model, None, routine_cfg["checkpoint_dir"])
    model = model.to(device)
    model.eval()

    final = _collect_embedding_store(model, loaders, device, routine_cfg)

    raw_save_path = routine_cfg.get("save_path") or routine_cfg.get("smiles_dict_save_path")
    if raw_save_path:
        _save_tensor_artifact(final, raw_save_path)
        print(f"[Embeddings] saved {len(final)} keys to: {raw_save_path} (epoch={epoch}, best_val={best_val})")

    reaction_df = pd.read_csv(routine_cfg["reaction_csv_path"])
    emb, mask = _build_indexed_embedding_tensors(
        reaction_df=reaction_df,
        ts_reaction_dict=final,
        reaction_column=routine_cfg.get("reaction_column", "Reaction"),
        index_column=routine_cfg.get("index_column", "r_idx"),
        dtype=getattr(torch, routine_cfg.get("dtype", "float16")),
        mask_dtype=getattr(torch, routine_cfg.get("mask_dtype", "uint8")),
        strict_rest_shape=bool(routine_cfg.get("strict_rest_shape", True)),
        verbose=bool(routine_cfg.get("verbose", True)),
    )
    emb_save_path = routine_cfg["emb_save_path"]
    mask_save_path = routine_cfg["mask_save_path"]
    _save_tensor_artifact(emb, emb_save_path)
    _save_tensor_artifact(mask, mask_save_path)
    print(
        f"[Indexed TS] saved emb to: {emb_save_path}, mask to: {mask_save_path} "
        f"(epoch={epoch}, best_val={best_val})"
    )

