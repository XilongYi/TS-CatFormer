from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from .data import build_dataloaders, build_datasets
from .model import build_model
from .routines import export_embeddings, export_indexed_embeddings, train_bond_change, train_regression
from .utils import cosine_with_warmup_lambda, resolve_device, set_seed


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
CONFIG_ROOT = PACKAGE_ROOT / "configs"


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone TS encoder runner")
    parser.add_argument(
        "--config",
        required=True,
        help="Config name under ts_encoder/configs or a yaml path",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted override, e.g. --override model.hidden_channels=900",
    )
    return parser.parse_args()


def _resolve_config_path(config_arg: str) -> Path:
    path = Path(config_arg)
    if path.exists():
        return path.resolve()
    candidate = CONFIG_ROOT / config_arg
    if candidate.suffix != ".yaml":
        candidate = candidate.with_suffix(".yaml")
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"config not found: {config_arg}")


def _deep_update(config: dict, key_path: str, value):
    parts = key_path.split(".")
    current = config
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _parse_override(raw: str):
    key, value = raw.split("=", 1)
    return key, yaml.safe_load(value)


def _resolve_relative_path(key: str, value: str) -> str:
    return str((PROJECT_ROOT / value).resolve())


def _resolve_project_paths(obj):
    path_keys = {
        "data_path",
        "data_folder",
        "cache_path",
        "checkpoint_dir",
        "load_model_path",
        "save_path",
        "reaction_csv_path",
        "emb_save_path",
        "mask_save_path",
        "smiles_dict_save_path",
    }
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if key in path_keys and isinstance(value, str) and value and not Path(value).is_absolute():
                out[key] = _resolve_relative_path(key, value)
            else:
                out[key] = _resolve_project_paths(value)
        return out
    if isinstance(obj, list):
        return [_resolve_project_paths(item) for item in obj]
    return obj


def load_config(config_arg: str, overrides: list[str]) -> dict:
    with open(_resolve_config_path(config_arg), "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = copy.deepcopy(config)
    for raw in overrides:
        key, value = _parse_override(raw)
        _deep_update(config, key, value)
    return _resolve_project_paths(config)


def main():
    args = parse_args()
    config = load_config(args.config, args.override)
    set_seed(int(config["seed"]))
    device = resolve_device(config.get("device"), use_cuda=bool(config.get("use_cuda", True)))
    print(f"Using device: {device}")

    datasets = build_datasets(config)
    loaders = build_dataloaders(config, datasets)

    ref_dataset = next(ds for ds in [datasets.train, datasets.val, datasets.test] if len(ds) > 0)
    model = build_model(
        config,
        num_node_features=ref_dataset.num_node_features,
        num_edge_features=ref_dataset.num_edge_features,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")

    config["scheduler_factory"] = cosine_with_warmup_lambda
    task_type = config["task"]["type"]
    if task_type == "regression_train":
        train_regression(config, loaders, model, device)
    elif task_type == "bond_change_train":
        train_bond_change(config, loaders, model, device)
    elif task_type == "export_embeddings":
        export_embeddings(config, loaders, model, device)
    elif task_type == "export_indexed_embeddings":
        export_indexed_embeddings(config, loaders, model, device)
    else:
        raise ValueError(f"unsupported task type: {task_type}")

