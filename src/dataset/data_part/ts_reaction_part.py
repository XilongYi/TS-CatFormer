from __future__ import annotations

from pathlib import Path

import torch
import yaml

from ts_encoder.data import build_dataloaders, build_datasets
from ts_encoder.main import PROJECT_ROOT, load_config
from ts_encoder.model import build_model
from ts_encoder.routines import export_indexed_embeddings
from ts_encoder.utils import resolve_device, set_seed


TS_MODE_CONFIG_ROOT = PROJECT_ROOT / "configs" / "dataset" / "ts_encoder"


def _try_load_ts_cache(emb_path: Path, mask_path: Path):
    if not (emb_path.exists() and mask_path.exists()):
        return None
    return {
        "emb": torch.load(emb_path, map_location="cpu"),
        "mask": torch.load(mask_path, map_location="cpu"),
    }


def _load_mode_config(ts_mode: str) -> dict:
    path = TS_MODE_CONFIG_ROOT / f"{ts_mode}.yaml"
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    available = sorted(p.stem for p in TS_MODE_CONFIG_ROOT.glob("*.yaml"))
    raise FileNotFoundError(
        f"Unknown TS encoder mode '{ts_mode}'. Available modes: {available}"
    )


def _build_output_paths(main_set, ts_mode: str) -> tuple[Path, Path, Path]:
    save_dir = Path(main_set.embedding_save_dir_base) / main_set.dataset_embedding_name / "ts_reaction"
    save_dir.mkdir(parents=True, exist_ok=True)
    raw_path = save_dir / f"ts_reaction_{ts_mode}.pt"
    emb_path = save_dir / f"ts_reaction_{ts_mode}_emb.pt"
    mask_path = save_dir / f"ts_reaction_{ts_mode}_mask.pt"
    return raw_path, emb_path, mask_path


def _run_ts_encoder_export(main_set, reaction_csv_path: str, ts_mode: str, mode_cfg: dict, raw_path: Path, emb_path: Path, mask_path: Path) -> None:
    if reaction_csv_path is None or str(reaction_csv_path).strip() == "":
        raise FileNotFoundError(
            f"reaction_csv_path is required to build TS reaction embeddings for mode '{ts_mode}'"
        )

    if mode_cfg.get("requires_extra_atom_features", False):
        raise NotImplementedError(
            f"TS encoder mode '{ts_mode}' requires extra atom features / coordinates. "
            "A smiles-only reaction CSV is not sufficient for this mode."
        )

    base_config = load_config(mode_cfg["base_config"], [])
    base_config["data"]["source_type"] = "single_csv"
    base_config["data"]["data_path"] = str(Path(reaction_csv_path).resolve())
    base_config["data"]["train_ratio"] = float(mode_cfg.get("train_ratio", 0.8))
    base_config["data"]["val_ratio"] = float(mode_cfg.get("val_ratio", 0.1))
    base_config["data"]["column_mapping"] = {
        "smiles": mode_cfg.get("smiles_column", "Reaction"),
        "ts_label": None,
        "reaction_class_label": None,
    }
    base_config["data"]["dropna"] = True
    base_config["dataset"]["use_bond_change_label"] = False
    base_config["dataset"]["cache_name_prefix"] = f"ts_export_{ts_mode}"
    base_config["task"] = {
        "type": "export_indexed_embeddings",
        "checkpoint_dir": mode_cfg.get("checkpoint_dir", base_config["task"]["checkpoint_dir"]),
        "save_path": str(raw_path),
        "reaction_csv_path": str(Path(reaction_csv_path).resolve()),
        "reaction_column": mode_cfg.get("reaction_column", "Reaction"),
        "index_column": mode_cfg.get("index_column", "r_idx"),
        "emb_save_path": str(emb_path),
        "mask_save_path": str(mask_path),
        "splits": mode_cfg.get("splits", ["train", "val", "test"]),
        "use_ptr_if_available": bool(mode_cfg.get("use_ptr_if_available", True)),
        "keep_duplicates_as_list": bool(mode_cfg.get("keep_duplicates_as_list", True)),
        "dtype": mode_cfg.get("dtype", "float16"),
        "mask_dtype": mode_cfg.get("mask_dtype", "uint8"),
        "strict_rest_shape": bool(mode_cfg.get("strict_rest_shape", True)),
        "verbose": bool(mode_cfg.get("verbose", True)),
    }

    set_seed(int(base_config["seed"]))
    device = resolve_device(base_config.get("device"), use_cuda=bool(base_config.get("use_cuda", True)))
    print(f"[TS Reaction] Export mode '{ts_mode}' using device: {device}")

    datasets = build_datasets(base_config)
    loaders = build_dataloaders(base_config, datasets)
    ref_dataset = next(ds for ds in [datasets.train, datasets.val, datasets.test] if len(ds) > 0)
    model = build_model(
        base_config,
        num_node_features=ref_dataset.num_node_features,
        num_edge_features=ref_dataset.num_edge_features,
    ).to(device)
    export_indexed_embeddings(base_config, loaders, model, device)


def get_or_build_ts_reaction_embedding(main_set, reaction_csv_path: str):
    ts_mode = main_set.use_ts_reaction
    if ts_mode is None:
        return None

    raw_path, emb_path, mask_path = _build_output_paths(main_set, ts_mode)
    cache = _try_load_ts_cache(emb_path, mask_path)
    if cache is not None:
        return cache

    mode_cfg = _load_mode_config(ts_mode)
    _run_ts_encoder_export(main_set, reaction_csv_path, ts_mode, mode_cfg, raw_path, emb_path, mask_path)
    return _try_load_ts_cache(emb_path, mask_path)
