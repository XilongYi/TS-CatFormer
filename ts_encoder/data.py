from __future__ import annotations

import pathlib
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from .representation import CGRRepresentation
from .utils import DataSplit


def npz_to_df(path: str | Path) -> pd.DataFrame:
    path = pathlib.Path(path)
    with open(path, "rb") as handle:
        head = handle.read(24)
    if head.startswith(b"version https"):
        raise RuntimeError(f"{path} is a Git-LFS pointer file")
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"{path} is not a valid npz file")
    arrs = np.load(path)
    return pd.DataFrame({"key": arrs.files, "array": [arrs[k] for k in arrs.files]})


def _rename_columns(data: pd.DataFrame, column_mapping: dict[str, str | None], dropna: bool) -> pd.DataFrame:
    active_mapping = {new: old for new, old in column_mapping.items() if old is not None}
    if "array" in data.columns:
        return data
    if dropna and "smiles" in active_mapping and active_mapping["smiles"] in data.columns:
        data = data.dropna(subset=active_mapping["smiles"])
    missing = [old for old in active_mapping.values() if old not in data.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    return data[list(active_mapping.values())].rename(columns={old: new for new, old in active_mapping.items()})


def load_data_split(config: dict) -> DataSplit:
    source_cfg = config["data"]
    mapping = source_cfg["column_mapping"]
    source_type = source_cfg["source_type"]
    dropna = bool(source_cfg.get("dropna", True))

    if source_type == "presplit_coordinate":
        data_folder = Path(source_cfg["data_folder"])
        train = _rename_columns(pd.read_csv(data_folder / "train.csv"), mapping, dropna)
        val = _rename_columns(pd.read_csv(data_folder / "val.csv"), mapping, dropna)
        test = _rename_columns(pd.read_csv(data_folder / "test.csv"), mapping, dropna)
        return DataSplit(
            train=train,
            val=val,
            test=test,
            train_coord=npz_to_df(data_folder / source_cfg["train_coordinate"]),
            val_coord=npz_to_df(data_folder / source_cfg["val_coordinate"]),
            test_coord=npz_to_df(data_folder / source_cfg["test_coordinate"]),
        )

    if source_type == "single_csv":
        df = _rename_columns(pd.read_csv(source_cfg["data_path"]), mapping, dropna)
        random_df = df.sample(frac=1, random_state=int(config["seed"])).reset_index(drop=True)
        train_ratio = float(source_cfg.get("train_ratio", 0.8))
        val_ratio = float(source_cfg.get("val_ratio", 0.1))
        train_size = int(len(random_df) * train_ratio)
        val_size = int(len(random_df) * val_ratio)
        return DataSplit(
            train=random_df[:train_size].reset_index(drop=True),
            val=random_df[train_size:train_size + val_size].reset_index(drop=True),
            test=random_df[train_size + val_size:].reset_index(drop=True),
        )

    raise ValueError(f"unsupported data source type: {source_type}")


def _empty_like_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    return df.iloc[0:0].copy()


def _combine_for_embedding_export(raw: DataSplit) -> DataSplit:
    frames = [df for df in [raw.train, raw.val, raw.test] if df is not None]
    coords = [df for df in [raw.train_coord, raw.val_coord, raw.test_coord] if df is not None]

    combined_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined_coord = pd.concat(coords, ignore_index=True) if coords else None

    return DataSplit(
        train=combined_df,
        val=_empty_like_df(combined_df),
        test=_empty_like_df(combined_df),
        train_coord=combined_coord,
        val_coord=_empty_like_df(combined_coord),
        test_coord=_empty_like_df(combined_coord),
    )


class GraphDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        representation: CGRRepresentation,
        split: str,
        coordinates: pd.DataFrame | None = None,
        precompute_all: bool = True,
        cache: bool = False,
        cache_path: str | None = None,
        cache_name_prefix: str | None = None,
    ):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.dataframe = dataframe.reset_index(drop=True)
        self.representation = representation
        self.coordinates = coordinates
        self.precompute_all = precompute_all
        self.cache = cache
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache_name_prefix = cache_name_prefix
        self.split = split
        self.precomputed_items: list[tuple[Data, dict[str, torch.Tensor]]] = []
        self._skip_counter: Counter[str] = Counter()
        self._build()

    def _cache_file(self) -> Path | None:
        if not self.cache or self.cache_path is None or self.dataframe.empty:
            return None
        self.cache_path.mkdir(parents=True, exist_ok=True)
        base = self.cache_name_prefix or "graph_cache"
        return self.cache_path / f"{base}_{self.split}.pt"

    def _build(self) -> None:
        cache_file = self._cache_file()
        if self.precompute_all and cache_file is not None and cache_file.exists():
            print(f"INFO: loading dataset cache: {cache_file}")
            self.precomputed_items = torch.load(cache_file, map_location="cpu", weights_only=False)
            return

        total = len(self.dataframe)
        for idx in range(total):
            try:
                row = self.dataframe.iloc[idx]
                extra = None
                if self.coordinates is not None:
                    extra = self.coordinates["array"].iloc[idx]
                data = self.representation(smiles=row["smiles"], extra_atom_features=extra)
                labels = {
                    "ts_label": torch.tensor(
                        float(row["ts_label"]) if "ts_label" in row and pd.notna(row["ts_label"]) else -1.0,
                        dtype=torch.float,
                    ),
                    "reaction_class_label": torch.tensor(
                        int(row["reaction_class_label"])
                        if "reaction_class_label" in row and pd.notna(row["reaction_class_label"])
                        else -1,
                        dtype=torch.long,
                    ),
                }
                self.precomputed_items.append((data, labels))
            except Exception as exc:
                self._skip_counter[type(exc).__name__] += 1

        if cache_file is not None:
            print(f"INFO: saving dataset cache: {cache_file}")
            torch.save(self.precomputed_items, cache_file)

        print("\n====== Dataset preprocessing summary ======")
        print(f"Total samples   : {total}")
        print(f"Kept samples    : {len(self.precomputed_items)}")
        print(f"Dropped samples : {total - len(self.precomputed_items)}")
        if self._skip_counter:
            print("Drop reasons:")
            for key, value in self._skip_counter.most_common():
                print(f"  {key:>20s} : {value}")
        print("==========================================\n")

    def __len__(self) -> int:
        return len(self.precomputed_items)

    def __getitem__(self, idx: int):
        return self.precomputed_items[idx]

    @property
    def mean(self) -> float:
        if "ts_label" not in self.dataframe.columns:
            return -1.0
        series = pd.to_numeric(self.dataframe["ts_label"], errors="coerce")
        return float(series.mean())

    @property
    def std(self) -> float:
        if "ts_label" not in self.dataframe.columns:
            return 1.0
        series = pd.to_numeric(self.dataframe["ts_label"], errors="coerce")
        value = float(series.std(ddof=0))
        return value if value != 0.0 else 1.0

    @property
    def num_node_features(self) -> int:
        return int(self.precomputed_items[0][0].num_node_features)

    @property
    def num_edge_features(self) -> int:
        return int(self.precomputed_items[0][0].num_edge_features)


def build_datasets(config: dict) -> DataSplit:
    raw = load_data_split(config)
    export_mode = config.get("task", {}).get("type") in {"export_embeddings", "export_indexed_embeddings"}
    if export_mode:
        raw = _combine_for_embedding_export(raw)
    representation = CGRRepresentation(use_bond_change_label=bool(config["dataset"].get("use_bond_change_label", False)))
    dataset_cfg = config["dataset"]
    cache_enabled = bool(dataset_cfg.get("cache", False)) and not export_mode
    return DataSplit(
        train=GraphDataset(
            raw.train,
            representation=representation,
            split="train",
            coordinates=raw.train_coord,
            precompute_all=bool(dataset_cfg.get("precompute_all", True)),
            cache=cache_enabled,
            cache_path=dataset_cfg.get("cache_path"),
            cache_name_prefix=dataset_cfg.get("cache_name_prefix"),
        ),
        val=GraphDataset(
            raw.val,
            representation=representation,
            split="val",
            coordinates=raw.val_coord,
            precompute_all=bool(dataset_cfg.get("precompute_all", True)),
            cache=cache_enabled,
            cache_path=dataset_cfg.get("cache_path"),
            cache_name_prefix=dataset_cfg.get("cache_name_prefix"),
        ),
        test=GraphDataset(
            raw.test,
            representation=representation,
            split="test",
            coordinates=raw.test_coord,
            precompute_all=bool(dataset_cfg.get("precompute_all", True)),
            cache=cache_enabled,
            cache_path=dataset_cfg.get("cache_path"),
            cache_name_prefix=dataset_cfg.get("cache_name_prefix"),
        ),
    )


def build_dataloaders(config: dict, datasets: DataSplit):
    dataloader_cfg = config["dataloader"]
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))

    def make_loader(dataset, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=int(dataloader_cfg.get("batch_size", 32)),
            shuffle=shuffle,
            num_workers=int(dataloader_cfg.get("num_workers", 0)),
            pin_memory=bool(dataloader_cfg.get("pin_memory", True)),
            generator=generator,
        )

    shuffle_train = bool(dataloader_cfg.get("shuffle_train", False))
    return DataSplit(
        train=make_loader(datasets.train, shuffle_train),
        val=make_loader(datasets.val, False),
        test=make_loader(datasets.test, False),
    )
