from .data_part.complex_part import pre_aa_complex
from .data_part.base_set.lmdb_dataset import LmdbDataset
from .data_part.base_set.embedding_dataset import EmbeddingDataset
from .data_part.base_set.index_filtered_dataset import IndexFilteredDataset
from .data_part.base_set.row_id_dataset import (
    RowIdInjectedDataset,
    resolve_row_ids_from_df,
)
from tqdm import tqdm
import pandas as pd
import os
import json
import hashlib
import numpy as np
import torch
from pathlib import Path


class ComplexOnlyData:
    CACHE_META_VERSION = 1

    def __init__(
        self,
        istrain=True,
        n_jobs=1,
        shared_context=None,
        **kwargs,
    ):
        self.__dict__.update(shared_context.__dict__)

        train_ds = valid_ds = test_ds = None
        train_cmoplex_df = valid_cmoplex_df = test_cmoplex_df = None

        if istrain:
            train_ds, train_cmoplex_df = self._build_split(
                "train", self.train_csv_path, n_jobs, filter_ts=self.filter_ts_valid
            )
            valid_ds, valid_cmoplex_df = self._build_split(
                "valid", self.valid_csv_path, n_jobs, filter_ts=self.filter_ts_valid
            )
        else:
            test_ds, test_cmoplex_df = self._build_split(
                "test", self.inference_csv_path, n_jobs, filter_ts=self.filter_ts_valid
            )

        self.datasets = (train_ds, valid_ds, test_ds)
        sizes = []
        names = ["train", "valid", "test"]

        for name, ds in zip(names, self.datasets):
            if ds is None:
                sizes.append(0)
                print(f"[ConformationData] {name}: None")
            else:
                n = len(ds)
                sizes.append(n)
                print(f"[ConformationData] {name}: {n}")

        print(f"[ConformationData] Total samples: {sum(sizes)}")
        self.all_df_list = [
            df for df in [train_cmoplex_df, valid_cmoplex_df, test_cmoplex_df]
            if df is not None
        ]

    def _build_split(
        self,
        split_name: str,
        csv_path: str | None,
        n_jobs: int,
        filter_ts: bool = False,
    ):
        if csv_path is None:
            print(f"[{split_name} Set] csv_path is None, skip.")
            return None, None

        tmp_dir = os.path.join(self.base_tmp_dir, split_name)
        os.makedirs(tmp_dir, exist_ok=True)

        base_ds, cmoplex_df = self._get_or_build_lmdb(split_name, csv_path, tmp_dir, n_jobs)
        base_ds = self._filter_required_reaction_idx(base_ds, split_name)

        merged_ds = EmbeddingDataset(
            base_ds,
            sequence_em=self.sequence_em,
            sequence_pocket_aa_em=getattr(self, "sequence_pocket_aa_em", None),
            sequence_pocket_aa_mask=getattr(self, "sequence_pocket_aa_mask", None),
            sequence_pocket_aa_label=getattr(self, "sequence_pocket_aa_label", None),
            sequence_pocket_aa_valid=getattr(self, "sequence_pocket_aa_valid", None),
            sequence_pocket_active_sites=getattr(self, "sequence_pocket_active_sites", None),
            pocket_em=self.pocket_em,
            substrate_em=self.substrate_em,
            product_em=self.product_em,
            ts_reaction_em=self.ts_reaction_em,
            reaction_fp_em=self.reaction_fp_em,
            use_pocket_embedding=self.use_pocket_embedding,
            use_pocket_aa_embedding=getattr(self, "use_pocket_aa_embedding", False),
            use_product_embedding=self.use_product_embedding,
            ts_key="tsr_em",
            ts_mask_key="tsr_mask",
            seq_key="p_em",
            pocket_aa_key="p_aa_em",
            pocket_aa_mask_key="p_aa_mask",
            pocket_aa_label_key="asites_label",
            pocket_aa_valid_key="asites_valid",
            active_sites_key="active_sites",
            pocket_key="pocket_graph",
            sub_key="s_em",
            product_key="prod_em",
            reaction_fp_key="rxn_fp_em",
            clone_item=False,
        )

        if filter_ts:
            keep = self._get_or_build_ts_keep_indices(
                merged_ds=merged_ds,
                split_name=split_name,
                tmp_dir=tmp_dir,
                ts_key="tsr_em",
                mask_key="tsr_mask",
                treat_all_zero_as_invalid=True,
                show_progress=True,
            )
            print(
                f"[{split_name} Set] keep {len(keep)}/{len(merged_ds)} "
                f"by merged_ds tsr_em/tsr_mask"
            )
            merged_ds = IndexFilteredDataset(merged_ds, keep)
        else:
            print(f"[{split_name} Set] skip ts_reaction_em filtering.")

        return merged_ds, cmoplex_df

    @staticmethod
    def _normalize_idx(idx):
        if idx is None:
            return None
        try:
            if pd.isna(idx):
                return None
        except Exception:
            pass
        return int(idx)

    def _reaction_upper_bound(self):
        upper_bounds = []
        if self.reaction_fp_em is not None:
            upper_bounds.append(int(self.reaction_fp_em["emb"].shape[0]))
        if self.ts_reaction_em is not None:
            upper_bounds.append(int(self.ts_reaction_em["emb"].shape[0]))
        if self.use_product_embedding and self.product_em is not None:
            upper_bounds.append(int(self.product_em.shape[0]))
        return min(upper_bounds) if upper_bounds else None

    def _filter_required_reaction_idx(self, base_ds, split_name):
        if not self.filter_ts_valid:
            return base_ds

        upper_bound = self._reaction_upper_bound()
        if upper_bound is None:
            return base_ds

        keep = []
        invalid = 0
        for i in range(len(base_ds)):
            _, item, _ = base_ds[i]
            if item is None:
                invalid += 1
                continue
            r_idx = self._normalize_idx(item.get("r_idx", None))
            if r_idx is None or r_idx < 0 or r_idx >= upper_bound:
                invalid += 1
                continue
            keep.append(i)

        if invalid > 0:
            print(
                f"[{split_name} Set] ts_reaction.filter_valid dropped invalid r_idx: "
                f"kept {len(keep)}/{len(base_ds)}"
            )
            return IndexFilteredDataset(base_ds, keep)
        return base_ds

    def _get_keep_index_cache_path(
        self,
        split_name: str,
        tmp_dir: str,
        ts_key: str,
        mask_key: str,
        treat_all_zero_as_invalid: bool,
    ):
        """Build the cache path for TS keep indices."""
        dataset_embedding_name = getattr(self, "dataset_embedding_name", "default_embedding")
        zero_flag = "drop_allzero" if treat_all_zero_as_invalid else "keep_allzero"

        filename = (
            f"keep_idx__{dataset_embedding_name}"
            f"__{split_name}"
            f"__{ts_key}"
            f"__{mask_key}"
            f"__{zero_flag}.pt"
        )
        return os.path.join(tmp_dir, filename)

    def _cache_meta_path(self, tmp_dir: str):
        return os.path.join(tmp_dir, "cache_meta.json")

    @staticmethod
    def _compute_file_signature(csv_path: str):
        path = Path(csv_path).expanduser().resolve()
        stat = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "csv_path": str(path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": digest.hexdigest(),
        }

    def _expected_cache_meta(self, split_name: str, csv_path: str):
        return {
            "version": self.CACHE_META_VERSION,
            "cache_kind": "complex_only",
            "split_name": split_name,
            "label_key": self.label_key,
            "csv": self._compute_file_signature(csv_path),
        }

    def _validate_cache(self, split_name: str, csv_path: str, tmp_dir: str):
        meta_path = self._cache_meta_path(tmp_dir)
        complex_df_path = os.path.join(tmp_dir, "complex_df.csv")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"[{split_name}] Cache metadata not found: {meta_path}")
        if not os.path.exists(complex_df_path):
            raise FileNotFoundError(f"[{split_name}] Cached complex_df not found: {complex_df_path}")

        with open(meta_path, "r", encoding="utf-8") as handle:
            cache_meta = json.load(handle)

        expected = self._expected_cache_meta(split_name, csv_path)
        if cache_meta != expected:
            raise ValueError(
                f"[{split_name}] Cache metadata mismatch. "
                f"Expected {expected}, got {cache_meta}"
            )

    def _write_cache_meta(self, split_name: str, csv_path: str, tmp_dir: str):
        meta_path = self._cache_meta_path(tmp_dir)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(self._expected_cache_meta(split_name, csv_path), handle, indent=2, sort_keys=True)

    def _get_or_build_ts_keep_indices(
        self,
        merged_ds,
        split_name: str,
        tmp_dir: str,
        ts_key="tsr_em",
        mask_key="tsr_mask",
        treat_all_zero_as_invalid=True,
        show_progress=True,
    ):
        keep_path = self._get_keep_index_cache_path(
            split_name=split_name,
            tmp_dir=tmp_dir,
            ts_key=ts_key,
            mask_key=mask_key,
            treat_all_zero_as_invalid=treat_all_zero_as_invalid,
        )

        if os.path.exists(keep_path):
            keep = torch.load(keep_path, map_location="cpu")
            print(f"[{split_name} Set] loaded keep index from: {keep_path}")
            return keep

        print(f"[{split_name} Set] keep index not found, building: {keep_path}")
        keep = self._filter_indices_by_ts_from_dataset(
            merged_ds,
            ts_key=ts_key,
            mask_key=mask_key,
            treat_all_zero_as_invalid=treat_all_zero_as_invalid,
            show_progress=show_progress,
        )

        torch.save(keep, keep_path)
        print(f"[{split_name} Set] saved keep index to: {keep_path}")
        return keep

    def _get_or_build_lmdb(self, split_name: str, csv_path: str, tmp_dir: str, n_jobs: int):
        """Load cached LMDB or build it with pre_aa_complex."""
        try:
            self._validate_cache(split_name, csv_path, tmp_dir)
            ds = LmdbDataset(cache_dir=tmp_dir, lmdb_name="complex_aa.lmdb", readonly=True)
            cmoplex_df = pd.read_csv(os.path.join(tmp_dir, "complex_df.csv"))
            ds = RowIdInjectedDataset(ds, resolve_row_ids_from_df(cmoplex_df))
            print(f"[{split_name}] Loaded cache: {tmp_dir}")
            return ds, cmoplex_df

        except Exception as e:
            print(f"[{split_name}] Cache load failed: {e}")
            print(f"[{split_name}] Building LMDB from csv...")

            ds, cmoplex_df = pre_aa_complex(self, csv_path, tmp_dir, n_jobs)
            self._write_cache_meta(split_name, csv_path, tmp_dir)
            return ds, cmoplex_df

    def _is_valid_ts_item(
        self,
        item,
        ts_key="tsr_em",
        mask_key="tsr_mask",
        treat_all_zero_as_invalid=True,
    ):
        ts = item.get(ts_key, None)
        mask = item.get(mask_key, None)

        if ts is None:
            return False

        if isinstance(ts, np.ndarray):
            if ts.size == 0:
                return False
            if np.isnan(ts).all():
                return False
            if treat_all_zero_as_invalid and np.all(ts == 0):
                return False

            if mask is not None:
                if isinstance(mask, np.ndarray):
                    if mask.size == 0 or mask.sum() == 0:
                        return False
                else:
                    mask_t = torch.as_tensor(mask)
                    if mask_t.numel() == 0 or mask_t.sum().item() == 0:
                        return False
            return True

        if torch.is_tensor(ts):
            if ts.numel() == 0:
                return False
            if torch.isnan(ts).all().item():
                return False
            if treat_all_zero_as_invalid and (ts == 0).all().item():
                return False

            if mask is not None:
                if torch.is_tensor(mask):
                    if mask.numel() == 0 or mask.sum().item() == 0:
                        return False
                else:
                    mask_t = torch.as_tensor(mask)
                    if mask_t.numel() == 0 or mask_t.sum().item() == 0:
                        return False
            return True

        return False

    def _filter_indices_by_ts_from_dataset(
        self,
        merged_ds,
        ts_key="tsr_em",
        mask_key="tsr_mask",
        treat_all_zero_as_invalid=True,
        show_progress=True,
    ):
        keep = []
        it = range(len(merged_ds))
        if show_progress:
            it = tqdm(it, desc="Filtering by merged_ds ts validity")

        for i in it:
            item = merged_ds[i][1]
            if self._is_valid_ts_item(
                item,
                ts_key=ts_key,
                mask_key=mask_key,
                treat_all_zero_as_invalid=treat_all_zero_as_invalid,
            ):
                keep.append(i)

        return keep
