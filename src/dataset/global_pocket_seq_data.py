from torch.utils.data import Dataset
import pandas as pd
import os
import json
import hashlib
from tqdm import tqdm
import torch
from pathlib import Path

from .data_part.base_set.lmdb_dataset import LmdbDataset
from .data_part.base_set.embedding_dataset import EmbeddingDataset
from .data_part.base_set.index_filtered_dataset import IndexFilteredDataset
from .data_part.base_set.row_id_dataset import (
    RowIdInjectedDataset,
    resolve_row_ids_from_df,
)

class GlobalPocketSeqDataset(Dataset):
    CACHE_META_VERSION = 1

    def __init__(self, main_dataset, istrain=True, existing_df=None):
        self.main_dataset = main_dataset
        if existing_df is not None:
            self.existing_df = existing_df
        else:
            all_df_list = getattr(self.main_dataset, "all_df_list", None)
            self.existing_df = pd.concat(all_df_list) if all_df_list else None
    
        self.base_tmp_dir = self.main_dataset.base_tmp_dir
        train_data = valid_data = test_data = None
        (
            self.ts_reaction_em,
            self.substrate_em,
            self.sequence_em,
            self.sequence_pocket_aa_em,
            self.sequence_pocket_aa_mask,
            self.sequence_pocket_aa_label,
            self.sequence_pocket_aa_valid,
            self.sequence_pocket_active_sites,
            self.pocket_em,
            self.product_em,
            self.reaction_fp_em,
        ) = (
            main_dataset.ts_reaction_em,
            main_dataset.substrate_em,
            main_dataset.sequence_em,
            getattr(main_dataset, "sequence_pocket_aa_em", None),
            getattr(main_dataset, "sequence_pocket_aa_mask", None),
            getattr(main_dataset, "sequence_pocket_aa_label", None),
            getattr(main_dataset, "sequence_pocket_aa_valid", None),
            getattr(main_dataset, "sequence_pocket_active_sites", None),
            main_dataset.pocket_em,
            main_dataset.product_em,
            main_dataset.reaction_fp_em,
        )
        if istrain:
            train_data = self._get_split_data("train", self.main_dataset.train_csv_path)
            valid_data = self._get_split_data("valid", self.main_dataset.valid_csv_path)
        else:
            test_data = self._get_split_data("test", self.main_dataset.inference_csv_path)

        self.datasets = (train_data, valid_data, test_data)
        sizes = []
        names = ["train", "valid", "test"]

        for name, ds in zip(names, self.datasets):
            if ds is None:
                sizes.append(0)
                print(f"[NoStructData] {name}: None")
            else:
                n = len(ds)
                sizes.append(n)
                print(f"[NoStructData] {name}: {n}")


    def _get_split_data(self, split_name, csv_path):
        tmp_dir_save = os.path.join(self.base_tmp_dir, split_name)
        seq_df_path = os.path.join(tmp_dir_save, "seq_df.csv")
    
        if csv_path is None:
            print(f"Warning: {split_name} CSV path is None. Skipping.")
            return None

        os.makedirs(tmp_dir_save, exist_ok=True)

        # Try cache first.
        try:
            self._validate_cache(split_name, csv_path, tmp_dir_save, seq_df_path)
            filtered_ds = LmdbDataset(cache_dir=tmp_dir_save, lmdb_name="data_no_complex.lmdb", readonly=True)
            print(f"[{split_name}] Loaded cache with {len(filtered_ds)} rows")
            if os.path.exists(seq_df_path):
                seq_df = pd.read_csv(seq_df_path)
                filtered_ds = RowIdInjectedDataset(filtered_ds, resolve_row_ids_from_df(seq_df))
            filtered_ds = self._filter_required_reaction_idx(filtered_ds, split_name)
            merged_ds = EmbeddingDataset(
                filtered_ds,
                sequence_em=self.sequence_em,
                sequence_pocket_aa_em=self.sequence_pocket_aa_em,
                sequence_pocket_aa_mask=self.sequence_pocket_aa_mask,
                sequence_pocket_aa_label=self.sequence_pocket_aa_label,
                sequence_pocket_aa_valid=self.sequence_pocket_aa_valid,
                sequence_pocket_active_sites=self.sequence_pocket_active_sites,
                pocket_em=self.pocket_em,
                substrate_em=self.substrate_em,
                product_em=self.product_em,
                ts_reaction_em=self.ts_reaction_em,
                reaction_fp_em=self.reaction_fp_em,
                use_pocket_embedding=self.main_dataset.use_pocket_embedding,
                use_pocket_aa_embedding=self.main_dataset.use_pocket_aa_embedding,
                use_product_embedding=self.main_dataset.use_product_embedding,
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
            return merged_ds
        except Exception as e:
            print(f"[{split_name}] Cache load failed: {e}")

        seq_df = self._filter_new_df(split_name, csv_path)
        if seq_df is None:
            return None
        
        # Build base rows.
        data_list = []
        print('[NoStructData] Start sequence & substrate construction...')
        for i in tqdm(range(len(seq_df)), desc=f"Construction {split_name}"):
            row = seq_df.iloc[i]

            item = {
                "label": row[self.main_dataset.label_key],
                "r_idx": row["r_idx"],
                "s_idx": row["s_idx"],
                "p_idx": row["p_idx"],
                "target": row["Target"],
                "row_id": row["row_id"],
            }

            data_list.append(item)

        mask = [x is not None for x in data_list]

        n_total_after = len(data_list)
        n_valid_after = sum(mask)

        print(f'[SequenceData] After TS embedding: total={n_total_after}, non-None={n_valid_after}')

        data_list = [x for x in data_list if x is not None]

        seq_df = seq_df.loc[mask].reset_index(drop=True)

        filtered_ds = self.main_dataset.assign_label2x(self.main_dataset.pic50_key, 
                self.main_dataset.label_key, self.main_dataset.sites_key, seq_df, data_list)
        if filtered_ds is None:
            print(f"[{split_name}] Empty after label assignment")
            return None

        seq_df.to_csv(os.path.join(tmp_dir_save, "seq_df.csv"), index=False)
        LmdbDataset.write(rows=filtered_ds, cache_dir=tmp_dir_save, lmdb_name="data_no_complex.lmdb")
        self._write_cache_meta(split_name, csv_path, tmp_dir_save)
        filtered_ds = LmdbDataset(cache_dir=tmp_dir_save, lmdb_name="data_no_complex.lmdb", readonly=True)
        filtered_ds = self._filter_required_reaction_idx(filtered_ds, split_name)

        merged_ds = EmbeddingDataset(
            filtered_ds,
            sequence_em=self.sequence_em,
            sequence_pocket_aa_em=self.sequence_pocket_aa_em,
            sequence_pocket_aa_mask=self.sequence_pocket_aa_mask,
            sequence_pocket_aa_label=self.sequence_pocket_aa_label,
            sequence_pocket_aa_valid=self.sequence_pocket_aa_valid,
            sequence_pocket_active_sites=self.sequence_pocket_active_sites,
            pocket_em=self.pocket_em,
            substrate_em=self.substrate_em,
            product_em=self.product_em,
            ts_reaction_em=self.ts_reaction_em,
            reaction_fp_em=self.reaction_fp_em,
            use_pocket_embedding=self.main_dataset.use_pocket_embedding,
            use_pocket_aa_embedding=self.main_dataset.use_pocket_aa_embedding,
            use_product_embedding=self.main_dataset.use_product_embedding,
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

        return merged_ds

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

    @staticmethod
    def _coerce_series_to_str(series):
        return series.map(lambda value: "" if pd.isna(value) else str(value).strip())

    def _standardize_input_df(self, df):
        for column in [self.main_dataset.protein_id, self.main_dataset.substrate_id, "Target"]:
            if column and column in df.columns:
                df[column] = self._coerce_series_to_str(df[column])

        return df

    def _cache_meta_path(self, tmp_dir_save):
        return os.path.join(tmp_dir_save, "cache_meta.json")

    @staticmethod
    def _compute_file_signature(csv_path):
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

    def _expected_cache_meta(self, split_name, csv_path):
        return {
            "version": self.CACHE_META_VERSION,
            "cache_kind": "global_pocket_seq",
            "split_name": split_name,
            "label_key": self.main_dataset.label_key,
            "csv": self._compute_file_signature(csv_path),
        }

    def _validate_cache(self, split_name, csv_path, tmp_dir_save, seq_df_path):
        meta_path = self._cache_meta_path(tmp_dir_save)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"[{split_name}] Cache metadata not found: {meta_path}")
        if not os.path.exists(seq_df_path):
            raise FileNotFoundError(f"[{split_name}] Cached seq_df not found: {seq_df_path}")

        with open(meta_path, "r", encoding="utf-8") as handle:
            cache_meta = json.load(handle)

        expected = self._expected_cache_meta(split_name, csv_path)
        if cache_meta != expected:
            raise ValueError(
                f"[{split_name}] Cache metadata mismatch. "
                f"Expected {expected}, got {cache_meta}"
            )

    def _write_cache_meta(self, split_name, csv_path, tmp_dir_save):
        meta_path = self._cache_meta_path(tmp_dir_save)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(self._expected_cache_meta(split_name, csv_path), handle, indent=2, sort_keys=True)

    def _reaction_upper_bound(self):
        upper_bounds = []
        if self.reaction_fp_em is not None:
            upper_bounds.append(int(self.reaction_fp_em["emb"].shape[0]))
        if self.ts_reaction_em is not None:
            upper_bounds.append(int(self.ts_reaction_em["emb"].shape[0]))
        if self.main_dataset.use_product_embedding and self.product_em is not None:
            upper_bounds.append(int(self.product_em.shape[0]))
        return min(upper_bounds) if upper_bounds else None

    def _filter_required_reaction_idx(self, base_ds, split_name):
        if not self.main_dataset.filter_ts_valid:
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
                f"[{split_name}] ts_reaction.filter_valid dropped invalid r_idx: "
                f"kept {len(keep)}/{len(base_ds)}"
            )
            return IndexFilteredDataset(base_ds, keep)
        return base_ds
    

    
    def _filter_new_df(self, split_name, csv_path):
        df = pd.read_csv(csv_path)
        df = self._standardize_input_df(df)
        df["row_id"] = resolve_row_ids_from_df(df)
        label_key = self.main_dataset.label_key
        if "Target" not in df.columns:
            required_columns = [self.main_dataset.protein_id, self.main_dataset.substrate_id]
            missing_columns = [column for column in required_columns if column not in df.columns]
            if missing_columns:
                raise KeyError(
                    f"[{split_name}] Missing required columns for Target construction: {missing_columns}. "
                    f"Actual columns: {list(df.columns)}"
                )
            df["Target"] = df[self.main_dataset.protein_id] + "_" + df[self.main_dataset.substrate_id]

        if label_key and label_key not in df.columns:
            if split_name == "test":
                df[label_key] = 0
            else:
                raise KeyError(
                    f"[{split_name}] Missing required label column: {label_key}. "
                    f"Actual columns: {list(df.columns)}"
                )

        print(f"[{split_name}] Loaded {len(df)} rows")

        if self.existing_df is None or len(self.existing_df) == 0:
            print(f"[{split_name}] Standalone no-struct mode; use all rows")
            return df

        common_cols = self.existing_df.columns

        df["_merge_key"] = df[common_cols].astype(str).agg("|".join, axis=1)
        existing_keys = set(
            self.existing_df[common_cols].astype(str).agg("|".join, axis=1)
        )

        seq_df = df[~df["_merge_key"].isin(existing_keys)].copy()
        seq_df.drop(columns=["_merge_key"], inplace=True)

        print(f"[{split_name}] {len(seq_df)} rows need encoding")
        if len(seq_df) == 0:
            print(f"[{split_name}] No new rows to encode")
            return None

        return seq_df
    
