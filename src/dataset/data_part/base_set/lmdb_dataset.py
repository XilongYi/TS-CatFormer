import os
import glob
import shutil
import pickle
import lmdb
import numpy as np
import pandas as pd


class LmdbDataset:
    def __init__(
        self,
        cache_dir: str,
        lmdb_name: str = "data.lmdb",
        readonly: bool = True,
        lock: bool = False,
        readahead: bool = False,
        meminit: bool = False,
        max_readers: int = 256,
    ):
        self.cache_dir = cache_dir
        self.lmdb_path = os.path.join(cache_dir, lmdb_name)

        self.env = None
        self.txn = None
        self.keys = None
        self._len = None

        if readonly:
            self.open(
                readonly=True,
                lock=lock,
                readahead=readahead,
                meminit=meminit,
                max_readers=max_readers,
            )

        self.meta_df = None
        self.labels = None
        self.targets = None

    @staticmethod
    def exists(cache_glob: str) -> bool:
        """Check if cached path exists (supports glob like your old code)."""
        hits = glob.glob(cache_glob)
        return len(hits) > 0

    @staticmethod
    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def clear_dir(path: str):
        if os.path.exists(path):
            shutil.rmtree(path)

    def open(self, readonly=True, lock=False, readahead=False, meminit=False, max_readers=256):
        """Open LMDB env. Each worker should open its own env in practice."""
        self.ensure_dir(self.cache_dir)

        self.env = lmdb.open(
            self.lmdb_path,
            subdir=False,
            readonly=readonly,
            lock=lock,
            readahead=readahead,
            meminit=meminit,
            max_readers=max_readers,
        )
        self.txn = self.env.begin(write=False)

        self.keys = list(self.txn.cursor().iternext(values=False))
        self._len = len(self.keys)

    def close(self):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        self.env, self.txn, self.keys, self._len = None, None, None, None

    @property
    def csv_path(self) -> str:
        return os.path.join(self.cache_dir, "data.csv")

    def save_metadata(self, df: pd.DataFrame, filename: str = "data.csv"):
        """Write metadata CSV to cache_dir."""
        self.ensure_dir(self.cache_dir)
        path = os.path.join(self.cache_dir, filename)
        df.to_csv(path, index=False)

    def load_metadata(
        self,
        filename: str = "data.csv",
        label_col: str | None = None,
        target_cols: list[str] | None = None,
        adjust_label_fn=None,
        return_df: bool = False,
    ):
        """
        Generic metadata loader (optional).
        - label_col: column name for labels (optional)
        - target_cols: list of column names to build targets (optional)
        - adjust_label_fn: callable(y: np.ndarray)->np.ndarray (optional)
        """
        path = os.path.join(self.cache_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"metadata not found: {path}")

        df = pd.read_csv(path)
        self.meta_df = df

        if label_col is not None and label_col in df.columns:
            y = df[label_col].to_numpy()
            if adjust_label_fn is not None:
                y = adjust_label_fn(y)
            self.labels = y

        if target_cols is not None:
            missing = [c for c in target_cols if c not in df.columns]
            if missing:
                raise KeyError(f"missing target columns in metadata: {missing}")
            self.targets = (
                df[target_cols].astype(str).agg("_".join, axis=1).tolist()
                if len(target_cols) > 1
                else df[target_cols[0]].astype(str).tolist()
            )

        return df if return_df else None

    @staticmethod
    def stats_data(rows: list[dict]):
        pass

    @classmethod
    def write(
        cls,
        rows: list,
        cache_dir: str,
        lmdb_name: str = "data.lmdb",
        map_size: int = 1099511627776,
    ) -> "LmdbDataset":
        """
        Create cache_dir, write LMDB and optional metadata CSV, then return an opened dataset.
        """
        if rows is None:
            rows = []
        cls.ensure_dir(cache_dir)

        lmdb_path = os.path.join(cache_dir, lmdb_name)

        try:
            cls.stats_data(rows)
        except Exception:
            pass

        env = lmdb.open(lmdb_path, subdir=False, map_size=map_size)
        txn = env.begin(write=True)

        for i, row in enumerate(rows):
            key = str(i).encode()
            val = pickle.dumps(row, protocol=pickle.HIGHEST_PROTOCOL)
            txn.put(key=key, value=val)

        txn.commit()
        env.close()

        return cls(cache_dir=cache_dir, lmdb_name=lmdb_name, readonly=True)

    def __len__(self):
        if self._len is None:
            return 0
        return self._len

    def __getitem__(self, idx: int):
        """
        Fast path assumes keys are '0..N-1' strings.
        If you ever stored non-sequential keys, change to:
            key = self.keys[idx]
        """
        if self.txn is None:
            raise RuntimeError("LMDB is not opened. Call open() first.")

        key = str(idx).encode()
        buf = self.txn.get(key)
        if buf is None:
            if self.keys is not None and 0 <= idx < len(self.keys):
                buf = self.txn.get(self.keys[idx])
            if buf is None:
                raise KeyError(f"missing key for idx={idx}")

        x = pickle.loads(buf)
        return f"Complex:{idx}", x, idx