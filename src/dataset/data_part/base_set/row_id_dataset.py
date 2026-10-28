from torch.utils.data import Dataset


def resolve_row_ids_from_df(df):
    if "row_id" in df.columns:
        return df["row_id"].tolist()
    if "row id" in df.columns:
        return df["row id"].tolist()
    return df.index.tolist()


class RowIdInjectedDataset(Dataset):
    def __init__(self, base_ds, row_ids):
        self.base_ds = base_ds
        self.row_ids = list(row_ids)

        if len(self.base_ds) != len(self.row_ids):
            raise ValueError(
                f"row_ids length mismatch: len(base_ds)={len(self.base_ds)} vs len(row_ids)={len(self.row_ids)}"
            )

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        name, item, item_idx = self.base_ds[idx]
        if item is None:
            return name, item, item_idx

        if item.get("row_id", None) != self.row_ids[idx]:
            item = dict(item)
            item["row_id"] = self.row_ids[idx]

        return name, item, item_idx
