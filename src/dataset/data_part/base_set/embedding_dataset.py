import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict
from pandas import isna


class EmbeddingDataset(Dataset):
    """Attach embeddings to base dataset items."""

    def __init__(
        self,
        base_ds,
        sequence_em: torch.Tensor | None,
        pocket_em: Dict | None,
        substrate_em: torch.Tensor | None,
        product_em: torch.Tensor | None,
        ts_reaction_em: Dict | None,
        reaction_fp_em: Dict | None,
        sequence_pocket_aa_em: torch.Tensor | None = None,
        sequence_pocket_aa_mask: torch.Tensor | None = None,
        sequence_pocket_aa_label: torch.Tensor | None = None,
        sequence_pocket_aa_valid: torch.Tensor | None = None,
        sequence_pocket_active_sites=None,
        use_pocket_embedding=True,
        use_pocket_aa_embedding=False,
        use_product_embedding=True,
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
    ):
        self.ds = base_ds
        self.sequence_em = sequence_em
        self.sequence_pocket_aa_em = sequence_pocket_aa_em
        self.sequence_pocket_aa_mask = sequence_pocket_aa_mask
        self.sequence_pocket_aa_label = sequence_pocket_aa_label
        self.sequence_pocket_aa_valid = sequence_pocket_aa_valid
        self.sequence_pocket_active_sites = sequence_pocket_active_sites
        self.pocket_em = pocket_em
        self.substrate_em = substrate_em
        self.product_em = product_em
        self.ts_reaction_em = ts_reaction_em
        self.reaction_fp_em = reaction_fp_em
        self.use_pocket_embedding = use_pocket_embedding
        self.use_pocket_aa_embedding = use_pocket_aa_embedding
        self.use_product_embedding = use_product_embedding
        self.ts_key = ts_key
        self.ts_mask_key = ts_mask_key
        self.seq_key = seq_key
        self.pocket_aa_key = pocket_aa_key
        self.pocket_aa_mask_key = pocket_aa_mask_key
        self.pocket_aa_label_key = pocket_aa_label_key
        self.pocket_aa_valid_key = pocket_aa_valid_key
        self.active_sites_key = active_sites_key
        self.pocket_key = pocket_key
        self.sub_key = sub_key
        self.product_key = product_key
        self.reaction_fp_key = reaction_fp_key
        self.clone_item = clone_item

    @staticmethod
    def _safe_get_idx(idx):
        if idx is None or isna(idx):
            return None
        try:
            if isinstance(idx, float) and np.isnan(idx):
                return None
        except Exception:
            pass
        return int(idx)

    @staticmethod
    def _get_required_em(em: torch.Tensor, idx, name: str):
        """Return a required embedding."""
        if em is None:
            raise ValueError(f"{name} embedding tensor is None")

        idx = EmbeddingDataset._safe_get_idx(idx)
        if idx is None:
            raise ValueError(f"{name} index is None")

        if idx < 0 or idx >= em.shape[0]:
            raise IndexError(f"{name} idx {idx} out of range (0, {em.shape[0]-1})")

        return em[idx]

    @staticmethod
    def _get_optional_ts(em: torch.Tensor | None, idx, name: str):
        """Return TS embedding or a zero fallback."""
        if em is None:
            return None

        idx = EmbeddingDataset._safe_get_idx(idx)
        if idx is None or idx < 0 or idx >= em.shape[0]:
            return torch.zeros_like(em[0])

        return em[idx]

    @staticmethod
    def _get_required_graph(cache: Dict | None, idx, name: str):
        if cache is None:
            raise ValueError(f"{name} cache is None")

        idx = EmbeddingDataset._safe_get_idx(idx)
        if idx is None:
            raise ValueError(f"{name} index is None")

        graphs = cache.get("graphs", None)
        if graphs is None:
            raise KeyError(f"{name} cache missing 'graphs'")

        if idx < 0 or idx >= len(graphs):
            raise IndexError(f"{name} idx {idx} out of range (0, {len(graphs)-1})")

        graph = graphs[idx]
        graph_p_idx = EmbeddingDataset._safe_get_idx(graph.get("p_idx", None))
        if graph_p_idx != idx:
            raise ValueError(f"{name} graph p_idx mismatch: graph_p_idx={graph_p_idx}, idx={idx}")
        return graph

    @staticmethod
    def _get_optional_em(em: torch.Tensor | None, idx, name: str):
        if em is None:
            return None

        idx = EmbeddingDataset._safe_get_idx(idx)
        if idx is None:
            return None

        if idx < 0 or idx >= em.shape[0]:
            return None

        return em[idx]

    @staticmethod
    def _get_optional_item(seq, idx):
        if seq is None:
            return None
        idx = EmbeddingDataset._safe_get_idx(idx)
        if idx is None:
            return None
        if idx < 0 or idx >= len(seq):
            return None
        return seq[idx]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        name, item, idx = self.ds[i]
        if item is None:
            return name, None, idx

        if self.clone_item:
            item = dict(item)

        p_idx = item.get("p_idx", None)
        s_idx = item.get("s_idx", None)
        r_idx = item.get("r_idx", None)

        item[self.seq_key] = self._get_optional_em(self.sequence_em, p_idx, "sequence_em")
        if item.get(self.pocket_aa_key, None) is None and self.use_pocket_aa_embedding:
            item[self.pocket_aa_key] = self._get_optional_em(
                self.sequence_pocket_aa_em,
                p_idx,
                "sequence_pocket_aa_em",
            )
        if item.get(self.pocket_aa_mask_key, None) is None and self.use_pocket_aa_embedding:
            item[self.pocket_aa_mask_key] = self._get_optional_em(
                self.sequence_pocket_aa_mask,
                p_idx,
                "sequence_pocket_aa_mask",
            )
        if item.get(self.pocket_aa_label_key, None) is None and self.use_pocket_aa_embedding:
            label = self._get_optional_em(
                self.sequence_pocket_aa_label,
                p_idx,
                "sequence_pocket_aa_label",
            )
            mask = self._get_optional_em(
                self.sequence_pocket_aa_mask,
                p_idx,
                "sequence_pocket_aa_mask",
            )
            if label is not None:
                if mask is not None:
                    label = label[mask.bool()]
                item[self.pocket_aa_label_key] = label.detach().cpu().tolist()
        if item.get(self.pocket_aa_valid_key, None) is None and self.use_pocket_aa_embedding:
            valid = self._get_optional_em(
                self.sequence_pocket_aa_valid,
                p_idx,
                "sequence_pocket_aa_valid",
            )
            if valid is not None:
                item[self.pocket_aa_valid_key] = bool(valid.item())
        if item.get(self.active_sites_key, None) is None and self.use_pocket_aa_embedding:
            item[self.active_sites_key] = self._get_optional_item(
                self.sequence_pocket_active_sites,
                p_idx,
            )
        item[self.pocket_key] = (
            self._get_required_graph(self.pocket_em, p_idx, "pocket_em")
            if self.use_pocket_embedding
            else None
        )
        item["pocket_graph_valid"] = bool(
            self.use_pocket_embedding
            and item[self.pocket_key] is not None
            and item[self.pocket_key].get("is_valid", False)
        )
        item[self.sub_key] = self._get_optional_em(self.substrate_em, s_idx, "substrate_em")
        item[self.product_key] = (
            self._get_required_em(self.product_em, r_idx, "product_em")
            if self.use_product_embedding
            else None
        )

        if self.reaction_fp_em is not None:
            item[self.reaction_fp_key] = self._get_required_em(
                self.reaction_fp_em["emb"], r_idx, "reaction_fp_em"
            )

        if self.ts_reaction_em is not None:
            item[self.ts_key] = self._get_optional_ts(
                self.ts_reaction_em["emb"], r_idx, "ts_em"
            )

            ts_mask = self.ts_reaction_em.get("mask", None)
            if ts_mask is not None:
                mask = self._get_optional_ts(ts_mask.unsqueeze(-1), r_idx, "ts_mask")
                if mask is not None:
                    mask = mask.squeeze(-1)
                item[self.ts_mask_key] = mask
            else:
                item[self.ts_mask_key] = None

        return name, item, idx
