from pathlib import Path

import torch


def get_or_build_reaction_fp_embedding(main_set, rxn_fp_emb_path, rxn_fp_mask_path=None):
    rxn_fp_emb_path = Path(rxn_fp_emb_path)
    if not rxn_fp_emb_path.exists():
        raise FileNotFoundError(f"Reaction FP emb file not found: {rxn_fp_emb_path}")

    try:
        emb = torch.load(rxn_fp_emb_path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load emb: {rxn_fp_emb_path}") from e

    if not isinstance(emb, torch.Tensor):
        raise TypeError(f"emb is not torch.Tensor, got {type(emb)}")

    return {"emb": emb, "mask": None}
