from pathlib import Path
import torch
import pandas as pd


def _try_load_tensor_cache(save_path: Path):
    if not save_path.exists():
        return None
    try:
        return torch.load(save_path, map_location="cpu")
    except Exception:
        return None


def _get_or_build_smiles_embedding(
    main_set,
    csv_path: str,
    save_name: str,
    smiles_col: str | None = None,
):
    save_dir = Path(main_set.embedding_save_dir_base) / main_set.dataset_embedding_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / save_name

    smiles_em = _try_load_tensor_cache(save_path)
    if smiles_em is not None:
        return smiles_em

    smiles_df = pd.read_csv(csv_path)
    smiles_em = main_set.embeddingmodel.embed_smiles_molformer_batch(
        df=smiles_df,
        batch_size=64,
        smiles_col=smiles_col,
    )
    torch.save(smiles_em.detach().cpu(), save_path)
    return smiles_em


def get_or_build_substrate_embedding(main_set, substrate_csv_path: str):
    substrate_em = _get_or_build_smiles_embedding(
        main_set,
        substrate_csv_path,
        "substrate_molformer.pt",
        smiles_col="Substrate",
    )
    return substrate_em


def get_or_build_product_embedding(main_set, product_csv_path: str):
    product_em = _get_or_build_smiles_embedding(
        main_set,
        product_csv_path,
        "product_molformer.pt",
        smiles_col="Product",
    )
    return product_em
