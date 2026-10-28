from pathlib import Path

import pandas as pd
import torch


def get_or_build_sequence_embedding(
    main_set,
    sequence_csv_path: str,
    output_name: str = "t5_clip_em",
):
    if output_name not in {"t5_global_em", "t5_clip_em", "sequence_em"}:
        raise ValueError(
            f"Unknown sequence embedding output '{output_name}'. "
            "Expected one of: ['t5_global_em', 't5_clip_em', 'sequence_em']"
        )

    save_dir = Path(main_set.embedding_save_dir_base) / main_set.dataset_embedding_name
    save_dir.mkdir(parents=True, exist_ok=True)
    seq_clip_path = save_dir / "sequence_t5_clip.pt"
    seq_t5_path = save_dir / "sequence_t5_global.pt"

    cache = _try_load_base_sequence_cache(
        seq_t5_path=seq_t5_path,
        seq_clip_path=seq_clip_path,
    )
    if cache is None:
        sequence_df = pd.read_csv(sequence_csv_path)
        if "Protein_id" not in sequence_df.columns:
            raise KeyError(f"sequence csv missing required column 'Protein_id': {sequence_csv_path}")
        if "Protein" not in sequence_df.columns:
            raise KeyError(f"sequence csv missing required column 'Protein': {sequence_csv_path}")

        t5_global_em, t5_clip_em = main_set.embeddingmodel.embed_sequences_t5_clip_batch(
            df=sequence_df,
            batch_size=8,
            maxlen=1500,
            output="t5_clip",
        )
        t5_global_em = t5_global_em.detach().cpu()
        t5_clip_em = t5_clip_em.detach().cpu()
        torch.save(t5_clip_em, seq_clip_path)
        torch.save(t5_global_em, seq_t5_path)
    else:
        t5_global_em = cache["t5_global_em"]
        t5_clip_em = cache["t5_clip_em"]

    if output_name == "t5_global_em":
        return t5_global_em
    return t5_clip_em


def _try_load_base_sequence_cache(seq_t5_path: Path, seq_clip_path: Path):
    if not (seq_t5_path.exists() and seq_clip_path.exists()):
        return None
    try:
        return {
            "t5_global_em": torch.load(seq_t5_path, map_location="cpu"),
            "t5_clip_em": torch.load(seq_clip_path, map_location="cpu"),
        }
    except Exception:
        return None
