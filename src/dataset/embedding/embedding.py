from transformers import T5EncoderModel, T5Tokenizer
from transformers import AutoModel, AutoTokenizer
import re
import torch
import numpy as np
import pandas as pd
import onnxruntime as ort
from tqdm import tqdm
import os

def l2_normalize(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    L2 normalize last dimension.
    Supports 1D or 2D tensors (and more).
    """
    return X / (X.norm(p=2, dim=-1, keepdim=True) + eps)


class ONNXModel:
    """Wrapper for an ONNX model. Supports both (D,) and (B,D) inputs."""

    def __init__(self, path: str):
        path = str(path)
        self.model = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def predict(self, x: np.ndarray, apply_norm: bool = True) -> np.ndarray:
        """
        x: (D,) or (B,D)
        returns: (D2,) or (B,D2)
        """
        if apply_norm:
            if x.ndim == 1:
                x = x / (np.linalg.norm(x) + 1e-12)
            elif x.ndim == 2:
                x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
            else:
                raise ValueError("x must be 1D or 2D")

        if x.dtype != np.float32:
            x = x.astype(np.float32)

        if x.ndim == 1:
            out = self.model.run(None, {"input": x[None, :]})[0]
            return out.squeeze(0)
        else:
            # batch
            out = self.model.run(None, {"input": x})[0]
            return out


def get_t5_clip_model(device: torch.device = torch.device("cuda:0")):
    t5_model_path = "checkpoints/prot_t5_xl_half_uniref50-enc"
    if not os.path.exists(t5_model_path) or not os.listdir(t5_model_path):
        raise FileNotFoundError(
            f"No models found in the directory: {t5_model_path}\n"
            f"You need to follow the instructions in the README to download models"
        )

    model_t5 = T5EncoderModel.from_pretrained(t5_model_path)
    tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
    model_t5.eval().to(device)

    clip_model_path = "checkpoints/clip/proteinclip_prott5.onnx"
    clip_model = ONNXModel(clip_model_path)

    return model_t5, tokenizer, clip_model


class EmbeddingModel:
    def __init__(self, tmp_dir: str, work_path: str):
        self.protein_model = "t5_clip"
        self.smiles_model = "molformer"
        self.device = torch.device("cuda:0")


        self.prot_t5_model, self.prot_t5_tokenizer, self.clip_model = get_t5_clip_model(self.device)

        MoLFormer_path = "checkpoints/MoLFormer-model/"
        self.molformer_model = AutoModel.from_pretrained(
            MoLFormer_path, deterministic_eval=True, trust_remote_code=True
        )
        self.molformer_tokenizer = AutoTokenizer.from_pretrained(MoLFormer_path, trust_remote_code=True)
        self.molformer_model.eval()

    @staticmethod
    def _truncate_seqs(seqs, maxlen: int):
        if isinstance(seqs, str):
            return seqs[:maxlen]
        if isinstance(seqs, list):
            return [("" if s is None else str(s))[:maxlen] for s in seqs]
        raise TypeError("sequence must be str or list[str]")

    @staticmethod
    def _sanitize_seqs(seqs: list[str]) -> list[str]:
        # Replace uncommon AA with X, then add spaces for ProtT5 tokenizer
        return [" ".join(list(re.sub(r"[UZOB]", "X", s))) for s in seqs]

    def _t5_encode(self, seqs: list[str], device=None, maxlen: int = 1500):
        """
        seqs: list[str] raw amino acid sequences (no spaces)
        returns:
          rep:  [B, L, D] on device
          attn: [B, L] on device
        """
        if device is None:
            device = self.device

        san = self._sanitize_seqs(seqs)
        ids = self.prot_t5_tokenizer(
            san,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=maxlen,
            return_tensors="pt",
        )
        input_ids = ids["input_ids"].to(device)
        attn = ids["attention_mask"].to(device)

        with torch.no_grad():
            rep = self.prot_t5_model(input_ids=input_ids, attention_mask=attn).last_hidden_state
        return rep, attn

    @staticmethod
    def _pool_global(rep: torch.Tensor, attn: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        rep:  [B, L, D]
        attn: [B, L]
        return:
          global_vec: [B, D] mean over non-pad tokens + L2 norm
        """
        mask = attn.unsqueeze(-1).float()  # [B, L, 1]
        summed = (rep * mask).sum(dim=1)   # [B, D]
        denom = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
        global_vec = summed / denom
        global_vec = l2_normalize(global_vec, eps=eps)
        return global_vec

    @staticmethod
    def _extract_residue(rep: torch.Tensor, attn: torch.Tensor, eps: float = 1e-12):
        """
        rep:  [B, L, D]
        attn: [B, L]
        return list of residue tensors (CPU):
          residue_list[i] shape: [Li, D] (Li = attn sum for sample i), token-level L2 norm
        """
        residue_list = []
        B = rep.shape[0]
        for i in range(B):
            Li = int(attn[i].sum().item())
            aa = rep[i, :Li, :].detach().cpu()
            aa = l2_normalize(aa, eps=eps)
            residue_list.append(aa)
        return residue_list

    def _clip_from_global(self, global_vec: torch.Tensor) -> torch.Tensor:
        """
        global_vec: [B, D] torch tensor (GPU/CPU)
        returns clip_vec: [B, D2] torch CPU tensor
        """
        x = global_vec.detach().cpu().numpy().astype(np.float32)
        out = self.clip_model.predict(x)  # (B,D2)
        if out.ndim == 1:
            out = out[None, :]
        return torch.from_numpy(out)

    @staticmethod
    def _expand_global_as_residue_tokens(global_vec: torch.Tensor, length: int) -> torch.Tensor:
        if length <= 0:
            length = 1
        return global_vec.unsqueeze(0).repeat(length, 1).detach().cpu()

    @staticmethod
    def _match_pdb_residues_to_residue_em(
        residue_em: torch.Tensor,
        global_em: torch.Tensor,
        sequence: str,
        pdb_residue_keys,
    ):
        mismatch_count = 0
        out_of_range_count = 0

        if not pdb_residue_keys:
            return (
                EmbeddingModel._expand_global_as_residue_tokens(global_em, 1),
                1,
                0,
                True,
            )

        matched = []
        use_global_fallback = False
        seq_len = len(sequence)
        residue_len = int(residue_em.size(0))
        hidden_dim = int(global_em.size(0))

        for residx, res1 in pdb_residue_keys:
            if residx <= 0 or residx > seq_len or residx > residue_len:
                out_of_range_count += 1
                use_global_fallback = True
                break

            if sequence[residx - 1] != res1:
                mismatch_count += 1
                use_global_fallback = True
                break

            vec = residue_em[residx - 1, :].detach().cpu()
            if vec.numel() != hidden_dim:
                mismatch_count += 1
                use_global_fallback = True
                break
            matched.append(vec)

        if use_global_fallback:
            fallback = EmbeddingModel._expand_global_as_residue_tokens(global_em, len(pdb_residue_keys))
            return fallback, mismatch_count, out_of_range_count, True

        if len(matched) == 0:
            fallback = EmbeddingModel._expand_global_as_residue_tokens(global_em, 1)
            return fallback, mismatch_count, out_of_range_count, True

        return torch.stack(matched, dim=0).detach().cpu(), mismatch_count, out_of_range_count, False


    def get_pocket_from_residue_em(self, residue_em: torch.Tensor, aa_dict: dict, maxlen: int, sequence: str):
        """
        residue_em: [L, D] CPU or GPU tensor; we will .detach().cpu() inside
        aa_dict: { "115_R": atom_ids, ... }  (atom_ids unused here, kept for compatibility)
        sequence: raw sequence (no spaces)
        """
        pocket_aa_map_em = {}
        out_of_range = 0
        mismatch = 0

        for key, atom_ids in aa_dict.items():
            idx_str, res1 = key.split("_")
            idx = int(idx_str)

            if 1 <= idx <= maxlen:
                try:
                    seq_res = sequence[idx - 1]  # 1-based -> 0-based
                except IndexError:
                    out_of_range += 1
                    continue

                if seq_res == res1:
                    try:
                        vec = residue_em[idx - 1, :].detach().cpu()
                        pocket_aa_map_em[key] = vec
                    except Exception:
                        mismatch += 1
                else:
                    mismatch += 1
            else:
                out_of_range += 1

        if mismatch > 0 or out_of_range > 0:
            print(f"Mismatch: {mismatch}, out of range: {out_of_range}")

        return pocket_aa_map_em

    def embed_sequences_t5_clip_batch(
        self,
        df: pd.DataFrame,
        batch_size: int = 4,
        device: torch.device = torch.device("cuda:0"),
        maxlen: int = 1500,
        output: str = "t5_clip",
    ):
        """
        Returns:
          if output == "t5":       global_t5 [N, D] (CPU)
          if output == "t5_clip":  (global_t5 [N,D], clip [N,D2]) (CPU, CPU)
        """
        all_global = []
        all_clip = []

        for idx in tqdm(range(0, len(df), batch_size), desc="Embedding sequences with ProtT5 Clip"):
            batch_df = df.iloc[idx: idx + batch_size]
            seqs = batch_df["Protein"].tolist()
            seqs = self._truncate_seqs(seqs, maxlen)

            rep, attn = self._t5_encode(seqs, device=device, maxlen=maxlen)
            global_vec = self._pool_global(rep, attn)  # [B, D] on device
            all_global.append(global_vec.detach().cpu())

            if output == "t5_clip":
                clip_vec = self._clip_from_global(global_vec)  # [B, D2] CPU
                all_clip.append(clip_vec)

        global_out = torch.cat(all_global, dim=0)

        if output == "t5":
            return global_out.detach().cpu().to(dtype=torch.float16).contiguous()

        clip_out = torch.cat(all_clip, dim=0) if len(all_clip) else None
        return global_out.detach().cpu().to(dtype=torch.float16).contiguous(), clip_out.detach().cpu().to(dtype=torch.float16).contiguous()

    def embed_sequences_t5_clip_pocket_aa_batch(
        self,
        df: pd.DataFrame,
        pocket_residue_keys: list,
        batch_size: int = 4,
        device: torch.device = torch.device("cuda:0"),
        maxlen: int = 1500,
        output: str = "t5_clip",
    ):
        """
        Returns:
          if output == "t5":
            global_t5 [N, D], pocket_aa [N, Lmax, D], pocket_aa_mask [N, Lmax]
          if output == "t5_clip":
            global_t5 [N, D], clip [N, D2], pocket_aa [N, Lmax, D], pocket_aa_mask [N, Lmax]
        """
        if len(df) != len(pocket_residue_keys):
            raise ValueError(
                f"len(df)={len(df)} must equal len(pocket_residue_keys)={len(pocket_residue_keys)}"
            )

        all_global = []
        all_clip = []
        pocket_aa_list = []
        mismatch_protein_count = 0
        mismatch_residue_count = 0
        out_of_range_residue_count = 0
        global_fallback_count = 0

        for start in tqdm(range(0, len(df), batch_size), desc="Embedding sequences with ProtT5 Pocket AA"):
            batch_df = df.iloc[start: start + batch_size]
            seqs = self._truncate_seqs(batch_df["Protein"].tolist(), maxlen)
            batch_keys = pocket_residue_keys[start: start + len(batch_df)]

            rep, attn = self._t5_encode(seqs, device=device, maxlen=maxlen)
            global_vec = self._pool_global(rep, attn).detach().cpu()
            residue_list = self._extract_residue(rep, attn)
            all_global.append(global_vec)

            if output == "t5_clip":
                all_clip.append(self._clip_from_global(global_vec))

            for local_idx, (seq, residue_em, pdb_keys) in enumerate(zip(seqs, residue_list, batch_keys)):
                matched, mismatch_cnt, out_of_range_cnt, used_global = self._match_pdb_residues_to_residue_em(
                    residue_em=residue_em,
                    global_em=global_vec[local_idx],
                    sequence=seq,
                    pdb_residue_keys=pdb_keys,
                )
                pocket_aa_list.append(matched.detach().cpu().to(dtype=torch.float16).contiguous())
                mismatch_residue_count += mismatch_cnt
                out_of_range_residue_count += out_of_range_cnt
                if mismatch_cnt > 0 or out_of_range_cnt > 0:
                    mismatch_protein_count += 1
                if used_global:
                    global_fallback_count += 1

        global_out = torch.cat(all_global, dim=0).detach().cpu().to(dtype=torch.float16).contiguous()
        hidden_dim = int(global_out.size(-1))
        lengths = [int(x.size(0)) for x in pocket_aa_list]
        lmax = max(lengths) if lengths else 1
        padded = []
        masks = []
        for aa in pocket_aa_list:
            cur = aa
            cur_mask = torch.zeros(lmax, dtype=torch.bool)
            cur_mask[: cur.size(0)] = True
            if cur.size(0) < lmax:
                pad = torch.zeros((lmax - cur.size(0), hidden_dim), dtype=cur.dtype)
                cur = torch.cat([cur, pad], dim=0)
            padded.append(cur)
            masks.append(cur_mask)
        pocket_aa_out = torch.stack(padded, dim=0).contiguous()
        pocket_aa_mask = torch.stack(masks, dim=0).contiguous()

        print(f"[PocketAA] mismatched proteins: {mismatch_protein_count}")
        print(f"[PocketAA] mismatched residues: {mismatch_residue_count}")
        print(f"[PocketAA] out-of-range residues: {out_of_range_residue_count}")
        print(f"[PocketAA] global fallback proteins: {global_fallback_count}")

        if output == "t5":
            return global_out, pocket_aa_out, pocket_aa_mask

        clip_out = torch.cat(all_clip, dim=0).detach().cpu().to(dtype=torch.float16).contiguous()
        return global_out, clip_out, pocket_aa_out, pocket_aa_mask

    def molformer_embedding(self, smiles: str) -> np.ndarray:
        """
        returns numpy array shape (1, D)
        """
        smiles = "" if smiles is None else str(smiles)
        smiles_input = self.molformer_tokenizer([smiles], padding=True, return_tensors="pt")
        with torch.no_grad():
            output = self.molformer_model(**smiles_input)
            em = output.pooler_output.detach().cpu().numpy()
        return em


    def embed_smiles_molformer_batch(
        self,
        df: pd.DataFrame,
        batch_size: int = 64,
        smiles_col: str | None = None,
    ):
        """
        Returns:
          em_all: [N, D] (CPU torch)
        """
        if smiles_col is None:
            for candidate in ["Substrate", "Product", "SMILES", "smiles"]:
                if candidate in df.columns:
                    smiles_col = candidate
                    break
        if smiles_col is None or smiles_col not in df.columns:
            raise KeyError(
                f"SMILES column not found. Expected one of ['Substrate', 'Product', 'SMILES', 'smiles'], got {list(df.columns)}"
            )

        all_em = []

        for idx in tqdm(range(0, len(df), batch_size), desc="Embedding smiles with MoLFormer"):
            batch_df = df.iloc[idx: idx + batch_size]
            smiles_list = batch_df[smiles_col].tolist()
            smiles_list = [("" if s is None else str(s)) for s in smiles_list]

            smiles_input = self.molformer_tokenizer(smiles_list, padding=True, return_tensors="pt")
            with torch.no_grad():
                output = self.molformer_model(**smiles_input)
                em = output.pooler_output.detach()
                em = l2_normalize(em)
            all_em.append(em.cpu())

        return torch.cat(all_em, dim=0).detach().cpu().to(dtype=torch.float16).contiguous()
    
    def get_ts_reaction_embedding(
        self,
        df: pd.DataFrame,
        ts_reaction_save: dict,
        dtype=torch.float16,
        mask_dtype=torch.uint8,
        strict_rest_shape=True,
        verbose=True,
    ):
        reactions = df["Reaction"].tolist()
        N = len(reactions)
        out = {}

        for ts_key, ts_reaction_save_path in ts_reaction_save.items():
            # Allow empty paths.
            if ts_reaction_save_path is None or str(ts_reaction_save_path).strip() == "":
                if verbose:
                    print(f"[{ts_key}] ts_reaction_save_path is empty, return None embeddings.")
                out[ts_key] = {"emb": None, "mask": None}
                continue

            try:
                ts_reaction_dict = torch.load(ts_reaction_save_path, map_location="cpu")
            except FileNotFoundError:
                if verbose:
                    print(f"[{ts_key}] file not found: {ts_reaction_save_path}, return None embeddings.")
                out[ts_key] = {"emb": None, "mask": None}
                continue

            # Scan shapes.
            Lmax = 0
            rest_shape = None
            found = 0
            miss_key = 0
            bad_type = 0
            bad_rest = 0

            for smi in reactions:
                e = ts_reaction_dict.get(smi, None)
                if e is None:
                    miss_key += 1
                    continue

                t = e[0] if isinstance(e, (list, tuple)) else e
                if not isinstance(t, torch.Tensor):
                    bad_type += 1
                    continue
                if t.dim() < 2:
                    bad_type += 1
                    continue

                this_rest = tuple(t.shape[1:])
                if rest_shape is None:
                    rest_shape = this_rest
                elif this_rest != rest_shape:
                    bad_rest += 1
                    if strict_rest_shape:
                        continue
                    else:
                        continue

                Li = int(t.shape[0])
                if Li > Lmax:
                    Lmax = Li
                found += 1

            if rest_shape is None or Lmax == 0 or found == 0:
                out[ts_key] = {"emb": None, "mask": None}
                if verbose:
                    print(f"[{ts_key}] no valid tensors found. miss_key={miss_key}, bad_type={bad_type}, bad_rest={bad_rest}")
                continue

            if verbose:
                print(
                    f"[{ts_key}] N={N}, found={found}, Lmax={Lmax}, rest_shape={rest_shape}, "
                    f"miss_key={miss_key}, bad_type={bad_type}, bad_rest={bad_rest}"
                )

            # Allocate output.
            emb = torch.zeros((N, Lmax, *rest_shape), dtype=dtype)
            mask = torch.zeros((N, Lmax), dtype=mask_dtype)

            # Fill output.
            filled = 0
            skipped_rest = 0

            for i, smi in enumerate(reactions):
                e = ts_reaction_dict.get(smi, None)
                if e is None:
                    continue

                t = e[0] if isinstance(e, (list, tuple)) else e
                if not isinstance(t, torch.Tensor) or t.dim() < 2:
                    continue

                if tuple(t.shape[1:]) != rest_shape:
                    skipped_rest += 1
                    continue

                t = t.detach().cpu().to(dtype=dtype).contiguous()
                Li = int(t.shape[0])
                Li_clip = min(Li, Lmax)

                emb[i, :Li_clip] = t[:Li_clip]
                mask[i, :Li_clip] = 1
                filled += 1

            if verbose:
                print(f"[{ts_key}] filled={filled}, skipped_rest={skipped_rest}")

            out[ts_key] = {"emb": emb, "mask": mask}

        return out

        
