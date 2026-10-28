import math

import torch
import torch.nn as nn


def normalize_ts_batch(ts_batch: torch.Tensor, ts_key: str):
    ts_div = {
        "3d_features_no_mask": 1.0e5,
        "no_coordinates_mask_only": 1.0,
        "no_coordinates_mask": 1.944,
        "no_coordinates_no_mask": 1.0,
    }
    ref_max = 3.421556
    clamp_k = 10.0
    div = ts_div.get(ts_key, 1.0)

    dtype0 = ts_batch.dtype
    x = ts_batch.float()
    nz = x != 0
    if nz.any():
        x[nz] = x[nz] / div
        x[nz] = x[nz].clamp(-ref_max * clamp_k, ref_max * clamp_k)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.to(dtype0)


def normalize_global_emb(x, last_dim, name="emb"):
    if x is None:
        raise ValueError(f"{name} is None")
    if x.ndim == 1:
        if x.shape[0] != last_dim:
            raise ValueError(f"{name} shape error: got {tuple(x.shape)}, expect [{last_dim}]")
        return x.unsqueeze(0)
    if x.ndim == 2:
        if x.shape[-1] != last_dim:
            raise ValueError(f"{name} shape error: got {tuple(x.shape)}, expect [B,{last_dim}]")
        return x
    if x.ndim == 3 and x.shape[1] == 1 and x.shape[2] == last_dim:
        return x.squeeze(1)
    raise ValueError(f"{name} shape error: got {tuple(x.shape)}")


def normalize_ts_emb(ts, ts_mask, ts_dim=600):
    if ts is None:
        raise ValueError("tsr_em is None")
    if ts_mask is None:
        raise ValueError("tsr_mask is None")
    if ts.ndim != 3 or ts.shape[-1] != ts_dim:
        raise ValueError(f"tsr_em shape error: got {tuple(ts.shape)}, expect [B,L,{ts_dim}]")
    if ts_mask.ndim != 2:
        raise ValueError(f"tsr_mask shape error: got {tuple(ts_mask.shape)}, expect [B,L]")
    if ts.shape[:2] != ts_mask.shape:
        raise ValueError(
            f"tsr_em / tsr_mask mismatch: tsr_em[:2]={tuple(ts.shape[:2])}, tsr_mask={tuple(ts_mask.shape)}"
        )
    return ts, ts_mask.bool()


def masked_sum(x, mask, dim=1):
    mask_f = mask.to(x.dtype).unsqueeze(-1)
    return (x * mask_f).sum(dim=dim)


def masked_mean(x, mask, dim=1, eps=1e-8):
    mask_f = mask.to(x.dtype).unsqueeze(-1)
    x_sum = (x * mask_f).sum(dim=dim)
    denom = mask_f.sum(dim=dim).clamp_min(eps)
    return x_sum / denom


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv, mask=None):
        B, Lq, H = q.shape
        _, Lk, _ = kv.shape

        q_in = self.norm_q(q)
        kv_in = self.norm_kv(kv)

        Q = self.q_proj(q_in).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv_in).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv_in).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_mask = mask[:, None, None, :].bool()
            attn = attn.masked_fill(~attn_mask, float("-inf"))
            all_invalid = (~mask).all(dim=1)
            if all_invalid.any():
                attn[all_invalid] = 0.0

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, Lq, H)
        out = self.o_proj(out)

        out = q + self.dropout(out)
        out = out + self.ffn(self.norm_out(out))
        return out


class ReactionContextEncoder(nn.Module):
    def __init__(self, s_dim, ts_dim, model_dim, hidden_dim, dropout, marker_dim):
        super().__init__()
        self.s_proj = nn.Linear(s_dim, model_dim)
        self.ts_proj = nn.Linear(ts_dim, model_dim)
        self.valid_proj = nn.Sequential(
            nn.Linear(1, marker_dim),
            nn.GELU(),
            nn.Linear(marker_dim, model_dim),
        )
        self.pre_fuse = nn.Sequential(
            nn.Linear(model_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
            nn.GELU(),
        )

    def forward(self, sub_emb, ts_batch, ts_mask, dtype):
        s = self.s_proj(sub_emb)
        ts_sum_raw = masked_sum(ts_batch, ts_mask)
        ts_sum = self.ts_proj(ts_sum_raw)
        valid_marker = (ts_sum_raw.abs().sum(dim=-1, keepdim=True) > 0).float().to(dtype)
        v = self.valid_proj(valid_marker)
        fused = self.pre_fuse(torch.cat([ts_sum, s, v], dim=-1))
        return s, ts_sum, v, fused


class CatalyticReactionFeasibilityHead(nn.Module):
    def __init__(self, p_in_dim, hidden_dim, dropout_rate, num_heads, ts_in_dim, substrate_in_dim):
        super().__init__()
        self.p_in_dim = p_in_dim
        self.s_dim = substrate_in_dim
        self.ts_dim = ts_in_dim
        self.model_dim = hidden_dim

        self.p_proj = nn.Linear(p_in_dim, hidden_dim)
        self.virtual_node_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.virtual_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_encoder = ReactionContextEncoder(
            s_dim=substrate_in_dim,
            ts_dim=ts_in_dim,
            model_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout_rate,
            marker_dim=32,
        )
        self.cross_attn = CrossAttentionBlock(dim=hidden_dim, num_heads=num_heads, dropout=dropout_rate)
        self.pocket_ts_attn = CrossAttentionBlock(dim=hidden_dim, num_heads=num_heads, dropout=dropout_rate)
        self.pocket_sub_attn = CrossAttentionBlock(dim=hidden_dim, num_heads=num_heads, dropout=dropout_rate)
        self.pocket_valid_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, hidden_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 10, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2),
        )
        self.supports_seq_only_input = True

    def _encode_pocket(self, pocket_outputs, batch_size, dtype, device):
        if pocket_outputs is None:
            tokens = torch.zeros(batch_size, 1, self.model_dim, device=device, dtype=dtype)
            mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
            valid = torch.zeros(batch_size, dtype=torch.bool, device=device)
            return tokens, mask, valid

        tokens = pocket_outputs.get("tokens", None)
        mask = pocket_outputs.get("mask", None)
        valid = pocket_outputs.get("valid_mask", None)

        if tokens is None or mask is None:
            tokens = torch.zeros(batch_size, 1, self.model_dim, device=device, dtype=dtype)
            mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        else:
            tokens = tokens.to(device=device, dtype=dtype)
            mask = mask.to(device=device).bool()

        if valid is None:
            valid = mask.any(dim=1)
        else:
            valid = valid.to(device=device).bool() & mask.any(dim=1)

        tokens = tokens * mask.to(dtype).unsqueeze(-1)
        return tokens, mask, valid

    def _build_pocket_interactions(self, pocket_tokens, pocket_mask, pocket_valid, s, ts_sum):
        B = pocket_tokens.size(0)
        dtype = pocket_tokens.dtype
        device = pocket_tokens.device

        ts_token = ts_sum.unsqueeze(1)
        sub_token = s.unsqueeze(1)
        single_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

        pocket_ts = self.pocket_ts_attn(q=pocket_tokens, kv=ts_token, mask=single_mask)
        pocket_sub = self.pocket_sub_attn(q=pocket_tokens, kv=sub_token, mask=single_mask)

        valid_scale = pocket_valid.to(dtype).unsqueeze(-1)
        pocket_feat = masked_mean(pocket_tokens, pocket_mask) * valid_scale
        pocket_ts = masked_mean(pocket_ts, pocket_mask) * valid_scale
        pocket_sub = masked_mean(pocket_sub, pocket_mask) * valid_scale
        pocket_valid_feat = self.pocket_valid_proj(valid_scale)
        return pocket_feat, pocket_ts, pocket_sub, pocket_valid_feat

    def forward(self, new_node_feats, pocket_outputs, new_inter_edge_dense, batched_data):
        prot_emb = normalize_global_emb(batched_data["p_em"], self.p_in_dim, name="p_em")
        sub_emb = normalize_global_emb(batched_data["s_em"], self.s_dim, name="s_em")
        ts_batch = normalize_ts_batch(batched_data["tsr_em"], batched_data["ts_key"])
        ts_batch, ts_mask = normalize_ts_emb(ts_batch, batched_data["tsr_mask"], ts_dim=self.ts_dim)

        dtype = prot_emb.dtype
        device = prot_emb.device
        batch_size = prot_emb.size(0)

        p = self.p_proj(prot_emb)
        if new_node_feats is not None:
            if new_node_feats.ndim != 3 or new_node_feats.size(1) == 0:
                raise ValueError(
                    f"new_node_feats shape error: got {tuple(new_node_feats.shape)}, expect [B,N,H] with N>0"
                )
            virtual_node = self.virtual_node_proj(new_node_feats[:, 0, :].to(dtype))
            p = p + self.virtual_fuse(torch.cat([p, virtual_node], dim=-1))
        s, ts_sum, v, fused = self.context_encoder(
            sub_emb=sub_emb,
            ts_batch=ts_batch,
            ts_mask=ts_mask,
            dtype=dtype,
        )

        pocket_tokens, pocket_mask, pocket_valid = self._encode_pocket(
            pocket_outputs=pocket_outputs,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        pocket_feat, pocket_ts, pocket_sub, pocket_valid_feat = self._build_pocket_interactions(
            pocket_tokens=pocket_tokens,
            pocket_mask=pocket_mask,
            pocket_valid=pocket_valid,
            s=s,
            ts_sum=ts_sum,
        )

        fused_kv = fused.unsqueeze(1)
        fused_mask = torch.ones(batch_size, 1, device=device, dtype=torch.bool)
        p_attn = self.cross_attn(q=p.unsqueeze(1), kv=fused_kv, mask=fused_mask).squeeze(1)

        feat = torch.cat(
            [p, s, ts_sum, v, fused, p_attn, pocket_feat, pocket_ts, pocket_sub, pocket_valid_feat],
            dim=-1,
        )
        logits = self.head(feat)

        final = [0.0, 0.0, 0.0, 0.0, None]
        final.append(logits)
        return final
