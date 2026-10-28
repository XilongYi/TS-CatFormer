import torch.nn as nn
import torch.nn.functional as F
import torch

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention with gated residual fusion.
    A: [B, LA, H] or [B, H]
    B: [B, LB, H]
    Output: same shape as A
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, gated=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.gated = gated

        # Linear projections.
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.normA = nn.LayerNorm(hidden_dim)
        self.normB = nn.LayerNorm(hidden_dim)
        self.alignA = nn.Linear(hidden_dim, hidden_dim)
        self.alignB = nn.Linear(hidden_dim, hidden_dim)

        # Learnable gate.
        if gated:
            # Broadcast over sequence dimensions.
            self.alpha = nn.Parameter(torch.ones(1, 1, hidden_dim) * 1e-3)

    def forward(self, A, B, mask=None):
        # Support [B, H] inputs.
        unsqueeze_later = False
        if A.dim() == 2:   # [B, H]
            A = A.unsqueeze(1)
            unsqueeze_later = True

        A = self.alignA(self.normA(A))
        B = self.alignB(self.normB(B))

        Bsz, LA, H = A.shape
        LB = B.shape[1]
        Hh = H // self.num_heads

        # Project inputs.
        Q = self.q_proj(A)
        K = self.k_proj(B)
        V = self.v_proj(B)

        Q = Q.view(Bsz, LA, self.num_heads, Hh).transpose(1, 2)
        K = K.view(Bsz, LB, self.num_heads, Hh).transpose(1, 2)
        V = V.view(Bsz, LB, self.num_heads, Hh).transpose(1, 2)

        # Attention.
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (Hh ** 0.5)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask[:, None, :, :], -1e4)

        attn_scores = torch.clamp(attn_scores, -50, 50)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(Bsz, LA, H)
        out = self.out_proj(out)
        out = torch.nan_to_num(out, nan=0.0)

        # Gated residual fusion.
        if self.gated:
            out = A + self.alpha * out
            # print(self.alpha)
        else:
            out = A + out

        # Restore shape.
        if unsqueeze_later:
            out = out.squeeze(1)

        return out