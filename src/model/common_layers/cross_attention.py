import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    """Protein-ligand cross attention."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv):
        # q, kv: [B, 1, dim].
        attn_out, _ = self.attn(q, kv, kv)
        q = q + self.dropout(attn_out)
        q = self.norm(q)
        return q
    