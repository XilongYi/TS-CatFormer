import torch
import torch.nn as nn

from src.model.common_layers.feed_forward import FeedForwardNetwork
from src.model.common_layers.multi_head_attention import MultiHeadAttention
from .gvp_core import GVP, GVPConvLayer, LayerNorm


def init_last_residual_layer(layer):
    layer.weight.data.zero_()
    layer.bias.data.zero_()


class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads, num_layer):
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = MultiHeadAttention(
            hidden_dim, attention_dropout_rate, num_heads, num_layer
        )
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForwardNetwork(hidden_dim, ffn_dim, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

        self.edge_attn_norm = nn.LayerNorm(hidden_dim)
        self.edge_ffn_norm = nn.LayerNorm(hidden_dim)
        self.edge_ffn = FeedForwardNetwork(hidden_dim, ffn_dim, dropout_rate)

        self.init_param()

    def init_param(self):
        init_last_residual_layer(self.ffn.layer2)
        init_last_residual_layer(self.edge_ffn.layer2)

    def forward(self, x, e, attn_bias=None):
        y = self.self_attention_norm(x)
        e_hat = self.edge_attn_norm(e)
        y, e_hat = self.self_attention(y, y, y, e_hat, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y

        e_hat = self.self_attention_dropout(e_hat)
        e = e + e_hat
        e_hat = self.edge_ffn_norm(e)
        e_hat = self.edge_ffn(e_hat)
        e_hat = self.ffn_dropout(e_hat)
        e = e + e_hat
        return x, e


class PocketGVPEncoder(nn.Module):
    def __init__(
        self,
        node_in_dim=(6, 3),
        node_h_dim=(64, 16),
        edge_in_dim=(32, 1),
        edge_h_dim=(32, 1),
        num_layers=3,
        drop_rate=0.1,
    ):
        super().__init__()
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None)),
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
        )
        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers)
        )
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (2 * ns, 0)),
        )

    def forward(self, h_V, edge_index, h_E):
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        return self.W_out(h_V)


class PocketInteractionLayer(nn.Module):
    def __init__(
        self,
        n_intra_encoder,
        n_pocket_encoder,
        n_inter_encoder,
        hidden_dim,
        ffn_dim,
        dropout_rate,
        attention_dropout_rate,
        num_heads,
        pocket_node_in_dim=(6, 3),
        pocket_node_h_dim=(64, 16),
        pocket_edge_in_dim=(32, 1),
        pocket_edge_h_dim=(32, 1),
        pocket_num_layers=3,
    ):
        super().__init__()
        self.intra_encoder = nn.ModuleList(
            [
                EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads, i)
                for i in range(n_intra_encoder)
            ]
        )
        self.pocket_encoder = nn.ModuleList(
            [
                EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads, i)
                for i in range(n_pocket_encoder)
            ]
        )
        self.inter_encoder = nn.ModuleList(
            [
                EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads, i)
                for i in range(n_inter_encoder)
            ]
        )
        self.pocket_gvp_encoder = PocketGVPEncoder(
            node_in_dim=pocket_node_in_dim,
            node_h_dim=pocket_node_h_dim,
            edge_in_dim=pocket_edge_in_dim,
            edge_h_dim=pocket_edge_h_dim,
            num_layers=pocket_num_layers,
            drop_rate=dropout_rate,
        )
        self.pocket_proj = nn.Linear(pocket_node_h_dim[0] * 2, hidden_dim)

    def forward(
        self,
        batched_data,
        node_feats,
        intra_edge_dense,
        inter_edge_dense,
        intra_attn_bias,
        inter_attn_bias,
        pocket_attn_bias,
    ):
        if (
            node_feats is None
            or intra_edge_dense is None
            or inter_edge_dense is None
            or intra_attn_bias is None
            or inter_attn_bias is None
            or pocket_attn_bias is None
        ):
            pocket_outputs = self._encode_pocket_graph(batched_data, dtype=torch.float32)
            return None, pocket_outputs, None, batched_data

        for i in range(len(self.intra_encoder)):
            intra_node, _ = self.intra_encoder[i](node_feats, intra_edge_dense, intra_attn_bias)

        if len(self.pocket_encoder) == 0:
            intra_pocket_node = intra_node
        else:
            for i in range(len(self.pocket_encoder)):
                intra_pocket_node, _ = self.pocket_encoder[i](
                    intra_node, inter_edge_dense, pocket_attn_bias
                )

        for i in range(len(self.inter_encoder)):
            new_node_feats, new_inter_edge_dense = self.inter_encoder[i](
                intra_pocket_node, inter_edge_dense, inter_attn_bias
            )

        pocket_outputs = self._encode_pocket_graph(batched_data, dtype=new_node_feats.dtype)
        return new_node_feats, pocket_outputs, new_inter_edge_dense, batched_data

    def _encode_pocket_graph(self, batched_data, dtype):
        pocket_graph = batched_data.get("pocket_graph", None)
        valid_mask = batched_data.get("pocket_graph_valid", None)

        if pocket_graph is None or pocket_graph.get("h_V", None) is None:
            return {"tokens": None, "mask": None, "valid_mask": valid_mask}

        node_output = self.pocket_gvp_encoder(
            pocket_graph["h_V"],
            pocket_graph["edge_index"],
            pocket_graph["h_E"],
        )
        pocket_tokens, pocket_mask = self._pack_pocket_tokens(
            node_output=node_output,
            batch_index=pocket_graph["batch"],
            batch_size=len(batched_data["target"]),
        )
        pocket_tokens = self.pocket_proj(pocket_tokens.to(dtype))

        if valid_mask is not None:
            valid_mask = valid_mask.bool().to(pocket_tokens.device)
            pocket_mask = pocket_mask & valid_mask.unsqueeze(1)
            pocket_tokens = pocket_tokens * pocket_mask.to(dtype).unsqueeze(-1)

        return {
            "tokens": pocket_tokens,
            "mask": pocket_mask,
            "valid_mask": valid_mask,
        }

    def _pack_pocket_tokens(self, node_output, batch_index, batch_size):
        hidden = node_output.size(-1)
        counts = torch.bincount(batch_index, minlength=batch_size)
        lmax = max(int(counts.max().item()), 1)

        tokens = node_output.new_zeros(batch_size, lmax, hidden)
        mask = torch.zeros(batch_size, lmax, dtype=torch.bool, device=node_output.device)

        for b in range(batch_size):
            cur = node_output[batch_index == b]
            cur_len = cur.size(0)
            if cur_len == 0:
                continue
            tokens[b, :cur_len] = cur
            mask[b, :cur_len] = True
        return tokens, mask
