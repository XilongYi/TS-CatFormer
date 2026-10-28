from __future__ import annotations

from typing import Generic, TypeVar

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing, global_add_pool, global_max_pool, global_mean_pool


T = TypeVar("T")


class DirectedEdgeEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bond_mask_rate: float = 0.0):
        super().__init__()
        self.bond_mask_rate = float(bond_mask_rate)
        self.edge_init = nn.Linear(in_channels + 1, out_channels)
        self.bondtype_slice = (11, 22)
        self.prefer_changed_only = True

    def forward(self, batch: Batch) -> Batch:
        row, _ = batch.edge_index
        edge_attr = batch.edge_attr
        device = edge_attr.device
        edge_mask = torch.zeros(edge_attr.size(0), dtype=torch.bool, device=device)
        mask_flag = torch.zeros(edge_attr.size(0), 1, device=device)

        if self.bond_mask_rate > 0.0:
            edge_mask = self._sample_edge_mask(batch)
            mask_flag = edge_mask.float().unsqueeze(-1)
            if edge_mask.any():
                edge_attr = edge_attr.clone()
                start, end = self.bondtype_slice
                edge_attr[edge_mask, start:end] = 0.0

        batch.edge_mask = edge_mask
        batch.edge_mask_flag = mask_flag.squeeze(-1)
        batch.edge_attr_masked = edge_attr
        batch.h_0 = F.relu(self.edge_init(torch.cat([batch.x[row], edge_attr, mask_flag], dim=1)))
        batch.h = batch.h_0
        return batch

    @torch.no_grad()
    def _sample_edge_mask(self, batch: Batch) -> torch.Tensor:
        edge_count = batch.edge_attr.size(0)
        device = batch.edge_attr.device
        candidate = None

        if self.prefer_changed_only and hasattr(batch, "bond_change_label"):
            labels = batch.bond_change_label.to(device)
            if edge_count == 2 * labels.numel():
                changed = (labels != 0).repeat_interleave(2)
                idx = torch.where(changed)[0]
                if idx.numel() > 0:
                    candidate = idx
            elif labels.numel() == edge_count:
                idx = torch.where(labels != 0)[0]
                if idx.numel() > 0:
                    candidate = idx

        if candidate is None:
            candidate = torch.arange(edge_count, device=device)

        picked = candidate[torch.rand(candidate.numel(), device=device) < self.bond_mask_rate]
        mask = torch.zeros(edge_count, dtype=torch.bool, device=device)
        mask[picked] = True
        return mask


class FeatureEncoderRelu(nn.Module):
    def __init__(
        self,
        feature_in_channels: int,
        feature_hidden_channels: int,
        feature_out_channels: int,
        features_dropout: float,
        modified_in_channels: int,
        edge_embedding_size: int,
    ):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_in_channels, feature_hidden_channels),
            nn.ReLU(),
            nn.Dropout(features_dropout),
            nn.Linear(feature_hidden_channels, feature_out_channels),
        )
        self.edge_init = nn.Linear(modified_in_channels, edge_embedding_size)

    def forward(self, batch: Batch) -> Batch:
        batch.extra_atom_features = self.feature_encoder(batch.extra_atom_features)
        batch.x1 = torch.cat([batch.x, batch.extra_atom_features], dim=1)
        row, _ = batch.edge_index
        batch.h_0 = F.relu(self.edge_init(torch.cat([batch.x1[row], batch.edge_attr], dim=1)))
        batch.h = batch.h_0
        return batch


class DMPNNConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        separate_nn: bool = False,
    ):
        super().__init__()
        self.separate_nn = separate_nn
        self.lin_real = nn.Linear(in_channels, out_channels)
        if separate_nn:
            self.lin_artificial = nn.Linear(in_channels, out_channels)

    def forward(self, batch: Batch) -> Batch:
        edge_index = batch.edge_index
        is_real_bond = getattr(batch, "is_real_bond", None)
        row, _ = edge_index
        aggregated_messages = self.propagate(edge_index, edge_attr=batch.h)
        rev_messages = self._compute_reverse_messages(batch.h)

        if self.separate_nn and is_real_bond is not None:
            batch.h = torch.where(
                is_real_bond.unsqueeze(1),
                self.lin_real(aggregated_messages[row] - rev_messages),
                self.lin_artificial(aggregated_messages[row] - rev_messages),
            )
        else:
            batch.h = self.lin_real(aggregated_messages[row] - rev_messages)
        return batch

    def message(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return edge_attr

    def _compute_reverse_messages(self, edge_attr: torch.Tensor) -> torch.Tensor:
        try:
            return torch.flip(edge_attr.view(edge_attr.size(0) // 2, 2, -1), dims=[1]).view(edge_attr.size(0), -1)
        except Exception:
            return torch.zeros_like(edge_attr)


class DMPNNBlock(nn.Module):
    def __init__(self, hidden_channels: int, dropout: float, residual: bool):
        super().__init__()
        self.graph_conv = DMPNNConv(hidden_channels, hidden_channels, separate_nn=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.use_residual = residual

    def forward(self, batch: Batch) -> Batch:
        residual = batch.h_0
        batch = self.graph_conv(batch)
        batch.h = F.relu(batch.h)
        batch.h = self.dropout(batch.h)
        if self.use_residual:
            batch.h = batch.h + residual
        return batch


class LayerStack(nn.Module, Generic[T]):
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: T) -> T:
        for layer in self.layers:
            x = layer(x)
        return x


class EdgeToNodeEmbedding(nn.Module):
    def __init__(self, embedding_size: int, num_node_features: int, aggr: str = "add"):
        super().__init__()
        self.linear = nn.Linear(num_node_features + embedding_size, embedding_size)
        self.aggr = aggr

    def forward(self, batch: Batch) -> Batch:
        h = batch.h
        target = batch.edge_index[1]
        h_aggr = torch.zeros(batch.num_nodes, h.size(-1), device=h.device, dtype=h.dtype)
        h_aggr.index_add_(0, target, h)
        if self.aggr == "mean" and target.numel() > 0:
            counts = torch.bincount(target, minlength=batch.num_nodes).clamp_min(1).unsqueeze(-1)
            h_aggr = h_aggr / counts
        batch.q = torch.cat([batch.x, h_aggr], dim=1)
        batch.x = self.linear(batch.q)
        batch.x = F.relu(batch.x)
        return batch


class DMPNNStack(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_node_features: int,
        depth: int,
        dropout: float,
        residual: bool = True,
    ):
        super().__init__()
        self.dmpnn_blocks = LayerStack(
            [DMPNNBlock(hidden_channels=hidden_channels, dropout=dropout, residual=residual) for _ in range(depth)]
        )
        self.edge_to_node_embedding = EdgeToNodeEmbedding(
            embedding_size=hidden_channels,
            num_node_features=num_node_features,
            aggr="add",
        )

    def forward(self, batch: Batch) -> Batch:
        batch = self.dmpnn_blocks(batch)
        batch = self.edge_to_node_embedding(batch)
        return batch


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int,
        num_hidden_layers: int,
        dropout: float,
    ):
        super().__init__()
        layers = []
        current = in_channels
        for _ in range(max(0, num_hidden_layers)):
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(current, hidden_size))
            layers.append(nn.ReLU())
            current = hidden_size
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(current, out_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class BondMaskHead(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, dropout: float = 0.1, pooling: str = "mean"):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, batch: Batch):
        h = batch.h if hasattr(batch, "h") else batch.h_0
        edge_count, hidden = h.shape
        if hidden != self.hidden_size:
            raise ValueError(f"hidden size mismatch: expected {self.hidden_size}, got {hidden}")
        if edge_count % 2 != 0:
            raise ValueError(f"edge count must be even, got {edge_count}")

        if hasattr(batch, "edge_mask"):
            edge_mask = batch.edge_mask.bool()
            bond_mask = edge_mask[0::2] | edge_mask[1::2]
        else:
            bond_mask = torch.zeros(edge_count // 2, dtype=torch.bool, device=h.device)

        h0, h1 = h[0::2], h[1::2]
        bond_h = 0.5 * (h0 + h1) if self.pooling == "mean" else (h0 + h1)
        logits = self.classifier(self.dropout(bond_h))
        return logits, bond_mask, batch.bond_change_label


class GNNModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        layer_stack: nn.Module,
        pool: str,
        head: nn.Module,
        bond_mask_head: nn.Module | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.layer_stack = layer_stack
        self.pool = pool
        self.head = head
        self.bond_mask_head = bond_mask_head

    def _pool(self, batch: Batch) -> torch.Tensor:
        if self.pool == "mean":
            return global_mean_pool(batch.x, batch.batch)
        if self.pool == "max":
            return global_max_pool(batch.x, batch.batch)
        return global_add_pool(batch.x, batch.batch)

    def forward(self, batch: Batch, return_batch_embedding: bool = False):
        batch = self.encoder(batch)
        batch = self.layer_stack(batch)
        if return_batch_embedding:
            return batch
        if self.bond_mask_head is not None:
            return self.bond_mask_head(batch)
        graph_h = self._pool(batch)
        return self.head(graph_h)


def build_model(config: dict, num_node_features: int, num_edge_features: int) -> GNNModel:
    model_cfg = config["model"]
    hidden_channels = int(model_cfg["hidden_channels"])

    if model_cfg.get("encoder_type") == "feature_relu":
        feature_out = int(model_cfg["feature_out_channels"])
        encoder = FeatureEncoderRelu(
            feature_in_channels=int(model_cfg["feature_in_channels"]),
            feature_hidden_channels=int(model_cfg["feature_hidden_channels"]),
            feature_out_channels=feature_out,
            features_dropout=float(model_cfg["features_dropout"]),
            modified_in_channels=int(model_cfg.get("modified_in_channels", feature_out + num_node_features + num_edge_features)),
            edge_embedding_size=hidden_channels,
        )
    else:
        encoder = DirectedEdgeEncoder(
            in_channels=num_node_features + num_edge_features,
            out_channels=hidden_channels,
            bond_mask_rate=float(model_cfg.get("bond_mask_rate", 0.0)),
        )

    layer_stack = DMPNNStack(
        hidden_channels=hidden_channels,
        num_node_features=num_node_features,
        depth=int(model_cfg["depth"]),
        dropout=float(model_cfg["block_dropout"]),
        residual=bool(model_cfg.get("residual", True)),
    )
    head = MLP(
        in_channels=hidden_channels,
        out_channels=int(model_cfg.get("out_channels", 1)),
        hidden_size=int(model_cfg["head_hidden_size"]),
        num_hidden_layers=int(model_cfg["head_num_hidden_layers"]),
        dropout=float(model_cfg["head_dropout"]),
    )
    bond_mask_head = None
    if config["task"]["type"] == "bond_change_train":
        bond_mask_head = BondMaskHead(
            num_classes=int(model_cfg.get("bond_mask_num_classes", 31)),
            hidden_size=hidden_channels,
            dropout=float(model_cfg.get("bond_mask_dropout", 0.1)),
        )

    return GNNModel(
        encoder=encoder,
        layer_stack=layer_stack,
        pool=str(model_cfg.get("pool", "add")),
        head=head,
        bond_mask_head=bond_mask_head,
    )
