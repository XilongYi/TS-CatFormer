import torch.nn as nn
from src.model.common_layers.positionwise_feed_forward import PositionwiseFeedForward


def init_embedding(layer):
    if isinstance(layer, nn.Embedding):
        if len(layer.weight) > 1:
            layer.weight.data[0].zero_()
    if isinstance(layer, nn.Linear) and layer.bias is not None:
        layer.bias.data.zero_()

class AtomHead(nn.Module):
    def __init__(self, hidden_dim, node_feat_size, dropout_rate=0.2):
        super(AtomHead, self).__init__()
        self.node_ln = nn.LayerNorm(hidden_dim)
        self.atom_type_proj = PositionwiseFeedForward(hidden_dim, hidden_dim * 4, d_out=node_feat_size,
                                                      dropout_rate=dropout_rate)
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
        self.init_param()

    def init_param(self):
        self.apply(lambda module: init_embedding(module))

    def forward(self, node_feats, node_x):
        node_x = node_x.squeeze(-1).long()
        node_feats = node_feats[:, 1:, :]  # exclude vn node
        atom_type_hat = self.atom_type_proj(node_feats)
        atom_type_hat = atom_type_hat.transpose(1, 2)
        ce_loss = 0.001 * self.loss_fn(atom_type_hat, node_x)
        return ce_loss
