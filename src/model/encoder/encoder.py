import torch.nn as nn
from src.model.encoder.encoder_module.complex_encoder import ComplexEncoder

class Encoder(nn.Module):
    def __init__(self, node_feat, edge_feat, hidden_dim, num_heads, intput_dropout_rate,
                                angle_feat_size, K, rbf_cutoff, get_inter_edge_dense):
        super(Encoder, self).__init__()
        self.get_inter_edge_dense  = get_inter_edge_dense
        self.complex_feat_layer = ComplexEncoder(node_feat, edge_feat, hidden_dim, num_heads,
                                              intput_dropout_rate, angle_feat_size, K, rbf_cutoff)
        
    def forward(self, batched_data):
        if batched_data.get("edata", None) is None:
            return None, None, None, None, None, None

        node_feats = self.complex_feat_layer(batched_data)
        intra_edge_dense = self.complex_feat_layer.edge_feat(batched_data['intra_D'], batched_data)  # [b, h, n, n]
        inter_attn_bias = self.complex_feat_layer.wrap_bias(batched_data['attn_bias']) #[B,H,N+1,N+1]=[1,8,309,309]

        intra_attn_bias = inter_attn_bias.clone()
        intra_attn_bias[:, :, 1:, 1:].masked_fill_(batched_data['pair_mask'].permute(0, 3, 1, 2), float('-inf'))

        pocket_attn_bias = inter_attn_bias.clone()
        pocket_attn_bias[:, :, 1:, 1:].masked_fill_(~batched_data['pocket_mask'].permute(0, 3, 1, 2), float('-inf'))

        # valid_mask = batched_data['ligand_mask']
        inter_edge_dense = self.get_inter_edge_dense(intra_edge_dense, batched_data, self.complex_feat_layer) 
        return node_feats, intra_edge_dense, inter_edge_dense, intra_attn_bias, inter_attn_bias, pocket_attn_bias
