import torch.nn as nn
import torch
from src.model.common_layers.rbf_layer import RBFLayer


def init_embedding(layer):
    if isinstance(layer, nn.Embedding):
        if len(layer.weight) > 1:
            layer.weight.data[0].zero_()
    if isinstance(layer, nn.Linear) and layer.bias is not None:
        layer.bias.data.zero_()


class ComplexEncoder(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, hidden_dim, num_heads,
                 intput_dropout_rate, angle_feat_size,
                 K=128, rbf_cutoff=10., use_affine_linear=False,
                 num_atom_types=29):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # Node Feat
        self.atom_encoder = nn.Embedding(512 * node_feat_size + 1,
                                         hidden_dim,
                                         padding_idx=0)
        # Node LayerNorm.
        # self.node_ln = nn.LayerNorm(hidden_dim)

        self.input_dropout = nn.Dropout(intput_dropout_rate)

        # virtual Node
        self.graph_token = nn.Embedding(1, hidden_dim)
        self.mol_graph_token = nn.Embedding(1, hidden_dim)

        # Edge feat
        if edge_feat_size > 0:
            self.edge_encoder = nn.Embedding(512 * edge_feat_size + 1,
                                             hidden_dim,
                                             padding_idx=0)
            # Edge LayerNorm.
            # self.edge_ln = nn.LayerNorm(hidden_dim)

        # Pair Affine
        self.num_atom_types = num_atom_types

        # Gaussian Kernels (atom level)
        self.rbf = RBFLayer(K, cutoff=rbf_cutoff,
                            use_affine_linear=use_affine_linear,
                            edge_types=self.num_atom_types ** 2)
        self.rel_pos_3d_proj = nn.Linear(K, hidden_dim)

        # Molecular RBF placeholder.
        # self.mol_rbf = RBFLayer(K, cutoff=15,
        #                         use_affine_linear=use_affine_linear,
        #                         edge_types=2)
        # self.mol_rel_pos_3d_proj = nn.Linear(K, hidden_dim)

        # molecular pair-type embedding: 0=pad, 1=ligand, 2=aa-aa
        self.mol_edge_encoder = nn.Embedding(4, hidden_dim)  # 0 pad, 1 ligand-edge, 2 aa-aa, 3 virtual
        self.mol_edge_ln = nn.LayerNorm(hidden_dim)

        self.init_param()

    def init_param(self):
        self.apply(lambda module: init_embedding(module))

    def atom_type2pair_type(self, x: torch.Tensor) -> torch.Tensor:
        atoms_type_x = x % self.num_atom_types  # 29-th is the 0-idx in pocket-atoms-type
        pair_type = atoms_type_x[:, :, None, :] * self.num_atom_types + atoms_type_x[:, None, :]
        return pair_type

    def wrap_bias(self, attn_bias):
        attn_bias = attn_bias.clone().unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        return attn_bias

    def edge_feat(self, D, batched_data):
        x = batched_data['x'].long()  
        # Distance with RBF layer, convert 3d pos to an unlimited space
        D = D.float()
        # Pair-Type-Aware distance
        D, pair_type = D.squeeze(-1), self.atom_type2pair_type(x).squeeze(-1)

        real_3d_input = self.rel_pos_3d_proj(self.rbf(D, pair_type))
        edge_feats = real_3d_input
        # Edge feature[PLIP]
        if hasattr(self, 'edge_encoder'):
            edata = batched_data['edata'].long()
            # Keep sum(dim=-2).
            edge_input = self.edge_encoder(edata).sum(dim=-2)
            edge_feats = edge_feats + edge_input

            # Edge LayerNorm.
            # B, N, N2, H = edge_feats.shape
            # edge_feats = self.edge_ln(edge_feats.view(B * N * N2, H)).view(B, N, N2, H)

        # Edge-feat
        # [n_graph, n_node+1, n_node+1, d]
        graph_edge_feats = torch.zeros(
            [edge_feats.size(0), edge_feats.size(1) + 1,
             edge_feats.size(1) + 1, edge_feats.size(3)],
            device=edge_feats.device,
            dtype=edge_feats.dtype
        )
        graph_edge_feats[:, 1:, 1:] = edge_feats
        return graph_edge_feats

    def molecular_edge_feat(self, mol_D_mean=None):
        """
        Build three molecular edge types:
            1 = ligand <-> any node
            2 = aa-aa
            3 = virtual node (graph_token) <-> any node
        Output:
            mol_edge_feats: [B, N_mol+1, N_mol+1, hidden_dim]
        """
        # Get shapes.
        B, N, _, _ = mol_D_mean.shape
        device = mol_D_mean.device
        H = self.hidden_dim

        # Edge types.
        #
        # Default to AA-AA edges.
        #
        pair_type = torch.full((B, N, N), 2, dtype=torch.long, device=device)

        # Ligand edges.
        # Assume the ligand node is index 0.
        pair_type[:, 0, :] = 1
        pair_type[:, :, 0] = 1

        # Embed edge types.
        mol_edge_input = self.mol_edge_encoder(pair_type)

        # Add the virtual node.
        #
        # Output: [B, N+1, N+1, H].
        #
        mol_edge_feats = torch.zeros(
            [B, N + 1, N + 1, H],
            device=device,
            dtype=mol_edge_input.dtype
        )

        # Copy real-node edges.
        mol_edge_feats[:, 1:, 1:, :] = mol_edge_input

        # Virtual-node edges.
        # graph_token <-> real nodes.
        virtual_pair = torch.full((B, N + 1), 3, dtype=torch.long, device=device)

        # Virtual-node row.
        mol_edge_feats[:, 0, :, :] = self.mol_edge_encoder(virtual_pair)
        # Virtual-node column.
        mol_edge_feats[:, :, 0, :] = self.mol_edge_encoder(virtual_pair)

        # virtual->virtual is written twice but is harmless.

        # Apply LayerNorm.
        B2, N2, N3, H = mol_edge_feats.shape
        mol_edge_feats = self.mol_edge_ln(mol_edge_feats.view(B2 * N2 * N3, H)).view(B2, N2, N3, H)

        return mol_edge_feats
    def build_molecular_token(self, molecular_node):
        B = molecular_node.size(0)
        graph_token_feature = self.mol_graph_token.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, 1, H]
        molecular_node_with_virtual = torch.cat([graph_token_feature, molecular_node], dim=1)
        return molecular_node_with_virtual

    def forward(self, batched_data):
        # Only Output Node Feats
        x = batched_data['x'].long()
        n_graph, n_node = x.size()[:2]
        # Keep sum-based node features.
        node_feature = self.atom_encoder(x).sum(dim=-2)
        # Node LayerNorm.
        # node_feature = self.node_ln(node_feature)
        # Virtual Node Embedding
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
        graph_node_feature = self.input_dropout(graph_node_feature)
        return graph_node_feature
