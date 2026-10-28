import torch
import torch.nn as nn


class TSCatformer(nn.Module):
    def __init__(
        self,
        complex_feat_layer,
        pocket_interaction_layer,
        task_head,
        **kwargs,
    ):
        super().__init__()

        self.complex_feat_layer = complex_feat_layer
        self.pocket_interaction_layer = pocket_interaction_layer
        self.task_head = task_head
        self.freeze_graphormer = bool(kwargs.get("freeze_graphormer", False))
        if self.freeze_graphormer:
            self.freeze_model()

    @staticmethod
    def _freeze_module(module: nn.Module, set_eval: bool = True):
        """Freeze params; optionally set eval mode (useful for dropout/bn)."""
        for p in module.parameters():
            p.requires_grad = False
        if set_eval:
            module.eval()

    def freeze_model(self, set_eval: bool = True, verbose: bool = True):
        self._freeze_module(self.complex_feat_layer, set_eval=set_eval)
        if verbose:
            print(f"[freeze] complex_feat_layer frozen (requires_grad=False, eval={set_eval})")

        non_gvp_modules = [
            ("pocket_interaction_layer.intra_encoder", self.pocket_interaction_layer.intra_encoder),
            ("pocket_interaction_layer.pocket_encoder", self.pocket_interaction_layer.pocket_encoder),
            ("pocket_interaction_layer.inter_encoder", self.pocket_interaction_layer.inter_encoder),
        ]
        for name, module in non_gvp_modules:
            self._freeze_module(module, set_eval=set_eval)
            if verbose:
                print(f"[freeze] {name} frozen (requires_grad=False, eval={set_eval})")

    def forward(self, batched_data, perturb=None, istrain=True):
        node_feats, intra_edge_dense, inter_edge_dense, intra_attn_bias, inter_attn_bias, pocket_attn_bias = (
            self.complex_feat_layer(batched_data)
        )
        new_node_feats, pocket_outputs, new_inter_edge_dense, batched_data = self.pocket_interaction_layer(
            batched_data,
            node_feats,
            intra_edge_dense,
            inter_edge_dense,
            intra_attn_bias,
            inter_attn_bias,
            pocket_attn_bias,
        )
        final = self.task_head(new_node_feats, pocket_outputs, new_inter_edge_dense, batched_data)
        return final
