import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class RBFLayer(nn.Module):
    def __init__(self, K=64, cutoff=10., use_affine_linear=False,
                 edge_types=29 ** 2, dtype=torch.float):
        super().__init__()
        self.cutoff = cutoff

        # Init centers.
        centers = torch.tensor(
            softplus_inverse(np.linspace(1.0, np.exp(-cutoff), K)),
            dtype=dtype
        )
        self.centers = nn.Parameter(F.softplus(centers))

        # Init widths.
        widths = torch.tensor(
            [softplus_inverse(0.5 / ((1.0 - np.exp(-cutoff) / K)) ** 2)] * K,
            dtype=dtype
        )
        self.widths = nn.Parameter(F.softplus(widths))

        # pair-type affine
        self.use_affine_linear = use_affine_linear
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.constant_(self.bias.weight, 0.)
        nn.init.constant_(self.mul.weight, 1.)

    def cutoff_fn(self, D):
        x = D / self.cutoff
        x3, x4, x5 = x**3, x**4, x**5
        return torch.where(x < 1, 1 - 6 * x5 + 15 * x4 - 10 * x3,
                           torch.zeros_like(x))

    def forward(self, D, edge_types):
        # D: [B, N, N] -> [B, N, N, 1]
        D = D.unsqueeze(-1)

        if self.use_affine_linear:
            mul = self.mul(edge_types)
            bias = self.bias(edge_types)
            D = mul * D + bias

        # Clamp distances.
        D = torch.clamp(D, min=0.0, max=self.cutoff)

        # Clamp learned parameters.
        widths = torch.clamp(self.widths, 1e-3, 50.0)
        centers = torch.clamp(self.centers, 1e-3, 50.0)

        # Keep the original RBF form.
        exp_neg_D = torch.exp(-D)
        diff = exp_neg_D - centers
        rbf = torch.exp(-widths * diff**2)

        rbf_D = self.cutoff_fn(D) * rbf
        return rbf_D

def softplus_inverse(x):
    return x + np.log(-np.expm1(-x))