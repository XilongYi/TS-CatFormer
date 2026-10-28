import torch
import torch.nn as nn


class BinaryCELoss(nn.Module):
    def __init__(self, w_neg=1.0, w_pos=1.0):
        super().__init__()
        self.register_buffer("ce_weight", torch.tensor([w_neg, w_pos], dtype=torch.float))

    def forward(self, logits, y_r):
        if logits is None:
            device = y_r.device if torch.is_tensor(y_r) else self.ce_weight.device
            return torch.zeros((), device=device, dtype=torch.float32)

        y_r = y_r.view(-1).long()
        return nn.functional.cross_entropy(logits, y_r, weight=self.ce_weight.to(logits.device))
