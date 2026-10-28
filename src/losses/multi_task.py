import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fns = []

    def add_metric(self, name, weight=1.0, loss_fn=None, pred_index=None, target_index=None):
        self.loss_fns.append((name, loss_fn, weight, pred_index, target_index))

    def _infer_reference_device(self, y_hat, y_gt):
        for items in (y_hat, y_gt):
            for item in items:
                if torch.is_tensor(item):
                    return item.device
        return torch.device("cpu")

    def forward(self, y_hat, y_gt):
        ref_device = self._infer_reference_device(y_hat, y_gt)
        all_losses = []
        losses_list = []

        for i, (_name, loss_fn, weight, pred_index, target_index) in enumerate(self.loss_fns):
            pred_i = i if pred_index is None else pred_index
            target_i = i if target_index is None else target_index
            if loss_fn:
                loss = loss_fn(y_hat[pred_i], y_gt[target_i])
            else:
                loss = y_hat[pred_i]
            if not torch.is_tensor(loss):
                loss = torch.tensor(loss, device=ref_device, dtype=torch.float32)
            losses_list.append(loss)
            all_losses.append(loss * weight)

        total_loss = sum(all_losses)
        return total_loss, losses_list
