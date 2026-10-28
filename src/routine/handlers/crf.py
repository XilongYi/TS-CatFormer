import torch
import torch.nn.functional as F
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryAccuracy,
    BinaryMatthewsCorrCoef,
)
from ..utils.train_utils import _rank0_print
from .base import ValHandler, PredictHandler, StepLogHandler


def _to_prob1(logits_or_score: torch.Tensor) -> torch.Tensor:
    """
    Supports:
      - [B,2] logits -> softmax[:,1]
      - [B] / [B,1] logit -> sigmoid
    Returns: prob1 [B]
    """
    if logits_or_score.dim() == 2 and logits_or_score.size(-1) == 2:
        return F.softmax(logits_or_score, dim=-1)[:, 1]
    return torch.sigmoid(logits_or_score.view(-1))


class CRFValHandler(ValHandler):
    def on_epoch_start(self, pl_module):
        # Keep buffers lightweight.
        pl_module._crf_buf = {"prob1": [], "y": []}

    def collect(self, pl_module, batch, y_hat, loss, losses_list, dataloader_idx=None):
        X_new, Y_aa, Y_r, target, _id = batch
        y_hat_crf = y_hat[5] if isinstance(y_hat, (list, tuple)) else y_hat["crf_logit"]
        if y_hat_crf is None or Y_r is None:
            return

        prob1 = _to_prob1(y_hat_crf.detach())
        y = Y_r.detach().view(-1).int()

        pl_module._crf_buf["prob1"].append(prob1)
        pl_module._crf_buf["y"].append(y)

    def on_epoch_end(self, pl_module):
        prob1 = torch.cat(pl_module._crf_buf["prob1"], dim=0)
        y     = torch.cat(pl_module._crf_buf["y"], dim=0).int()

        # Gather predictions across ranks.
        prob1_all = pl_module.all_gather(prob1).reshape(-1)
        y_all     = pl_module.all_gather(y).reshape(-1).int()

        # Match device and dtype for torchmetrics.
        device = prob1_all.device
        prob1_all = prob1_all.float()
        y_all = y_all.int()

        # TorchMetrics.
        auroc = BinaryAUROC().to(device)
        auprc = BinaryAveragePrecision().to(device)
        f1    = BinaryF1Score(threshold=0.5).to(device)
        prec  = BinaryPrecision(threshold=0.5).to(device)
        rec   = BinaryRecall(threshold=0.5).to(device)
        acc   = BinaryAccuracy(threshold=0.5).to(device)
        mcc   = BinaryMatthewsCorrCoef(threshold=0.5).to(device)

        val_auroc = auroc(prob1_all, y_all).float()
        val_auprc = auprc(prob1_all, y_all).float()
        val_f1    = f1(prob1_all, y_all).float()
        val_prec  = prec(prob1_all, y_all).float()
        val_rec   = rec(prob1_all, y_all).float()
        val_acc   = acc(prob1_all, y_all).float()
        val_mcc   = mcc(prob1_all, y_all).float()

        # Metrics are already global.
        pl_module.log("val_auc_crf", val_auroc, on_epoch=True, sync_dist=False)
        pl_module.log("val_auprc_crf", val_auprc, on_epoch=True, sync_dist=False)
        pl_module.log("val_f1_crf", val_f1, on_epoch=True, sync_dist=False)
        pl_module.log("val_precision_crf", val_prec, on_epoch=True, sync_dist=False)
        pl_module.log("val_recall_crf", val_rec, on_epoch=True, sync_dist=False)
        pl_module.log("val_acc_crf", val_acc, on_epoch=True, sync_dist=False)
        pl_module.log("val_mcc_crf", val_mcc, on_epoch=True, sync_dist=False)

        _rank0_print(
            pl_module,
            f"\n[Validation Epoch {pl_module.current_epoch}] CRF | "
            f"AUROC={float(val_auroc):.4f} AUPRC={float(val_auprc):.4f} "
            f"F1={float(val_f1):.4f} P={float(val_prec):.4f} R={float(val_rec):.4f} "
            f"ACC={float(val_acc):.4f} MCC={float(val_mcc):.4f}\n"
            + "#" * 120
        )

        pl_module._crf_buf["prob1"].clear()
        pl_module._crf_buf["y"].clear()


class CRFPredictHandler(PredictHandler):
    def predict(self, pl_module, batch, y_hat):
        X_new, Y_aa, Y_r, target, ids = batch
        logits = y_hat[5]

        prob1 = _to_prob1(logits.detach())
        pred  = (prob1 > 0.5).long()

        # Optional argmax for [B, 2] outputs.
        if logits.dim() == 2 and logits.size(-1) == 2:
            pred_argmax = logits.argmax(dim=-1).detach().cpu()
        else:
            pred_argmax = pred.detach().cpu()

        return {
            "ids": ids,
            "crf_logit": logits.detach().cpu(),
            "crf_prob1": prob1.detach().cpu(),
            "crf_pred": pred.detach().cpu(),
            "crf_pred_argmax": pred_argmax,
            "Y_r": (Y_r.detach().cpu() if Y_r is not None else None),
        }


class CRFStepLogger(StepLogHandler):
    def train_step_log(self, pl_module, losses_list, batch_size):
        pass

    def val_step_log(self, pl_module, losses_list, batch_size):
        pass
