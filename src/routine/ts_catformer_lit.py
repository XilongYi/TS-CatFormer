import pytorch_lightning as pl
from src.losses import (
    BinaryCELoss,
    MultiTaskLoss,
)
import torch
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau


class TSCatformerLit(pl.LightningModule):
    def __init__(
        self,
        loss_weight,
        loader_builder,
        dataset,
        model,
        val_handler,
        predict_handler,
        step_logger,
        optimizer_builder,
        peak_lr,
        weight_decay,
        warmup_ratio,
        end_lr,
        bce_w_neg,
        bce_w_pos,
        grad_guard=None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "loader_builder",
                "dataset",
                "model",
                "val_handler",
                "predict_handler",
                "step_logger",
                "optimizer_builder",
            ]
        )

        self.loader_builder = loader_builder
        self.dataset = dataset
        self.model = model

        self.val_handler = val_handler
        self.predict_handler = predict_handler
        self.step_logger = step_logger

        self.optimizer_builder = optimizer_builder
        self.peak_lr = peak_lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.end_lr = end_lr
        self.grad_guard = grad_guard or {}
        self._grad_guard_triggered = False

        # Support the custom warmup+plateau scheduler.
        self._manual_scheduler = None

        # loss
        self.mt_loss = MultiTaskLoss()
        for k, v in loss_weight.items():
            if k == "crf_loss":
                self.mt_loss.add_metric(
                    k,
                    weight=v,
                    loss_fn=BinaryCELoss(w_neg=bce_w_neg, w_pos=bce_w_pos),
                    pred_index=5,
                    target_index=5,
                )
            else:
                self.mt_loss.add_metric(k, weight=v)

    def setup(self, stage=None):
        if getattr(self, "_did_setup", False):
            return

        self.bind_train, self.bind_valid, self.bind_test = self.dataset.datasets
        self._did_setup = True

    def train_dataloader(self):
        if not getattr(self, "_did_setup", False):
            self.setup("fit")

        train_dataloader = self.loader_builder(
            self.bind_train,
            shuffle=True,
            drop_last=True,
            mode="sequential",
        )
        return train_dataloader

    def val_dataloader(self):
        if not getattr(self, "_did_setup", False):
            self.setup("fit")

        val_dataloader = self.loader_builder(
            self.bind_valid,
            shuffle=True,
            drop_last=True,
            mode="sequential",
        )
        return val_dataloader

    def test_dataloader(self):
        if not getattr(self, "_did_setup", False):
            self.setup("test")

        test_dataloader = self.loader_builder(
            self.bind_test,
            shuffle=False,
            drop_last=False,
            mode="sequential",
        )
        return test_dataloader

    def find_used_parameters(self, loss):
        loss.backward(retain_graph=True)
        used = [name for name, p in self.named_parameters() if p.grad is not None]
        return used

    def on_train_epoch_start(self):
        dl = self.trainer.train_dataloader
        loaders = getattr(dl, "loaders", None)
        if isinstance(loaders, dict):
            for name, ldr in loaders.items():
                sampler = getattr(ldr, "sampler", None)
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(self.current_epoch)

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        if isinstance(batch, (list, tuple)) and len(batch) == 3 and not torch.is_tensor(batch[1]):
            batch = batch[0]

        if isinstance(batch, dict):
            splits = list(batch.items())
        else:
            splits = [("single", batch)]

        total_loss = None
        used_splits = 0
        total_batch_size = 0

        for split_name, split_batch in splits:
            if split_batch is None:
                continue

            X_new, Y_aa, Y_r, target, _id = split_batch

            y_hat = self.model(X_new)
            y_gt = [None, None, None, None, Y_aa, Y_r]

            loss, losses_list = self.mt_loss(y_hat, y_gt)
            self.step_logger.train_step_log(self, losses_list, batch_size=len(target))

            total_loss = loss if total_loss is None else total_loss + loss
            used_splits += 1
            total_batch_size += len(target)

        if used_splits == 0:
            return None

        total_loss = total_loss / used_splits

        self.log(
            "train_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=total_batch_size,
        )

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "lr_step",
            lr,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=False,
        )

        return {"loss": total_loss}

    def on_validation_epoch_start(self):
        self.val_handler.on_epoch_start(self)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        if isinstance(batch, (list, tuple)) and len(batch) == 3 and not torch.is_tensor(batch[1]):
            batch = batch[0]

        X_new, Y_aa, Y_r, target, _id = batch

        y_hat = self.model(X_new)
        y_gt = [None, None, None, None, Y_aa, Y_r]

        loss, losses_list = self.mt_loss(y_hat, y_gt)
        self.step_logger.val_step_log(self, losses_list, batch_size=len(target))

        self.val_handler.collect(self, batch, y_hat, loss, losses_list, dataloader_idx)
        return loss

    def on_validation_epoch_end(self):
        self.val_handler.on_epoch_end(self)

        # Support WarmupThenPlateauScheduler.
        if self._manual_scheduler is not None:
            metric = self.trainer.callback_metrics.get("val_auc_crf")
            if metric is not None:
                self._manual_scheduler.step_metric(float(metric))

    def on_after_backward(self):
        if self._grad_guard_triggered:
            return

        guard_cfg = self.grad_guard or {}
        if not bool(guard_cfg.get("enabled", False)):
            return

        total_sq = 0.0
        has_grad = False
        nonfinite_found = False

        for param in self.parameters():
            grad = param.grad
            if grad is None:
                continue
            has_grad = True
            grad_detached = grad.detach()
            if bool(guard_cfg.get("check_finite", True)) and not torch.isfinite(grad_detached).all():
                nonfinite_found = True
                break
            grad_norm = grad_detached.float().norm(2).item()
            if not math.isfinite(grad_norm):
                nonfinite_found = True
                break
            total_sq += grad_norm * grad_norm

        if not has_grad:
            return

        total_norm = math.sqrt(total_sq)
        explode_threshold = float(guard_cfg.get("explode_norm_threshold", 1.0e6))
        vanish_threshold = float(guard_cfg.get("vanish_norm_threshold", 1.0e-12))

        should_stop = False
        reason = None

        if nonfinite_found:
            should_stop = True
            reason = "non-finite gradients detected"
        elif bool(guard_cfg.get("stop_on_exploding", True)) and total_norm > explode_threshold:
            should_stop = True
            reason = f"gradient explosion detected (global_norm={total_norm:.4e} > {explode_threshold:.4e})"
        elif bool(guard_cfg.get("stop_on_vanishing", True)) and total_norm < vanish_threshold:
            should_stop = True
            reason = f"gradient vanishing detected (global_norm={total_norm:.4e} < {vanish_threshold:.4e})"

        if should_stop:
            self._grad_guard_triggered = True
            if self.trainer is not None:
                self.trainer.should_stop = True
            self.print(f"[GradGuard] Stop current seed at epoch={self.current_epoch}, step={self.global_step}: {reason}")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X_new, *_ = batch
        with torch.no_grad():
            y_hat = self.model(X_new)
        return self.predict_handler.predict(self, batch, y_hat)

    def configure_optimizers(self):
        optimizer, scheduler = self.optimizer_builder(
            named_parameters=self.named_parameters(),
            peak_lr=self.hparams.peak_lr,
            weight_decay=self.hparams.weight_decay,
            estimated_stepping_batches=self.trainer.estimated_stepping_batches,
            warmup_ratio=self.hparams.warmup_ratio,
            end_lr=self.hparams.end_lr,
        )

        if scheduler is None:
            return optimizer

        # Case 1: custom warmup + plateau wrapper.
        if hasattr(scheduler, "step_step") and hasattr(scheduler, "step_metric"):
            self._manual_scheduler = scheduler
            return optimizer

        # Case 2: native ReduceLROnPlateau.
        if isinstance(scheduler, ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_auc_crf",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        # Case 3: step-based scheduler.
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, *args, **kwargs):
        optimizer.step(closure=optimizer_closure)

        # Step only the custom warmup + plateau scheduler here.
        if self._manual_scheduler is not None:
            self._manual_scheduler.step_step()

        optimizer.zero_grad()
