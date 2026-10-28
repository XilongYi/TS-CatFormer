import os
import math
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.losses import (
    BinaryCELoss,
    MultiTaskLoss,
)


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
        valid_seq_csv,
        valid_query_col,
        valid_label_col,
        valid_target_col,
        valid_candidate_col,
        **kwargs,
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

        # Keep the interface.
        self.val_handler = val_handler
        self.predict_handler = predict_handler
        self.step_logger = step_logger

        self.optimizer_builder = optimizer_builder
        self.peak_lr = peak_lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.end_lr = end_lr

        self.valid_seq_csv = valid_seq_csv
        self.valid_query_col = valid_query_col
        self.valid_label_col = valid_label_col
        self.valid_target_col = valid_target_col
        self.valid_candidate_col = valid_candidate_col

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

        # validation cache
        self._valid_df = None
        self._val_records = None

    def setup(self, stage=None):
        if getattr(self, "_did_setup", False):
            return

        self.bind_train, self.bind_valid, self.bind_test = self.dataset.datasets

        if not os.path.exists(self.valid_seq_csv):
            raise FileNotFoundError(f"valid_seq_csv not found: {self.valid_seq_csv}")

        self._valid_df = pd.read_csv(self.valid_seq_csv)

        required_cols = [
            self.valid_label_col,
            self.valid_query_col,
            self.valid_target_col,
        ]
        missing = [c for c in required_cols if c not in self._valid_df.columns]
        if missing:
            raise KeyError(
                f"Missing columns in valid csv: {missing}. "
                f"Available columns: {self._valid_df.columns.tolist()}"
            )

        # Normalize targets to strings.
        self._valid_df[self.valid_target_col] = self._valid_df[self.valid_target_col].astype(str)

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
            shuffle=False,
            drop_last=False,
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

    def on_train_epoch_start(self):
        dl = self.trainer.train_dataloader
        loaders = getattr(dl, "loaders", None)
        if isinstance(loaders, dict):
            for _, ldr in loaders.items():
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

    # Validation ranking.
    def on_validation_epoch_start(self):
        self._val_records = []

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        if isinstance(batch, (list, tuple)) and len(batch) == 3 and not torch.is_tensor(batch[1]):
            batch = batch[0]

        X_new, Y_aa, Y_r, target, _id = batch

        y_hat = self.model(X_new)

        y_gt = [None, None, None, None, Y_aa, Y_r]
        loss, losses_list = self.mt_loss(y_hat, y_gt)
        self.step_logger.val_step_log(self, losses_list, batch_size=len(target))

        pred_score = self._extract_crf_score(y_hat)

        targets = X_new.get("target", target)
        if targets is None:
            raise ValueError("Validation batch has no target field in X_new or target output.")

        targets = [str(t) for t in targets]

        if len(targets) != pred_score.shape[0]:
            raise ValueError(
                f"Target count != pred_score batch size: {len(targets)} vs {pred_score.shape[0]}"
            )

        for t, sc in zip(targets, pred_score):
            self._val_records.append((t, float(sc)))

        return loss

    def on_validation_epoch_end(self):
        gathered_records = self._gather_object(self._val_records)
        is_sanity = getattr(self.trainer, "sanity_checking", False)

        if self.global_rank == 0:
            merged_records = []
            for part in gathered_records:
                if part is not None:
                    merged_records.extend(part)

            if is_sanity:
                print(f"[Sanity Check] collected {len(merged_records)} validation predictions, skip ranking metrics.")
                metrics = {
                    "val_sr_at_1": 0.0,
                    "val_sr_at_3": 0.0,
                    "val_sr_at_5": 0.0,
                    "val_sr_at_10": 0.0,
                    "val_ef_at_1pct": 0.0,
                    "val_ef_at_2pct": 0.0,
                    "val_dcg_at_10": 0.0,
                }
            else:
                metrics = self._compute_valid_ranking_metrics(merged_records)
        else:
            metrics = None

        metrics = self._broadcast_metrics(metrics)

        for k, v in metrics.items():
            self.log(
                k,
                float(v),
                on_step=False,
                on_epoch=True,
                prog_bar=(k in ["val_sr_at_1", "val_sr_at_10", "val_dcg_at_10"]),
                sync_dist=True,
            )

        if (not is_sanity) and self._manual_scheduler is not None:
            metric = metrics.get("val_sr_at_10", None)
            if metric is not None:
                self._manual_scheduler.step_metric(float(metric))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X_new, *_ = batch
        with torch.no_grad():
            y_hat = self.model(X_new)
        return self.predict_handler.predict(self, batch, y_hat)

    # Optimizer and scheduler.
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

        if hasattr(scheduler, "step_step") and hasattr(scheduler, "step_metric"):
            self._manual_scheduler = scheduler
            return optimizer

        if isinstance(scheduler, ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_sr_at_10",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

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

        if self._manual_scheduler is not None:
            self._manual_scheduler.step_step()

        optimizer.zero_grad()

    # Helpers.
    def _extract_crf_score(self, y_hat: object) -> torch.Tensor:
        score = None

        if isinstance(y_hat, dict):
            for key in ["crf", "crf_logits", "y_r", "reaction", "reaction_logits", "pred_r"]:
                if key in y_hat:
                    score = y_hat[key]
                    break
            if score is None:
                raise KeyError(f"Cannot find CRF score key in y_hat dict. Keys: {list(y_hat.keys())}")

        elif isinstance(y_hat, (list, tuple)):
            score = y_hat[-1]

        elif torch.is_tensor(y_hat):
            score = y_hat

        else:
            raise TypeError(f"Unsupported y_hat type: {type(y_hat)}")

        if not torch.is_tensor(score):
            raise TypeError(f"Extracted score is not tensor: {type(score)}")

        if score.ndim == 2 and score.shape[1] == 1:
            score = score.squeeze(1)
            score = torch.sigmoid(score)
        elif score.ndim == 2 and score.shape[1] == 2:
            score = torch.softmax(score, dim=-1)[:, 1]
        elif score.ndim == 1:
            score = torch.sigmoid(score)
        else:
            raise ValueError(f"Unsupported CRF score shape: {tuple(score.shape)}")

        return score.detach().float().cpu()

    def _gather_object(self, obj):
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, obj)
            return gathered
        return [obj]

    def _broadcast_metrics(self, metrics: dict | None):
        metric_keys = [
            "val_sr_at_1",
            "val_sr_at_3",
            "val_sr_at_5",
            "val_sr_at_10",
            "val_ef_at_1pct",
            "val_ef_at_2pct",
            "val_dcg_at_10",
        ]

        if dist.is_available() and dist.is_initialized():
            if self.global_rank == 0:
                tensor = torch.tensor([float(metrics[k]) for k in metric_keys], device=self.device)
            else:
                tensor = torch.zeros(len(metric_keys), device=self.device)

            dist.broadcast(tensor, src=0)
            out = {k: float(v) for k, v in zip(metric_keys, tensor.detach().cpu().tolist())}
            return out

        return metrics

    def _compute_valid_ranking_metrics(self, merged_records):
        """
        merged_records: list[(Target, pred_score)]
        Merge by Target, drop missing rows, and compute ranking metrics.
        """
        if len(merged_records) == 0:
            raise RuntimeError("No validation records collected.")

        df = self._valid_df.copy()

        pred_df = pd.DataFrame(merged_records, columns=[self.valid_target_col, "pred_score"])
        pred_df[self.valid_target_col] = pred_df[self.valid_target_col].astype(str)

        # Drop duplicate targets after DDP padding.
        before_pred = len(pred_df)
        pred_df = pred_df.drop_duplicates(subset=[self.valid_target_col], keep="first").reset_index(drop=True)
        after_pred = len(pred_df)
        if after_pred < before_pred:
            print(f"[Validation] drop duplicated prediction targets: {before_pred - after_pred} ({before_pred} -> {after_pred})")

        before_merge_n = len(df)
        df = df.merge(pred_df, on=self.valid_target_col, how="left")

        before_dropna = len(df)
        df = df.dropna(subset=["pred_score"]).reset_index(drop=True)
        after_dropna = len(df)

        dropped = before_dropna - after_dropna
        print(f"[Validation] merge by {self.valid_target_col}: kept {after_dropna}/{before_merge_n}, dropped {dropped} rows without prediction.")

        if after_dropna == 0:
            raise RuntimeError("All validation rows were dropped after merge on Target. Please check whether X_new['target'] matches valid csv Target column.")

        query_col = self.valid_query_col
        label_col = self.valid_label_col

        if query_col not in df.columns:
            raise KeyError(f"Missing query col {query_col}. Available: {df.columns.tolist()}")
        if label_col not in df.columns:
            raise KeyError(f"Missing label col {label_col}. Available: {df.columns.tolist()}")

        sr1_list, sr3_list, sr5_list, sr10_list = [], [], [], []
        ef1_list, ef2_list, dcg10_list = [], [], []

        grouped = df.groupby(query_col, sort=False)

        n_queries = 0
        for query, g in grouped:
            g = g.sort_values("pred_score", ascending=False, kind="mergesort").reset_index(drop=True)

            labels = g[label_col].astype(int).to_numpy()
            n = len(labels)
            p = int(labels.sum())

            if p <= 0 or n <= 0:
                continue

            n_queries += 1

            sr1_list.append(float(labels[:1].sum() > 0))
            sr3_list.append(float(labels[:3].sum() > 0))
            sr5_list.append(float(labels[:5].sum() > 0))
            sr10_list.append(float(labels[:10].sum() > 0))

            top10 = labels[:10]
            denom = np.log2(np.arange(2, len(top10) + 2))
            dcg10 = float((top10 / denom).sum())
            dcg10_list.append(dcg10)

            k1 = max(1, int(math.ceil(n * 0.01)))
            k2 = max(1, int(math.ceil(n * 0.02)))

            base_rate = p / n
            ef1 = float((labels[:k1].sum() / k1) / base_rate)
            ef2 = float((labels[:k2].sum() / k2) / base_rate)

            ef1_list.append(ef1)
            ef2_list.append(ef2)

        if n_queries == 0:
            raise RuntimeError("No valid queries with positive labels found in validation csv after merge/dropna.")

        metrics = {
            "val_sr_at_1": float(np.mean(sr1_list)),
            "val_sr_at_3": float(np.mean(sr3_list)),
            "val_sr_at_5": float(np.mean(sr5_list)),
            "val_sr_at_10": float(np.mean(sr10_list)),
            "val_ef_at_1pct": float(np.mean(ef1_list)),
            "val_ef_at_2pct": float(np.mean(ef2_list)),
            "val_dcg_at_10": float(np.mean(dcg10_list)),
        }

        print("[Validation Ranking Metrics]")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")

        return metrics
