import os
import pandas as pd
import pytorch_lightning as pl
import torch


class TSCatformerCRFPredictLit(pl.LightningModule):
    def __init__(
        self,
        loader_builder,
        dataset,
        model,
        input_csv_path,
        output_csv_path,
        output_score_col="Score",
        target_col="target",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "loader_builder",
                "dataset",
                "model",
            ]
        )

        self.loader_builder = loader_builder
        self.dataset = dataset
        self.model = model

        self.input_csv_path = input_csv_path
        self.output_csv_path = output_csv_path
        self.output_score_col = output_score_col
        self.target_col = target_col

        self._test_records = []
        self._did_setup = False

    @staticmethod
    def _resolve_input_row_ids(df_input):
        if "row_id" in df_input.columns:
            return df_input["row_id"]
        if "row id" in df_input.columns:
            return df_input["row id"]
        return df_input.index

    def setup(self, stage=None):
        if self._did_setup:
            return

        self.bind_train, self.bind_valid, self.bind_test = self.dataset.datasets
        self._did_setup = True

    def test_dataloader(self):
        if not self._did_setup:
            self.setup("test")

        test_dataloader = self.loader_builder(
            self.bind_test,
            shuffle=False,
            drop_last=False,
            mode="sequential",
        )
        return test_dataloader

    def forward(self, X):
        return self.model(X)

    def on_test_epoch_start(self):
        self._test_records = []

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        if isinstance(batch, (list, tuple)) and len(batch) == 3 and not torch.is_tensor(batch[1]):
            batch = batch[0]

        X_new, Y_aa, Y_r, target, _id = batch

        with torch.no_grad():
            y_hat = self.model(X_new)

        # Extract prediction scores.
        pred_score = self._extract_pred_score(y_hat)

        if pred_score is None:
            raise ValueError("Cannot extract pred_score from model output y_hat.")

        # Convert to per-sample probabilities.
        pred_score = self._to_positive_class_probability(pred_score)

        if isinstance(X_new, dict):
            row_ids = X_new.get("row_id", _id)
        else:
            row_ids = _id

        if row_ids is None:
            raise ValueError("Prediction batch has no row_id field in X_new or id output.")

        row_ids = [str(row_id) for row_id in row_ids]

        if len(row_ids) != pred_score.shape[0]:
            raise ValueError(
                f"row_id count != pred_score batch size: {len(row_ids)} vs {pred_score.shape[0]}"
            )

        for row_id, sc in zip(row_ids, pred_score):
            self._test_records.append((row_id, float(sc)))

        return None

    def on_test_epoch_end(self):
        if len(self._test_records) == 0:
            raise ValueError("No prediction records collected in test epoch.")

        df_input = pd.read_csv(self.input_csv_path)
        merge_key_col = "__row_id__"
        df_input[merge_key_col] = self._resolve_input_row_ids(df_input).astype(str)

        df_pred = pd.DataFrame(
            self._test_records,
            columns=["row_id", self.output_score_col]
        )

        df_input[merge_key_col] = df_input[merge_key_col].astype(str)
        df_pred["row_id"] = df_pred["row_id"].astype(str)

        df_pred = df_pred.drop_duplicates(subset=["row_id"], keep="first")

        df_out = df_input.merge(
            df_pred,
            left_on=merge_key_col,
            right_on="row_id",
            how="left",
        )

        if "row_id" in df_out.columns:
            df_out = df_out.drop(columns=["row_id"])
        if merge_key_col in df_out.columns:
            df_out = df_out.drop(columns=[merge_key_col])

        out_dir = os.path.dirname(self.output_csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        df_out.to_csv(self.output_csv_path, index=False)

        print(f"[CRF Predict] Input CSV : {self.input_csv_path}")
        print(f"[CRF Predict] Output CSV: {self.output_csv_path}")
        print(f"[CRF Predict] Total predictions collected: {len(self._test_records)}")
        print(f"[CRF Predict] Output rows: {len(df_out)}")
        print(f"[CRF Predict] Matched rows: {df_out[self.output_score_col].notna().sum()}")

    def configure_optimizers(self):
        # Prediction-only mode.
        return None

    def _extract_pred_score(self, y_hat):
        """Extract scores from common output formats."""

        if isinstance(y_hat, dict):
            for key in [
                "pred_score",
                "score",
                "crf_score",
                "y_r",
                "logit",
                "logits",
            ]:
                if key in y_hat:
                    return y_hat[key]

            for v in y_hat.values():
                if torch.is_tensor(v):
                    return v

        if isinstance(y_hat, (list, tuple)):
            # Prefer 1D or 2D tensors.
            for v in y_hat:
                if torch.is_tensor(v) and v.ndim in (1, 2):
                    return v
            for v in y_hat:
                if torch.is_tensor(v):
                    return v

        if torch.is_tensor(y_hat):
            return y_hat

        return None

    def _to_positive_class_probability(self, pred_score):
        """Convert model output to positive-class probabilities."""

        if not torch.is_tensor(pred_score):
            pred_score = torch.as_tensor(pred_score, dtype=torch.float32)
        else:
            pred_score = pred_score.detach().float()

        # [B, 2] -> softmax.
        if pred_score.ndim == 2 and pred_score.shape[1] == 2:
            pred_score = torch.softmax(pred_score, dim=1)[:, 1]

        # [B, 1] -> [B].
        elif pred_score.ndim == 2 and pred_score.shape[1] == 1:
            pred_score = pred_score.squeeze(1)
            pred_score = self._maybe_sigmoid_1d(pred_score)

        # [B].
        elif pred_score.ndim == 1:
            pred_score = self._maybe_sigmoid_1d(pred_score)

        else:
            raise ValueError(f"Unexpected pred_score shape: {tuple(pred_score.shape)}")

        return pred_score.cpu()

    def _maybe_sigmoid_1d(self, x):
        """Apply sigmoid only when the input looks like logits."""
        if x.numel() == 0:
            return x

        xmin = x.min().item()
        xmax = x.max().item()

        if xmin >= 0.0 and xmax <= 1.0:
            return x

        return torch.sigmoid(x)
