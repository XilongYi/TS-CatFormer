import os
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping

from src.routine.utils.train_utils import load_weights_non_strict
from src.utils.hydra import safe_instantiate


def _configure_seed(seed: int, deterministic: bool):
    pl.seed_everything(int(seed), workers=True)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def _format_predict_log(
    seed: int,
    ckpt_path: str,
    output_csv_path: str,
    load_log_text: str = "",
):
    lines = [
        f"seed: {int(seed)}",
        f"checkpoint_path: {ckpt_path}",
        f"output_csv_path: {output_csv_path}",
    ]
    if load_log_text:
        lines.append("checkpoint_load_log:")
        lines.extend(load_log_text.rstrip("\n").splitlines())
    return "\n".join(lines) + "\n"


def _write_predict_log(run_cfg: DictConfig, seed: int, log_text: str, output_csv_path: str):
    log_dir = run_cfg.get("predict_logdir", None)
    if not log_dir:
        return None

    output_stem = Path(str(output_csv_path)).stem
    log_dir = Path(str(log_dir)).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{output_stem}.log"
    log_path.write_text(log_text, encoding="utf-8")
    print(f"INFO: Predict log written to {log_path}")
    return str(log_path)


def _load_checkpoint_with_log(lit_module, ckpt_path: str):
    buffer = StringIO()
    with redirect_stdout(buffer):
        load_weights_non_strict(lit_module, ckpt_path, verbose=None)
    load_log_text = buffer.getvalue()
    if load_log_text:
        print(load_log_text, end="")
    return load_log_text


def _trim_output_csv_columns(output_csv_path: str):
    keep_columns = ["Protein_id", "r_idx", "s_idx", "p_idx", "Score"]
    label_candidates = ["label", "Label"]
    if not output_csv_path or not os.path.exists(output_csv_path):
        raise FileNotFoundError(f"Predicted output csv not found: {output_csv_path}")

    df = pd.read_csv(output_csv_path)

    selected_columns = []
    for column in label_candidates:
        if column in df.columns:
            selected_columns.append(column)
            break
    selected_columns.extend(column for column in keep_columns if column in df.columns)

    missing_columns = [column for column in keep_columns if column not in df.columns]
    if not any(column in df.columns for column in label_candidates):
        missing_columns.append("label/Label")

    if not selected_columns:
        expected_columns = label_candidates + keep_columns
        raise KeyError(
            f"None of the requested columns exist in output csv: {expected_columns}. "
            f"Actual columns: {list(df.columns)}"
        )

    df = df[selected_columns]
    df.to_csv(output_csv_path, index=False)
    print(f"INFO: Trimmed output CSV columns to {selected_columns}")
    if missing_columns:
        print(f"INFO: Missing output CSV columns skipped: {missing_columns}")


def _run_once(cfg: DictConfig, seed: int, run_name: str):
    run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.set_struct(run_cfg, False)
    run_cfg.experiment_name = cfg.experiment_name
    run_cfg.current_seed = int(seed)

    checkpoint_callback = safe_instantiate(run_cfg.callback)
    callbacks = [checkpoint_callback]

    early_stop_cfg = run_cfg.get("early_stop", {})
    if bool(early_stop_cfg.get("enabled", False)):
        callbacks.append(
            EarlyStopping(
                monitor=early_stop_cfg.get("monitor", run_cfg.mode.monitor.metric),
                mode=early_stop_cfg.get("mode", run_cfg.mode.monitor.mode),
                patience=int(early_stop_cfg.get("patience", 15)),
                min_delta=float(early_stop_cfg.get("min_delta", 0.0)),
                check_on_train_epoch_end=bool(early_stop_cfg.get("check_on_train_epoch_end", False)),
            )
        )

    ckpt_path = run_cfg.checkpoint.get("path", None)
    is_train = run_cfg.run.get("is_train", True)
    if not ckpt_path:
        raise ValueError("checkpoint.path must point to a single checkpoint file in predict_main.py")

    ckpt_path_obj = Path(str(ckpt_path)).expanduser()
    if ckpt_path_obj.is_dir():
        raise IsADirectoryError(
            f"checkpoint.path must be a single .ckpt file, but got directory: {ckpt_path_obj}"
        )

    ckpt_path = str(ckpt_path_obj)
    run_cfg.checkpoint.path = ckpt_path
    ckpt_mode = run_cfg.checkpoint.get("mode", None)
    output_csv_path = str(run_cfg.predict.get("output_csv_path", ""))
    run_cfg.predict.output_csv_path = output_csv_path

    logger = safe_instantiate(run_cfg.logger)
    trainer = safe_instantiate(
        run_cfg.trainer,
        logger=logger,
        callbacks=callbacks,
    )

    dataset = safe_instantiate(run_cfg.dataset)
    loader_builder = safe_instantiate(run_cfg.dataloader)
    model = safe_instantiate(run_cfg.model)

    lit_module_factory = safe_instantiate(
        run_cfg.routine,
        loader_builder=loader_builder,
        dataset=dataset,
        model=model,
    )
    lit_module = lit_module_factory()

    trainer.logger.log_hyperparams(lit_module.hparams)

    load_log_text = ""
    if ckpt_path is not None:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ckpt_path not found: {ckpt_path}")

        if ckpt_mode == "weights" or not is_train:
            load_log_text = _load_checkpoint_with_log(lit_module, ckpt_path)

    predict_log_text = _format_predict_log(
        seed=seed,
        ckpt_path=ckpt_path,
        output_csv_path=output_csv_path,
        load_log_text=load_log_text,
    )
    _write_predict_log(run_cfg, seed, predict_log_text, output_csv_path)

    if is_train:
        print(f"INFO: Running TRAIN mode with seed={seed}")
        if ckpt_path is not None and ckpt_mode != "weights":
            trainer.fit(lit_module, ckpt_path=ckpt_path)
        else:
            trainer.fit(lit_module)
    else:
        print(f"INFO: Preparing run `{run_name}`")
        print(f"INFO: Checkpoint: {ckpt_path}")
        print(f"INFO: Predicted CSV output: {output_csv_path}")
        trainer.test(lit_module)
        _trim_output_csv_columns(output_csv_path)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg, resolve=False))
    seed_cfg = cfg.get("seed", {})
    seeds = seed_cfg.get("values", [0])
    deterministic = bool(seed_cfg.get("deterministic", False))

    if isinstance(seeds, int):
        seeds = [seeds]
    seeds = [int(seed) for seed in seeds]
    if len(seeds) == 0:
        raise ValueError("seed.values must contain at least one seed.")
    if len(seeds) > 1:
        raise ValueError(
            f"predict_main.py only supports a single seed/model per run, but got seed.values={seeds}"
        )

    seed = seeds[0]
    run_name = cfg.experiment_name
    _configure_seed(seed, deterministic)
    _run_once(cfg, seed, run_name)


if __name__ == "__main__":
    main()
