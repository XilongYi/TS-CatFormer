import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omegaconf import DictConfig, OmegaConf
from src.utils.hydra import safe_instantiate
from src.routine.utils.train_utils import load_weights_non_strict
import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping


def _configure_seed(seed: int, deterministic: bool):
    pl.seed_everything(int(seed), workers=True)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


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

    ckpt_path = run_cfg.checkpoint.get("path", None)
    ckpt_mode = run_cfg.checkpoint.get("mode", None)
    is_train = run_cfg.run.get("is_train", True)

    if ckpt_path is not None:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ckpt_path not found: {ckpt_path}")

        if ckpt_mode == "weights" or not is_train:
            load_weights_non_strict(lit_module, ckpt_path, verbose=None)

    if is_train:
        print(f"INFO: Running TRAIN mode with seed={seed}")
        if ckpt_path is not None and ckpt_mode != "weights":
            trainer.fit(lit_module, ckpt_path=ckpt_path)
        else:
            trainer.fit(lit_module)
    else:
        print(f"INFO: Running PREDICT/TEST mode with seed={seed}")
        trainer.test(lit_module)


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

    multi_seed = len(seeds) > 1
    for seed in seeds:
        run_name = cfg.experiment_name if not multi_seed else f"{cfg.experiment_name}/seed_{seed}"
        print(f"INFO: Preparing run `{run_name}`")
        _configure_seed(seed, deterministic)
        _run_once(cfg, seed, run_name)


if __name__ == "__main__":
    main()
