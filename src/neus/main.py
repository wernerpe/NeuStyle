import os
from pathlib import Path

import hydra
import torch
import wandb
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src.neus",),
    ("beartype", "beartype"),
):
    from src.neus.dataset.DataModule import DataModule
    from src.neus.model.ModelWrapper import ModelWrapper


@hydra.main(
    version_base=None,
    config_path="../../config/neus",
    config_name="main",
)
def train(cfg: DictConfig):
    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(f"Saving outputs to {output_dir}")
    latest_run = output_dir.parents[1] / "latest-run"
    os.system(f"rm {latest_run}")
    os.system(f"ln -s {output_dir} {latest_run}")

    # Set up logging with wandb.
    callbacks = []
    if cfg.wandb.mode != "disabled":
        logger = WandbLogger(
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            name=f"{cfg.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            log_model="all",
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg),
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            print(f"wandb mode: {wandb.run.settings.mode}")
            wandb.run.log_code(".")
    else:
        logger = None

    # Set up checkpointing.
    if "checkpointing" in cfg.training:
        callbacks.append(
            ModelCheckpoint(
                output_dir / "checkpoints",
                **cfg.training.checkpointing,
            )
        )

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        val_check_interval=cfg.validation.interval,
        inference_mode=False,  # needed for the forward pass gradient in NeuS to work
    )
    model = ModelWrapper(cfg)
    data_module = DataModule(cfg)

    if cfg.mode == "train":
        trainer.fit(
            model,
            datamodule=data_module,
        )
    elif cfg.mode == "render":
        trainer.predict(
            model,
            datamodule=data_module,
            ckpt_path=cfg.rendering.checkpoint,
        )
    else:
        raise ValueError(f'Unrecognized mode "{cfg.mode}"')


if __name__ == "__main__":
    train()
