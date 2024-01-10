from pytorch_lightning import Trainer
from models.model import VGG
import pytorch_lightning as pl
import torch
import hydra
import os
import wandb

torch.manual_seed(42)


@hydra.main(config_path=".", config_name="config")
def main(cfg):
    model = VGG(
        101,
        batch_size=cfg.hyperparameters.batch_size,
        num_workers=cfg.hyperparameters.num_workers,
        learning_rate=cfg.hyperparameters.learning_rate,
    )
    size_limiter = cfg.hyperparameters.data_fraction

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)
    # checkpoint_callback = pl.Callback.ModelCheckpoint(
    #     dirpath="./models", monitor="val_loss", mode="min"
    # )
    trainer = Trainer(
        max_epochs=cfg.hyperparameters.max_epochs,
        limit_train_batches=size_limiter,
        limit_test_batches=size_limiter,
        limit_val_batches=size_limiter,
        logger=pl.loggers.WandbLogger(project=cfg.wandb.project, name=cfg.wandb.name),
        # callbacks=[checkpoint_callback],
    )
    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    main()
