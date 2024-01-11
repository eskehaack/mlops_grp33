from pytorch_lightning import Trainer
from MLops_project.models.model import VGG
import pytorch_lightning as pl
import torch
import hydra
import os
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint


torch.manual_seed(42)


@hydra.main(config_path=".", config_name="config")
def main(cfg):
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)

    model = VGG(
        101,
        batch_size=cfg.hyperparameters.batch_size,
        num_workers=cfg.hyperparameters.num_workers,
        learning_rate=cfg.hyperparameters.learning_rate,
    )
    size_limiter = cfg.hyperparameters.data_fraction

    checkpoint_callback = ModelCheckpoint(
        dirpath="models/",  # Path where checkpoints will be saved
        filename="{epoch}-{val_loss:.2f}",  # Checkpoint file name
        save_top_k=1,  # Save the top k models
        verbose=True,  # Print a message when a checkpoint is saved
        monitor="val_acc",  # Metric to monitor for deciding the best model
        mode="min",  # Mode for the monitored quantity for model selection
    )
    trainer = Trainer(
        max_epochs=cfg.hyperparameters.max_epochs,
        limit_train_batches=size_limiter,
        limit_test_batches=size_limiter,
        limit_val_batches=size_limiter,
        check_val_every_n_epoch=1,
        logger=pl.loggers.WandbLogger(project=cfg.wandb.project, name=cfg.wandb.name),
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    main()
