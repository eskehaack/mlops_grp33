from pytorch_lightning import Trainer
from models.model import VGG
import pytorch_lightning as pl
import torch
import hydra

torch.manual_seed(42)


@hydra.main(config_name="config.yaml")
def main(cfg):
    model = VGG(
        101,
        batch_size=cfg.hyperparameters.batch_size,
        num_workers=cfg.hyperparameters.num_workers,
        learning_rate=cfg.hyperparameters.learning_rate,
    )
    size_limiter = 0.05
    trainer = Trainer(
        max_epochs=2,
        limit_train_batches=size_limiter,
        limit_test_batches=size_limiter,
        limit_val_batches=size_limiter,
        logger=pl.loggers.WandbLogger(project=cfg.wandb.project, name=cfg.wandb.name),
    )
    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    main()
