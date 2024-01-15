from pytorch_lightning import Trainer
from MLops_project import VGG
import pytorch_lightning as pl
import torch
import hydra
import os
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
from google.cloud import secretmanager

torch.manual_seed(42)


def access_secret_version():
    """
    Access the secret version and return the payload.
    """
    client = secretmanager.SecretManagerServiceClient()
    name = "projects/735066189170/secrets/WANDB-KEY/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")


@hydra.main(config_path=".", config_name="config")
def main(cfg):
    wandb_api_key = access_secret_version() if condition else os.environ.get("WANDB_API_KEY")
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
        logger=pl.loggers.WandbLogger(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            entity=cfg.wandb.entity,
        ),
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model)
    trainer.test(model)


def update_yaml_config(file_path, new_run_dir):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    config["hydra"]["run"]["dir"] = new_run_dir

    with open(file_path, "w") as file:
        yaml.safe_dump(config, file)


if __name__ == "__main__":
    condition = os.path.exists("/gcs")
    new_run_dir = (
        "/gcs/dtu_mlops_grp33_processed_data/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"
        if condition
        else "./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"
    )

    config_path = "./MLops_project/config.yaml"
    update_yaml_config(config_path, new_run_dir)

    main()
