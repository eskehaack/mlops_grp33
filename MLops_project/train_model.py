from pytorch_lightning import Trainer
from models.model import VGG
import pytorch_lightning as pl
import torch
torch.manual_seed(42)
model = VGG(101)
size_limiter = 0.05
trainer = Trainer(
    max_epochs=2,
    limit_train_batches=size_limiter,
    limit_test_batches=size_limiter,
    limit_val_batches=size_limiter,
    logger=pl.loggers.WandbLogger(project="dtu_mlops", name="test_run"),
)
trainer.fit(model)

trainer.test(model)
