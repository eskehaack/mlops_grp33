from pytorch_lightning import Trainer
from models.model import VGG

model = VGG(101)
trainer = Trainer(max_epochs=1, limit_train_batches=0.05)
trainer.fit(model)
