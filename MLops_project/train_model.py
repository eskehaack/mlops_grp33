from pytorch_lightning import Trainer
from models.model import VGG

model = VGG(101)
trainer = Trainer(max_epochs=5)
trainer.fit(model)
