from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch import optim
from data.dataload import food101_dataloader


class VGG(pl.LightningModule):
    """Python Lightning module for simple VGG16 implementation.

    Args:
        num_classes: number of labels to predict across

    """

    def __init__(
        self,
        num_classes: int = 101,
        batch_size: int = 64,
        num_workers: int = 2,
        learning_rate: float = 1e-3,
        load_datasets: bool = True,
    ) -> None:
        super().__init__()

        self.VGG16 = self._make_layers()
        self.classifier = nn.Sequential(
            # Note that the 2*2 is the dimensions of the output of the CNN layers. For 256 this is 8*8.
            # I have had to downscale to 2*2 for 64*64 images
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.lr = learning_rate
        self.criterium = nn.CrossEntropyLoss()

        self.load_datasets = load_datasets
        if self.load_datasets:
            self.train_loader, self.val_loader, self.test_loader = food101_dataloader(
                batch_size=batch_size, num_workers=num_workers
            )

    def _make_layers(batch_norm=False) -> nn.Sequential:
        layers = []
        in_channels = 3
        cfg = [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            "M",
        ]
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # type: ignore
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)  # type: ignore
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]  # type: ignore
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]  # type: ignore
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        x = self.VGG16(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()

        self.log("val_acc", acc)
        return loss

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("test_acc", acc)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        if self.load_datasets:
            return self.train_loader
        return None

    def val_dataloader(self):
        if self.load_datasets:
            return self.val_loader
        return None

    def test_dataloader(self):
        if self.load_datasets:
            return self.test_loader
        return None


if __name__ == "__main__":
    model = VGG(101)
    model.parameters()
