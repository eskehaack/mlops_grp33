import torch
from torchvision import datasets
from torch.utils.data import DataLoader


def dataloader(train=True, batch_size=64, num_workers=4):
    batch_size = 64
    dataset = datasets.ImageFolder(root="data/processed/food-101")
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers
    )

    return data_loader


if __name__ == "__main__":
    pass
