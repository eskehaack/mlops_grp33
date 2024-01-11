import torch
from torch.utils.data import DataLoader, Dataset
import os
from typing import List
from hydra.utils import get_original_cwd


class ProcessedFood101(Dataset):
    def __init__(self, processed_folder):
        if "outputs" in os.getcwd():
            self.processed_folder = os.path.join(get_original_cwd(), processed_folder)
        else:
            self.processed_folder = os.path.join(os.getcwd(), processed_folder)

        # Assuming there is only one file per folder
        file_name = os.listdir(self.processed_folder)[0]
        self.data, self.labels = torch.load(os.path.join(self.processed_folder, file_name))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def food101_dataloader(batch_size=64, num_workers=4) -> List[DataLoader]:
    train = ProcessedFood101("data/processed/train")
    val = ProcessedFood101("data/processed/val")
    test = ProcessedFood101("data/processed/test")
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = food101_dataloader(batch_size=8)
    for batch in train_loader:
        images, labels = batch
        print(images.shape)
        print(labels.shape)
        break
