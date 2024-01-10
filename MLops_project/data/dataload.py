import torch
from torch.utils.data import DataLoader
import os
from typing import List
from hydra.utils import get_original_cwd


class ProcessedFood101(torch.utils.data.Dataset):
    def __init__(self, processed_folder):
        if "outputs" in os.getcwd():
            self.processed_folder = os.path.join(get_original_cwd(), processed_folder)
        else:
            self.processed_folder = os.path.join(os.getcwd(), processed_folder)

        self.file_names = os.listdir(self.processed_folder)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        return torch.load(os.path.join(self.processed_folder, file_name))


def food101_dataloader(batch_size=64, num_workers=4) -> List[DataLoader]:
    """Custom dataloader creation function.
      Takes processed data from the 3 splits: "train","val" and "test" from the data/processed/ folder
    and packages it into 3 dataloaders for each split.

    Returns:
        List[DataLoader]: 3 loaders for each split in the respective order: Train, Validation, Test
    """
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
