import os
import torch
from torchvision import datasets, transforms

from tqdm import tqdm
import json
from typing import Dict, List


def load_statistics(stat_filename: str) -> Dict[str, float]:
    with open(f"data/processed/{stat_filename}", "r") as file:
        stats = json.load(file)

    return stats


def calculate_mean_std() -> Dict[str, List[float]]:
    """Iterates through the training dataset to calcuate the mean and std of the full
    dataset, to determine key statistics for normalization.

    Returns:
        Dict[str,float]: Keys are the metric names, values being list of the metrics themselves
    """
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    dataset = datasets.Food101(root="data/raw/", split="train", download=False, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # Initilize the stat varialbes with 0 tensors of size 3 for each of the R,G,B channels
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images_count = 0
    for images, _ in tqdm(loader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count
    stats = {"mean": mean.tolist(), "std": std.tolist()}
    return stats


def process_data(split: str, normalization_constants: Dict[str, float], seed: int, batch_size=1000):
    """Process and save the dataset in batches.

    Args:
        split (str): 'train' or 'test'.
        normalization_constants (Dict[str, float]): Mean and std for normalization.
        seed (int): Seed for reproducibility.
        batch_size (int): Number of images to process in each batch.
    """
    # Calculate mean and std
    mean, std = normalization_constants.values()

    # Apply the normalization to the raw data
    transform_normalized = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    dataset = datasets.Food101(root="data/raw/", split=split, download=False, transform=transform_normalized)

    datasets_to_process = {}
    # Split the dataset into training and validation if the split is 'train'
    if split == "train":
        train_size = int(0.95 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        datasets_to_process = {"train": train_dataset, "val": val_dataset}
    else:
        datasets_to_process = {split: dataset}

    for subset, data in datasets_to_process.items():
        processed_folder = f"./data/processed/{subset}"
        if not os.path.exists(processed_folder):
            os.makedirs(processed_folder)

        # Initialize lists to store all images and labels
        all_images = []
        all_labels = []

        # Aggregate images and labels
        for image, label in tqdm(data, total=len(data)):
            all_images.append(image)
            all_labels.append(label)

        # Save all images and labels in a single file
        torch.save(
            (all_images, all_labels),
            os.path.join(processed_folder, f"{subset}_data.pt"),
        )


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    stat_filename = "training_split_stats.json"
    if stat_filename not in os.listdir("data/processed"):
        stats = calculate_mean_std()
        with open(f"data/processed/{stat_filename}", "w") as file:
            json.dump(stats, file)
    else:
        stats = load_statistics(stat_filename)

    print("Processing train split of raw data, into processed train and validation..")
    process_data("train", stats, seed)
    print("Train split saved at data/processed/train")
    print("Validation split saved at data/processed/val")

    print("Processing test split of raw into test processed..")
    process_data("test", stats, seed)
    print("Test split saved at data/processed/test")
