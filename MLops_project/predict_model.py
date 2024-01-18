import torch
from torchvision import transforms
from MLops_project import load_statistics, VGG
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json
from typing import List, Tuple


def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str):
    """Run prediction for a given model and dataloader, returning predictions and filenames.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns:
        Tuple of two elements:
        - Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model
        - List of filenames corresponding to each sample
    """
    model.to(device)
    model.eval()
    predictions = []
    filenames = []
    with torch.no_grad():
        for batch, fname in dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            predictions.append(outputs)
            filenames.extend(fname)

    return torch.cat(predictions, 0), filenames


class CustomImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.images = [file for file in os.listdir(folder_path) if file.endswith(("jpg", "png", "jpeg"))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.folder_path, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_name


def load_data(data_path):
    data_path = os.path.join(os.getcwd(), data_path)
    mean, std = load_statistics("training_split_stats.json").values()
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    dataset = CustomImageDataset(folder_path=data_path, transform=transform)
    return dataset


def predict_main(checkpoint_path: str, data_path: str) -> List[Tuple[str, str]]:
    """
    Load a pre-trained model, perform predictions on a dataset, and return the results.

    Args:
        checkpoint_path (str): Path to the pre-trained model checkpoint.
        data_path (str): Path to the dataset directory.

    Returns:
        List[Tuple[str, str]]: A list of tuples, where each tuple contains the filename
                               and its corresponding predicted label.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained model
    model = VGG.load_from_checkpoint(checkpoint_path, load_datasets=False).to(device)
    model.eval()  # Set the model to evaluation mode

    # Load the data
    dataset = load_data(data_path)  # Ensure load_data is properly defined elsewhere
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

    # Predict
    predictions, filenames = predict(model, dataloader, device)  # Ensure predict is properly defined elsewhere

    with open("data/processed/label_mapping.json", "r") as json_file:
        labels = json.load(json_file)

    # Convert predictions to actual labels
    labeled_predictions = [labels[torch.argmax(pred).item()] for pred in predictions]

    # Combine filenames with their corresponding predictions
    results = list(zip(filenames, labeled_predictions))

    return results


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1]
    data_path = sys.argv[2]
    predictions = predict_main(model_path, data_path)
    print(predictions)
