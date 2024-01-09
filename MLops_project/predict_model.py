import torch
import numpy as np
import pickle
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    return torch.cat([model(batch) for batch in dataloader], 0)


def load_data(data_path):
    if data_path.endswith(".npy"):
        images = np.load(data_path)
        return torch.utils.data.TensorDataset(torch.from_numpy(images))
    elif data_path.endswith(".pkl"):
        with open(data_path, "rb") as f:
            images = pickle.load(f)
        return torch.utils.data.TensorDataset(torch.from_numpy(images))
    else:  # assuming a folder of images
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Resize images if necessary
                transforms.ToTensor(),
            ]
        )
        dataset = ImageFolder(root=data_path, transform=transform)
        return dataset


def main(model_path, data_path):
    # Load the pre-trained model
    model = torch.load(model_path)
    model.eval()  # Set the model to evaluation mode

    # Load the data
    dataset = load_data(data_path)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

    # Predict
    predictions = predict(model, dataloader)

    return predictions


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1]
    data_path = sys.argv[2]
    predictions = main(model_path, data_path)
    print(predictions)
