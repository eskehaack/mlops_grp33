import torch
import numpy as np
from MLops_project import food101_dataloader, calculate_mean_std
from MLops_project.predict_model import predict_main, load_data
from torchvision import transforms
import os

def load_org_data(path):
    # load all .pt files in the folders (takes a while)
    val_path = os.path.join(os.getcwd(), path + "/val")
    test_path = os.path.join(os.getcwd(), path + "/test") 
    train_path = os.path.join(os.getcwd(), path + "/train")
    val_filenames = [os.path.join(val_path, f) for f in os.listdir(val_path) if f.endswith(".pt")]
    test_filenames = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith(".pt")]
    train_filenames = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith(".pt")]

    # load all tensors
    val_tensors = [torch.load(f) for f in val_filenames]
    test_tensors = [torch.load(f) for f in test_filenames]
    train_tensors = [torch.load(f) for f in train_filenames]

    # Combine all tensors
    tensors = val_tensors.extend(train_tensors)
    tensors = val_tensors.extend(test_tensors)

    # Get labels and images
    images = [t[0] for t in tensors]
    labels = [t[1] for t in tensors]

    return images, labels

def get_features(images, labels):
    # Convert images to grayscale
    transform = transforms.Grayscale()

    images = [transform(t) for t in images]
    # Compute average brightness of all images
    brightness = [torch.mean(t) for t in images]

    # Compute contrast of all images
    contrast = [torch.std(t) for t in images]

    # Compute Image Sharpness
    sharpness = [torch.var(t) for t in images]

    # Save all metrics in a csv file
    metrics = np.array([brightness, contrast, sharpness, labels]).T
    np.savetxt("org_metrics.csv", metrics, delimiter=",")


def load_drift_data(data_path, model_path):
    data = load_data(data_path)
    predictions = predict_main(data_path, model_path)
    predictions = predictions[1].tolist()

    get_features(data, predictions)

if __name__ == "__main__":
    #load_drift_data("data", "model/model.pt")
    images, labels = load_org_data("data/processed")
    print("Done loading data")
    get_features(images, labels)



