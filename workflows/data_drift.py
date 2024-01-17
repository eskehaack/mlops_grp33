import torch
import numpy as np
from MLops_project.predict_model import predict_main, load_data
from torchvision import transforms
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfDriftedColumns, TestShareOfDriftedColumns
import pandas as pd
import json
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


def get_features(images, labels, name="org_metrics.csv"):
    # Convert images to grayscale
    transform = transforms.Grayscale()

    images = [transform(t) for t in images]
    # Convert images to tensors
    images = [transforms.ToTensor()(t) for t in images]

    # Compute average brightness of all images
    brightness = [torch.mean(t) for t in images]

    # Compute contrast of all images
    contrast = [torch.std(t) for t in images]

    # Compute Image Sharpness
    sharpness = [torch.var(t) for t in images]

    # Save all metrics in a csv file
    metrics = np.array([brightness, contrast, sharpness, labels]).T
    np.savetxt(f"tests/{name}", metrics, delimiter=",")


def load_drift_data(model_path, data_path):
    # Load data and model and create predictions
    data = load_data(data_path)
    predictions = predict_main(model_path, data_path)
    predictions = [item[1] for item in predictions]
    print(predictions)

    # Load label mappings
    if os.path.exists("/gcs"):
        gcs_string = "/gcs/dtu_mlops_grp33_processed_data/"
    else:
        gcs_string = ""
    with open(gcs_string + "data/processed/label_mapping.json", "r") as json_file:
        labels = json.load(json_file)
        labels = list(labels)

    # Convert predictions to integer representation of labels
    predictions = [labels.index(pred) for pred in predictions]

    # Turn into PIL images
    transform = transforms.ToPILImage()
    images = [transform(t[0]) for t in data]

    # Get features and save to csv file
    get_features(images, predictions, name="drift_metrics.csv")


def evidently_tests(reference: str, current: str) -> None:
    reference = pd.read_csv("workflows/" + reference)
    current = pd.read_csv("workflows/" + current)

    report = TestSuite(
        tests=[
            TestNumberOfDriftedColumns(),
            TestShareOfDriftedColumns(),
        ]
    )
    report.run(reference_data=reference, current_data=current)
    report.save_html("./report.html")


if __name__ == "__main__":
    # images, labels = load_org_data("data/processed")
    # get_features(images, labels, name="org_metrics.csv")

    # model_path = "/home/lachlan/mlops_grp33/outputs/2024-01-16/11-14-24/models/model.ckpt"
    # data_path = "/home/lachlan/mlops_grp33/workflows/test_images"
    # load_drift_data(model_path, data_path)

    evidently_tests("org_metrics.csv", "drift_metrics.csv")
