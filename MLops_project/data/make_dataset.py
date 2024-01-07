from torchvision import datasets


def main():
    train = datasets.Food101(root="data/raw/", split="train", download=True)
    test = datasets.Food101(root="data/raw/", split="test", download=True)
    return train, test


if __name__ == "__main__":
    main()
