from torchvision import datasets


def main():
    data = datasets.Food101(root="data/raw/", download=True)
    return data


if __name__ == "__main__":
    main()
