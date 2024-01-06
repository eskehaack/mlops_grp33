from torchvision import datasets


def main():
    data = datasets.Places365(root="data/raw/", download=True)
    return data


if __name__ == "__main__":
    main()
