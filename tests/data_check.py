import os


def data_check():
    file_check = all(
        [
            os.path.exists("data/processed/test"),
            os.path.exists("data/processed/train"),
            os.path.exists("data/processed/val"),
        ]
    ) and all(
        [
            len(os.listdir("data/processed/test/")) >= 1,
            len(os.listdir("data/processed/train/")) >= 1,
            len(os.listdir("data/processed/val/")) >= 1,
        ]
    )

    return file_check
