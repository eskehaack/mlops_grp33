import os


def data_check():
    parent_path_check = all(
        [
            os.path.exists("data/processed/test"),
            os.path.exists("data/processed/train"),
            os.path.exists("data/processed/val"),
        ]
    )
    subpath_check = all(
        [
            len(os.listdir("data/processed/test/")) >= 1,
            len(os.listdir("data/processed/train/")) >= 1,
            len(os.listdir("data/processed/val/")) >= 1,
        ]
    )

    return all([check for check in [parent_path_check, subpath_check]])
