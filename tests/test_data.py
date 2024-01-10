import pytest
from torch.utils.data import DataLoader


from MLops_project.data.dataload import food101_dataloader
from data_check import data_check


@pytest.mark.skipif(
    not data_check(),
    reason="Data files not found",
)
def test_food101_dataloader():
    bt_size = 64
    data = food101_dataloader(batch_size=bt_size)
    assert len(data) == 3, "Dataloader function did not return 3 objects"
    assert all([type(dl) == DataLoader for dl in data]), "Dataloader function did not return DataLoader objects"
    for i in range(3):
        images, labels = next(iter(data[i]))
        assert (
            len(images) == bt_size
        ), f"Dataloader image batch size is {len(images)} instead of the specified {bt_size}"


if __name__ == "__main__":
    test_food101_dataloader()
