import os


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
    assert all(
        [type(dl) == DataLoader for dl in DataLoader]
    ), "Dataloader function did not return DataLoader objects"
    for i in range(3):
        for batch in data[i]:
            assert (
                len(batch) == bt_size
            ), f"Dataloader batch size is {len(batch)} instead of the specified {bt_size}"
            break
