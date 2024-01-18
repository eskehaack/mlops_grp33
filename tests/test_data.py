import pytest
from torch.utils.data import DataLoader


from MLops_project import food101_dataloader, calculate_mean_std
from data_check import data_check


@pytest.mark.skipif(
    not data_check(),
    reason="Data files not found",
)
def test_food101_dataloader():
    """
    Testing of the data loader.
    This test checks if:
        The output of the dataloader has the correct format.
        The batching works properly.
    """
    bt_size = 64
    data = food101_dataloader(batch_size=bt_size)
    assert len(data) == 3, "Dataloader function did not return 3 objects"
    assert all([type(dl) == DataLoader for dl in data]), "Dataloader function did not return DataLoader objects"
    for i in range(3):
        images, labels = next(iter(data[i]))
        assert (
            len(images) == bt_size
        ), f"Dataloader image batch size is {len(images)} instead of the specified {bt_size}"


@pytest.mark.skipif(
    not data_check(),
    reason="Data files not found",
)
def test_calculate_mean_std():
    """
    Testing of the standerdization.
    This test checks if:
        The output of the standerdizer is the correct type.
        The output of the standerdizer has the correct keys.
        The output of the standerdizer has the correct dimension.
    """
    keys = ["mean", "std"]
    parameters = calculate_mean_std()
    assert isinstance(parameters, dict), f"Parameter output is not of type dict: {type(parameters)}"
    assert all(
        [key in parameters.keys() for key in keys]
    ), f"Both mean and std should be in keys. Current keys {list(parameters.keys())}"
    assert all(
        [len(parameters[key]) == 3 for key in keys]
    ), f"Parameters should be of length 3 (same as # of channels). Current dimension {[len(parameters[key]) for key in keys]}"


if __name__ == "__main__":
    test_food101_dataloader()
