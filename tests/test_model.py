import pytest
from torch import ones, Tensor, LongTensor
import numpy as np

from MLops_project import VGG
from data_check import data_check


DATA = ones((1, 3, 64, 64))


@pytest.mark.skipif(
    not data_check(),
    reason="Data files not found",
)
def test_VGG_output():
    """
    Testing of the model.
    This test checks if:
        The model outputs the data in the correct dimension.
    """
    out_dim = 101
    inp_data = DATA
    model = VGG(out_dim, 64, 4, 0.001)
    out = model(inp_data)

    assert out.shape == (
        1,
        out_dim,
    ), f"Model output is in an incorrect shape {out.shape}"


@pytest.mark.skipif(
    not data_check(),
    reason="Data files not found",
)
def test_VGG_training():
    """
    Testing of the training function.
    This test checks if:
        The training function can produce a loss.
        The training does not produce a NAN or NAN-like loss
        The training loss is zero or positive
    """
    out_dim = 101
    inp_data = DATA
    model = VGG(out_dim, 64, 4, 0.001)
    loss = model.training_step((inp_data, LongTensor([1])), Tensor([1]))

    float(loss)  # If this fails, the test is failed
    assert not np.isnan(loss)
    assert loss >= 0, f"Loss less than zero: {loss}"


@pytest.mark.skipif(
    not data_check(),
    reason="Data files not found",
)
def test_VGG_validation():
    """
    Testing of the validation function.
    This test checks if:
        The validation function can produce a loss.
        The validation does not produce a NAN or NAN-like loss
        The validation loss is zero or positive
    """
    out_dim = 101
    inp_data = DATA
    model = VGG(out_dim, 64, 4, 0.001)
    loss = model.validation_step((inp_data, LongTensor([1])), Tensor([1]))

    float(loss)  # If this fails, the test is failed

    assert loss > 0, f"Loss less than zero: {loss}"


@pytest.mark.skipif(
    not data_check(),
    reason="Data files not found",
)
def test_VGG_test():
    """
    Testing of the test function.
    This test checks if:
        The test function can produce a loss.
        The test does not produce a NAN or NAN-like loss
        The test loss is zero or positive
    """
    out_dim = 101
    inp_data = DATA
    model = VGG(out_dim, 64, 4, 0.001)
    loss = model.test_step((inp_data, LongTensor([1])), Tensor([1]))

    float(loss)  # If this fails, the test is failed

    assert loss > 0, f"Loss less than zero: {loss}"
