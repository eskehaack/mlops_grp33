import os

import pytest
from torch import ones

from MLops_project.models.model import VGG
from data_check import data_check


@pytest.mark.skipif(
    not data_check(),
    reason="Data files not found",
)
def test_VGG_output():
    out_dim = 101
    inp_data = ones((1, 1, 64, 64))
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
    out_dim = 101
    inp_data = ones((1, 1, 64, 64))
    model = VGG(out_dim, 64, 4, 0.001)
    loss = model.training_step((inp_data, 0), 0)

    try:
        float(loss)
    except:
        ValueError(f"Loss could not be converted to float: {loss}")

    assert loss > 0, f"Loss less than zero: {loss}"


@pytest.mark.skipif(
    not data_check(),
    reason="Data files not found",
)
def test_VGG_validation():
    out_dim = 101
    inp_data = ones((1, 1, 64, 64))
    model = VGG(out_dim, 64, 4, 0.001)
    loss = model.validation_step((inp_data, 0), 0)

    try:
        float(loss)
    except:
        ValueError(f"Loss could not be converted to float: {loss}")

    assert loss > 0, f"Loss less than zero: {loss}"


@pytest.mark.skipif(
    not data_check(),
    reason="Data files not found",
)
def test_VGG_test():
    out_dim = 101
    inp_data = ones((1, 1, 64, 64))
    model = VGG(out_dim, 64, 4, 0.001)
    loss = model.test_step((inp_data, 0), 0)

    try:
        float(loss)
    except:
        ValueError(f"Loss could not be converted to float: {loss}")

    assert loss > 0, f"Loss less than zero: {loss}"
