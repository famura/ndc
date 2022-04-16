import math
from typing import List, Union

import numpy
import pytest
import torch

import ndc
from tests.conftest import _noisy_nonlin_fcn, m_needs_cuda, m_needs_pyplot


def test_ill_parameterization():
    # Non-unique stencils.
    with pytest.raises(ValueError):
        ndc.compute_coefficients(stencils=[0, 1, 1], order=1, step_size=1)

    # Float order.
    with pytest.raises(TypeError):
        ndc.compute_coefficients(stencils=[0, 1], order=1.0, step_size=1)

    # Non-positive order.
    with pytest.raises(ValueError):
        ndc.compute_coefficients(stencils=[0, 1], order=0, step_size=1)

    # Non-positive step size.
    with pytest.raises(ValueError):
        ndc.compute_coefficients(stencils=[0, 1], order=1, step_size=0)

    # Number of dimensions for the signal tensor > 3.
    with pytest.raises(ValueError):
        ndc.differentiate_numerically(torch.randn(1, 2, 3, 4), stencils=[0, 1], order=1, step_size=1)


@pytest.mark.parametrize(
    "s",
    [[-1, 0, 1], torch.tensor([[-1, 0, 1]]), torch.tensor([-3, -1, 0])],
    ids=["central_list", "central_tensor", "backward_tensor"],
)
@pytest.mark.parametrize("d", [1, 2], ids=["order_1", "order_2"])
@pytest.mark.parametrize("h", [1, 1e-3], ids=["unit_step", "small_step"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=m_needs_cuda)], ids=["cpu", "cuda"])
def test_compute_coefficients(s: Union[torch.LongTensor, List[int]], d: int, h: Union[float, int], device: str):
    c = ndc.compute_coefficients(stencils=s, order=d, step_size=h, device=device)
    assert torch.isclose(torch.sum(c), torch.tensor(0.0), atol=1e-6)
    assert c.device.type == device


@pytest.mark.parametrize(
    "s, d, h, c_des",
    [
        ([-3, -2, -1, 0, 1, 2, 3], 3, 1, torch.tensor([1 / 8, -1, 13 / 8, 0, -13 / 8, 1, -1 / 8])),
        ([0, 1, 2, 3, 4], 2, 1, torch.tensor([35 / 12, -26 / 3, 19 / 2, -14 / 3, 11 / 12])),
        ([-1, 0], 1, 1, torch.tensor([-1.0, 1])),
    ],
    ids=["central_d3_acc4", "forward_d2_acc3", "backward_d1_acc1"],
)
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=m_needs_cuda)], ids=["cpu", "cuda"])
def test_compute_coefficients_cases(
    s: Union[torch.LongTensor, List[int]], d: int, h: Union[float, int], c_des: torch.Tensor, device: str
):
    # See https://en.wikipedia.org/wiki/Finite_difference_coefficient
    c = ndc.compute_coefficients(stencils=s, order=d, step_size=h, device=device)
    assert torch.allclose(c, c_des, atol=1e-5)
    assert c.device.type == device


@pytest.mark.parametrize(
    "x",
    [
        torch.linspace(0, 2, math.floor(1.75 / 0.01)).reshape(-1, 1),
        torch.meshgrid(torch.linspace(0, 1.75, math.floor(2 / 0.01)), torch.linspace(0, 1, 2))[0],
    ],
    ids=["1dim", "2dim"],
)
@pytest.mark.parametrize(
    "stencils, order", [([-1, 0, 1], 1), ([0, 1, 2, 3, 4], 1)], ids=["o1_acc2_central", "o1_acc4_forward"]
)
@pytest.mark.parametrize("padding", [True, False], ids=["padding", "no_padding"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=m_needs_cuda)], ids=["cpu", "cuda"])
@pytest.mark.parametrize("visual", [pytest.param(True, marks=[m_needs_pyplot, pytest.mark.visual]), False])
def test_numerical_differentiation(
    x: torch.Tensor, stencils: Union[torch.LongTensor, List[int]], order: int, padding: bool, device: str, visual: bool
):
    x = x.clone()
    num_steps, dim_data = x.shape
    x.requires_grad_(True)
    dx = float(x[1, 0] - x[0, 0])
    y = _noisy_nonlin_fcn(x)
    y.backward(torch.ones_like(y))
    dy_dx_anal = x.grad.data
    x = x.detach()
    y = y.detach()

    dy_dx_num = ndc.differentiate_numerically(y, stencils, order, step_size=dx, padding=padding)
    assert dy_dx_num.device.type == y.device.type

    # Check the non-padded elements.
    num_pad_init = sum(s < 0 for s in stencils)
    num_pad_end = sum(s > 0 for s in stencils)
    if padding:
        assert torch.allclose(dy_dx_num[num_pad_init:-num_pad_end], dy_dx_anal[num_pad_init:-num_pad_end], atol=dx)
        assert dy_dx_num.shape == (num_steps, dim_data)
    else:
        assert torch.allclose(dy_dx_num, dy_dx_anal[num_pad_init:-num_pad_end], atol=dx)
        assert dy_dx_num.shape == (num_steps - sum(s != 0 for s in stencils), dim_data)

    # Plot.
    if visual:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(dim_data, 1, figsize=(12, 8))
        axs = numpy.atleast_1d(axs)

        for idx in range(dim_data):
            axs[idx].plot(x[:, idx], y[:, idx], label="y(x)")
            axs[idx].plot(x[:, idx], dy_dx_anal[:, idx], label="dy/dx analytical")
            if padding:
                axs[idx].plot(x[:, idx], dy_dx_num[:, idx], label="dy/dx numerical", ls="-.")
            else:
                axs[idx].plot(x[num_pad_init:-num_pad_end, idx], dy_dx_num[:, idx], label="dy/dx numerical", ls="-.")

        fig.suptitle(f"padding = {padding}")
        plt.legend()
        # plt.show()
