import math
from typing import List, Union

import torch


def compute_coefficients(
    stencils: Union[torch.Tensor, List[int]],
    order: int,
    step_size: Union[float, int],
    device: Union[torch.device, str] = "cpu",
) -> torch.Tensor:
    r"""
    Compute the coefficients for discrete time numerical differentiation. These can later be convolved with the signal.

    Args:
        stencils: Stencil points :math:`s` of length :math:`N`, negative means accessing data point in the past. For
            example pass `[-1, 0, 1]` for a central difference or `[-3, -1, 0]` for a backward difference without
            equally space stencil points.
            Will be cast into an 8bit integer :external:class:`~torch.Tensor` of shape (N,).
        order: Order :math:`d` of the derivative, must be an integer lower than the number of stencil points.
        step_size: Time step size :math:`h`. Will be cast into a `float`.
        device: Device to cast the :external:class:`~torch.Tensor`s to before solving the linear system of equations.
            By default, the CPU is used.
    Returns:
        Coefficients for computing the derivative with the oder of the approximation :math:`\mathcal{O}(h^{N-d})`.
    """
    stencils = torch.as_tensor(stencils, dtype=torch.int64).reshape(-1)
    step_size = float(step_size)
    num_eqs = len(stencils)  # number of equations for the solver N
    stencils = torch.unique(stencils)  # remove double stencils

    if len(stencils) != num_eqs:  # if the length was reduced, there have been non-unique elements
        raise ValueError("The tensor of stencil points must only contain unique elements!")
    if not isinstance(order, int):
        raise TypeError(f"The order must be of type int, but is of type {type(order)}!")
    if not 0 < order < num_eqs:
        raise ValueError(f"The order must be greater than 0 and less than {num_eqs}, but is {order}!")
    if step_size <= 0:
        raise ValueError(f"The step size must be greater than 0, but is {step_size}!")

    A = torch.stack([torch.pow(stencils, idx) for idx in range(num_eqs)], dim=0).float()  # shape (num_eqs, num_eqs)
    b = torch.zeros(num_eqs)
    b[order] = math.factorial(order) / step_size**order

    # Compute the coefficients which are the solution to a linear system of equations (see reference).
    return torch.linalg.solve(A.to(device=device), b.to(device=device))


def differentiate_numerically(
    signal: torch.Tensor,
    stencils: Union[torch.Tensor, List[int]],
    order: int,
    step_size: Union[float, int],
    padding: bool = True,
) -> torch.Tensor:
    r"""
    Compute an arbitrary derivative of the given signal, leveraging :meth:`torch.nn.functional.conv1d`.

    Note:
        Even though :meth:`torch.nn.functional.conv1d` has an option for padding, this function is designed to implement
        its own padding logic. The reason is that the stencils could indicate a forward or backward derivative of any
        order which is a custom use case not covered by PyTorch.

    Args:
        signal: Time series of shape (num_steps, dim_data) to differentiate. The signal's device type is forwarded to
            :meth:`~ndc.numerical_differentiation.compute_coefficients`.
        stencils: Stencil points :math:`s` of length :math:`N`. Forwarded to
            :meth:`~ndc.numerical_differentiation.compute_coefficients`.
        order: Order :math:`d` of the derivative.
            Forwarded to :meth:`~ndc.numerical_differentiation.compute_coefficients`.
        step_size: Time step size :math:`h`. Forwarded to :meth:`~ndc.numerical_differentiation.compute_coefficients`.
        padding: If `True`, a custom padding scheme is applied. Based on how many stencils are negative/positive,
            the respective number of initial or final derivative values is repeated to restore the length of the signal.
    Returns:
        Derivative of the signal. If `padding=False` the shape is (num_steps - :math:`M`, dim_data) where
        :math:`M = count(s \neq 0)`.  If `padding=True` the shape is (num_steps, dim_data).
    """
    dim_data = signal.size(1)

    # Prepare the input for the convolution.
    sigal_conv = torch.atleast_3d(signal)  # in our context this is typically (num_samples, dim_data, 1)
    if sigal_conv.ndim > 3:
        raise ValueError(
            f"The input for the convolution must have 3 or less 3 dimensions, but it is of shape {signal.shape}!"
        )
    sigal_conv = sigal_conv.permute(2, 1, 0)  # of shape (num_minibatch, in_channels, -1)

    # Compute the weights for the convolution. The weights are the same for every dim, thus we repeat them.
    weights_conv = compute_coefficients(stencils, order, step_size, signal.device)
    weights_conv = weights_conv.repeat(dim_data, 1, 1)  # of shape (out_channels, in_channels_per_group, -1)

    # Compute the derivative via a convolution (without using a padding).
    derivative = torch.conv1d(sigal_conv, weights_conv, padding=0, dilation=1, groups=dim_data)

    # Reshape to be of similar shape (num_steps - M, dim_data) as the input signal.
    derivative = derivative.squeeze(0).permute(1, 0)

    # Pad by repeating values. This is an approximation.
    if padding:
        num_pad_init = sum(s < 0 for s in stencils)
        num_pad_end = sum(s > 0 for s in stencils)
        pad_init = derivative[0].repeat(num_pad_init, 1)  # of shape (num_pad_init, dim_data)
        pad_end = derivative[-1].repeat(num_pad_end, 1)  # of shape (num_pad_end, dim_data)
        derivative = torch.cat((pad_init, derivative, pad_end))

    return derivative
