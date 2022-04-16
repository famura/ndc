"""
This file is found by pytest and contains fixtures that can be used for all tests.
"""
import pytest
import torch

# Check if optional packages are available.
try:
    from matplotlib import pyplot as plt

    m_needs_pyplot = pytest.mark.skipif(False, reason="matplotlib.pyplot can be imported.")

except (ImportError, ModuleNotFoundError):
    m_needs_pyplot = pytest.mark.skip(reason="matplotlib.pyplot is not supported in this setup.")

# Check if CUDA support is available.
m_needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not supported in this setup.")


def _noisy_nonlin_fcn(t: torch.Tensor, freq: float = 1.0, noise_std: float = 0.0) -> torch.Tensor:
    """
    A 1-dim function (sinus superposed with polynomial), representing some singal.

    Args:
        t: Function argument, e.g. time.
        noise_std: Scale of the additive noise sampled from a standard normal distribution.
        freq: Frequency of the sinus wave.
    Returns:
        Function value.
    """
    return -torch.sin(2 * torch.pi * freq * t) - torch.pow(t, 2) + 0.7 * t + noise_std * torch.randn_like(t)
