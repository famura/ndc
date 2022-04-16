import math

import matplotlib.pyplot as plt
import numpy
import torch

import ndc

# Configure.
stencils = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
order = 1
padding = True
t = torch.linspace(0, 1.5, math.floor(1.5 / 0.01)).reshape(-1, 1)


numpy.set_printoptions(precision=6, sign=" ", linewidth=200, suppress=True)


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


if __name__ == "__main__":
    # Get from configuration.
    dt = float(t[1, 0] - t[0, 0])
    num_steps, dim_data = t.shape

    # Create a clean signal.
    t.requires_grad_(True)
    y = _noisy_nonlin_fcn(t)

    # Differentiate analytically
    y.backward(torch.ones_like(y))
    dy_dt_anal = t.grad.data
    t = t.detach()
    y = y.detach()

    # Create a noisy signal.
    z = _noisy_nonlin_fcn(t, noise_std=0.02)

    # Compute the filter coefficients, only for display here.
    w = ndc.compute_coefficients(stencils, order, dt)
    print(f"The convolution kernel's filter coefficients are:\n{w.numpy()}")

    # Differentiate numerically.
    dy_dt_num = ndc.differentiate_numerically(y, stencils, order, dt, padding)
    dz_dt_num = ndc.differentiate_numerically(z, stencils, order, dt, padding)
    dz_dt_num_naive = torch.diff(z, dim=0) / dt
    if padding:
        dz_dt_num_naive = torch.cat((dz_dt_num_naive, dz_dt_num_naive[-1].unsqueeze(1)))

    # Compute signal-to-noise ratio.
    power_signal = torch.mean(torch.pow(y, 2))
    power_noise = torch.mean(torch.pow(z- y, 2))
    snr = power_signal / power_noise
    snr_db = 10 * torch.log10(snr)
    print(f"The signal-to-noise ration is: {snr.item():.4} = {snr_db.item():.4} dB")

    # Plot.
    fig_c, axs_c = plt.subplots(dim_data, 1, figsize=(12, 8))
    fig_n, axs_n = plt.subplots(dim_data, 1, figsize=(12, 8))
    axs_c = numpy.atleast_1d(axs_c)
    axs_n = numpy.atleast_1d(axs_n)

    fig_c.suptitle("Clean Signal")
    for idx in range(dim_data):
        axs_c[idx].plot(t[:, idx], y[:, idx], label="y(t)")
        if padding:
            axs_c[idx].plot(t[:, idx], dy_dt_num[:, idx], label="dy/dt numerical", ls="-")
        else:
            num_pad_init = sum(s < 0 for s in stencils)
            num_pad_end = sum(s > 0 for s in stencils)
            axs_c[idx].plot(t[num_pad_init:-num_pad_end, idx], dy_dt_num[:, idx], label="dy/dt numerical", ls="-.")
            axs_c[idx].plot(t[:, idx], dy_dt_anal[:, idx], label="dy/dt analytical", ls="--")
        axs_c[idx].legend()

    fig_n.suptitle("Noisy Signal")
    for idx in range(dim_data):
        axs_n[idx].plot(t[:, idx], z[:, idx], label="z(t)")
        if padding:
            axs_n[idx].plot(t[:, idx], dz_dt_num[:, idx], label="dz/dt numerical (ndc)", ls="-")
            axs_n[idx].plot(t[:, idx], dz_dt_num_naive[:, idx], label="dz/dt numerical (naive)", ls="-.")
        else:
            num_pad_init = sum(s < 0 for s in stencils)
            num_pad_end = sum(s > 0 for s in stencils)
            axs_n[idx].plot(t[num_pad_init:-num_pad_end, idx], dz_dt_num[:, idx], label="dz/dt numerical", ls="-.")
        axs_n[idx].plot(t[:, idx], dy_dt_anal[:, idx], label="dy/dt (clean) analytical", ls="--")
        axs_n[idx].legend()

    plt.show()
