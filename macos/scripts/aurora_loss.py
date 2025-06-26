"""Loss functions for Aurora model training."""

import torch


def mae(x_hat_t, x_t):
    lamb = 2
    vs_va = 9
    surface = {
        "2t": 3.0,
        "msl": 1.5,
        "10u": 0.77,
        "10v": 0.66,
    }
    atmos = {
        "z": 2.8,
        "q": 0.78,
        "t": 1.7,
        "u": 0.87,
        "v": 0.6,
    }
    foo = sum(
        [
            (v / (720 * 1440))
            * torch.sum(
                torch.abs(x_hat_t.surf_vars[k] - x_t.surf_vars[k][:, :, :720, :])
            )
            for k, v in surface.items()
        ]
    )
    bar = sum(
        [
            (v / (720 * 1440 * 13))
            * torch.sum(
                torch.abs(x_hat_t.atmos_vars[k] - x_t.atmos_vars[k][:, :, :, :720, :])
            )
            for k, v in atmos.items()
        ]
    )

    alpha = 0.25

    return (lamb / vs_va) * ((alpha * foo) + bar)
