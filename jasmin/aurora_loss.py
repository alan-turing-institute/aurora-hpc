"""Loss functions for Aurora model training."""

import torch

from aurora import Batch


def mae(x_hat_t: Batch, x_t: Batch) -> torch.Tensor:

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

    foo = torch.zeros((), device=next(iter(x_hat_t.surf_vars.values())).device)
    for key, weight in surface.items():
        pred = x_hat_t.surf_vars[key]
        target = x_t.surf_vars[key][..., : pred.shape[-2], : pred.shape[-1]]
        if pred.device != target.device:
            # Move whichever isn't on the CPU to the CPU
            pred = pred.to("cpu")
            target = target.to("cpu")

        foo = foo + weight * torch.mean(torch.abs(pred - target))

    bar = torch.zeros((), device=next(iter(x_hat_t.atmos_vars.values())).device)
    for key, weight in atmos.items():
        pred = x_hat_t.atmos_vars[key]
        target = x_t.atmos_vars[key][
            ...,
            : pred.shape[-3],
            : pred.shape[-2],
            : pred.shape[-1],
        ]
        if pred.device != target.device:
            # Move whichever isn't on the CPU to the CPU
            pred = pred.to("cpu")
            target = target.to("cpu")
        bar = bar + weight * torch.mean(torch.abs(pred - target))

    alpha = 0.25

    return (lamb / vs_va) * ((alpha * foo) + bar)
