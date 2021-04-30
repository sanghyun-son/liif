import typing

import torch

from srwarp import transform
from srwarp import crop
from srwarp import warp

@torch.no_grad()
def quantize(x: torch.Tensor) -> torch.Tensor:
    x = 127.5 * (x + 1)
    x = x.clamp(min=0, max=255)
    x = x.round()
    x = x / 127.5 - 1
    return x

@torch.no_grad()
def get_input(
        gt: torch.Tensor,
        m_inv: torch.Tensor,
        stochastic: bool=True) -> typing.Tuple[torch.Tensor, torch.Tensor]:

    m_inv, sizes, _ = transform.compensate_matrix(gt, m_inv)
    m = transform.inverse_3x3(m_inv)

    ignore_value = -255
    lr = warp.warp_by_function(
        gt,
        m,
        sizes=sizes,
        kernel_type='bicubic',
        adaptive_grid=True,
        fill=ignore_value,
    )

    patch_max = 96
    lr_crop, iy, ix = crop.valid_crop(
        lr,
        ignore_value,
        patch_max=patch_max,
        stochastic=stochastic,
    )
    lr_crop = quantize(lr_crop)
    m = transform.compensate_offset(m, ix, iy)
    return lr_crop, m