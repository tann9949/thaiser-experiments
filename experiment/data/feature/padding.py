import torch
from torch import Tensor


def pad_dup(x: Tensor, max_len: int) -> Tensor:
    """Pad an Arguments feature upto specified length.
    The Arguments is repeated until max_len is reached.

    Arguments
    ---------
    x: Tensor
        Input feature to pad
    max_len: int
        Maximum sequence length to pad (in number of frame)

    Return
    ------
    padded_x: Tensor
        Padded value of x
    """
    time_dim: int = x.shape[-1];
    tmp: Tensor = x.clone();
    num_repeat: int = int(max_len / time_dim);
    remainder: int = max_len - num_repeat * time_dim;
    x_rem: Tensor = x[:, :remainder];
    for _ in range(num_repeat - 1):
        x: Tensor = torch.cat([x, tmp], dim=-1);
    x_pad: Tensor = torch.cat([x, x_rem], dim=-1);
    return x_pad;


def pad_constant(x: Tensor, max_len: int, constant: float) -> Tensor:
    """Pad Arguments feature up to specified length.
    The padded values are zero.
    
    Arguments
    ---------
    x: Tensor
        Input feature to pad
    max_len: int
        Maximum sequence length to pad (in number of frame)

    Return
    ------
    padded_x: Tensor
        Padded value of x
    """
    zeros: Tensor = torch.zeros([x.shape[0], max_len - x.shape[1]]) + constant;
    x_pad: Tensor = torch.cat([x, zeros], dim=-1);
    return x_pad;
