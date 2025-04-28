"""Implement 2d gradient descent."""

import numpy as np
import torch as th
from torch.func import grad

from util import write_movie


def bumpy_function(pos: th.Tensor) -> th.Tensor:
    """Return values from an even bumpier function.

    This even bumpier function is hard to optimize.
    It will require momentum.

    Args:
        pos (np.ndarray): The position array [x, y].

    Returns:
        np.ndarray: The height value z.
    """
    return (
        pos[0] * pos[0]
        + pos[1] * pos[1]
        + th.cos(pos[0] * 2 * th.pi)
        + th.sin(pos[1] * 2 * th.pi)
        + (pos[0] > 0).type(pos.dtype) * 0.5
        + th.tanh(th.sqrt(pos[0] ** 2 + pos[1] ** 2)) * 10
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import cm

    # TODO: use torch to find the gradient.

    nx, ny = (1001, 1001)
    x = th.linspace(-3, 3, nx)
    y = th.linspace(-3, 3, ny)
    mx, my = th.meshgrid(x, y)
    pos = th.stack((mx, my))
    mz = bumpy_function(pos)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(mx, my, mz, cmap=cm.coolwarm)
    fig.colorbar(surf)

    plt.show()

    plt.contourf(mx, my, mz)
    plt.colorbar()

    start_pos = th.tensor((2.9, -2.9))
    step_size = 0.0  # TODO: Choose your step size.
    alpha = 0.0  # TODO: Choose your momentum term.
    step_total = 100

    pos_list = [start_pos]
    velocity_vec = th.tensor((0.0, 0.0))
    # TODO: Implement gradient descent with momentum.

    for pos in pos_list:
        plt.plot(pos[0], pos[1], ".r")
    plt.show()

    write_movie(
        np.array(mx),
        np.array(my),
        np.array(mz),
        pos_list,
        "writer_grad_bumpy_plot_torch",
    )
