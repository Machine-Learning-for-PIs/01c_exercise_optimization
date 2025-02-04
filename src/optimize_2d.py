"""Implement gradient descent in 2d."""

import numpy as np

from util import write_movie


def paraboloid(pos: np.ndarray) -> np.ndarray:
    """Return values from a paraboloid function.

    Args:
        pos (np.ndarray): The position array [x, y].

    Returns:
        np.ndarray: The height value z.
    """
    return pos[0] * pos[0] + pos[1] * pos[1]


def grad_paraboloid(pos: np.ndarray) -> np.ndarray:
    """Return the gradient of the paraboloid.

    Args:
        pos (np.ndarray): The position array [x, y].

    Returns:
        np.ndarray: The gradient at position pos.
    """
    # TODO: implement me!
    return np.zeros_like(pos)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    nx, ny = (301, 301)
    x = np.linspace(-3, 3, nx)
    y = np.linspace(-3, 3, ny)
    mx, my = np.meshgrid(x, y)
    pos = np.stack((mx, my))
    mz = paraboloid(pos)

    plt.contourf(mx, my, mz)
    plt.colorbar()

    start_pos = np.array((2.9, -2.9))
    step_size = 0.0  # TODO: choose your step size.
    step_total = 1  # TODO: choose your step total.

    pos_list = [start_pos]
    for _ in range(step_total):
        pos = pos_list[-1] - step_size * grad_paraboloid(pos_list[-1])
        pos_list.append(pos)

    for pos in pos_list:
        plt.plot(pos[0], pos[1], ".r")
    plt.show()

    write_movie(mx, my, mz, pos_list, "writer_grad_parabola_plot")
