"""
Test gradients.

Rule 1:
"Difference Qutiones may sometimes be useful, too"
Andreas Griewank, Andrea Walther - Evaluating Derivatives.
"""

import sys
from typing import Callable, Optional, Union

import numpy as np
import pytest

sys.path.insert(0, "./src/")

from src.optimize_1d import derivative_parabola, parabola
from src.optimize_2d import grad_paraboloid, paraboloid
from src.optimize_2d_momentum_bumpy import bumpy_function, bumpy_grad


def finite_difference(
    fun: Callable, pos: Union[float, np.ndarray], h: Optional[float] = 1e-9
) -> np.ndarray:
    """Estimate gradients using finite differences.

    More about finite differences at:
    https://en.wikipedia.org/wiki/Finite_difference_method

    Args:
        fun (Callable): The function differentiate.
        pos (Union[float, np.ndarray]): The point at which we are seeking the gradient.
        h (float, optional): The estimation height. Defaults to 1e-9.

    Returns:
        np.ndarray: A crude estimation of the gradient.
    """
    if type(pos) is float:
        pos = np.array((pos,))
    h = np.ones_like(pos) * h
    if len(pos) == 1:
        return (fun(pos + h) - fun(pos)) / h
    else:
        # evaluate along all possible directions.
        grad = np.zeros_like(pos)
        for direction_index in range(len(pos)):
            direction = np.zeros_like(pos)
            direction[direction_index] = 1.0
            grad += (
                (fun((pos + h) * direction) - fun(pos * direction)) / h
            ) * direction
        return grad


@pytest.mark.parametrize("pos", (5.0, 4.0, 0.0, -4.0, -5.0))
def test_grad_parabola(pos):
    """Test the parabola gradient."""
    ddx = derivative_parabola(pos)
    fdddx = finite_difference(parabola, pos)
    assert np.allclose(ddx, fdddx)


@pytest.mark.parametrize(
    "pos",
    (
        np.array((-1.0, -1.0)),
        np.array((0.0, 0.0)),
        np.array((2.9, -2.9)),
        np.array((-2.9, 2.9)),
        np.array((2.9, 2.9)),
    ),
)
def test_grad_paraboloid(pos):
    """Test the paraboloid gradient."""
    nabla_x = grad_paraboloid(pos)
    fd_nabla_x = finite_difference(paraboloid, pos)
    assert np.allclose(nabla_x, fd_nabla_x)


@pytest.mark.parametrize(
    "pos",
    (
        np.array((-1.0, -1.0)),
        np.array((0.0, 0.0)),
        np.array((2.9, -2.9)),
        np.array((-2.9, 2.9)),
        np.array((2.9, 2.9)),
    ),
)
def test_grad_bumpy(pos):
    """Test the bumpy gradient."""
    nabla_x = bumpy_grad(pos)
    fd_nabla_x = finite_difference(bumpy_function, pos)
    assert np.allclose(nabla_x, fd_nabla_x)
