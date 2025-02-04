"""Code to export gradient descent sequences into a movie."""

from typing import Optional

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np


def write_movie(
    mx: np.ndarray,
    my: np.ndarray,
    mz: np.ndarray,
    pos_list: list,
    name: Optional[str] = "grad_movie",
    xlim: Optional[int] = 3,
    ylim: Optional[int] = 3,
):
    """Write the optimization steps into a mp4-movie file.

    Args:
        mx (np.ndarray): A x-value grid. Required for the background.
        my (np.ndarray): A y-value grid. Required for the background.
        mz (np.ndarray): A z-value grid. Required for the background.
        pos_list (list): A list of optimization positions.
        name (str, optional): The name of the movie file. Defaults to "grad_movie".
        xlim (int, optional): Largest x value in the data. Defaults to 3.
        ylim (int, optional): Largest y value in the data. Defaults to 3.
    
    Raises:
        RuntimeError: If conda ffmpeg package is not installed.
    """
    try:
        ffmpeg_writer = manimation.writers["ffmpeg"]
    except RuntimeError:
        raise RuntimeError(
            "RuntimeError: If you are using anaconda or miniconda there might "
            "be a missing package named ffmpeg. Try installing it with "
            "'conda install -c conda-forge ffmpeg' in your terminal."
        )

    metadata = dict(
        title="Gradient descent", artist="Matplotlib", comment="Minimization movie!"
    )
    writer = ffmpeg_writer(fps=15, metadata=metadata)

    fig = plt.figure()
    plt.contourf(mx, my, mz)
    plt.colorbar()
    (l,) = plt.plot([], [], ".r")

    plt.xlim(-xlim, xlim)
    plt.ylim(-ylim, ylim)

    with writer.saving(fig, f"{name}.gif", 100):
        for pos in pos_list:
            l.set_data([pos[0]], [pos[1]])
            writer.grab_frame()
