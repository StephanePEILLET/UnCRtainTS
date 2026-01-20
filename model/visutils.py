import math
from typing import List, Literal, Optional, Tuple, Union

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor

COLORMAPS = {
    "rgb": None,
    "cloud_prob": "YlOrBr",
    "binary_mask": "gray",
    "error": "Reds",
    "att": "magma",
    "s1": "gray",
}


def apply_brightness_factor(
    data: np.ndarray | Tensor, factor: float = 3.0
) -> np.ndarray | Tensor:

    data = data * factor

    if isinstance(data, np.ndarray):
        data = np.where(data > 1, 1, data)
    else:
        data = torch.where(data > 1, torch.ones_like(data), data)

    return data


def gallery(
    data: np.ndarray | Tensor,
    ncols: int = 10,
    border_thickness: int = 2,
    border_color: Literal["black", "white"] = "black",
    brightness_factor: float = 3.0,
) -> np.ndarray | Tensor:

    if border_color not in ["black", "white"]:
        raise ValueError("Choose either 'white' or 'black' as border color.\n")

    if isinstance(data, torch.Tensor):
        seq_length, n_channels, height, width = data.size()
        data = data.permute((0, 2, 3, 1))
    else:
        seq_length, n_channels, height, width = data.shape
        data = data.transpose((0, 2, 3, 1))

    # Apply brightness factor to the images
    if brightness_factor is not None:
        data = apply_brightness_factor(data, brightness_factor)

    ncols = min(data.shape[0], ncols)
    nrows = math.ceil(seq_length / ncols)

    if border_color == "black":
        padded = np.zeros(
            (
                seq_length,
                height + 2 * border_thickness,
                width + 2 * border_thickness,
                n_channels,
            )
        )
    else:
        padded = np.ones(
            (
                seq_length,
                height + 2 * border_thickness,
                width + 2 * border_thickness,
                n_channels,
            )
        )

    padded[
        :, border_thickness:-border_thickness, border_thickness:-border_thickness, :
    ] = data

    if nrows * ncols != seq_length:
        # Add frames to complement the last row of the gallery
        if border_color == "black":
            dummy_frames = np.zeros(
                (
                    nrows * ncols - seq_length,
                    height + 2 * border_thickness,
                    width + 2 * border_thickness,
                    n_channels,
                )
            )
        else:
            dummy_frames = np.ones(
                (
                    nrows * ncols - seq_length,
                    height + 2 * border_thickness,
                    width + 2 * border_thickness,
                    n_channels,
                )
            )
        padded = np.concatenate((padded, dummy_frames), axis=0)

    height += 2 * border_thickness
    width += 2 * border_thickness

    result = (
        padded.reshape(nrows, ncols, height, width, n_channels)
        .swapaxes(1, 2)
        .reshape(height * nrows, width * ncols, n_channels)
    )

    if isinstance(data, torch.Tensor):
        result = torch.from_numpy(result)

    return result


def sequence2gallery(
    data: np.ndarray | Tensor,
    variable: Literal["rgb", "binary_mask", "att", "cloud_prob", "s1"] = "rgb",
    ncols: int = 10,
    border_thickness: int = 2,
    border_color: Literal["black", "white"] = "black",
    indices_rgb: List[int] | List[float] | Tensor = [0, 1, 2],
    brightness_factor: float = 1,
    dpi: int = 300,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    return_grid: bool = False,
) -> Union[
    matplotlib.figure.Figure, Tuple[matplotlib.figure.Figure, np.ndarray | Tensor]
]:
    """
    Plots an image time series as a grid.

    Args:
        data:               torch.Tensor or np.ndarray, input time series, (T x C x H x W) or (T x H x W).
        variable:           str, image type, choose among:
                                'rgb':          RGB image composites
                                'binary_mask':  binary masks
                                'att':          attention masks
                                'cloud_prob':   cloud probability maps
                                's1':           2-channel Sentinel-1 imagery
        ncols:              int, number of columns of the grid.
        border_thickness:   int, thickness of the grid lines.
        border_color:       str, color of the grid lines, choose among ['black', 'white'].
        indices_rgb:        list of int or list of float or torch.Tensor, indices of the RGB channels.
        brightness_factor:  float, brightness factor applied to all images in the sequence.
        dpi:                int, dpi value.
        vmin:               float, minimum data range.
        vmax:               float, maximum data range.
        return_grid:        bool, True to return the grid image.

    Returns:
        matplotlib.pyplot.
    """

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    assert variable in ["rgb", "binary_mask", "att", "cloud_prob", "s1"]
    if variable == "rgb":
        assert data.dim() == 4 and data.shape[1] >= 3
    elif variable in ["att", "s1"]:
        assert data.dim() == 4 or (data.dim() == 3 and data.shape[1] != 1)

    if variable == "rgb":
        data = data[:, indices_rgb, :, :]
        cmap = COLORMAPS["rgb"]
    elif variable in ["att", "binary_mask", "cloud_prob", "s1"]:
        if data.dim() == 3:
            data = data.unsqueeze(dim=1)
        cmap = COLORMAPS[variable]
    elif variable == "binary_mask":
        cmap = COLORMAPS["binary_mask"]
    elif variable == "cloud_prob":
        cmap = COLORMAPS["cloud_prob"]
    else:
        cmap = None

    if variable == "att":
        brightness_factor = 1

    grid = gallery(
        data,
        ncols=ncols,
        border_thickness=border_thickness,
        border_color=border_color,
        brightness_factor=brightness_factor,
    )

    fig = plt.figure(dpi=dpi)
    plt.imshow(grid, cmap, vmin=vmin, vmax=vmax)
    plt.axis("off")

    if return_grid:
        return fig, grid
    return fig


import math
from typing import Dict

from rich.console import Console
from rich.table import Table


def display_metrics(results: Dict[str, float]):
    """
    Displays result metrics in a formatted table using rich.
    The table structure adapts based on whether detailed metrics are provided.

    Args:
        results (Dict[str, float]): The dictionary containing metric names and their values.
    """
    console = Console()

    # Check if detailed (occluded/observed) metrics are present in the results
    has_detailed_metrics = any(
        "_occluded_input_pixels" in key or "_observed_input_pixels" in key for key in results.keys()
    )

    table = Table(
        title="[bold bright_blue]Cloud Reconstruction Metrics[/bold bright_blue]",
        show_header=True,
        header_style="bold magenta",
    )

    # Define table columns based on the content of the results dictionary
    table.add_column("Metric", style="cyan", no_wrap=True, justify="right")
    if has_detailed_metrics:
        table.add_column("Overall", style="white", justify="center")
        table.add_column("Occluded Pixels", style="yellow", justify="center")
        table.add_column("Observed Pixels", style="green", justify="center")
    else:
        table.add_column("Value", style="white", justify="center")

    # Get a sorted list of base metrics for consistent display order
    base_metrics = sorted([k for k in results.keys() if not ("_occluded" in k or "_observed" in k)])

    # Helper function to format numbers into strings, handling NaN values
    def format_value(v: float) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "[dim]N/A[/dim]"
        return f"{v:.4f}"

    for metric in base_metrics:
        # If detailed metrics exist, populate all columns
        if has_detailed_metrics:
            # Handle the special case for 'ssim' key naming
            occluded_key = (
                f"{metric}_images_occluded_input_pixels" if metric == "ssim" else f"{metric}_occluded_input_pixels"
            )
            observed_key = (
                f"{metric}_images_observed_input_pixels" if metric == "ssim" else f"{metric}_observed_input_pixels"
            )

            # Get values, which will be None if the key is missing
            val_global = results.get(metric)
            val_occluded = results.get(occluded_key)
            val_observed = results.get(observed_key)

            table.add_row(
                metric.upper(),
                format_value(val_global),
                format_value(val_occluded),
                format_value(val_observed),
            )
        # Otherwise, populate the simple table
        else:
            val_global = results.get(metric)
            table.add_row(metric.upper(), format_value(val_global))

    console.print(table)