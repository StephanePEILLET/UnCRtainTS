import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Union

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.io import DatasetReader
from scipy.ndimage import gaussian_filter

from utils_misc.detect_cloudshadow import get_cloud_mask, get_shadow_mask

# Type aliases for better readability
ImageArray = NDArray[np.float32]
MaskArray = NDArray[np.float32]
CloudDetectorType = Union[str, Any]  # Can be either string or cloud detector instance
RescaleMethod = Literal["default", "resnet"]


def to_date(date_string: str) -> datetime:
    """
    Convert a date string to datetime object.

    Args:
        date_string: Date string in format "YYYY-MM-DD"

    Returns:
        Corresponding datetime object

    Example:
        >>> to_date("2020-01-01")
        datetime.datetime(2020, 1, 1, 0, 0)
    """
    return datetime.strptime(date_string, "%Y-%m-%d")


def read_tif(img_path: Union[str, Path]) -> DatasetReader:
    """
    Read a GeoTIFF file using rasterio.

    Args:
        img_path: Path to the GeoTIFF file

    Returns:
        Rasterio DatasetReader object

    Raises:
        rasterio.errors.RasterioIOError: If file cannot be read
    """
    return rasterio.open(img_path)


def read_img(tif: DatasetReader) -> ImageArray:
    """
    Read image data from rasterio DatasetReader and convert to float32.

    Args:
        tif: Rasterio DatasetReader object

    Returns:
        Image data as float32 numpy array
    """
    return tif.read().astype(np.float32)


def rescale(img: ImageArray, old_min: float, old_max: float) -> ImageArray:
    """
    Rescale image values from [old_min, old_max] to [0, 1] range.

    Args:
        img: Input image array
        old_min: Minimum value of original range
        old_max: Maximum value of original range

    Returns:
        Rescaled image array

    Example:
        >>> rescale(np.array([5000, 7500, 10000]), 0, 10000)
        array([0.5, 0.75, 1.0], dtype=float32)
    """
    old_range = old_max - old_min
    return (img - old_min) / old_range


def process_MS(img: ImageArray, method: RescaleMethod = "default") -> ImageArray:
    """
    Process multispectral (MS) image data with specified normalization method.

    Args:
        img: Input multispectral image
        method: Processing method ("default" or "resnet")
            - "default": Clip to [0, 10000] and rescale to [0, 1]
            - "resnet": Clip to [0, 10000] and divide by 2000

    Returns:
        Processed image array with NaN values replaced by 0
    """
    if method == "default":
        intensity_min, intensity_max = 0, 10000
        img = np.clip(img, intensity_min, intensity_max)
        img = rescale(img, intensity_min, intensity_max)
    elif method == "resnet":
        intensity_min, intensity_max = 0, 10000
        img = np.clip(img, intensity_min, intensity_max)
        img /= 2000  # Scale to [0, 5] range

    return np.nan_to_num(img)


def process_SAR(img: ImageArray, method: RescaleMethod = "default") -> ImageArray:
    """
    Process SAR image data with specified normalization method.

    Args:
        img: Input SAR image (2 channels: VV and VH)
        method: Processing method ("default" or "resnet")
            - "default": Clip to [-25, 0] dB and rescale to [0, 1]
            - "resnet": Process each channel with different ranges

    Returns:
        Processed SAR image array with NaN values replaced by 0
    """
    if method == "default":
        dB_min, dB_max = -25, 0
        img = np.clip(img, dB_min, dB_max)
        img = rescale(img, dB_min, dB_max)
    elif method == "resnet":
        # Process VV and VH channels separately
        dB_min, dB_max = [-25.0, -32.5], [0, 0]
        processed_channels = []
        for i in range(2):
            channel = img[i]
            clipped = np.clip(channel, dB_min[i], dB_max[i])
            scaled = 2 * (clipped - dB_min[i]) / (dB_max[i] - dB_min[i])
            processed_channels.append(scaled[None, ...])
        img = np.concatenate(processed_channels, axis=0)

    return np.nan_to_num(img)


def get_cloud_cloudshadow_mask(img: ImageArray, cloud_threshold: float = 0.2) -> MaskArray:
    """
    Generate combined cloud and cloud shadow mask for an image.

    Args:
        img: Input multispectral image
        cloud_threshold: Threshold for cloud detection (default: 0.2)

    Returns:
        Binary mask where 1 indicates clouds/shadows and 0 indicates clear sky
    """
    cloud_mask = get_cloud_mask(img, cloud_threshold, binarize=True)
    shadow_mask = get_shadow_mask(img)

    # Combine and binarize masks
    combined_mask = np.zeros_like(cloud_mask)
    combined_mask[shadow_mask < 0] = -1
    combined_mask[cloud_mask > 0] = 1
    combined_mask[combined_mask != 0] = 1  # Binarize

    return combined_mask


def iterdict(dictionary: Dict[str, Any], func: Callable[[Any], Any]) -> Dict[str, Any]:
    """
    Recursively apply a function to all values in a nested dictionary.

    Args:
        dictionary: Input dictionary (potentially nested)
        func: Function to apply to each non-dictionary value

    Returns:
        Dictionary with function applied to all values

    Example:
        >>> d = {'a': 1, 'b': {'c': 2}}
        >>> iterdict(d, lambda x: x*2)
        {'a': 2, 'b': {'c': 4}}
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = iterdict(value, func)
        else:
            dictionary[key] = func(value)
    return dictionary


def get_cloud_map(img: ImageArray, detector: CloudDetectorType, instance: Optional[Any] = None) -> MaskArray:
    """
    Generate cloud mask using specified detection method.

    Args:
        img: Input multispectral image
        detector: Cloud detection method or string identifier
        instance: Optional detector instance for some methods

    Returns:
        Cloud probability mask or binary cloud mask

    Raises:
        Warning: If detector method not implemented or blank image encountered
    """
    img = np.clip(img, 0, 10000)
    mask = np.ones((img.shape[-1], img.shape[-1]), dtype=np.float32)

    # Skip processing if image is blank
    if img.mean() < 1e-5 and img.std() < 1e-5:
        warnings.warn("Encountered a blank sample, defaulting to cloudy mask.")
        return mask

    if detector == "cloud_cloudshadow_mask":
        threshold = 0.2
        mask = get_cloud_cloudshadow_mask(img, threshold)
    elif detector == "s2cloudless_map":
        threshold = 0.5
        cloud_prob = instance.get_cloud_probability_maps(np.moveaxis(img / 10000, 0, -1)[None, ...])[0, ...]
        cloud_prob[cloud_prob < threshold] = 0
        mask = gaussian_filter(cloud_prob, sigma=2)
    elif detector == "s2cloudless_mask":
        mask = instance.get_cloud_masks(np.moveaxis(img / 10000, 0, -1)[None, ...])[0, ...]
    else:
        warnings.warn(f"Method {detector} not yet implemented!")

    return mask.astype(np.float32)
