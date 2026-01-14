import datetime
import itertools
from pathlib import Path
from typing import List, Literal, Tuple, Union

import numpy as np
import rasterio
from rasterio.windows import Window

# Constants for the number of channels in Sentinel-2 and Sentinel-1 data
S2_N_CHANNELS = 12
S1_N_CHANNELS = 4


class SentinelDataProcessor:
    """
    Utility class for processing Sentinel data.
    """

    @staticmethod
    def read_MS(path_raster: str, window: Window) -> np.ndarray:
        """
        Reads and processes multispectral (MS) data from a raster file.

        Parameters:
        - path_raster (str): Path to the raster file.
        - window (rasterio.windows.Window): Window defining the region of interest.

        Returns:
        - np.ndarray: Processed MS data with shape (T, H, W, C), where:
            - T: Number of time steps.
            - H: Height of the patch.
            - W: Width of the patch.
            - C: Number of spectral channels.
        """
        with rasterio.open(path_raster) as src_S2:
            patch_S2_array = src_S2.read(window=window)
            patch_S2_array = SentinelDataProcessor.reshape_sentinel(
                patch_S2_array, chunk_size=S2_N_CHANNELS
            )
            patch_S2_array = patch_S2_array.transpose(0, 2, 3, 1)
            return patch_S2_array

    @staticmethod
    def read_SAR(path_raster: str, window: Window) -> np.ndarray:
        """
        Reads and processes SAR data from a raster file.

        Parameters:
        - path_raster (str): Path to the raster file.
        - window (rasterio.windows.Window): Window defining the region of interest.

        Returns:
        - np.ndarray: Processed SAR data with shape (T, H, W, C), where:
            - T: Number of time steps.
            - H: Height of the patch.
            - W: Width of the patch.
            - C: Number of SAR channels.
        """
        with rasterio.open(path_raster) as src_S1:
            patch_S1_array = src_S1.read(window=window)
            patch_S1_array = SentinelDataProcessor.reshape_sentinel(
                patch_S1_array, chunk_size=S1_N_CHANNELS
            )  # (T * C, H, W) => (T, C, H, W)
            patch_S1_array = patch_S1_array.transpose(
                0, 2, 3, 1
            )  # (T, C, H, W)=> (T, H, W, C)
            return patch_S1_array

    @staticmethod
    def read_cloud_prob(path_raster: str, window: Window, cloud_band_index: int = 10) -> np.ndarray:
        with rasterio.open(path_raster) as src_S2:
            patch_S2_array = src_S2.read(window=window)
            patch_S2_array = SentinelDataProcessor.reshape_sentinel(
                patch_S2_array, chunk_size=S2_N_CHANNELS
            )
            patch_S2_array = patch_S2_array.transpose(0, 2, 3, 1) # T x H x W x C
            return patch_S2_array[:, :, :, cloud_band_index]

    @staticmethod
    def read_cloud_mask(path_raster: str, window: Window, cloud_band_index: int = 10) -> np.ndarray:
        cloud_prob = SentinelDataProcessor.read_cloud_prob(path_raster=path_raster, window=window, cloud_band_index=cloud_band_index)
        return (cloud_prob != 0).astype(int)

    @staticmethod
    def reshape_sentinel(arr: np.ndarray, chunk_size: int = 10) -> np.ndarray:
        """
        Reshapes a temporally stacked Sentinel array into chunks.

        Parameters:
        - arr (np.ndarray): Input array with temporal data.
        - chunk_size (int, optional): Number of time steps per chunk. Defaults to 10.

        Returns:
        - np.ndarray: Reshaped array with shape (n_chunks, chunk_size, height, width).
        """
        first_dim_size = arr.shape[0] // chunk_size
        return arr.reshape((first_dim_size, chunk_size, *arr.shape[1:]))

    @staticmethod
    def get_img_windows_list(
        img_shape: Tuple[int, int], tile_size: int, overlap: int = 0
    ) -> List[Tuple[int, int, int, int]]:
        """
        Compute patches windows from an image with overlap on all sides.
        Return a list of coordinates for each window. All patches are entirely within the image.

        Parameters:
        - img_shape (Tuple[int, int]): Size of the input image (height, width).
        - tile_size (int): Size of the output patches (patches are squared).
        - overlap (int, optional): Number of pixels to overlap between patches on all sides. Defaults to 0.

        Returns:
        - List[Tuple[int, int, int, int]]: List of coordinates (col_off, row_off, width, height).
        """
        height, width = img_shape
        stride = tile_size - overlap  # Calculate stride based on overlap

        # Calculate the starting points for rows and columns
        col_steps = list(range(0, width - tile_size + 1, stride))
        row_steps = list(range(0, height - tile_size + 1, stride))

        # Ensure the last patch covers the edge of the image
        if (width - tile_size) % stride != 0:
            col_steps.append(width - tile_size)
        if (height - tile_size) % stride != 0:
            row_steps.append(height - tile_size)

        # Generate all combinations of row and column steps
        windows_list = [
            (col, row, tile_size, tile_size)
            for col, row in itertools.product(col_steps, row_steps)
        ]

        return windows_list

    @staticmethod
    def split_raster_into_windows(
        path_raster: Union[str, Path],
        patch_size: int,
        overlap: int = 0,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Splits a raster into windows of a specified patch size.

        Parameters:
        - path_raster (Union[str, Path]): Path to the raster file.
        - patch_size (int): Size of the patches to create.
        - overlap (int, optional): Number of pixels to overlap between patches on all sides. Defaults to 0.

        Returns:
        - List[Tuple[int, int, int, int]]: List of window coordinates.
        """
        with rasterio.open(path_raster) as dataset:
            return SentinelDataProcessor.get_img_windows_list(
                img_shape=(dataset.height, dataset.width),
                tile_size=patch_size,
                overlap=overlap,
            )

    @staticmethod
    def get_datetime(date: str) -> datetime.datetime:
        """
        Converts a date string in 'YYYYMMDD' format to a datetime object.

        Parameters:
        - date (str): Date string in 'YYYYMMDD' format.

        Returns:
        - datetime.datetime: Corresponding datetime object.
        """
        return datetime.datetime.strptime(date, "%Y%m%d")

    @staticmethod
    def cloud_mask_correction(
        input_mask: np.ndarray, threshold: int = 50
    ) -> np.ndarray:
        """
        Corrects cloud masks by identifying and removing inconsistent pixels based on temporal statistics.

        Parameters:
        - input_mask (np.ndarray): Input cloud masks with shape (T, H, W), where:
            - T: Number of time steps.
            - H: Height of the mask.
            - W: Width of the mask.
        - threshold (int, optional): Percentile threshold for identifying inconsistent pixels. Defaults to 60.

        Returns:
        - np.ndarray: Corrected cloud masks with the same shape as input_mask.
        """

        def compute_persistence_mask(array: np.ndarray) -> np.ndarray:
            """
            Binarizes the input array (non-zero values become 1) and sums along the time axis.

            Parameters:
            - array (np.ndarray): Input array with shape (T, H, W).

            Returns:
            - np.ndarray: Summed array with shape (H, W).
            """
            binary_array = (array != 0).astype(int)  # Binarisation
            return binary_array.sum(axis=0)

        cloud_masks = input_mask.copy()
        persistence_mask = compute_persistence_mask(
            cloud_masks
        )  # Stationnarité temporelle
        pixel_threshold = np.percentile(persistence_mask, threshold)
        error_indexes = np.where(persistence_mask > pixel_threshold)
        cloud_masks[:, error_indexes[0], error_indexes[1]] = 0
        return np.stack(cloud_masks, axis=0)

    @staticmethod
    def filter_dates(
        masks: np.ndarray,
        max_fraction_covered: float = 0.05,
    ) -> np.ndarray:
        """
        Filters dates based on cloud and snow coverage.

        Parameters:
        - masks (np.ndarray): Array containing cloud and snow masks.
        - max_cloud_value (int, optional): Maximum allowed cloud value. Defaults to 10.
        - max_snow_value (int, optional): Maximum allowed snow value. Defaults to 10.
        - max_fraction_covered (float, optional): Maximum fraction of the image covered by clouds or snow. Defaults to 0.05.

        Returns:
        - np.ndarray: Indices of the selected dates.
        """
        MAX_CLOUD_VALUE = MAX_SNOW_VALUE = 1
        T, H, W, _ = masks.shape
        select = (masks[:, :, :, 0] <= MAX_SNOW_VALUE) & (
            masks[:, :, :, 1] <= MAX_CLOUD_VALUE
        )
        num_pix = H * W
        threshold = (1 - max_fraction_covered) * num_pix
        selected_days = np.sum(select, axis=(1, 2)) >= threshold
        # Boucle while permettant de diminuer le seuillage de sélection tant que l'on obtient un nombre de dates filtrées trop bas.
        while np.sum(selected_days) <= 0.1 * T and max_fraction_covered < 1.0:
            max_fraction_covered += 0.05
            threshold = (1 - max_fraction_covered) * num_pix
            selected_days = np.sum(select, axis=(1, 2)) >= threshold
        return np.where(selected_days)[0]

    @staticmethod
    def extract_and_transform_S2(
        S2_array: np.ndarray, dates: List[str], S2_channels_selected: List[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts and transforms Sentinel-2 data by filtering cloudy dates and correcting cloud masks.

        Parameters:
        - S2_array (np.ndarray): Input Sentinel-2 data with shape (T, H, W, C), where:
            - T: Number of time steps.
            - H: Height of the patch.
            - W: Width of the patch.
            - C: Number of spectral bands and masks.
        - dates (List[str]): List of dates corresponding to the time steps in S2_array.
        - S2_channels_selected (List[str]): List of bands index to extract.
        Returns:
        - Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - patch_S2_data: Filtered Sentinel-2 data with shape (T_filtered, H, W, 10).
            - dates_filtered: Filtered dates corresponding to the remaining time steps.
            - cloud_masks_corrected: Corrected cloud masks with shape (T_filtered, H, W).
        """
        # Filtrage des dates S2 nuageuses
        masks = S2_array[:, :, :, -2:]
        snow_masks, cloud_masks = masks[:, :, :, 0], masks[:, :, :, 1]

        if S2_channels_selected is None:
            patch_S2_data = S2_array[:, :, :, 0:10]
        else:
            patch_S2_data = S2_array[:, :, :, S2_channels_selected]

        cloud_masks_corrected = SentinelDataProcessor.cloud_mask_correction(cloud_masks)
        index_S2_curated = SentinelDataProcessor.filter_dates(
            np.stack([snow_masks, cloud_masks_corrected], axis=-1)
        )
        return (
            patch_S2_data[index_S2_curated],
            np.asarray(dates)[index_S2_curated],
            cloud_masks_corrected[index_S2_curated],
        )

    @staticmethod
    def get_pairedS1(
        dates_S2: List[str],
        dates_S1_asc: List[str],
        dates_S1_desc: List[str],
    ) -> Tuple[List[str], List[int], str]:
        """
        Pairs Sentinel-1 data with Sentinel-2 data based on the closest dates.

        Parameters:
        - dates_S2 (List[str]): List of Sentinel-2 dates.
        - dates_S1_asc (List[str]): List of Sentinel-1 ascendant dates.
        - dates_S1_desc (List[str]): List of Sentinel-1 descendant dates.

        Returns:
        - Tuple[List[str], List[int], str]: A tuple containing the curated dates, indices, and the orbit type of the radar file.
        """
        deltas_S1_ASC, deltas_S1_DESC = [], []
        index_S1_ASC, index_S1_DESC = [], []
        dates_S1_ASC_curated, dates_S1_DESC_curated = [], []

        for date_S2 in dates_S2:
            deltas_asc = [
                (
                    SentinelDataProcessor.get_datetime(date_S2)
                    - SentinelDataProcessor.get_datetime(date_S1)
                ).days
                for date_S1 in dates_S1_asc
            ]
            deltas_desc = [
                (
                    SentinelDataProcessor.get_datetime(date_S2)
                    - SentinelDataProcessor.get_datetime(date_S1)
                ).days
                for date_S1 in dates_S1_desc
            ]
            deltas_S1_ASC.append(np.min(np.abs(deltas_asc)))
            deltas_S1_DESC.append(np.min(np.abs(deltas_desc)))

            idx_min_asc = np.argmin(np.abs(deltas_asc))
            index_S1_ASC.append(idx_min_asc)
            dates_S1_ASC_curated.append(dates_S1_asc[idx_min_asc])

            idx_min_desc = np.argmin(np.abs(deltas_desc))
            index_S1_DESC.append(idx_min_desc)
            dates_S1_DESC_curated.append(dates_S1_desc[idx_min_desc])

        if sum(deltas_S1_ASC) <= sum(deltas_S1_DESC):
            return dates_S1_ASC_curated, index_S1_ASC, "ASC"
        else:
            return dates_S1_DESC_curated, index_S1_DESC, "DESC"

    @staticmethod
    def rescale(img: np.ndarray, old_min: float, old_max: float) -> np.ndarray:
        """
        Rescales an image from the range [old_min, old_max] to the range [0, 1].

        Parameters:
        - img (np.ndarray): Input image to rescale.
        - old_min (float): Minimum value of the original range.
        - old_max (float): Maximum value of the original range.

        Returns:
        - np.ndarray: Rescaled image in the range [0, 1].
        """
        old_range = old_max - old_min  # Calculate the range of the original values
        img = (img - old_min) / old_range  # Rescale to [0, 1]
        return img

    @staticmethod
    def process_MS(
        img: np.ndarray, method: Literal["default", "resnet"] = "default"
    ) -> np.ndarray:
        """
        Processes multispectral (MS) data by clipping and rescaling intensities.

        Parameters:
        - img (np.ndarray): Input multispectral image.
        - method (str): Processing method. Options: 'default' or 'resnet'. Defaults to 'default'.

        Returns:
        - np.ndarray: Processed image.
        """
        if method == "default":
            intensity_min, intensity_max = 0, 10000

            img = np.clip(img, intensity_min, intensity_max)
            # Rescale to [0, 1] while preserving global intensities
            img = SentinelDataProcessor.rescale(img, intensity_min, intensity_max)
        elif method == "resnet":
            intensity_min, intensity_max = 0, 10000
            img = np.clip(img, intensity_min, intensity_max)
            # Rescale to [0, 5] for ResNet compatibility
            img /= 2000

        img = np.nan_to_num(img)  # Replace NaN values with 0
        return img

    @staticmethod
    def process_SAR(
        img: np.ndarray, method: Literal["default", "resnet"] = "default"
    ) -> np.ndarray:
        """
        Processes SAR data by clipping and rescaling dB values.

        Parameters:
        - img (np.ndarray): Input SAR image.
        - method (str): Processing method. Options: 'default' or 'resnet'. Defaults to 'default'.

        Returns:
        - np.ndarray: Processed image.
        """
        if method == "default":
            dB_min, dB_max = -25, 0
            img = np.clip(img, dB_min, dB_max)
            img = SentinelDataProcessor.rescale(
                img, dB_min, dB_max
            )  # Rescale to [0, 1] while preserving global intensities
        elif method == "resnet":
            dB_min, dB_max = [-25.0, -32.5], [0, 0]
            img = np.concatenate(
                [
                    (
                        2
                        * (np.clip(img[0], dB_min[0], dB_max[0]) - dB_min[0])
                        / (dB_max[0] - dB_min[0])
                    )[
                        None, ...
                    ],  # Channel 1
                    (
                        2
                        * (np.clip(img[1], dB_min[1], dB_max[1]) - dB_min[1])
                        / (dB_max[1] - dB_min[1])
                    )[
                        None, ...
                    ],  # Channel 2
                ],
                axis=0,
            )  # Clip and rescale each channel independently

        img = np.nan_to_num(img)  # Replace NaN values with 0
        return img
