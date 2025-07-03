import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
# s2cloudless: see https://github.com/sentinel-hub/sentinel2-cloud-detector
import os
from typing import Dict, List, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray
from pyproj.datadir import get_data_dir

# Vérifie et configure le chemin PROJ_DATA
try:
    from pyproj import Proj

    Proj(init="epsg:4326")
except Exception:
    proj_data = os.path.join(os.path.dirname(get_data_dir()), "proj.db")
    if os.path.exists(proj_data):
        os.environ["PROJ_LIB"] = os.path.dirname(proj_data)
    else:
        # Essaye de trouver le chemin alternatif
        import pyproj

        proj_data = os.path.join(os.path.dirname(pyproj.__file__), "proj_dir", "share", "proj")
        if os.path.exists(proj_data):
            os.environ["PROJ_LIB"] = proj_data

# Maintenant importer s2cloudless
from s2cloudless import S2PixelCloudDetector

from data.circa_dataloader import CIRCA_from_HDF5
from data.utils.process_functions import (
    S1_LAUNCH,
    get_cloud_map,
    process_MS,
    process_SAR,
)
from data.utils.sampling_functions import sampler

SEN12MSCRTS_SEQ_LENGTH: int = 30  # Length of the Sentinel time series
CLEAR_THRESHOLD: float = 1e-3  # Threshold for considering a scene as cloud-free

# Type aliases for better readability
CloudMaskType = Optional[Literal["cloud_cloudshadow_mask", "s2cloudless_map", "s2cloudless_mask"]]
SampleType = Literal["generic", "cloudy_cloudfree"]
RescaleMethod = Literal["default", "minmax", "standard"]
SamplerType = Literal["fixed", "random", "stratified"]


class UnCRtainTS_from_hdf5(CIRCA_from_HDF5):
    """
    A dataset class for loading and processing Sentinel-1 and Sentinel-2 time series data from HDF5 files,
    with cloud masking and temporal sampling capabilities.

    Inherits from CIRCA_from_HDF5 and extends it with cloud-aware sampling functionality.

    Attributes:
        modalities (List[str]): Supported satellite modalities (S1 and S2)
        compute_cloud_mask (bool): Flag to compute cloud masks on-the-fly
        time_points (range): Indices for the time series
        cloud_masks (CloudMaskType): Type of cloud mask being used
        sample_type (SampleType): Sampling strategy type
        sampling (SamplerType): Sampler method being used
        n_input_t (int): Number of input time samples
        cloud_detector (Optional[S2PixelCloudDetector]): Cloud detector instance
        method (RescaleMethod): Data rescaling method
        min_cov (float): Minimum cloud coverage threshold
        max_cov (float): Maximum cloud coverage threshold
    """

    def __init__(
        self,
        phase: str = "all",
        hdf5_file: Optional[Union[str, Path]] = None,
        shuffle: bool = False,
        include_S1: bool = True,
        channels: Optional[str] = "all",
        cloud_masks: CloudMaskType = "s2cloudless_mask",
        sample_type: SampleType = "cloudy_cloudfree",
        sampler: SamplerType = "fixed",
        n_input_samples: int = 3,
        rescale_method: RescaleMethod = "default",
        min_cov: float = 0.0,
        max_cov: float = 1.0,
        compute_cloud_mask: bool = False,
    ) -> None:
        """
        Initialize the dataset with configuration parameters.

        Args:
            phase: Data phase ('train', 'val', 'test', or 'all')
            hdf5_file: Path to HDF5 file containing the data
            shuffle: Whether to shuffle the data
            include_S1: Whether to include Sentinel-1 SAR data
            channels: Which channels to include ('all' or specific subset)
            cloud_masks: Type of cloud mask to use
            sample_type: Sampling strategy ('generic' or 'cloudy_cloudfree')
            sampler: Sampler type ('fixed' or other implemented methods)
            n_input_samples: Number of input samples to use
            rescale_method: Method for rescaling data
            min_cov: Minimum cloud coverage threshold (0-1)
            max_cov: Maximum cloud coverage threshold (0-1)
            compute_cloud_mask: Whether to compute cloud masks on the fly

        Raises:
            AssertionError: If invalid parameters are provided
        """
        super().__init__(
            phase=phase,
            hdf5_file=hdf5_file,
            shuffle=shuffle,
            include_S1=include_S1,
            channels=channels,
        )

        # Validate inputs
        assert self.__len__() > 0, "No data found in the HDF5 file"
        assert sample_type in ["generic", "cloudy_cloudfree"], "Invalid sample type!"
        assert cloud_masks in [None, "cloud_cloudshadow_mask", "s2cloudless_map", "s2cloudless_mask"], "Unknown cloud mask type!"

        self.modalities: List[str] = ["S1", "S2"]
        self.compute_cloud_mask: bool = compute_cloud_mask
        self.time_points: range = range(SEN12MSCRTS_SEQ_LENGTH)
        self.cloud_masks: CloudMaskType = cloud_masks
        self.sample_type: SampleType = sample_type if self.cloud_masks is not None else "generic"
        self.sampling: SamplerType = sampler
        self.n_input_t: int = n_input_samples

        # Initialize cloud detector if needed
        self.cloud_detector: Optional[S2PixelCloudDetector] = None
        if self.cloud_masks in ["s2cloudless_map", "s2cloudless_mask"]:
            self.cloud_detector = S2PixelCloudDetector(threshold=0.4, all_bands=True, average_over=4, dilation_size=2)

        self.method: RescaleMethod = rescale_method
        self.min_cov: float = min_cov
        self.max_cov: float = max_cov

    def incr_epoch_count(self) -> None:
        """Increment the epoch counter for tracking training progress."""
        self.epoch_count += 1

    def get_sample(self, pdx: int) -> Dict[str, Union[NDArray, List[NDArray], List[int]]]:
        """
        Get a sample by index (wrapper around __getitem__).

        Args:
            pdx: Index of the sample to retrieve

        Returns:
            Dictionary containing the sample data
        """
        return self.__getitem__(pdx)

    def __getitem__(self, pdx: int) -> Dict[str, Union[NDArray, List[NDArray], Dict[str, Union[NDArray, List[NDArray], List[int]]]]]:
        """
        Get the time series of one patch with optional cloud processing.

        Args:
            pdx: Index of the patch to retrieve

        Returns:
            Dictionary containing processed satellite data, masks, and metadata.
            Structure depends on sample_type:
            - For 'generic': returns full time series
            - For 'cloudy_cloudfree': returns input/target split based on cloud coverage
        """
        patch_data: Dict = self.etl_item(item=pdx)

        # Extract SAR data and dates
        s1: NDArray = patch_data["S1"]["S1"]  # T * C * H * W
        dates_S1: List = patch_data["S1"]["S1_dates"]
        s1_td: List[int] = [(d - S1_LAUNCH).days for d in dates_S1]

        # Extract optical data and dates
        s2: NDArray = patch_data["S2"]["S2"]  # T * C * H * W
        dates_S2: List = patch_data["S2"]["S2_dates"]
        s2_td: List[int] = [(d - S1_LAUNCH).days for d in dates_S2]

        # Process cloud masks
        masks: NDArray
        if self.compute_cloud_mask:
            masks = np.asarray([get_cloud_map(img, self.cloud_masks, self.cloud_detector) for img in s2])
        else:
            masks = patch_data["S2"]["cloud_prob"][patch_data["idx_good_frames"]]

        coverage: List[float] = masks.mean(dim=(1, 2, 3)).tolist()

        # Process and normalize data
        s1 = np.asarray([process_SAR(img, self.method) for img in s1])
        s2 = np.asarray([process_MS(img, self.method) for img in s2])

        if self.sample_type == "generic":
            return {
                "S1": s1,
                "S2": [process_MS(img, self.method) for img in s2],
                "masks": masks,
                "coverage": coverage,
                "S1 TD": s1_td,
                "S2 TD": s2_td,
            }
        elif self.sample_type == "cloudy_cloudfree":
            # Sample cloud-free and cloudy dates
            inputs_idx: NDArray
            cloudless_idx: NDArray
            coverage_match: bool
            t_windows: int = len(coverage)
            inputs_idx, cloudless_idx, coverage_match = sampler(
                self.sampling,
                t_windows,
                self.n_input_t,
                self.min_cov,
                self.max_cov,
                coverage,
                clear_tresh=CLEAR_THRESHOLD,
            )

            # Prepare input and target data
            input_s1: NDArray = s1[inputs_idx]
            input_s2: NDArray = s2[inputs_idx]
            input_masks: NDArray = masks[inputs_idx]
            target_s1: NDArray = s1[cloudless_idx]
            target_s2: NDArray = s2[cloudless_idx]
            target_mask: NDArray = masks[cloudless_idx]

            return {
                "input": {
                    "S1": list(input_s1),
                    "S2": input_s2,
                    "masks": list(input_masks),
                    "coverage": input_masks.mean(dim=(1, 2, 3)).tolist(),
                    "S1 TD": [s1_td[idx] for idx in inputs_idx],
                    "S2 TD": [s2_td[idx] for idx in inputs_idx],
                    "idx": inputs_idx,
                },
                "target": {
                    "S1": [target_s1],
                    "S2": target_s2,
                    "masks": [target_mask],
                    "coverage": target_mask.mean(dim=(1, 2)).tolist(),
                    "S1 TD": [s1_td[cloudless_idx]],
                    "S2 TD": [s2_td[cloudless_idx]],
                    "idx": cloudless_idx,
                },
                "coverage bin": coverage_match,
            }


if __name__ == "__main__":
    # Example usage
    path_dataset_circa = Path("/home/SPeillet/Downloads/data")
    hdf5_file = path_dataset_circa / "circa_cloud_removal.hdf5"

    # Import data from HDF5 file
    dataset = UnCRtainTS_from_hdf5(
        hdf5_file=hdf5_file,
        phase="all",
        shuffle=False,
        channels="all",
    )
    sample = next(iter(dataset))
    print(sample.keys())
