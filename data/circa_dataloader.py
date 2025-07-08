import datetime as dt
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from data.constants.circa_constants import MGRSC_SPLITS

# Set multiprocessing sharing strategy
torch.multiprocessing.set_sharing_strategy("file_system")

# Constants
MAX_SEQ_LENGTH: int = 30
MIN_SEQ_LENGTH: int = 5
SEED: int = 42

DateArray = NDArray[dt.date]
TensorDict = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
SampleDict = Dict[str, Union[NDArray, Dict[str, NDArray], List[str]]]
PhaseType = Literal["train", "val", "test", "train+val", "all"]
ChannelType = Literal["all", "bgr-nir"]


def str2date(date_string: str) -> dt.date:
    """
    Convert a date string in format 'YYYYMMDD' to datetime object.

    Args:
        date_string: Date string in format 'YYYYMMDD'

    Returns:
        Corresponding datetime.date object

    Example:
        >>> str2date("20200101")
        datetime.date(2020, 1, 1)
    """
    return dt.datetime.strptime(date_string, "%Y%m%d")


class CIRCA_from_HDF5(Dataset):
    """
    A PyTorch Dataset class for loading CIRCA data from HDF5 files.

    This dataset handles Sentinel-1 (SAR) and Sentinel-2 (MSI) time series data
    with cloud mask information, supporting different data splits and channel configurations.

    Attributes:
        phase (PhaseType): Data phase/split being used
        shuffle (bool): Whether to shuffle the data
        use_sar (bool): Whether to include Sentinel-1 data
        rng (np.random.Generator): Random number generator
        hdf5_file (h5py.File): HDF5 file handle
        patches_dataset (pd.DataFrame): DataFrame containing patch metadata
        num_channels (int): Number of channels in the data
        c_index_rgb (torch.Tensor): Indices of RGB channels
        c_index_nir (torch.Tensor): Indices of NIR channel
        s2_channels (List[int]): List of Sentinel-2 channel indices to use
    """

    def __init__(
        self,
        phase: PhaseType = "all",
        hdf5_file: Optional[Union[str, Path]] = None,
        shuffle: bool = False,
        use_sar: bool = True,
        channels: ChannelType = "all",
    ) -> None:
        """
        Initialize the CIRCA dataset from HDF5 file.

        Args:
            phase: Data phase ('train', 'val', 'test', 'train+val', or 'all')
            hdf5_file: Path to the HDF5 file containing the data
            shuffle: Whether to shuffle the dataset
            use_sar: Whether to include Sentinel-1 SAR data
            channels: Which channels to include ('all' or 'bgr-nir')

        Raises:
            FileNotFoundError: If HDF5 file doesn't exist
            ValueError: If invalid channels or phase are specified
        """
        self.phase: PhaseType = phase
        self.shuffle: bool = shuffle
        self.use_sar: bool = use_sar
        self.rng: np.random.Generator = np.random.default_rng(seed=SEED)
        self.hdf5_file: h5py.File
        self.patches_dataset: pd.DataFrame
        self.hdf5_file, self.patches_dataset = self.setup_hdf5_file(hdf5_file)

        # Channel configuration
        self.num_channels: int
        self.c_index_rgb: torch.Tensor
        self.c_index_nir: torch.Tensor
        self.s2_channels: List[int]
        self.num_channels, self.c_index_rgb, self.c_index_nir, self.s2_channels = self.setup_channels(channels)

    def __len__(self) -> int:
        """Return the number of patches in the dataset."""
        return len(self.patches_dataset)

    def setup_hdf5_file(self, path_file: Optional[Union[str, Path]]) -> Tuple[h5py.File, pd.DataFrame]:
        """
        Initialize the HDF5 file and prepare the patches dataset.

        Args:
            path_file: Path to the HDF5 file

        Returns:
            Tuple containing the HDF5 file handle and patches DataFrame

        Raises:
            FileNotFoundError: If the HDF5 file doesn't exist
        """
        if Path(path_file).exists():
            f = h5py.File(path_file, "r", libver="latest", swmr=True)
            patches_dataset = self.list_files_in_hdf5(f)
            patches_dataset = self.splits_samples(patches_dataset, self.phase)
            return f, patches_dataset
        raise FileNotFoundError(f"HDF5 file {path_file} does not exist.")

    def setup_channels(self, channels: ChannelType) -> Tuple[int, torch.Tensor, torch.Tensor, List[int]]:
        """
        Configure channel settings based on the specified channel mode.

        Args:
            channels: Channel configuration ('all' or 'bgr-nir')

        Returns:
            Tuple containing:
            - Number of channels
            - RGB channel indices tensor
            - NIR channel index tensor
            - List of Sentinel-2 channel indices

        Raises:
            ValueError: If invalid channel configuration is specified
        """
        if channels == "all":
            num_channels = 10
            c_index_rgb = torch.tensor([2, 1, 0], dtype=torch.long)
            c_index_nir = torch.tensor([6], dtype=torch.long)
            s2_channels = list(range(10))
        elif channels == "bgr-nir":
            num_channels = 4
            c_index_rgb = torch.tensor([2, 1, 0], dtype=torch.long)
            c_index_nir = torch.tensor([6], dtype=torch.long)
            s2_channels = [0, 1, 2, 6]
        else:
            raise ValueError(f"Channels {channels} not recognized. Use 'all' or 'bgr-nir'.")

        if self.use_sar:
            num_channels += 4

        return num_channels, c_index_rgb, c_index_nir, s2_channels

    def list_files_in_hdf5(self, hdf5_file: h5py.File) -> pd.DataFrame:
        """
        List all files in the HDF5 file and return their details in a DataFrame.

        Args:
            hdf5_file: Opened HDF5 file handle

        Returns:
            DataFrame containing MGRS tiles, sub-tiles, and window information
        """
        patches_dataset = pd.DataFrame(columns=["mgrs", "mgrs25", "window"])

        for mgrs_level, mgrsc_list in hdf5_file.items():
            for mgrsc_level, mgrsc_data in mgrsc_list.items():
                for window in mgrsc_data.keys():
                    data = {
                        "mgrs": [mgrs_level],
                        "mgrs25": [mgrsc_level],
                        "window": [window],
                    }
                    patches_dataset = pd.concat([patches_dataset, pd.DataFrame(data)], ignore_index=True)

        if self.shuffle:
            patches_dataset = patches_dataset.sample(frac=1).reset_index(drop=True)

        return patches_dataset

    def splits_samples(self, patches_dataset: pd.DataFrame, phase: PhaseType) -> pd.DataFrame:
        """
        Filter samples based on the specified data split.

        Args:
            patches_dataset: DataFrame containing all patches
            phase: Data phase to filter for

        Returns:
            Filtered DataFrame containing only patches for the specified phase

        Raises:
            ValueError: If invalid phase is specified
        """
        if phase is None:
            raise ValueError("Phase is not defined. Use 'train', 'val', 'train+val', or 'all'.")

        if phase in MGRSC_SPLITS:
            patches_dataset = patches_dataset[patches_dataset["mgrs25"].isin(MGRSC_SPLITS[phase])]
        elif phase == "train+val":
            patches_dataset = patches_dataset[patches_dataset["mgrs25"].isin(MGRSC_SPLITS["train"] + MGRSC_SPLITS["val"])]
        elif phase != "all":
            raise ValueError(f"Phase {phase} not recognized. Use 'train', 'val', 'train+val', or 'all'.")

        return patches_dataset.reset_index(drop=True)

    def decode_dates(self, dates: NDArray[np.bytes_]) -> List[str]:
        """
        Decode byte strings in date array to UTF-8 strings.

        Args:
            dates: Array of date byte strings

        Returns:
            List of decoded date strings
        """
        return [el.decode("utf-8") for el in dates]

    def format_item(self, sample: SampleDict) -> TensorDict:
        """
        Format a sample dictionary into the correct tensor format for the model.

        Args:
            sample: Raw sample dictionary from HDF5

        Returns:
            Dictionary of tensors with properly formatted data types and shapes
        """
        return {
            "S1": {
                "S1": torch.from_numpy(sample["S1"]["S1"].astype(np.float32)),
                "S1_dates": np.array([str2date(date) for date in sample["S1"]["S1_dates"]]),
            },
            "S2": {
                "S2": torch.from_numpy(sample["S2"]["S2"].astype(np.float32)),
                "S2_dates": np.array([str2date(date) for date in sample["S2"]["S2_dates"]]),
                "cloud_mask": torch.from_numpy(np.expand_dims(sample["S2"]["cloud_mask"], axis=1).astype(np.float32)),
                "cloud_prob": torch.from_numpy(np.expand_dims(sample["S2"]["cloud_prob"], axis=1).astype(np.float32)),
            },
            "idx_cloudy_frames": torch.from_numpy(sample["idx_cloudy_frames"]),
            "idx_good_frames": torch.from_numpy(sample["idx_good_frames"]),
            "idx_impaired_frames": torch.from_numpy(sample["idx_impaired_frames"]),
            "valid_obs": torch.from_numpy(sample["valid_obs"]),
        }

    def etl_item(self, item: int) -> TensorDict:
        """
        Extract, transform, and load a single item from the dataset.

        Args:
            item: Index of the item to retrieve

        Returns:
            Formatted sample dictionary with tensor data
        """
        row = self.patches_dataset.iloc[item]
        patch = self.hdf5_file[f"{row.mgrs}/{row.mgrs25}/{row.window}"]

        sample: SampleDict = {
            "S1": {
                "S1": patch["S1/S1"][:],  # T * C * H * W
                "S1_dates": self.decode_dates(patch["S1/S1_dates"][:]),
            },
            "S2": {
                "S2": patch["S2/S2"][:],  # T * C * H * W
                "S2_dates": self.decode_dates(patch["S2/S2_dates"][:]),
                "cloud_mask": patch["S2/cloud_mask"][:],
                "cloud_prob": patch["S2/cloud_prob"][:],
            },
            "idx_cloudy_frames": patch["idx_cloudy_frames"][:],
            "idx_good_frames": patch["idx_good_frames"][:],
            "idx_impaired_frames": patch["idx_impaired_frames"][:],
            "valid_obs": patch["valid_obs"][:],
        }

        if self.num_channels != sample["S2"]["S2"].shape[1]:
            sample["S2"]["S2"] = sample["S2"]["S2"][:, self.s2_channels, :, :]

        return self.format_item(sample)

    def __getitem__(self, item: int) -> TensorDict:
        """
        Get an item from the dataset with proper channel selection.

        Args:
            item: Index of the item to retrieve

        Returns:
            Dictionary containing the sample data with selected channels
        """
        patch_data = self.etl_item(item=item)
        if self.num_channels != patch_data["S2"]["S2"].shape[1]:
            patch_data["S2"]["S2"] = patch_data["S2"]["S2"][:, self.s2_channels, :, :]
        return patch_data


if __name__ == "__main__":
    # Example usage
    path_dataset_circa = Path("/home/SPeillet/Downloads/data")
    hdf5_file = path_dataset_circa / "circa_cloud_removal.hdf5"

    # Import data from HDF5 file
    dataset = CIRCA_from_HDF5(
        hdf5_file=hdf5_file,
        phase="all",
        shuffle=False,
        channels="all",
    )

    sample = next(iter(dataset))
    print(sample.keys())
