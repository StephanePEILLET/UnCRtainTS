import datetime as dt
import json
import sys
from pathlib import Path
from typing import Dict  # noqa: F401
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import h5py
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from rasterio import Affine
from torch.utils.data import Dataset

from data.constants.circa_constants import MGRSC_SPLITS

# Set multiprocessing sharing strategy
torch.multiprocessing.set_sharing_strategy("file_system")

# Constants
SEED: int = 42
IMAGE_SIZE: Tuple[int] = (256, 256)  # Default image size for the dataset in the HDF5 files.

DateArray = np.ndarray[dt.date]
TensorDict = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
SampleDict = Dict[str, Union[NDArray, Dict[str, NDArray], List[str]]]
PhaseType = Literal["train", "val", "test", "train+val", "all"]
ChannelType = Literal["all", "bgr-nir"]
SarPairingType = Literal["asc+desc", "asc", "desc", "mix_closest"]


class CIRCA_from_HDF5(Dataset):
    """
    A PyTorch Dataset class for loading CIRCA data from HDF5 files.

    This dataset handles Sentinel-1 (SAR) and Sentinel-2 (MSI) time series data
    with cloud mask information, supporting different data splits and channel configurations.

    Attributes:
        phase (PhaseType): Data phase/split being used
        shuffle (bool): Whether to shuffle the data
        use_sar Union[bool| SarPairingType]: Whether to include Sentinel-1 data
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
        load_transforms: Optional[str] = None, # permet de charger des transformations de géoréférencement d'un patch
        shuffle: bool = False,
        use_sar: Union[bool | SarPairingType] = "mix_closest",
        channels: ChannelType = "all",
        image_size: Tuple[int] = IMAGE_SIZE,
    ) -> None:
        """
        Initialize the CIRCA dataset from HDF5 file.

        Args:
            phase: Data phase ('train', 'val', 'test', 'train+val', or 'all')
            hdf5_file: Path to the HDF5 file containing the data
            shuffle: Whether to shuffle the dataset
            use_sar: Whether to include Sentinel-1 SAR data ('asc+desc', 'asc', 'desc', 'mix_closest', or 'none')
            if asc use only ascending SAR data, if desc use only descending SAR data,
            if mix_closest use the closest SAR pair to the Sentinel-2 date, if none do not use SAR data.
            channels: Which channels to include ('all' or 'bgr-nir')

        Raises:
            FileNotFoundError: If HDF5 file doesn't exist
            ValueError: If invalid channels or phase are specified
        """
        self.phase: PhaseType = phase
        self.image_size: Tuple[int] = image_size
        self.shuffle: bool = shuffle
        self.use_sar: Union[bool | SarPairingType] = use_sar
        self.rng: np.random.Generator = np.random.default_rng(seed=SEED)
        self.hdf5_file: h5py.File
        self.patches_dataset: pd.DataFrame
        self.hdf5_file, self.patches_dataset = self.setup_hdf5_file(hdf5_file)
        if load_transforms is not None:
            self.load_transforms_from_file(load_transforms)

        # Channel configuration
        self.num_channels: int
        self.c_index_rgb: torch.Tensor
        self.c_index_nir: torch.Tensor
        self.s2_channels: List[int]
        self.num_channels, self.c_index_rgb, self.c_index_nir, self.s2_channels = self.setup_channels(channels)

    def __len__(self) -> int:
        """Return the number of patches in the dataset."""
        return len(self.patches_dataset)

    def load_transforms_from_file(self, path_file: str) -> None:
        """
        Load transformation settings from a JSON file containing info of the patches_dataset object.
        Args:
            path_file: Path to the JSON file with transformation settings

        Raises:
            FileNotFoundError: If the JSON file doesn't exist
        """
        import ast

        if Path(path_file).exists():
            df_transforms = pd.read_json(path_file)
            df_transforms.window = df_transforms.window.apply(lambda x: "_".join(map(str, x)))
            self.patches_dataset["patch"] = [f"patches_{row.mgrs25}_window_{row.window}" for i, row in self.patches_dataset.iterrows()]
            self.patches_dataset = pd.merge(self.patches_dataset, df_transforms)
            print(f"Loaded transforms from {path_file}.")
        else:
            raise FileNotFoundError(f"Transform file {path_file} does not exist.")

    def str2date(self, date_string: str) -> dt.date:
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
            if self.use_sar == "asc+desc":
                num_channels += 8
            elif self.use_sar == "asc" or self.use_sar == "desc" or self.use_sar == "mix_closest":
                num_channels += 4
            else:
                raise ValueError(f"SAR pairing {self.use_sar} not recognized. Use 'asc+desc', 'asc', 'desc' or 'mix_closest'.")
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
        data = {
            "info": sample["info"],
            "S2": {
                "S2": torch.from_numpy(sample["S2"]["S2"].astype(np.float32)),
                "S2_dates": np.array([self.str2date(date) for date in sample["S2"]["S2_dates"]]),
                "cloud_mask": torch.from_numpy(np.expand_dims(sample["S2"]["cloud_mask"], axis=1).astype(np.float32)),
                "cloud_prob": torch.from_numpy(np.expand_dims(sample["S2"]["cloud_prob"], axis=1).astype(np.float32)),
            },
            "idx_cloudy_frames": torch.from_numpy(sample["idx_cloudy_frames"]),
            "idx_good_frames": torch.from_numpy(sample["idx_good_frames"]),
            "idx_impaired_frames": torch.from_numpy(sample["idx_impaired_frames"]),
            "valid_obs": torch.from_numpy(sample["valid_obs"]),
        }
        if self.use_sar:
            if self.use_sar == "asc+desc":
                data.update(
                    {
                        "S1": {
                            "S1_asc": torch.from_numpy(sample["S1"]["S1_asc"].astype(np.float32)),
                            "S1_dates_asc": np.array([self.str2date(date) for date in sample["S1"]["S1_dates_asc"]]),
                            "S1_desc": torch.from_numpy(sample["S1"]["S1_desc"].astype(np.float32)),
                            "S1_dates_desc": np.array([self.str2date(date) for date in sample["S1"]["S1_dates_desc"]]),
                        }
                    }
                )
            else:
                data.update(
                    {
                        "S1": {
                            "S1": torch.from_numpy(sample["S1"]["S1"].astype(np.float32)),
                            "S1_dates": np.array([self.str2date(date) for date in sample["S1"]["S1_dates"]]),
                        }
                    }
                )
        if "idx_syn_aleatoire" in sample:
            data["idx_syn_aleatoire"] = torch.from_numpy(sample["idx_syn_aleatoire"])

        if "idx_syn_consecutif" in sample:
            data["idx_syn_consecutif"] = torch.from_numpy(sample["idx_syn_consecutif"])

        return data

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
            "info": {
                "mgrs": row.mgrs,
                "mgrs25": row.mgrs25,
                "window": row.window,
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
        if self.use_sar:
            sample.update(self.pairing_and_reconstruct_s1_to_s2_shape(method=self.use_sar, patch=patch, s2_dates=sample["S2"]["S2_dates"]))

        if patch.get("idx_syn_aleatoire", False):
            sample["idx_syn_aleatoire"] = patch["idx_syn_aleatoire"][:]

        if patch.get("idx_syn_consecutif", False):
            sample["idx_syn_consecutif"] = patch["idx_syn_consecutif"][:]

        if len(self.s2_channels) != sample["S2"]["S2"].shape[1]:
            sample["S2"]["S2"] = sample["S2"]["S2"][:, self.s2_channels, :, :]

        # if "meta" in row:
        #     sample["info"]["meta"] = row["meta"]
        return self.format_item(sample)

    def pairing_and_reconstruct_s1_to_s2_shape(self, method: SarPairingType, patch: dict, s2_dates) -> torch.Tensor:
        dict_pairing = json.loads(patch["S1/S2_S1_pairing"][()].decode("utf-8"))
        if method == "asc":
            s1_asc_collected = patch["S1/S1_asc"][:]
            s1_dates_asc_collected = self.decode_dates(patch["S1/S1_dates_asc"][:])
            s1_asc_reshape, s1_asc_dates = [], []
            true_index, old_index = [], []
            for s2_date in s2_dates:
                s1_date, _, x = dict_pairing["asc"][s2_date]
                assert s1_date in s1_dates_asc_collected
                s1_asc_reshape.append(s1_asc_collected[s1_dates_asc_collected.index(s1_date)])
                true_index.append(s1_dates_asc_collected.index(s1_date))
                old_index.append(x)
                s1_asc_dates.append(s1_date)
            return {
                "S1": {
                    "S1": np.stack(s1_asc_reshape, axis=0),
                    "S1_dates": s1_asc_dates,
                }
            }
        elif method == "desc":
            s1_desc_collected = patch["S1/S1_desc"][:]
            s1_dates_desc_collected = self.decode_dates(patch["S1/S1_dates_desc"][:])
            s1_desc_reshape, s1_desc_dates = [], []
            for s2_date in s2_dates:
                s1_date, _, _ = dict_pairing["desc"][s2_date]
                assert s1_date in s1_dates_desc_collected
                s1_desc_reshape.append(s1_desc_collected[s1_dates_desc_collected.index(s1_date)])
                s1_desc_dates.append(s1_date)
            s1_desc_reshape = np.stack(s1_desc_reshape, axis=0)
            return {
                "S1": {
                    "S1": s1_desc_reshape,
                    "S1_dates": s1_desc_dates,
                }
            }
        elif method == "asc+desc":
            s1_asc_collected = patch["S1/S1_asc"][:]
            s1_dates_asc_collected = self.decode_dates(patch["S1/S1_dates_asc"][:])
            s1_desc_collected = patch["S1/S1_desc"][:]
            s1_dates_desc_collected = self.decode_dates(patch["S1/S1_dates_desc"][:])
            s1_asc_reshape, s1_desc_reshape = [], []
            s1_asc_dates, s1_desc_dates = [], []
            for s2_date in s2_dates:
                # ASC pairing
                s1_date_asc, _, _ = dict_pairing["asc"][s2_date]
                assert s1_date_asc in s1_dates_asc_collected
                s1_asc_reshape.append(s1_asc_collected[s1_dates_asc_collected.index(s1_date_asc)])
                s1_asc_dates.append(s1_date_asc)
                # DESC pairing
                s1_date_desc, _, _ = dict_pairing["desc"][s2_date]
                assert s1_date_desc in s1_dates_desc_collected
                s1_desc_reshape.append(s1_desc_collected[s1_dates_desc_collected.index(s1_date_desc)])
                s1_desc_dates.append(s1_date_desc)
            return {
                "S1": {
                    "S1_asc": np.stack(s1_asc_reshape, axis=0),
                    "S1_desc": np.stack(s1_desc_reshape, axis=0),
                    "S1_dates_asc": s1_asc_dates,
                    "S1_dates_desc": s1_desc_dates,
                }
            }
        elif method == "mix_closest":
            s1_asc_collected = patch["S1/S1_asc"][:]
            s1_dates_asc_collected = self.decode_dates(patch["S1/S1_dates_asc"][:])
            s1_desc_collected = patch["S1/S1_desc"][:]
            s1_dates_desc_collected = self.decode_dates(patch["S1/S1_dates_desc"][:])
            s1_reshape, s1_dates = [], []
            for s2_date in s2_dates:
                s1_date, orbit_type, _ = dict_pairing["mix_closest"][s2_date]
                if orbit_type == "ASC":
                    assert s1_date in s1_dates_asc_collected
                    s1_reshape.append(s1_asc_collected[s1_dates_asc_collected.index(s1_date)])
                    s1_dates.append(s1_date)
                elif orbit_type == "DESC":
                    assert s1_date in s1_dates_desc_collected
                    s1_reshape.append(s1_desc_collected[s1_dates_desc_collected.index(s1_date)])
                    s1_dates.append(s1_date)
            return {
                "S1": {
                    "S1": np.stack(s1_reshape, axis=0),
                    "S1_dates": s1_dates,
                }
            }
        else:
            raise ValueError(f"SAR pairing {method} not recognized. Use 'asc+desc', 'asc', 'desc' or 'mix_closest'.")

    def __getitem__(self, item: int) -> TensorDict:
        """
        Get an item from the dataset with proper channel selection.

        Args:
            item: Index of the item to retrieve

        Returns:
            Dictionary containing the sample data with selected channels
        """
        return self.etl_item(item=item)


# if __name__ == "__main__":
#     # Example usage
#     path_dataset_circa = Path("/DATA_10TB/data_rpg/circa/hdf5")
#     # hdf5_file = path_dataset_circa / "new_circa_ligth.hdf5"
#     hdf5_file = path_dataset_circa / "CIRCA_CR_merged.hdf5"
#     # Import data from HDF5 file
#     dataset = CIRCA_from_HDF5(
#         hdf5_file=hdf5_file,
#         phase="all",
#         shuffle=False,
#         channels="all",
#         use_sar="asc+desc",
#         load_transforms="./data/CIRCA_patches_datasets_with_transforms.json",
#     )
#     # Get a sample
#     sample = next(iter(dataset))
#     print(sample.keys())
