import ast
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import rasterio

from rasterio.windows import Window
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from dataloader_CIRCA_old.tools.data_processor import SentinelDataProcessor
from dataloader_CIRCA_old.datasets import CircaPatchDataSet
import dataloader_CIRCA_old.tools.positional_encoding as encodings 


class UnCRtainTSDataset(CircaPatchDataSet):
    """
    A custom PyTorch Dataset designed to manage Sentinel-1 and Sentinel-2 data specifically for cloud
    reconstruction tasks. Dataset adapted for UnCRtainTS repo.
    """

    def __init__(
        self,
        data_optique: Union[str, Path],
        data_radar: Union[str, Path],
        patch_size: int = 256,
        overlap: Optional[int] = 0,
        load_dataset: Optional[str] = None,
        shuffle: bool = False,
        use_SAR: bool = True,
    ):
        """
        Initializes the dataset.

        Parameters:
        - data_optique (Union[str, Path]): Path to the optical data directory.
        - data_radar (Union[str, Path]): Path to the radar data directory.
        - patch_size (int, optional): Size of the patches to extract. Defaults to 256.
        - overlap (int, optional): Number of pixels to overlap between patches on all sides. Defaults to 0.
        - load_dataset (Optional[str], optional): Path to a pre-saved dataset CSV file. Defaults to None.
        - shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        - use_SAR (bool, optional): Whether to include SAR data in the dataset. Defaults to True.
        """
        super(UnCRtainTSDataset, self).__init__(
            data_optique=data_optique,
            data_radar=data_radar,
            patch_size=patch_size,
            overlap=overlap,
            load_dataset=load_dataset,
            shuffle=shuffle,
            use_SAR=use_SAR,
        )

    def __getitem__(self, item: int) -> Dict[str, Union[np.ndarray, str, List[str]]]:
        """
        Retrieves an item from the dataset.

        Parameters:
        - item (int): Index of the item to retrieve.

        Returns:
        - Dict[str, Union[np.ndarray, str, List[str]]]: A dictionary containing the data, name, masks, and dates.
        """
        patch_data = self.patches_dataset.iloc[item]
        patch_window = Window(*patch_data.window)
        patch_S2_array = SentinelDataProcessor.read_MS(
            patch_data.files[0], patch_window
        )  # Extraction données S2

        (
            patch_S2_curated,
            dates_S2_curated,
            cloud_masks,
        ) = SentinelDataProcessor.extract_and_transform_S2(
            patch_S2_array,
            self.dates_dict[patch_data.mgrs25]["S2"],
        )

        # Appariement des dates S1 ASC ou DESC les plus proches
        (
            dates_S1_curated,
            index_S1_curated,
            orbit_type,
        ) = SentinelDataProcessor.get_pairedS1(
            dates_S2_curated,
            self.dates_dict[patch_data.mgrs25]["S1"]["ASC"],
            self.dates_dict[patch_data.mgrs25]["S1"]["DESC"],
        )
        path_S1 = patch_data.files[1] if orbit_type == "ASC" else patch_data.files[2]
        patch_S1_array = SentinelDataProcessor.read_SAR(
            path_S1, patch_window
        )  # Extraction données S1
        bands_S1 = [patch_S1_array[t_index] for t_index in index_S1_curated]
        patch_S1_curated = np.stack(bands_S1, axis=0)
        patch = np.concatenate(
            [
                SentinelDataProcessor.process_MS(patch_S2_curated),
                SentinelDataProcessor.process_SAR(patch_S1_curated),
            ],
            axis=-1,
        )

        return {
            "data": patch,
            "name": patch_data.patch,
            "masks": cloud_masks,
            "dates_S2": dates_S2_curated,
            "dates_S1": dates_S1_curated,
        }


if __name__ == "__main__":
    store_dai = Path("/home/dl/speillet/Partage/store-dai/")
    path_dataset_CIRCA = store_dai / "projets/pac/3str/EXP_2"
    data_optique = path_dataset_CIRCA / "Data_Raster" / "optique_dataset"
    data_radar = path_dataset_CIRCA / "Data_Raster" / "radar_dataset_v2"
    patch_size = 256
    overlap = 0
    ds = CircaPatchDataSet(
        data_optique=data_optique,
        data_radar=data_radar,
        patch_size=patch_size,
        overlap=overlap,
    )

    # ds.setup()
    # ds.export_dataset()
    sample = next(iter(ds))
    print(sample.keys())
