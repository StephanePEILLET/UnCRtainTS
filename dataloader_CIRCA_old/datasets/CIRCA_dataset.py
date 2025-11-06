from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))

import ast
import json

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from rasterio.windows import Window
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from dataloader_CIRCA_old.tools.data_processor import SentinelDataProcessor


class CircaPatchDataSet(Dataset):
    """
    A custom PyTorch Dataset designed to manage Sentinel-1 and Sentinel-2 data specifically for cloud
    reconstruction tasks.
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
        no_filter: bool = False,
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
        self.data_optique = Path(data_optique)
        self.data_radar = Path(data_radar)
        self.patch_size = patch_size
        self.overlap = overlap
        self.load_dataset = load_dataset
        self.shuffle = shuffle
        self.zones_dataset, self.dates_dict = None, None
        self.patches_dataset = None
        self.use_SAR = use_SAR
        self.no_filter = no_filter
        self.setup()

    def __len__(self) -> int:
        """
        Returns the number of patches in the dataset.

        Returns:
        - int: Number of patches.
        """
        return len(self.patches_dataset)

    def setup(self, load_dataset: Optional[str] = None) -> None:
        """
        Sets up the dataset by either loading from a pre-saved file or processing the data.

        Parameters:
        - load_dataset (Optional[str], optional): Path to a pre-saved dataset CSV file. Defaults to None.
        """
        if self.load_dataset is not None or load_dataset is not None:
            self.load_exported_data(load_dataset or self.load_dataset)
        else:
            self.setup_zones()
            self.setup_patches()

        if self.shuffle:
            self.patches_dataset = self.patches_dataset.sample(frac=1).reset_index(
                drop=True
            )

    def load_exported_data(self, path_data: Union[str, Path]) -> None:
        """
        Loads the dataset from a pre-saved CSV file.

        Parameters:
        - path_data (Union[str, Path]): Path to the CSV file.
        """
        self.patches_dataset = pd.read_csv(path_data)
        cols_to_convert = [
            "window",
            "files",
            "dates_S2",
            "dates_S1_ASC",
            "dates_S1_DESC",
        ]
        self.patches_dataset[cols_to_convert] = self.patches_dataset[
            cols_to_convert
        ].map(ast.literal_eval)

        if self.patches_dataset.loc[0, "window"][2] != self.patch_size:
            print(
                """
                WARNING : Patch size load in the .csv file is not corresponding to the patch size requested 
                during the dataset initialization. New setup of the dataset will be done.."
                """
            )
            self.setup()

        # Recreate the dates dictionary after loading
        self.dates_dict = {
            mgrs25: {
                "S2": self.patches_dataset[self.patches_dataset["mgrs25"] == mgrs25][
                    "dates_S2"
                ].values[0],
                "S1": {
                    "ASC": self.patches_dataset[
                        self.patches_dataset["mgrs25"] == mgrs25
                    ]["dates_S1_ASC"].values[0],
                    "DESC": self.patches_dataset[
                        self.patches_dataset["mgrs25"] == mgrs25
                    ]["dates_S1_DESC"].values[0],
                },
            }
            for mgrs25 in self.patches_dataset["mgrs25"].unique()
        }

    def setup_patches(self) -> None:
        """
        Sets up the patches by processing the data and creating a DataFrame.
        """
        self.patches_dataset = pd.DataFrame(
            columns=["patch", "window", "mgrs", "mgrs25", "files"]
        ).astype(object)

        for _, row in tqdm(
            self.zones_dataset.iterrows(), total=len(self.zones_dataset), leave=False
        ):
            for window in row.windows:
                windows_str = "_".join(map(str, window))
                patch_df = pd.DataFrame(
                    {
                        "patch": f"patches_{row.mgrs25}_window_{windows_str}",
                        "window": [window],
                        "mgrs": row.mgrs,
                        "mgrs25": row.mgrs25,
                        "files": [row.files],  # 0: S2, 1: S1_ASC, 2: S1_DESC
                        "dates_S2": [row.dates_S2],
                        "dates_S1_ASC": [row.dates_S1_ASC],
                        "dates_S1_DESC": [row.dates_S1_DESC],
                    }
                )
                self.patches_dataset = pd.concat(
                    [self.patches_dataset, patch_df], ignore_index=True
                )

    def setup_zones(self) -> None:
        """
        Sets up the zones by processing the data and creating a DataFrame.
        """
        self.zones_dataset = pd.DataFrame(
            columns=["mgrs", "mgrs25", "files", "windows"]
        ).astype(object)
        self.dates_dict = {}

        for mgrs in tqdm(list(self.data_optique.iterdir()), leave=False, desc="mgrs"):
            for mgrs25 in tqdm(list(mgrs.iterdir()), leave=False, desc="mgrs25"):
                self.process_mgrs25_zone(mgrs, mgrs25)

    def process_mgrs25_zone(self, mgrs: Path, mgrs25: Path) -> None:
        """
        Processes a single MGRS25 zone.

        Parameters:
        - mgrs (Path): Path to the MGRS zone.
        - mgrs25 (Path): Path to the MGRS25 zone.
        """
        mgrs_name = mgrs.stem
        mgrs25_name = mgrs25.stem
        self.dates_dict[mgrs25_name] = {}

        # Process Sentinel-2 data
        list_tifs_optique = sorted(mgrs25.rglob("*.tif"))
        list_jsons_optique = sorted(mgrs25.rglob("*.json"))
        dates_S2 = json.load(open(list_jsons_optique[0]))
        self.dates_dict[mgrs25_name]["S2"] = dates_S2

        # Process Sentinel-1 data
        mgrs25_radar = self.data_radar / mgrs_name / mgrs25_name
        assert mgrs25_radar.exists()

        list_tifs_radar = sorted(mgrs25_radar.rglob("*.tif"))
        list_jsons_radar = sorted(mgrs25_radar.rglob("*.json"))

        tif_files = {"S2": list_tifs_optique[0]}
        for file in list_tifs_radar:
            if file.stem.endswith("ASC"):
                tif_files["S1_ASC"] = file
            else:
                tif_files["S1_DESC"] = file

        list_windows = SentinelDataProcessor.split_raster_into_windows(
            path_raster=tif_files["S2"],
            patch_size=self.patch_size,
            overlap=self.overlap,
        )

        self.dates_dict[mgrs25_name]["S1"] = {
            file.stem.split("_")[-1]: json.load(open(file)) for file in list_jsons_radar
        }

        data = {
            "mgrs": mgrs_name,
            "mgrs25": mgrs25_name,
            "files": [sorted(f.as_posix() for f in tif_files.values())],
            "windows": [list_windows],
            "dates_S2": [dates_S2],
            "dates_S1_ASC": [self.dates_dict[mgrs25_name]["S1"]["ASC"]],
            "dates_S1_DESC": [self.dates_dict[mgrs25_name]["S1"]["DESC"]],
        }
        df_temp = pd.DataFrame(data).astype(object)
        self.zones_dataset = pd.concat([self.zones_dataset, df_temp], ignore_index=True)

    def export_dataset(
        self, outpath: Union[str, Path] = "datasetCIRCAUnCRtainTS.csv"
    ) -> None:
        """
        Exports the dataset to a CSV file.

        Parameters:
        - outpath (Union[str, Path], optional): Path to save the CSV file. Defaults to "datasetCIRCAUnCRtainTS.csv".
        """
        self.patches_dataset.to_csv(outpath, index=False)

    def get_patches_per_mgrs(self, mgrs: str) -> pd.DataFrame:
        """
        Retrieves all patches associated with a given MGRS code.

        Parameters:
        - mgrs (str): The MGRS (Military Grid Reference System) code to filter by.

        Returns:
        - pd.DataFrame: A DataFrame containing only the patches matching the MGRS code.
        """
        return self.patches_dataset[self.patches_dataset["mgrs"] == mgrs]

    def get_patches_per_mgrs25(self, mgrs25: str) -> pd.DataFrame:
        """
        Retrieves all patches associated with a given MGRS25 code.

        Parameters:
        - mgrs25 (str): The MGRS25 code (a more precise version of MGRS) to filter by.

        Returns:
        - pd.DataFrame: A DataFrame containing only the patches matching the MGRS25 code.
        """
        return self.patches_dataset[self.patches_dataset["mgrs25"] == mgrs25]

    def get_patch_item(self, patch_name: str) -> Optional[int]:
        """
        Retrieves the index of a patch based on its name.

        Parameters:
        - patch_name (str): The name of the patch to search for.

        Returns:
        - Optional[int]: The index of the patch in the DataFrame. Returns None if the patch is not found.
        """
        matching_patches = self.patches_dataset[self.patches_dataset["patch"] == patch_name]
        if not matching_patches.empty:
            return matching_patches.index.values[0]
        return None  # Returns None if no matching patch is found
    
    def get_random_patch_by_mgrs(self, mgrs: str) -> Dict[str, Union[np.ndarray, str, list]]:
        """
        Retrieves a random patch associated with a given MGRS code.

        Parameters:
        - mgrs (str): The MGRS code to filter by.

        Returns:
        - Dict[str, Union[np.ndarray, str, List[str]]]: A dictionary containing the data, name, masks, and dates of the randomly selected patch.
        """
        df_mgrs = self.get_patches_per_mgrs(mgrs=mgrs)
        return self.__getitem__(np.random.choice(df_mgrs.index.values))

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

        if self.no_filter:
            patch_S2_curated = patch_S2_array[:, :, :, 0:10]
            dates_S2_curated = np.asarray(self.dates_dict[patch_data.mgrs25]["S2"])
            cloud_masks = patch_S2_array[:, :, :, 10]
        else:
            (
                patch_S2_curated,
                dates_S2_curated,
                cloud_masks,
            ) = SentinelDataProcessor.extract_and_transform_S2(
                patch_S2_array,
                self.dates_dict[patch_data.mgrs25]["S2"],
            )

        sample = {
            "name": patch_data.patch,
            "data_S2": patch_S2_curated,
            "dates_S2": dates_S2_curated,
            "masks": cloud_masks,
        }

        if self.use_SAR:
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

            path_S1 = (
                patch_data.files[1] if orbit_type == "ASC" else patch_data.files[2]
            )
            patch_S1_array = SentinelDataProcessor.read_SAR(
                path_S1, patch_window
            )  # Extraction données S1
            bands_S1 = [patch_S1_array[t_index] for t_index in index_S1_curated]
            patch_S1_curated = np.stack(bands_S1, axis=0)
            sample.update(
                {
                    "data_S1": patch_S1_curated,
                    "dates_S1": dates_S1_curated,
                }
            )
        return sample


if __name__ == "__main__":
    store_dai = Path("/home/SPeillet/Partage/store-dai")
    path_dataset_circa = store_dai / "projets/pac/3str/EXP_2"
    data_optique = path_dataset_circa / "Data_Raster" / "optique_dataset"
    data_radar = path_dataset_circa / "Data_Raster" / "radar_dataset_v4"
    PATCH_SIZE = 256
    OVERLAP = 0

    ds = CircaPatchDataSet(
        data_optique=data_optique,
        data_radar=data_radar,
        patch_size=PATCH_SIZE,
        overlap=OVERLAP,
        load_dataset="datasetCIRCAUnCRtainTS.csv"
    )

    # ds.setup()
    # ds.export_dataset()
    sample = next(iter(ds))
    print(sample.keys())
