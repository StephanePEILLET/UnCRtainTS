from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2]))
from typing import Tuple
import ast
import json
from typing import Dict, List, Optional, Union
import math
import numpy as np
import pandas as pd
import rasterio
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from rasterio.windows import Window
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.auto import tqdm
from dataloader_CIRCA_old.tools.data_processor import SentinelDataProcessor
from dataloader_CIRCA_old.datasets import CircaPatchDataSet
import h5py


class UTILISE_HDF5_Dataset(CircaPatchDataSet):

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

        super().__init__(
            data_optique=data_optique,
            data_radar=data_radar,
            patch_size=patch_size,
            overlap=overlap,
            load_dataset=load_dataset,
            shuffle=shuffle,
            use_SAR=use_SAR,
        )

    def __getitem__(self, item: int):

        patch_data = self.patches_dataset.iloc[item]
        patch_window = Window(*patch_data.window)
        S2_array = SentinelDataProcessor.read_MS(patch_data.files[0], patch_window)

        patch_S2_array = S2_array[:, :, :, 0:10]
        masks = S2_array[:, :, :, -2:]
        dates_S2 = self.dates_dict[patch_data.mgrs25]["S2"]

        snow_masks, cloud_prob = masks[:, :, :, 0], masks[:, :, :, 1]
        cloud_prob_corrected = SentinelDataProcessor.cloud_mask_correction(cloud_prob)
        TRESHOLD = 1
        cloud_mask = (cloud_prob_corrected > TRESHOLD).astype(np.float32)
        idx_good_frames = SentinelDataProcessor.filter_dates(np.stack([snow_masks, cloud_prob_corrected], axis=-1))
        idx_cloudy_frames = np.asarray([d for d in range(len(dates_S2)) if d not in idx_good_frames])
        dates_S1, index_S1, orbit_type = SentinelDataProcessor.get_pairedS1(
            dates_S2,
            self.dates_dict[patch_data.mgrs25]["S1"]["ASC"],
            self.dates_dict[patch_data.mgrs25]["S1"]["DESC"],
        )
        path_S1 = patch_data.files[1] if orbit_type == "ASC" else patch_data.files[2]
        patch_S1_data = SentinelDataProcessor.read_SAR(path_S1, patch_window)
        bands_S1 = [patch_S1_data[t_index] for t_index in index_S1]
        patch_S1_array = np.stack(bands_S1, axis=0)

        return {
            "S1": {
                "S1": patch_S1_array,
                "S1_dates": dates_S1,
            },
            "S2": { 
                "S2": patch_S2_array,
                "S2_dates": dates_S2,
                "cloud_mask": cloud_mask,
                "cloud_prob": cloud_prob_corrected,
                },
            "idx_cloudy_frames": idx_cloudy_frames.tolist(),
            "idx_good_frames": idx_good_frames.tolist(),
            "idx_impaired_frames": idx_cloudy_frames.tolist(),
        }


def pytorch_dict_2_hdf5(dataset, output_file, num_workers=8):
    dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            prefetch_factor=2,
    )
    progress_bar = tqdm(total=len(dataset))
    with h5py.File(output_file, 'w') as hf:
        data_group = hf.create_group('ROIs')
        hf.attrs['num_samples'] = len(dataset)
        for i, sample in enumerate(dataloader):
            sample_group = data_group.create_group(f'{i}')
            for key, value in sample.items():
                if isinstance(value, dict):
                    sample_subgroup = sample_group.create_group(f'{key}')
                    for meta_key, meta_value in value.items():
                        if isinstance(meta_value, torch.Tensor):
                            sample_subgroup.create_dataset(meta_key, data=meta_value, compression='gzip', compression_opts=9)
                        else:
                            sample_subgroup.create_dataset(meta_key, data=meta_value)
                else:
                    sample_group.create_dataset(key, data=value)
            progress_bar.update(1)


if __name__ == "__main__":
    store_dai = Path("/home/SPeillet/Partage/store-dai")
    path_dataset_circa = store_dai / "projets/pac/3str/EXP_2"
    data_optique = path_dataset_circa / "Data_Raster" / "optique_dataset"
    data_radar = path_dataset_circa / "Data_Raster" / "radar_dataset_v4"
    patch_size = 256
    overlap = 0

    SUBSET = True
    SUBSIZE = 1000

    dataset = UTILISE_HDF5_Dataset(
        data_optique=data_optique,
        data_radar=data_radar,
        patch_size=patch_size,
        overlap=overlap,
        # load_dataset="datasetCIRCAUnCRtainTS.csv",
    )
    output_file =  store_dai / "tmp/speillet/subset_1K_circa_utilise.hdf5"

    print("Conversion du dataset PyTorch en HDF5...")
    if SUBSET:
        dataset = Subset(dataset, np.arange(SUBSIZE))
        pytorch_dict_2_hdf5(dataset, output_file, num_workers=8)
    else:
        if not output_file.exists():
            pytorch_dict_2_hdf5(dataset, output_file, num_workers=8)
    print(f"Dataset converti et enregistré dans {output_file}")