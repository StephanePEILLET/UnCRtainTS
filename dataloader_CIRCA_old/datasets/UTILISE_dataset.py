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
from rasterio.windows import Window
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
import random

from omegaconf import DictConfig, ListConfig, OmegaConf

from dataloader_CIRCA_old.tools.data_processor import SentinelDataProcessor
import dataloader_CIRCA_old.tools.positional_encoding as encodings 
import dataloader_CIRCA_old.tools.torch_transforms as torch_transforms
from dataloader_CIRCA_old.datasets import CircaPatchDataSet
from dataloader_CIRCA_old.tools.mask_generation import overlay_seq_with_clouds


CHANNEL_CONFIG = ['bgr', 'bgr-nir', 'all', 'bgr-mask', 'bgr-nir-mask', 'all-mask']


class UTILISEDataset(CircaPatchDataSet):
    """
    A custom PyTorch Dataset designed to manage Sentinel-1 and Sentinel-2 data specifically for cloud
    reconstruction tasks. Dataset adapted for U-TILISE repo.
    """

    def __init__(
        self,
        data_optique: Union[str, Path],
        data_radar: Union[str, Path],
        patch_size: int = 256,
        overlap: Optional[int] = 0,
        load_dataset: Optional[str] = None,
        shuffle: bool = False,
        use_SAR: bool = False,
        mask_kwargs: dict = None,
        return_cloud_prob: bool = False,
        render_occluded_above_p: Optional[float] = None,
        channels: str = "all",
        augment: bool = False,
        seq_length: int = 30,
        pe_strategy: str = 'day-of-year',
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
        - channels(str): Channels to be extracted, choose among ['rgb', 'bgr', 'bgr-nir', 'bgr-mask', 'bgr-nir-mask'].

        """
        super(UTILISEDataset, self).__init__(
            data_optique=data_optique,
            data_radar=data_radar,
            patch_size=patch_size,
            overlap=overlap,
            load_dataset=load_dataset,
            shuffle=shuffle,
            use_SAR=use_SAR,
        )
        # Init des variables propres à U-TILISE
        self.seq_length = seq_length
        self.pe_strategy = pe_strategy
        self.return_cloud_prob = return_cloud_prob
        self.render_occluded_above_p = render_occluded_above_p
        self.set_mask_args(mask_kwargs)
        self.set_channels(channels)
        self.set_augment(augment)


    def set_mask_args(self, mask_kwargs):

        if isinstance(mask_kwargs, dict):
            mask_kwargs = OmegaConf.create(mask_kwargs)

        if mask_kwargs is not None:
            mask_kwargs.mask_type = mask_kwargs.get('mask_type', 'random_clouds')
            mask_kwargs.ratio_masked_frames = mask_kwargs.get('ratio_masked_frames', 0.5)
            mask_kwargs.ratio_fully_masked_frames = mask_kwargs.get('ratio_fully_masked_frames', 0.0)
            mask_kwargs.non_masked_frames = mask_kwargs.get('non_masked_frames', [])

            self.fill_type = mask_kwargs.get('fill_type', 'fill_value')
            self.fill_value = mask_kwargs.get('fill_value', 1)
            self.fixed_masking_ratio = mask_kwargs.get('fixed_masking_ratio', False)
            self.intersect_real_cloud_masks = mask_kwargs.get('intersect_real_cloud_masks', False)
            self.dilate_cloud_masks = mask_kwargs.get('dilate_cloud_masks', False)
            self.mask_kwargs = mask_kwargs
        else:
            self.mask_kwargs = None

        print(f"{self.mask_kwargs=}")

    def setup_mask_per_zone(self, ):
        
        # Retrouve par zone tous les fichiers appartenant à une mgrs 

        pass

    def set_augment(self, augment):
        if augment:
            self.augmentation_function = transforms.Compose([
                torch_transforms.Rotate(),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5)
            ])
        else:
            self.augmentation_function = None

    def set_channels(self, channels: str):
        """
            Save the number of channels, the indices of the RGB channels, and the index of the NIR channel
            self.channels: used to extract the relevant channels from the hdf5 file
            self.c_index_rgb and self.c_index_nir: indices of the RGB (B2, B3, B4) and NIR channels (B8), w.r.t. the
            output of the self.__getitem__() call
        """
        if channels not in CHANNEL_CONFIG:
            raise ValueError(f"Unknown channel configuration `{channels}`. Choose among {CHANNEL_CONFIG} to "
                             "specify `channels`.\n")
        else:
            self.channels = channels
        
        if 'bgr' == self.channels[:3]:
            # self.channels in ['bgr', 'bgr-nir', 'bgr-mask', 'bgr-nir-mask']
            self.num_channels = 3
            self.c_index_rgb = torch.Tensor([2, 1, 0]).long()
            self.s2_channels = list(np.arange(self.num_channels))                      # B2, B3, B4
        else:
            # self.channels in ['all', 'all-mask']
            self.num_channels = 10
            self.c_index_rgb = torch.Tensor([2, 1, 0]).long()
            self.s2_channels = list(np.arange(self.num_channels))               # all 13 bands

        if '-nir' in self.channels:
            # self.channels in ['bgr-nir', 'bgr-nir-mask']
            self.num_channels += 1
            self.c_index_nir = torch.Tensor([3]).long()
            self.s2_channels += [7]                              # B8
        elif 'all' in channels:
            self.c_index_nir = torch.Tensor([7]).long()
        else:
            self.c_index_nir = torch.from_numpy(np.array(np.nan))

        if '-mask' in self.channels:
            # self.channels in ['bgr-mask', 'bgr-nir-mask', 'all-mask']
            self.num_channels += 1

        if self.use_SAR:
            self.num_channels += 2

    def mask_images_with_cloud_coverage_above_p(self, cloud_mask: torch.Tensor) -> torch.Tensor:
        """
        Marks all pixels of an image as occluded if its cloud coverage exceeds `self.render_occluded_above_p` [-].

        Args:
            cloud_mask: torch.Tensor, (T x 1 x H x W), time series of cloud masks.

        Returns:
            cloud_mask: torch.Tensor, (T x 1 x H x W), updated time series of cloud masks.
        """
        coverage = np.mean(cloud_mask, axis=(1, 2))
        cloud_mask[coverage > self.render_occluded_above_p, :, :] = 1
        return cloud_mask

    def intersect_masks(self, masks: torch.Tensor, cloud_mask: torch.Tensor) -> torch.Tensor:
        """
        Intersects a randomly generated sequence of cloud masks `masks` with the actual cloud mask sequence of the
        image time series to be masked.

        Args:
            masks:      torch.Tensor, (T x 1 x H x W), sequence of randomly sampled cloud masks.
            cloud_mask: torch.Tensor, (T x 1 x H x W), actual time series of cloud masks.

        Returns:
            masks:      torch.Tensor, (T x 1 x H x W), intersection of `masks` with `cloud_mask`.
        """

        assert masks[0].shape == cloud_mask[0].shape, 'Cannot intersect two sequences of masks with unequal temporal ' \
                                                      'shape.'
        assert masks[-2:].shape == cloud_mask[-2:].shape, 'Cannot intersect two sequences of masks with unequal ' \
                                                      'spatial shape.'
        assert masks[1].shape == cloud_mask[1].shape, 'Cannot intersect two sequences of masks with unequal ' \
                                                      'spectral shape.'

        masks[np.logical_or(masks > 0., cloud_mask == 1)] = 1

        if self.render_occluded_above_p and self.render_occluded_above_p > 0.:
            masks = self.mask_images_with_cloud_coverage_above_p(masks)
        return masks

    def sample_cloud_masks_from_tiles(
            self, 
            patch_data, 
            n: int, 
            p: float = 0.1,
        ) -> torch.Tensor:
        """
        Randomly samples `n` cloud masks from a given tile.

        Args:
            patch_data:  h5py group
            n:      int, number of cloud masks to be sampled.
            p:      float, minimum cloud coverage [-] of the sampled cloud masks.

        Returns:
            cloud_mask:  torch.Tensor, n x 1 x H x W, sampled cloud masks.
        """
        # Extract all samples that originate from the same tile as the given input sample
        samples = self.patches_dataset[self.patches_dataset["mgrs25"] == patch_data.mgrs25].window.values

        # Randomly sample `n` cloud masks with cloud coverage of >= p
        cloud_mask = []
        while len(cloud_mask) < n:
            # Extract the cloud masks of a randomly drawn S2 image time series, T x 1 x H x W
            sample = random.choice(samples)
            seq = SentinelDataProcessor.read_cloud_mask(path_raster=patch_data.files[0], window=Window(*sample))

            # Compute cloud coverage per frame
            coverage = np.mean(seq, axis=(1, 2))

            indices = np.argwhere(coverage >= p).flatten()
            if len(indices) > 0:
                cloud_mask.append(seq[np.random.choice(indices), :, :])

        # n x 1 x H x W
        cloud_mask = np.stack(cloud_mask, axis=0)

        if self.render_occluded_above_p and self.render_occluded_above_p > 0.:
            cloud_mask = self.mask_images_with_cloud_coverage_above_p(cloud_mask)

        return cloud_mask

    def sample_indices_masked_frames(
            self,
            idx_valid_input_frames: np.ndarray,
            ratio_masked_frames: float = 0.5,
            ratio_fully_masked_frames: float = 0.0,
            non_masked_frames: Optional[List[int]] = None,
            fixed_masking_ratio: bool = True,
        ) -> Dict[str, np.ndarray]:
        """
        Generates a sequence of `masks` to synthetically mask an image time series. masks[t1, 0, y1, x1] == 1 will mask the
        spatio-temporal location (t1, y1, x1), whereas masks[t2, 0, y2, x2] == 0 will retain the observed reflectance at
        the spatio-temporal location (t2, y2,x2) (w.r.t. all spectral channels).

        Args:
            idx_valid_input_frames:      np.ndarray, indices of those frames that are available for masking.
            ratio_masked_frames:         float, ratio of (partially or fully) masked frames.
            ratio_fully_masked_frames:   float, ratio of fully masked frames.
            non_masked_frames:           list of int, indices of those frames that should be excluded from masking
                                        (e.g., first frame).
            fixed_masking_ratio:         bool, True to enforce the same ratio of masked frames across image time sequences,
                                        False to vary the ratio of masked frames across image time sequences.
                                        For varying sampling ratios: `ratio_masked_frames` and `ratio_fully_masked_frames`
                                        define upper bounds.

        Returns:
            dict, defines two mutually exclusive sets of frame indices sampled from `idx_valid_input_frames`:
                'indices_masked':        np.ndarray, indices of (partially) masked frames.
                'indices_fully_masked':  np.ndarray, indices of fully masked frames.
        """

        assert ratio_fully_masked_frames <= ratio_masked_frames, "Masking parameter `ratio_fully_masked_frames` needs to " \
                                                                "be smaller or equal to `ratio_masked_frames.`"

        # Upper bound: Maximum number of masked input frames (partially or fully masked)
        num_total = len(idx_valid_input_frames)

        if not fixed_masking_ratio:
            # Vary the sampling ratio by adjusting the number of frames available for masking
            # (at least one frame has to be masked)
            num_total = random.randint(1, num_total)

        # Number of masked frames (partially or fully masked)
        num_masked = math.ceil(ratio_masked_frames * num_total)

        # Number of fully masked frames
        num_fully_masked = math.ceil(ratio_fully_masked_frames * num_total)

        # Randomly select the indices of those frames that will be masked (partially or fully)
        if non_masked_frames is not None:
            non_masked_frames = np.asarray(non_masked_frames)
            if np.any(non_masked_frames < 0):
                # Account for negative indices
                indices_pos = non_masked_frames[non_masked_frames >= 0]
                indices_neg = idx_valid_input_frames[non_masked_frames[non_masked_frames < 0]]
                non_masked_frames = np.concatenate((indices_pos, indices_neg), axis=0)
            else:
                non_masked_frames = idx_valid_input_frames[non_masked_frames]
            list_frames = np.setdiff1d(idx_valid_input_frames, non_masked_frames)
            indices_masked = np.random.choice(list_frames, min(num_masked, list_frames.size), replace=False)
        else:
            indices_masked = np.random.choice(idx_valid_input_frames, num_masked, replace=False)

        # Randomly selected the frame indices of the fully masked frames
        indices_fully_masked = np.random.choice(indices_masked, num_fully_masked, replace=False)

        return {'indices_masked': indices_masked, 'indices_fully_masked': indices_fully_masked}

    def generate_masks(
            self, 
            patch_data, 
            input_frames, 
            input_masks,
            p: float = 0.1,
            intersect_real_cloud_masks: bool = True,
            t_masked : Optional[np.ndarray]= None,
        ):
        """

        """
        assert len(input_frames) == len(input_masks)
        if self.mask_kwargs.mask_type == 'random_clouds':

            # Permet de sélectionner les indices qui seront masquer parmis les images passer en entrées
            if t_masked is None:
                t_masked = self.sample_indices_masked_frames(
                    idx_valid_input_frames=np.arange(0, input_frames.shape[0]),
                    ratio_masked_frames=self.mask_kwargs.ratio_masked_frames,
                    ratio_fully_masked_frames=self.mask_kwargs.ratio_fully_masked_frames,
                    non_masked_frames=self.mask_kwargs.non_masked_frames,
                    fixed_masking_ratio=self.fixed_masking_ratio,
                )

            sampled_cloud_masks = self.sample_cloud_masks_from_tiles(patch_data=patch_data, n=len(t_masked['indices_masked']), p=p)

            # Generate a sequence of masks with synthetic cloud
            masks = np.zeros((input_masks.shape[0], *input_masks.shape[-2:]))
            masks[t_masked['indices_masked'], :, :] = sampled_cloud_masks
            
            if len(t_masked['indices_fully_masked']) > 0: 
                masks[t_masked['indices_fully_masked'], :, :] = [len(t_masked["indices_fully_masked"]) * np.ones(input_masks.shape[-2:])]

            # On applique les masques nuages sur les données S2 /S1 afin de les cachés
            images_masked, masks = overlay_seq_with_clouds(
                input_frames,
                masks,
                t_masked=None,
                fill_value=self.fill_value,
                dilate_cloud_masks=self.dilate_cloud_masks,
            )

            # Intersect the randomly generated sequence of cloud masks with the actual cloud masks of the sequence
            if intersect_real_cloud_masks:
                masks = self.intersect_masks(masks, input_masks)

        elif self.mask_kwargs.mask_type == 'real_clouds':
            # Cas où l'on utilise de vrais masks nuages dans le cas de l'inférence
            raise NotImplementedError
        else:
            raise NotImplementedError

        return t_masked, images_masked, masks


    def __getitem__(self, item: int, t_sampled:dict = None) -> Dict[str, Union[np.ndarray, str, List[str]]]:
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
            S2_channels_selected=self.s2_channels,
        )

        # Partie sur le sampling ou trim de dates
        masks_valid_obs = np.ones(patch_S2_curated.shape[0])

        frames_target = patch_S2_curated.copy()

        # Partie sur la génération de masks / t_sampled = indices des frames masqué par nuages synthétiques
        t_sampled, patch_S2_masked, masks = self.generate_masks(patch_data, patch_S2_curated, cloud_masks, t_masked=t_sampled)

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
            path_S1 = patch_data.files[1] if orbit_type == "ASC" else patch_data.files[2]
            patch_S1_array = SentinelDataProcessor.read_SAR(
                path_S1, patch_window
            )  # Extraction données S1
            bands_S1 = [patch_S1_array[t_index] for t_index in index_S1_curated]
            patch_S1_curated = np.stack(bands_S1, axis=0)

            frames_input = np.concatenate(
                [
                    SentinelDataProcessor.process_MS(patch_S2_masked),
                    SentinelDataProcessor.process_SAR(patch_S1_curated),
                ],
                axis=-1,
            )
        else:
            frames_input = SentinelDataProcessor.process_MS(patch_S2_masked)

        # Extract the number of days since the first observation in the sequence (= temporal sampling)
        days = encodings.get_position_for_positional_encoding(dates_S2_curated, 'day-within-sequence')
        # Get positions for positional encoding
        position_days = encodings.get_position_for_positional_encoding(dates_S2_curated, self.pe_strategy)

        if '-mask' in self.channels and self.mask_kwargs is not None:
            frames_input = torch.cat((frames_input, masks), dim=1)

        # Assemble output
        out = {
            'x': frames_input,  # (synthetically masked) S2 satellite image time series, (T x C x H x W), optionally including S1 bands
            'y': frames_target,  # observed/target satellite image time series, (T x C x H x W)
            'masks': masks,  # masks applied to `x`, (T x 1 x H x W); pixel with value 1 is masked, 0 otherwise
            'masks_valid_obs': masks_valid_obs,  # flag to indicate valid time steps, (T, ); 1 if valid, 0 if invalid
            'position_days': position_days,
            'days': days,    # temporal sampling, number of days since the first observation in the sequence, (T, )
            'sample_index': item,
            'filepath': self.patches_dataset.iloc[item].files,
            'c_index_rgb': self.c_index_rgb,
            'c_index_nir': self.c_index_nir,
            'S2_dates' : dates_S2_curated,
            'cloud_mask': cloud_masks,
        }

        if self.use_SAR:
            out['S1_dates'] = dates_S1_curated

        if t_sampled is not None:
            out['to_export'] = {'t_sampled': t_sampled}

        if self.return_cloud_prob:
            out['cloud_prob'] = cloud_masks

        return out


if __name__ == "__main__":

    store_dai = Path("/home/dl/speillet/Partage/store-dai/")
    path_dataset_CIRCA = store_dai / "projets/pac/3str/EXP_2"
    data_optique = path_dataset_CIRCA / "Data_Raster" / "optique_dataset"
    data_radar = path_dataset_CIRCA / "Data_Raster" / "radar_dataset_v4"
    patch_size = 256
    overlap = 0

    mask_kwargs = {
        "mask_type": "random_clouds",             # Strategy for synthetic data gap generation. ['random_clouds', 'real_clouds']
        "ratio_masked_frames": 0.5,               # Ratio of partially/fully masked images in a satellite image time series (upper bound).
        "ratio_fully_masked_frames": 0.0,         # Ratio of fully masked images in a satellite image time series (upper bound).
        "fixed_masking_ratio": True,              # False de base, True to vary the masking ratio across satellite image time series, False otherwise.
        "non_masked_frames": [0],                 # list of int, time steps to be excluded from masking.
        "intersect_real_cloud_masks": False,      # True to intersect randomly sampled cloud masks with the actual cloud mask sequence, False otherwise.
        "dilate_cloud_masks": False,              # True to dilate the cloud masks before masking, False otherwise.
        "fill_type": "fill_value",                # Strategy for initializing masked pixels. ['fill_value', 'white_noise', 'mean']
        "fill_value": 1,                          # Pixel value of masked pixels. Used if fill_type == 'fill_value'.
    }

    ds = UTILISEDataset(
        data_optique=data_optique,
        data_radar=data_radar,
        patch_size=patch_size,
        overlap=overlap,
        shuffle=False,
        use_SAR=False,
        mask_kwargs=mask_kwargs,
        channels="all",
        return_cloud_prob=False,
        render_occluded_above_p=None,
        augment=False,
        seq_length=30,
        pe_strategy='day-of-year',
        load_dataset="datasetCIRCAUnCRtainTS.csv"
    )

    # ds.setup()
    # ds.export_dataset()
    sample = next(iter(ds))
    print(sample.keys())
