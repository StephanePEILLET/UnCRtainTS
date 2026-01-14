import datetime as dt
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.utils.data
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import Tensor
from torchvision import transforms


class CRDatasetSampler:


    def sample_cloud_masks_from_tiles(self, sample: h5py._hl.group.Group, n: int, p: float = 0.1) -> torch.Tensor:
        """
        Randomly samples `n` cloud masks from a given tile.

        Args:
            sample:  h5py group
            n:      int, number of cloud masks to be sampled.
            p:      float, minimum cloud coverage [-] of the sampled cloud masks.

        Returns:
            cloud_mask:  torch.Tensor, n x 1 x H x W, sampled cloud masks.
        """
        # Extract all samples that originate from the same tile as the given input sample
        samples = self.tiles2samples[sample.parent.name]

        # Randomly sample `n` cloud masks with cloud coverage of >= p
        cloud_mask = []
        while len(cloud_mask) < n:
            # Extract the cloud masks of a randomly drawn S2 image time series, T x 1 x H x W
            seq = torch.from_numpy(self.f[random.choice(samples)]['S2/cloud_mask'][:]).float()

            if self.crop_settings.enabled:
                seq = self.crop_function(seq)

            # Compute cloud coverage per frame
            coverage = torch.mean(seq, dim=(1, 2, 3))

            indices = torch.argwhere(coverage >= p).flatten()
            if len(indices) > 0:
                cloud_mask.append(seq[np.random.choice(indices), :, :, :])

        # n x 1 x H x W
        cloud_mask = torch.stack(cloud_mask, dim=0)

        if self.render_occluded_above_p and self.render_occluded_above_p > 0.:
            cloud_mask = self._mask_images_with_cloud_coverage_above_p(cloud_mask)

        return cloud_mask

        def _subsample_sequence(self, sample: h5py._hl.group.Group, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Filters/Subsamples the image time series stored in `sample` as follows (cf. `self.filter_settings` and
            `self.max_seq_length`):
            1) Extracts cloud-free images or extracts the longest consecutive cloud-free subsequence,
            2) selects a subsequence of cloud-free images such that the temporal difference between consecutive cloud-free
            images is at most `self.filter_settings.max_t_sampling` days,
            3) trims the sequence to a maximum temporal length.

            Args:
                sample:           h5py group.
                seq_length:       int, temporal length of the sample.

            Returns:
                t_sampled:        torch.Tensor, length T.
                masks_valid_obs:  torch.Tensor, (T, ).
            """

            # Generate a mask to exclude invalid frames:
            # a value of 1 indicates a valid frame, whereas a value of 0 marks an invalid frame
            if self.filter_settings.type == 'cloud-free':
                # Indices of available and cloud-free images
                masks_valid_obs = torch.from_numpy(sample['valid_obs'][:])

                if self.filter_settings.max_t_sampling is not None:
                    subseq = self._longest_consecutive_seq_within_sampling_frequency(sample)
                    masks_valid_obs[:subseq['start']] = 0
                    masks_valid_obs[subseq['end'] + 1:] = 0

            elif self.filter_settings.type == 'cloud-free_consecutive':
                subseq = self._longest_consecutive_seq(sample)
                masks_valid_obs = torch.from_numpy(sample['valid_obs'][:])
                masks_valid_obs[:subseq['start']] = 0
                masks_valid_obs[subseq['end'] + 1:] = 0
            else:
                masks_valid_obs = torch.ones(seq_length, )

            if self.filter_settings.get('return_valid_obs_only', True):
                t_sampled = masks_valid_obs.nonzero().view(-1)
            else:
                t_sampled = torch.arange(0, len(masks_valid_obs))

            if self.max_seq_length is not None and len(t_sampled) > self.max_seq_length:
                # Randomly select `self.max_seq_length` consecutive frames
                t_start = np.random.choice(np.arange(0, len(t_sampled) - self.max_seq_length + 1))
                t_end = t_start + self.max_seq_length
                t_sampled = t_sampled[t_start:t_end]

            return t_sampled, masks_valid_obs[t_sampled]

        def _valid_sample(self, sample: h5py._hl.group.Group) -> bool:
            """
            Determines whether the h5py group `sample` defines a valid data sample. The following conditions have to be met:
            (i)   The h5py group stores S2 satellite imagery, i.e., it has a key 'S2' that refers to a h5py Dataset.
            (ii)  The h5py group stores S1 satellite imagery, i.e., it has a key 'S1' that refers to a h5py Dataset
                (only if self.include_S1 == True).
            (iii) If cloud filtering is enabled: the satellite image time series has a minimal sequence length, where
                the minimal length is given by `self.filter_settings.min_length`.
            (iv)  If cloud filtering is enabled and `self.filter_settings.max_t_sampling` is not None: the subsequence of
                cloud-free images includes at least `self.filter_settings.min_length` consecutive cloud-free images with
                at most `self.filter_settings.max_t_sampling` days between observations.

            Args:
                sample: h5py group.

            Returns:
                bool, True if the h5py group `sample` is a valid data sample, False otherwise.
            """

            if isinstance(sample, h5py.Group) and 'S2' in sample.keys():
                if self.include_S1 and 'S1' not in sample.keys():
                    return False
                if self.filter_settings.type is None:
                    return True
                if self.filter_settings.type == 'cloud-free':
                    seq_length = sample['idx_good_frames'].size

                    # Check number of valid frames
                    if seq_length >= self.filter_settings.min_length:
                        if self.filter_settings.max_num_consec_invalid is not None:
                            # Check number of consecutive invalid frames
                            max_num_consec_invalid = self._count_max_num_consecutive_invalid_frames(sample)

                            if max_num_consec_invalid <= self.filter_settings.max_num_consec_invalid:
                                # (i) Minimal sequence length is ok, and (ii) max. number of consecutive invalid frames
                                # is below the threshold
                                return True
                            if self.verbose == 1:
                                print(f"Too many consecutive invalid frames within the sequence "
                                    f"({max_num_consec_invalid} < {self.filter_settings.max_num_consec_invalid}): "
                                    f"{sample.name}")
                            return False

                        if self.filter_settings.max_t_sampling is not None:
                            count = self._longest_consecutive_seq_within_sampling_frequency(sample)['len']
                            if count >= self.filter_settings.min_length:
                                return True
                            if self.verbose == 1:
                                print(f"Less than {self.filter_settings.min_length} consecutive cloud-free frames with "
                                    f"temporal sampling of at most {self.filter_settings.max_t_sampling} days: "
                                    f"{sample.name}")
                            return False

                        # Minimal sequence length is ok
                        return True
                    if self.verbose == 1:
                        print(f"Too short sequence ({seq_length} < {self.filter_settings.min_length}): {sample.name}")
                    return False
                if self.filter_settings.type == 'cloud-free_consecutive':
                    seq_length = self._longest_consecutive_seq(sample)['len']
                    if seq_length >= self.filter_settings.min_length:

                        if self.filter_settings.max_t_sampling is not None:
                            count = self._longest_consecutive_seq_within_sampling_frequency(sample)['len']
                            if count >= self.filter_settings.min_length:
                                return True
                            if self.verbose == 1:
                                print(f"Less than {self.filter_settings.min_length} consecutive cloud-free frames with "
                                    f"temporal sampling of at most {self.filter_settings.max_t_sampling} days: "
                                    f"{sample.name}")
                            return False

                        return True
                    if self.verbose == 1:
                        print(f"Too short sequence ({seq_length} < {self.filter_settings.min_length}): {sample.name}")
                    return False
                raise NotImplementedError(f'Unknown sequence filter {self.filter_settings.type}.\n')

            return False

        def _longest_consecutive_seq_within_sampling_frequency(self, sample: h5py._hl.group.Group) -> Dict[str, int]:
            """
            Determines the longest subsequence of consecutive cloud-free images, where the temporal sampling between
            consecutive cloud-free images does not exceed `self.filter_settings.max_t_sampling` days.

            Args:
                sample:      h5py group.

            Returns:
                subseq:      dict, the longest subsequence of valid images. The dictionary has the following key-value
                            pairs:
                                'start':  int, index of the first image of the subsequence.
                                'end':    int, index of the last image of the subsequence.
                                'len':    int, temporal length of the subsequence.
            """

            # Extract the acquisition dates of the cloud-free images within the sequence
            s2_dates = [dataset_tools.str2date(date.decode("utf-8")) for date in sample['S2/S2_dates']]
            t_cloudfree = sample['idx_good_frames'][:]
            s2_dates = [s2_dates[t] for t in t_cloudfree]

            # Count number of consecutive cloud-free images with temporal sampling of at most
            # `self.filter_settings.max_t_sampling:`
            subseq = {'start': 0, 'end': 0, 'len': 0}
            count = 1
            start = 0

            for i in range(len(s2_dates) - 1):
                if (s2_dates[i+1] - s2_dates[i]).days <= self.filter_settings.max_t_sampling:
                    end = i + 1
                    count += 1
                    if count > subseq['len']:
                        subseq['start'] = t_cloudfree[start]
                        subseq['end'] = t_cloudfree[end]
                        subseq['len'] = count
                else:
                    start = i + 1
                    count = 1

            return subseq