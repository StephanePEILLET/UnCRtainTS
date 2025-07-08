import os

import torch
from torch.utils.data import DataLoader, Subset

from data.uncrtaints_adapter import UnCRtainTS_CIRCA_Adapter
from src.utils_training import seed_worker


class UnCRtainTS_datamodule:

    def __init__(self, config):
        self.config = config
        # Prepare the data module configuration
        self.max_samples_count, self.max_samples_frac = None, None
        if self.config.data.get("max_samples_count", False):
            self.max_samples_count = self.config.data.max_samples_count
            del self.config.data.max_samples_count
        if self.config.data.get("max_samples_frac", False):
            self.max_samples_frac = self.config.data.max_samples_frac
            del self.config.data.max_samples_frac
        self.dt_train, self.dt_val, self.dt_test = self.setup()

        self.collate_fn = self.collate_fn_mono if self.config.pretrain else self.collate_fn_multi
        self.device = torch.device(self.config.device) if self.config.get("device", False) else torch.device("cpu")

    # check for file of pre-computed statistics, e.g. indices or cloud coverage
    @staticmethod
    def import_from_path(split, config):
        if os.path.exists(os.path.join(os.path.dirname(os.getcwd()), "util", "precomputed")):
            import_path = os.path.join(
                os.path.dirname(os.getcwd()),
                "util",
                "precomputed",
                f"generic_{config.input_t}_{split}_{config.region}_s2cloudless_mask.npy",
            )
        else:
            import_path = os.path.join(
                config.precomputed,
                f"generic_{config.input_t}_{split}_{config.region}_s2cloudless_mask.npy",
            )
        import_data_path = import_path if os.path.isfile(import_path) else None
        return import_data_path

    def setup(self, stage=None):
        """
        Setup the data module, called once per process.
        """
        return self.get_datasets()

    def get_datasets(self):
        # define data sets
        dt_train = UnCRtainTS_CIRCA_Adapter(phase="train", **self.config.data)
        dt_val = UnCRtainTS_CIRCA_Adapter(phase="val", **self.config.data)
        dt_test = UnCRtainTS_CIRCA_Adapter(phase="test", **self.config.data)
        print(f"Train {len(dt_train)}, Val {len(dt_val)}, Test {len(dt_test)}")
        return dt_train, dt_val, dt_test

    def subsample_dataset(self, dataset):
        """
        Subsample the dataset based on max_samples_count and max_samples_frac.
        Wrap to allow for subsampling, e.g. for test runs etc
        """
        if self.max_samples_count is not None and self.max_samples_frac is not None:
            return Subset(
                dataset,
                range(0, min(self.max_samples_count, len(dataset), int(len(self.dt_train) * self.max_samples_frac))),
            )
        return dataset

    def train_dataloader(self):
        return DataLoader(
            self.subsample_dataset(self.dt_train),
            batch_size=self.config.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            num_workers=self.config.num_workers,
            # collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.subsample_dataset(self.dt_val),
            batch_size=self.config.batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            num_workers=self.config.num_workers,
            # collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.subsample_dataset(self.dt_test),
            batch_size=self.config.batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            num_workers=self.config.num_workers,
            # collate_fn=self.collate_fn,
        )

    def collate_fn_mono(self, batch):
        x = batch["input"]["S2"].to(self.device).unsqueeze(1)
        if self.config.use_sar:
            x = torch.cat((batch["input"]["S1"].to(self.device).unsqueeze(1), x), dim=2)
        m = batch["input"]["masks"].to(self.device).unsqueeze(1)
        y = batch["target"]["S2"].to(self.device).unsqueeze(1)
        dates = None
        return x, y, m, dates

    # def collate_fn_multi(self, batch):
    #     x = torch.stack([torch.from_numpy(sample["input"]["S2"]) for sample in batch]).to(self.device)
    #     dates = torch.stack([torch.Tensor(sample["input"]["S2 TD"]) for sample in batch]).to(self.device).float()
    #     m = torch.stack([torch.Tensor(sample["input"]["masks"]) for sample in batch]).swapaxes(0, 1).to(self.device)
    #     y = torch.stack([torch.from_numpy(sample["target"]["S2"]) for sample in batch]).to(self.device)

    #     if self.config.data.use_sar:
    #         in_S1 = torch.stack([torch.from_numpy(sample["input"]["S1"]) for sample in batch]).to(self.device)
    #         in_S1_td = torch.stack([torch.Tensor(sample["input"]["S1 TD"]) for sample in batch]).to(self.device).float()
    #         x = torch.cat([in_S1, x], dim=2)
    #         dates = torch.stack([in_S1_td, dates]).mean(dim=1).to(self.device)  # Mean on time dimension

    #     if self.config.batch_size > 1:
    #         dates = dates.T
    #     # SHAPES
    #     # x: (batch_size, len_seq, channels, height, width) => ex: ([4, 3, 14, 256, 256])
    #     # y: (batch_size, channels, height, width) => ex: ([4, 10, 256, 256])
    #     # m: (batch_size, len_seq, channels, height, width) => ex: ([4, 3, 1, 256, 256])
    #     # dates: (len_seq, batch_size) => ex: ([3, 4])
    #     return x, y, m, dates

    def collate_fn_multi(self, batch):
        from src.utils_training import recursive_todevice

        in_S2 = recursive_todevice(batch["input"]["S2"], self.device)
        in_S2_td = recursive_todevice(batch["input"]["S2 TD"], self.device)
        if self.config.batch_size > 1:
            in_S2_td = torch.stack(in_S2_td).T
        in_m = torch.stack(recursive_todevice(batch["input"]["masks"], self.device)).swapaxes(0, 1)  # .squeeze(2)
        target_S2 = recursive_todevice(batch["target"]["S2"], self.device)
        y = target_S2  # torch.cat(target_S2, dim=0).unsqueeze(1)

        if self.config.use_sar:
            in_S1 = recursive_todevice(batch["input"]["S1"], self.device)
            in_S1_td = recursive_todevice(batch["input"]["S1 TD"], self.device)
            if self.config.batch_size > 1:
                in_S1_td = torch.stack(in_S1_td).T
            x = torch.cat([in_S1, in_S2], dim=2)  # torch.cat((torch.stack(in_S1, dim=1), torch.stack(in_S2, dim=1)), dim=2)
            dates = torch.cat([in_S1_td, in_S2_td]).type(torch.float64).mean().to(self.device)
            # dates = torch.stack((torch.tensor(in_S1_td), torch.tensor(in_S2_td))).float().mean(dim=0).to(device)
        else:
            x = torch.stack(in_S2, dim=1)
            dates = torch.tensor(in_S2_td).float().to(self.device)

        return x, y, in_m, dates
