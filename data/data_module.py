import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from data.uncrtaints_adapter import UnCRtainTS_CIRCA_Adapter


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
        if self.max_samples_count is not None and self.max_samples_frac is not None:
            dataset = UnCRtainTS_CIRCA_Adapter(phase="all", **self.config.data)
            subset = Subset(
                dataset,
                range(0, min(self.max_samples_count, len(dataset), int(len(dataset) * self.max_samples_frac))),
            )
            dt_train = subset
            dt_val = subset
            dt_test = subset
        else:
            dt_train = UnCRtainTS_CIRCA_Adapter(phase="train", **self.config.data)
            dt_val = UnCRtainTS_CIRCA_Adapter(phase="val", **self.config.data)
            dt_test = UnCRtainTS_CIRCA_Adapter(phase="test", **self.config.data)
        print(f"Train {len(dt_train)}, Val {len(dt_val)}, Test {len(dt_test)}")
        return dt_train, dt_val, dt_test

    def train_dataloader(self):
        return DataLoader(
            self.dt_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dt_val,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dt_test,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
