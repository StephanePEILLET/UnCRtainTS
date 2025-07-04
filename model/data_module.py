import os

import torch

from data.uncrtaints_adapter import UnCRtainTS_CIRCA_Adapter
from model.utils_training import recursive_todevice, seed_worker


def prepare_data(batch, device, config):
    if config.pretrain:
        return prepare_data_mono(batch, device, config)
    else:
        return prepare_data_multi(batch, device, config)


def prepare_data_mono(batch, device, config):
    x = batch["input"]["S2"].to(device).unsqueeze(1)
    if config.use_sar:
        x = torch.cat((batch["input"]["S1"].to(device).unsqueeze(1), x), dim=2)
    m = batch["input"]["masks"].to(device).unsqueeze(1)
    y = batch["target"]["S2"].to(device).unsqueeze(1)
    return x, y, m


def prepare_data_multi(batch, device, config):
    in_S2 = recursive_todevice(batch["input"]["S2"], device)
    in_S2_td = recursive_todevice(batch["input"]["S2 TD"], device)
    if config.batch_size > 1:
        in_S2_td = torch.stack(in_S2_td).T
    in_m = torch.stack(recursive_todevice(batch["input"]["masks"], device)).swapaxes(0, 1)
    target_S2 = recursive_todevice(batch["target"]["S2"], device)
    y = torch.cat(target_S2, dim=0).unsqueeze(1)

    if config.use_sar:
        in_S1 = recursive_todevice(batch["input"]["S1"], device)
        in_S1_td = recursive_todevice(batch["input"]["S1 TD"], device)
        if config.batch_size > 1:
            in_S1_td = torch.stack(in_S1_td).T
        x = torch.cat((torch.stack(in_S1, dim=1), torch.stack(in_S2, dim=1)), dim=2)
        dates = torch.stack((torch.tensor(in_S1_td), torch.tensor(in_S2_td))).float().mean(dim=0).to(device)
    else:
        x = torch.stack(in_S2, dim=1)
        dates = torch.tensor(in_S2_td).float().to(device)

    return x, y, in_m, dates


# check for file of pre-computed statistics, e.g. indices or cloud coverage
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


def get_datasets(config):
    # define data sets
    dt_train = UnCRtainTS_CIRCA_Adapter(
        os.path.expanduser(config.root1),
        split="train",
        region=config.region,
        sample_type=config.sample_type,
        sampler="random" if config.vary_samples else "fixed",
        n_input_samples=config.input_t,
        import_data_path=import_from_path("train", config),
        min_cov=config.min_cov,
        max_cov=config.max_cov,
    )

    dt_val = UnCRtainTS_CIRCA_Adapter(
        os.path.expanduser(config.root2),
        split="val",
        region="all",
        sample_type=config.sample_type,
        n_input_samples=config.input_t,
        import_data_path=import_from_path("val", config),
    )

    dt_test = UnCRtainTS_CIRCA_Adapter(
        os.path.expanduser(config.root2),
        split="test",
        region="all",
        sample_type=config.sample_type,
        n_input_samples=config.input_t,
        import_data_path=import_from_path("test", config),
    )

    # wrap to allow for subsampling, e.g. for test runs etc
    dt_train = torch.utils.data.Subset(
        dt_train,
        range(
            0,
            min(
                config.max_samples_count,
                len(dt_train),
                int(len(dt_train) * config.max_samples_frac),
            ),
        ),
    )
    dt_val = torch.utils.data.Subset(
        dt_val,
        range(
            0,
            min(
                config.max_samples_count,
                len(dt_val),
                int(len(dt_train) * config.max_samples_frac),
            ),
        ),
    )
    dt_test = torch.utils.data.Subset(
        dt_test,
        range(
            0,
            min(
                config.max_samples_count,
                len(dt_test),
                int(len(dt_train) * config.max_samples_frac),
            ),
        ),
    )
    return dt_train, dt_val, dt_test


def get_dataloaders(config, dt_train, dt_val, dt_test):
    # seed generators for train & val/test dataloaders
    f, g = torch.Generator(), torch.Generator()
    f.manual_seed(config.rdm_seed + 0)  # note:  this may get re-seeded each epoch
    g.manual_seed(config.rdm_seed)  # keep this one fixed

    # instantiate dataloaders, note: worker_init_fn is needed to get reproducible random samples across runs if vary_samples=True
    train_loader = torch.utils.data.DataLoader(
        dt_train,
        batch_size=config.batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=f,
        num_workers=config.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        dt_val,
        batch_size=config.batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=config.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dt_test,
        batch_size=config.batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
        # num_workers=config.num_workers,
    )

    print(f"Train {len(dt_train)}, Val {len(dt_val)}, Test {len(dt_test)}")

    return train_loader, val_loader, test_loader
