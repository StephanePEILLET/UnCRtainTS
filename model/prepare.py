import torch

from model.utils_training import recursive_todevice


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
