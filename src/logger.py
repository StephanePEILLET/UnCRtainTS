import json
import os

import numpy as np
import torch

from src.plot import (
    continuous_matshow,
    discrete_matshow,
)


def log_aleatoric(writer, config, mode, step, var, name, img_meter=None):
    # if var is of shape [B x 1 x C x C x H x W] then it's a covariance tensor
    if len(var.shape) > 5:
        covar = var
        # get [B x 1 x C x H x W] variance tensor
        var = var.diagonal(dim1=2, dim2=3).moveaxis(-1, 2)

        # compute spatial-average to visualize patch-wise covariance matrices
        patch_covmat = covar.mean(dim=-1).mean(dim=-1).squeeze(dim=1)
        for bdx, img in enumerate(patch_covmat):  # iterate over [B x C x C] covmats
            img = img.detach().numpy()

            max_abs = max(abs(img.min()), abs(img.max()))
            scale_rel_left, scale_rel_right = -max_abs, +max_abs
            fig = continuous_matshow(img, min=scale_rel_left, max=scale_rel_right)
            writer.add_figure(f"Img/{mode}/patch covmat relative {bdx}", fig, step)
            scale_center0_absolute = 1 / 4 * 1**2  # assuming covmat has been rescaled already, this is an upper bound
            fig = continuous_matshow(img, min=-scale_center0_absolute, max=scale_center0_absolute)
            writer.add_figure(f"Img/{mode}/patch covmat absolute {bdx}", fig, step)

    # aleatoric uncertainty: comput during train, val and test
    # note: the quantile statistics are computed solely over the variances (and would be much different if involving covariances, e.g. in the isotopic case)
    avg_var = torch.mean(var, dim=2, keepdim=True)  # avg over bands, note: this only considers variances (else diag COV's avg would be tiny)
    q50 = avg_var[:, 0, ...].view(avg_var.shape[0], -1).median(dim=-1)[0].detach().clone()
    q75 = avg_var[:, 0, ...].view(avg_var.shape[0], -1).quantile(0.75, dim=-1).detach().clone()
    q50, q75 = q50[0], q75[0]  # take batch's first item as a summary
    binning = 256  # see: https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_histogram

    if config.loss in ["GNLL", "MGNLL"]:
        writer.add_image(
            f"Img/{mode}/{name}aleatoric [0,1]",
            avg_var[0, 0, ...].clip(0, 1),
            step,
            dataformats="CHW",
        )  # map image to [0, 1]
        writer.add_image(
            f"Img/{mode}/{name}aleatoric [0,q75]",
            avg_var[0, 0, ...].clip(0.0, q75) / q75,
            step,
            dataformats="CHW",
        )  # map image to [0, q75]
        writer.add_histogram(
            f"Hist/{mode}/{name}aleatoric",
            avg_var[0, 0, ...].flatten().clip(0, 1),
            step,
            bins=binning,
            max_bins=binning,
        )
    else:
        raise NotImplementedError

    writer.add_scalar(f"{mode}/{name}aleatoric median all", q50, step)
    writer.add_scalar(f"{mode}/{name}aleatoric q75 all", q75, step)
    if img_meter is not None:
        writer.add_scalar(f"{mode}/{name}UCE SE", img_meter.value()["UCE SE"], step)
        writer.add_scalar(f"{mode}/{name}AUCE SE", img_meter.value()["AUCE SE"], step)


def log_train(writer, config, model, step, x, out, y, in_m, name="", var=None):
    # logged loss is before rescaling by learning rate
    _, loss = model.criterion, model.loss_G.cpu()
    if name != "":
        name = f"model_{name}/"

    writer.add_scalar(f"train/{name}{config.loss}", loss, step)
    writer.add_scalar(f"train/{name}total", loss, step)
    # use add_images for batch-wise adding across temporal dimension
    if config.use_sar:
        writer.add_image(f"Img/train/{name}in_s1", x[0, :, [0], ...], step, dataformats="NCHW")
        writer.add_image(f"Img/train/{name}in_s2", x[0, :, [5, 4, 3], ...], step, dataformats="NCHW")
    else:
        writer.add_image(f"Img/train/{name}in_s2", x[0, :, [3, 2, 1], ...], step, dataformats="NCHW")
    writer.add_image(f"Img/train/{name}out", out[0, 0, [3, 2, 1], ...], step, dataformats="CHW")
    writer.add_image(f"Img/train/{name}y", y[0, 0, [3, 2, 1], ...], step, dataformats="CHW")
    writer.add_image(f"Img/train/{name}m", in_m[0, :, None, ...], step, dataformats="NCHW")

    # analyse cloud coverage

    # covered at ALL time points (AND) or covered at ANY time points (OR)
    # and_m, or_m = torch.prod(in_m[0,:, ...], dim=0, keepdim=True), torch.sum(in_m[0,:, ...], dim=0, keepdim=True).clip(0,1)
    and_m, or_m = (
        torch.prod(in_m, dim=1, keepdim=True),
        torch.sum(in_m, dim=1, keepdim=True).clip(0, 1),
    )
    writer.add_scalar(f"train/{name}OR m %", or_m.float().mean(), step)
    writer.add_scalar(f"train/{name}AND m %", and_m.float().mean(), step)
    writer.add_image(f"Img/train/{name}AND m", and_m, step, dataformats="NCHW")
    writer.add_image(f"Img/train/{name}OR m", or_m, step, dataformats="NCHW")

    and_m_gray = in_m.float().mean(axis=1).cpu()
    for bdx, img in enumerate(and_m_gray):
        fig = discrete_matshow(img, n_colors=config.input_t)
        writer.add_figure(f"Img/train/temp overlay m {bdx}", fig, step)

    if var is not None:
        # log aleatoric uncertainty statistics, excluding computation of ECE
        log_aleatoric(writer, config, "train", step, var, name, img_meter=None)


def export(arrs, mod, export_dir, file_id=None):
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    for tdx, arr in enumerate(arrs):  # iterate over temporal dimension
        num = "" if arrs.shape[0] == 1 else f"_t-{tdx}"
        np.save(os.path.join(export_dir, f"img-{file_id}_{mod}{num}.npy"), arr.cpu())


def prepare_output(config):
    os.makedirs(os.path.join(config.save_dir, config.experiment_name), exist_ok=True)


def checkpoint(log, config):
    with open(os.path.join(config.save_dir, config.experiment_name, "trainlog.json"), "w") as outfile:
        json.dump(log, outfile, indent=4)


def save_results(metrics, path, split="test"):
    with open(os.path.join(path, f"{split}_metrics.json"), "w") as outfile:
        json.dump(metrics, outfile, indent=4)
