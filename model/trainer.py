import os
import time

import torch
import torchnet as tnt
from matplotlib import pyplot as plt
from src.learning.metrics import avg_img_metrics, img_metrics
from tqdm import tqdm

from model.logger import (
    export,
    log_aleatoric,
    log_train,
)
from model.metrics import (
    compute_ece,
    compute_uce_auce,
)
from model.plot import (
    discrete_matshow,
    plot_discard,
    plot_img,
)
from model.prepare import prepare_data


def iterate(
    model,
    data_loader,
    config,
    writer,
    s2_bands,
    mode="train",
    epoch=None,
    device=None,
):

    if len(data_loader) == 0:
        raise ValueError("Received data loader with zero samples!")
    # loss meter, needs 1 meter per scalar (see https://tnt.readthedocs.io/en/latest/_modules/torchnet/meter/averagevaluemeter.html);
    loss_meter = tnt.meter.AverageValueMeter()
    img_meter = avg_img_metrics()

    # collect sample-averaged uncertainties and errors
    errs, errs_se, errs_ae, vars_aleatoric = [], [], [], []

    t_start = time.time()
    for i, batch in enumerate(tqdm(data_loader)):
        step = (epoch - 1) * len(data_loader) + i

        if config.sample_type == "cloudy_cloudfree":
            x, y, in_m, dates = prepare_data(batch, device, config)
        elif config.sample_type == "pretrain":
            x, y, in_m = prepare_data(batch, device, config)
            dates = None
        else:
            raise NotImplementedError
        inputs = {"A": x, "B": y, "dates": dates, "masks": in_m}

        if mode != "train":  # val or test
            with torch.no_grad():
                # compute single-model mean and variance predictions
                model.set_input(inputs)
                model.forward()
                model.get_loss_G()
                model.rescale()
                out = model.fake_B
                if hasattr(model.netG, "variance") and model.netG.variance is not None:
                    var = model.netG.variance
                    model.netG.variance = None
                else:
                    var = out[:, :, s2_bands:, ...]
                out = out[:, :, :s2_bands, ...]
                batch_size = y.size()[0]

                for bdx in range(batch_size):
                    # only compute statistics on variance estimates if using e.g. NLL loss or combinations thereof

                    if config.loss in ["GNLL", "MGNLL"]:
                        # if the variance variable is of shape [B x 1 x C x C x H x W] then it's a covariance tensor
                        if len(var.shape) > 5:
                            covar = var
                            # get [B x 1 x C x H x W] variance tensor
                            var = var.diagonal(dim1=2, dim2=3).moveaxis(-1, 2)

                        extended_metrics = img_metrics(y[bdx], out[bdx], var=var[bdx])
                        vars_aleatoric.append(extended_metrics["mean var"])
                        errs.append(extended_metrics["error"])
                        errs_se.append(extended_metrics["mean se"])
                        errs_ae.append(extended_metrics["mean ae"])
                    else:
                        extended_metrics = img_metrics(y[bdx], out[bdx])

                    img_meter.add(extended_metrics)
                    idx = i * batch_size + bdx  # plot and export every k-th item
                    if config.plot_every > 0 and idx % config.plot_every == 0:
                        plot_dir = os.path.join(
                            config.res_dir,
                            config.experiment_name,
                            "plots",
                            f"epoch_{epoch}",
                            f"{mode}",
                        )
                        plot_img(x[bdx], "in", plot_dir, file_id=idx)
                        plot_img(out[bdx], "pred", plot_dir, file_id=idx)
                        plot_img(y[bdx], "target", plot_dir, file_id=idx)
                        plot_img(
                            ((out[bdx] - y[bdx]) ** 2).mean(1, keepdims=True),
                            "err",
                            plot_dir,
                            file_id=idx,
                        )
                        plot_img(
                            discrete_matshow(
                                in_m.float().mean(axis=1).cpu()[bdx],
                                n_colors=config.input_t,
                            ),
                            "mask",
                            plot_dir,
                            file_id=idx,
                        )
                        if var is not None:
                            plot_img(
                                var.mean(2, keepdims=True)[bdx],
                                "var",
                                plot_dir,
                                file_id=idx,
                            )
                    if config.export_every > 0 and idx % config.export_every == 0:
                        export_dir = os.path.join(
                            config.res_dir,
                            config.experiment_name,
                            "export",
                            f"epoch_{epoch}",
                            f"{mode}",
                        )
                        export(out[bdx], "pred", export_dir, file_id=idx)
                        export(y[bdx], "target", export_dir, file_id=idx)
                        if var is not None:
                            try:
                                export(covar[bdx], "covar", export_dir, file_id=idx)
                            except Exception:
                                export(var[bdx], "var", export_dir, file_id=idx)
        else:  # training
            # compute single-model mean and variance predictions
            model.set_input(inputs)
            model.optimize_parameters()  # not using model.forward() directly
            out = model.fake_B.detach().cpu()

            # read variance predictions stored on generator
            if hasattr(model.netG, "variance") and model.netG.variance is not None:
                var = model.netG.variance.cpu()
            else:
                var = out[:, :, s2_bands:, ...]
            out = out[:, :, :s2_bands, ...]

            if config.plot_every > 0:
                plot_out = out.detach().clone()
                batch_size = y.size()[0]
                for bdx in range(batch_size):
                    idx = i * batch_size + bdx  # plot and export every k-th item
                    if idx % config.plot_every == 0:
                        plot_dir = os.path.join(
                            config.res_dir,
                            config.experiment_name,
                            "plots",
                            f"epoch_{epoch}",
                            f"{mode}",
                        )
                        plot_img(x[bdx], "in", plot_dir, file_id=i)
                        plot_img(plot_out[bdx], "pred", plot_dir, file_id=i)
                        plot_img(y[bdx], "target", plot_dir, file_id=i)

        if mode == "train":
            # periodically log stats
            if step % config.display_step == 0:
                out, x, y, in_m = out.cpu(), x.cpu(), y.cpu(), in_m.cpu()
                if config.loss in ["GNLL", "MGNLL"]:
                    var = var.cpu()
                    log_train(writer, config, model, step, x, out, y, in_m, var=var)
                else:
                    log_train(writer, config, model, step, x, out, y, in_m)

        # log the loss, computed via model.backward_G() at train time & via model.get_loss_G() at val/test time
        loss_meter.add(model.loss_G.item())
        # after each batch, close any leftover figures
        plt.close("all")

    # --- end of epoch ---
    # after each epoch, log the loss metrics
    t_end = time.time()
    total_time = t_end - t_start
    print(f"Epoch time : {total_time:.1f}s")
    metrics = {f"{mode}_epoch_time": total_time}
    # log the loss, only computed within model.backward_G() at train time
    metrics[f"{mode}_loss"] = loss_meter.value()[0]

    if mode == "train":  # after each epoch, update lr acc. to scheduler
        current_lr = model.optimizer_G.state_dict()["param_groups"][0]["lr"]
        writer.add_scalar("Etc/train/lr", current_lr, step)
        model.scheduler_G.step()

    if mode in {"test", "val"}:
        # log the metrics

        # log image metrics
        for key, val in img_meter.value().items():
            writer.add_scalar(f"{mode}/{key}", val, step)

        # any loss is currently only computed within model.backward_G() at train time
        writer.add_scalar(f"{mode}/loss", metrics[f"{mode}_loss"], step)

        # use add_images for batch-wise adding across temporal dimension
        if config.use_sar:
            writer.add_image(f"Img/{mode}/in_s1", x[0, :, [0], ...], step, dataformats="NCHW")
            writer.add_image(f"Img/{mode}/in_s2", x[0, :, [5, 4, 3], ...], step, dataformats="NCHW")
        else:
            writer.add_image(f"Img/{mode}/in_s2", x[0, :, [3, 2, 1], ...], step, dataformats="NCHW")
        writer.add_image(f"Img/{mode}/out", out[0, 0, [3, 2, 1], ...], step, dataformats="CHW")
        writer.add_image(f"Img/{mode}/y", y[0, 0, [3, 2, 1], ...], step, dataformats="CHW")
        writer.add_image(f"Img/{mode}/m", in_m[0, :, None, ...], step, dataformats="NCHW")

        # compute Expected Calibration Error (ECE)
        if config.loss in ["GNLL", "MGNLL"]:
            sorted_errors_se = compute_ece(vars_aleatoric, errs_se, len(data_loader.dataset), percent=5)
            sorted_errors = {"se_sortAleatoric": sorted_errors_se}
            plot_discard(sorted_errors["se_sortAleatoric"], config, mode, step, is_se=True)

            # compute ECE
            uce_l2, auce_l2 = compute_uce_auce(vars_aleatoric, errs, len(data_loader.dataset), percent=5, l2=True, mode=mode, step=step)

            # no need for a running mean here
            img_meter.value()["UCE SE"] = uce_l2.cpu().numpy().item()
            img_meter.value()["AUCE SE"] = auce_l2.cpu().numpy().item()

        if config.loss in ["GNLL", "MGNLL"]:
            log_aleatoric(writer, config, mode, step, var, "model/", img_meter)

        return metrics, img_meter.value()
    else:
        return metrics
