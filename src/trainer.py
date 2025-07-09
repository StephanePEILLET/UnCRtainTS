import os
import time

import torch
import torchnet as tnt
from matplotlib import pyplot as plt
from tqdm import tqdm

from data.constants.circa_constants import S2_BANDS
from src.logger import (
    export,
    log_train,
    log_validate,
)
from src.model.learning.metrics import avg_img_metrics, img_metrics
from src.plot import (
    discrete_matshow,
    plot_img,
)
from src.utils_training import recursive_todevice


def prepare_data(batch, device, config):
    if config.get("pretrain", False):
        return prepare_data_mono(batch, device, config)
    else:
        return prepare_data_multi(batch, device, config)


def prepare_data_mono(batch, device, config):
    x = batch["input"]["S2"].to(device).unsqueeze(1)
    if config.data.use_sar:
        x = torch.cat((batch["input"]["S1"].to(device).unsqueeze(1), x), dim=2)
    m = batch["input"]["masks"].to(device).unsqueeze(1)
    y = batch["target"]["S2"].to(device).unsqueeze(1)
    dates = None  # no dates for pre-training
    return x, y, m, dates


def prepare_data_multi(batch, device, config):
    in_S2 = recursive_todevice(batch["input"]["S2"], device)
    in_S2_td = recursive_todevice(batch["input"]["S2 TD"], device)
    if config.batch_size > 1:
        in_S2_td = torch.stack(in_S2_td).T
    in_m = recursive_todevice(batch["input"]["masks"], device).swapaxes(0, 1)  # .squeeze(2)
    target_S2 = recursive_todevice(batch["target"]["S2"], device)
    y = target_S2  # torch.cat(target_S2, dim=0).unsqueeze(1)

    if config.data.use_sar:
        in_S1 = recursive_todevice(batch["input"]["S1"], device)
        in_S1_td = recursive_todevice(batch["input"]["S1 TD"], device)
        if config.batch_size > 1:
            in_S1_td = torch.stack(in_S1_td).T
        x = torch.cat([in_S1, in_S2], dim=2)  # torch.cat((torch.stack(in_S1, dim=1), torch.stack(in_S2, dim=1)), dim=2)
        dates = torch.cat([in_S1_td, in_S2_td]).type(torch.float64).mean().to(device)
        # dates = torch.stack((torch.tensor(in_S1_td), torch.tensor(in_S2_td))).float().mean(dim=0).to(device)
    else:
        x = torch.stack(in_S2, dim=1)
        dates = torch.tensor(in_S2_td).float().to(device)

    return x, y, in_m, dates


def iterate(
    model,
    data_loader,
    config,
    writer,
    s2_bands=S2_BANDS,
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
        x, y, in_m, dates = prepare_data(batch, config=config, device=torch.device(config.device))
        inputs = {"A": x, "B": y, "dates": dates, "masks": in_m}

        if mode != "train":  # val or test
            if len(y.shape) == 4:
                y = y.unsqueeze(1)
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

                # print(f"Batch size: {batch_size}, Output shape: {out.shape}, Var shape: {var.shape if var is not None else 'None'}")
                # print(f"Input shape: {x.shape}, Target shape: {y.shape}, Mask shape: {in_m.shape}")
                for bdx in range(batch_size):
                    # only compute statistics on variance estimates if using e.g. NLL loss or combinations thereof

                    if config.loss in ["GNLL", "MGNLL"]:
                        # if the variance variable is of shape [B x 1 x C x C x H x W] then it's a covariance tensor
                        if len(var.shape) > 5:
                            covar = var
                            # get [B x 1 x C x H x W] variance tensor
                            var = var.diagonal(dim1=2, dim2=3).moveaxis(-1, 2)
                        
                        # print(f"out bdx shape: {out[bdx].shape}, y bdx shape: {y[bdx].shape}, var shape: {var[bdx].shape}")
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
                            config.save_dir,
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
                            config.save_dir,
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
                var = out[:, :, S2_BANDS:, ...]
            out = out[:, :, :S2_BANDS, ...]

            if config.plot_every > 0:
                plot_out = out.detach().clone()
                batch_size = y.size()[0]
                for bdx in range(batch_size):
                    idx = i * batch_size + bdx  # plot and export every k-th item
                    if idx % config.plot_every == 0:
                        plot_dir = os.path.join(
                            config.save_dir,
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
        img_meter = log_validate(
            writer=writer,config=config,img_meter=img_meter,metrics=metrics,mode=mode,step=step,x=x,out=out,
            y=y,in_m=in_m,var=var,data_loader=data_loader,errs=errs,vars_aleatoric=vars_aleatoric,errs_se=errs_se,
        )
        return metrics, img_meter.value()
    else:
        return metrics
