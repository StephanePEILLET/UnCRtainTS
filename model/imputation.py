"""
Main script for image reconstruction experiments
Author: Patrick Ebel (github/PatrickTUM), based on the scripts of
        Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""

import argparse
import json
import os
import pprint
import random
import sys
import time

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(dirname))

import torch
import torchnet as tnt
from parse_args import create_parser
from model.src.config_utils import print_config_rich
from omegaconf import OmegaConf

from src import losses, utils
from src.learning.metrics import avg_img_metrics, img_metrics
from src.learning.weight_init import weight_init
from src.model_utils import (
    freeze_layers,
    get_model,
    load_checkpoint,
    load_model,
    save_model,
)
from torch.utils.tensorboard import SummaryWriter
from data.uncrtaints_adapter import UnCRtainTS_CIRCA_Adapter
from data.dataLoader import SEN12MSCR, SEN12MSCRTS

import model.train_reconstruct as legacy

S2_BANDS = 10

def iterate_v2(model, data_loader, config, writer, mode="train", epoch=None, device=None):

    if len(data_loader) == 0:
        raise ValueError("Received data loader with zero samples!")

    t_start = time.time()
    for i, batch in enumerate(tqdm(data_loader)): 
        assert config.sample_type == "generic":

        if config.sample_type == "cloudy_cloudfree":
            x, y, in_m, dates = legacy.prepare_data(batch, device, config)
        elif config.sample_type == "pretrain":
            x, y, in_m = legacy.prepare_data(batch, device, config)
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
                    var = out[:, :, S2_BANDS:, ...]
                out = out[:, :, :S2_BANDS, ...]
                batch_size = y.size()[0]

                for bdx in range(batch_size):

        else:  # training
           raise NotImplementedError
        
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

    if mode in {"test", "val"}:
        # log the metrics

        # log image metrics
        for key, val in img_meter.value().items():
            writer.add_scalar(f"{mode}/{key}", val, step)

        # any loss is currently only computed within model.backward_G() at train time
        writer.add_scalar(f"{mode}/loss", metrics[f"{mode}_loss"], step)

        # use add_images for batch-wise adding across temporal dimension
        if config.use_sar:
            writer.add_image(
                f"Img/{mode}/in_s1", x[0, :, [0], ...], step, dataformats="NCHW"
            )
            writer.add_image(
                f"Img/{mode}/in_s2", x[0, :, [5, 4, 3], ...], step, dataformats="NCHW"
            )
        else:
            writer.add_image(
                f"Img/{mode}/in_s2", x[0, :, [3, 2, 1], ...], step, dataformats="NCHW"
            )
        writer.add_image(
            f"Img/{mode}/out", out[0, 0, [3, 2, 1], ...], step, dataformats="CHW"
        )
        writer.add_image(
            f"Img/{mode}/y", y[0, 0, [3, 2, 1], ...], step, dataformats="CHW"
        )
        writer.add_image(
            f"Img/{mode}/m", in_m[0, :, None, ...], step, dataformats="NCHW"
        )

        # compute Expected Calibration Error (ECE)
        if config.loss in ["GNLL", "MGNLL"]:
            sorted_errors_se = legacy.compute_ece(
                vars_aleatoric, errs_se, len(data_loader.dataset), percent=5
            )
            sorted_errors = {"se_sortAleatoric": sorted_errors_se}
            legacy.plot_discard(
                writer, sorted_errors["se_sortAleatoric"], config, mode, step, is_se=True
            )

            # compute ECE
            uce_l2, auce_l2 = legacy.compute_uce_auce(
                writer,
                vars_aleatoric,
                errs,
                len(data_loader.dataset),
                percent=5,
                l2=True,
                mode=mode,
                step=step,
            )

            # no need for a running mean here
            img_meter.value()["UCE SE"] = uce_l2.cpu().numpy().item()
            img_meter.value()["AUCE SE"] = auce_l2.cpu().numpy().item()

        if config.loss in ["GNLL", "MGNLL"]:
            legacy.log_aleatoric(writer, config, mode, step, var, "model/", img_meter)

        return metrics, img_meter.value()
    else:
        return metrics