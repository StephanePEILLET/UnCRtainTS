"""
Script for image reconstruction inference with pre-trained models
Author: Patrick Ebel (github/PatrickTUM), based on the scripts of
        Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""

import argparse
import json
import os
import pprint
import sys
from pathlib import Path

import torch
from parse_args import create_parser

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(dirname))

from src import utils
from src.model_utils import get_model, load_checkpoint
from torch.utils.tensorboard import SummaryWriter
from train_reconstruct import (
    # iterate,
    prepare_output,
    save_results,
    seed_packages,
)

from data.uncrtaints_adapter import UnCRtainTS_CIRCA_Adapter

parser = create_parser(mode="test")
test_config = parser.parse_args()
test_config.pid = os.getpid()

if "config_file" in test_config and os.path.isfile(test_config.config_file):
    from model.src.config_utils import read_config
    from omegaconf import OmegaConf
    params_conf_file = read_config(test_config.config_file)
    test_config.weight_folder = params_conf_file.weight_folder
    test_config.experiment_name = params_conf_file.experiment_name
    test_config.batch_size = params_conf_file.batch_size
else:
    params_conf_file = None

# load previous config from training directories
conf_path = (
    os.path.join(
        dirname, test_config.weight_folder, test_config.experiment_name, "conf.json"
    )
    if not test_config.load_config
    else test_config.load_config
)
if os.path.isfile(conf_path):
    with open(conf_path) as file:
        model_config = json.loads(file.read())
        t_args = argparse.Namespace()
        # do not overwrite the following flags by their respective values in the config file
        no_overwrite = [
            "pid",
            "device",
            "resume_at",
            "trained_checkp",
            "res_dir",
            "weight_folder",
            "root1",
            "root2",
            "root3",
            "max_samples_count",
            "batch_size",
            "display_step",
            "plot_every",
            "export_every",
            "input_t",
            "region",
            "min_cov",
            "max_cov",
        ]
        conf_dict = {
            key: val for key, val in model_config.items() if key not in no_overwrite
        }
        for key, val in vars(test_config).items():
            if key in no_overwrite:
                conf_dict[key] = val
        t_args.__dict__.update(conf_dict)
        config = parser.parse_args(namespace=t_args)
else:
    config = test_config  # otherwise, keep passed flags without any overwriting
config = utils.str2list(config, ["encoder_widths", "decoder_widths", "out_conv"])

# seed everything
seed_packages(config.rdm_seed)
# instantiate tensorboard logger

def main(config):
    device = torch.device(config.device)
    config.use_sar = True

    model = get_model(config)
    model = model.to(device)
    config.N_params = utils.get_ntrainparams(model)
    print(f"TOTAL TRAINABLE PARAMETERS: {config.N_params}\n")

    if params_conf_file is not None:
        config.__dict__.update(params_conf_file)
    prepare_output(config)
    experiment_dir = os.path.join(config.res_dir, config.experiment_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    with open(os.path.join(experiment_dir, "conf.json"), "w") as file:
        file.write(json.dumps(vars(config), indent=4))

    S1_LAUNCH: str = "2014-04-03"
    CLEAR_THRESHOLD: float = 1e-3  # Threshold for considering a scene as cloud-free

    circa_ds = UnCRtainTS_CIRCA_Adapter(
        phase="test",
        hdf5_file=config.hdf5_file,
        shuffle=config.shuffle,
        use_sar=config.use_sar,
        channels=config.channels,
        compute_cloud_mask=False,
        # paramaters specific to UnCRtainTS
        cloud_masks="s2cloudless_mask",
        sample_type=config.sample_type,
        sampler=config.sampler,
        n_input_samples=config.input_t,
        rescale_method="default",
        min_cov=0.0,
        max_cov=1.0,
        ref_date=S1_LAUNCH,
        seq_length=30, #TODO comment rendre cela adaptatif à la donnée ?
        clear_threshold=CLEAR_THRESHOLD,
        vary_samples=False,
    )

    dt_test = torch.utils.data.Subset(
        circa_ds, range(0, min(config.max_samples_count, len(circa_ds)))
    )

    test_loader = torch.utils.data.DataLoader(
        dt_test, 
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
    )

    # Load weights
    ckpt_n = f"_epoch_{config.resume_at}" if config.resume_at > 0 else ""
    # load_checkpoint(config, config.weight_folder, model, f"model{ckpt_n}")
    # /DATA_10TB/data_rpg/outputs/UnCRtainTS/results/train/UnCRtainTS_bs_7_MGNLL_experiment/model.pth.tar
    chckp_path = os.path.join(config.weight_folder, config.experiment_name, f"model.pth.tar") 
    print(f"Loading checkpoint {chckp_path}")
    checkpoint = torch.load(chckp_path, map_location=config.device)["state_dict"]

    try:  # try loading checkpoint strictly, all weights & their names must match
        model.load_state_dict(checkpoint, strict=True)
        print("Loaded checkpoint with matching keys")
    except:
        # rename keys
        #   in_block1 -> in_block0, out_block1 -> out_block0
        checkpoint_renamed = dict()
        for key, val in checkpoint.items():
            if "in_block" in key or "out_block" in key:
                strs = key.split(".")
                strs[1] = strs[1][:-1] + str(int(strs[1][-1]) - 1)
                strs[1] = ".".join([strs[1][:-1], strs[1][-1]])
                key = ".".join(strs)
            checkpoint_renamed[key] = val
        model.load_state_dict(checkpoint_renamed, strict=False)

    # Inference
    print("Testing . . .")
    model.eval()

    from model.imputation import iterate_full_sequence
    test_img_metrics = iterate_full_sequence(
        model=model,
        data_loader=test_loader,
        config=config,
        device=device,
    )
    print(f"\nTest image metrics: {test_img_metrics}")

    save_results(
        test_img_metrics,
        os.path.join(config.res_dir, config.experiment_name),
        split="test",
    )
    print(
        f"\nLogged test metrics to path {os.path.join(config.res_dir, config.experiment_name)}"
    )


if __name__ == "__main__":
    main(config)
    sys.exit()
