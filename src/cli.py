import os
import sys
from argparse import ArgumentParser

from omegaconf import OmegaConf

from data.constants.circa_constants import S2_BANDS
from utils_misc import config_utils

# Do not overwrite the following flags by their respective values in the config file
RESUME_TRAINING_NO_OVERWRITE_ARGS = [
    "pid",
    "num_workers",
    "root1",
    "root2",
    "root3",
    "resume_from",
    "trained_checkp",
    "epochs",
    "encoder_widths",
    "decoder_widths",
    "lr",
]

INFERENCE_NO_OVERWRITE_ARGS = [
    "pid",
    "device",
    "resume_at",
    "trained_checkp",
    "save_dir",
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


def setup_parser(mode="train"):
    parser = ArgumentParser(
        description="UnCRtainTS: Uncertainty Quantification for Cloud Removal in Optical Satellite Time Series (Training)",
    )
    parser.add_argument("config_file", type=str, help=f"yaml configuration file to augment/overwrite the settings in configs/_default_{mode}.yaml")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to the directory where models and logs should be saved")
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    return parser


def args_to_config(args, mode="train"):
    assert mode in ["train", "test"], "Mode can be only train or test"
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"ERROR: Cannot find the yaml configuration file: {args.config_file}")

    # Import the user configuration file
    cfg_custom = config_utils.read_config(args.config_file)

    if not cfg_custom:
        sys.exit(1)

    # Augment/overwrite the default parameter settings with the runtime arguments given by the user
    cfg_default = config_utils.read_config(f"configs/_default_{mode}.yaml")
    config = OmegaConf.merge(cfg_default, cfg_custom)
    config.save_dir = args.save_dir
    return handle_parameters_config(config, mode=mode)


def handle_parameters_config(config, mode="train"):
    if config.model in ["unet", "utae"]:
        assert len(config.encoder_widths) == len(config.decoder_widths)
        config.loss = "l2"
        if config.model == "unet":
            # train U-Net from scratch
            config.pretrain = True
            config.trained_checkp = ""
    if config.data.get("pretrain", False):  # pre-training is on a single time point
        config.input_t = config.n_head = 1
        config.sample_type = "pretrain"
        if config.model == "unet":
            config.batch_size = 32
        config.positional_encoding = False
    if config.loss in ["GNLL", "MGNLL"]:
        # for univariate losses, default to univariate mode (batched across channels)
        if config.loss in ["GNLL"]:
            config.covmode = "uni"
        if config.covmode == "iso":
            config.out_conv[-1] += 1
        elif config.covmode in ["uni", "diag"]:
            config.out_conv[-1] += S2_BANDS
            config.var_nonLinearity = "softplus"
    # grab the PID so we can look it up in the logged config for server-side process management
    config.pid = os.getpid()
    # import & re-load a previous configuration, e.g. to resume training
    if config.resume_from:
        if config.experiment_name != config.trained_checkp.split("/")[-2]:
            raise ValueError("Mismatch of loaded config file and checkpoints")
        load_conf = os.path.join(config.save_dir, config.experiment_name, "conf.json")
        cfg_resume = config_utils.read_config(load_conf)
        for key in RESUME_TRAINING_NO_OVERWRITE_ARGS:
            if key in cfg_resume:
                del cfg_resume[key]
        config = OmegaConf.merge(config, cfg_resume)

    # resume at a specified epoch and update optimizer accordingly
    if config.resume_at >= 0:
        config.lr = config.lr * config.gamma**config.resume_at

    return config


def parse_config(mode="train"):
    parser = setup_parser(mode=mode)
    return args_to_config(args=parser.parse_args(), mode=mode)
