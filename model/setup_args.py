import argparse
import json
import os

from data.constants.circa_constants import S2_BANDS
from model.src import utils


def setup_config(parser, list_args=["encoder_widths", "decoder_widths", "out_conv"]):
    config = utils.str2list(parser.parse_args(), list_args=list_args)
    if config.model in ["unet", "utae"]:
        assert len(config.encoder_widths) == len(config.decoder_widths)
        config.loss = "l2"
        if config.model == "unet":
            # train U-Net from scratch
            config.pretrain = True
            config.trained_checkp = ""
    if config.pretrain:  # pre-training is on a single time point
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
        load_conf = os.path.join(config.res_dir, config.experiment_name, "conf.json")
        if config.experiment_name != config.trained_checkp.split("/")[-2]:
            raise ValueError("Mismatch of loaded config file and checkpoints")
        with open(load_conf) as f:
            t_args = argparse.Namespace()
            # do not overwrite the following flags by their respective values in the config file
            no_overwrite = [
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
            conf_dict = {key: val for key, val in json.load(f).items() if key not in no_overwrite}
            for key, val in vars(config).items():
                if key in no_overwrite:
                    conf_dict[key] = val
            t_args.__dict__.update(conf_dict)
            config = parser.parse_args(namespace=t_args)
    config = utils.str2list(config, list_args=["encoder_widths", "decoder_widths", "out_conv"])

    # resume at a specified epoch and update optimizer accordingly
    if config.resume_at >= 0:
        config.lr = config.lr * config.gamma**config.resume_at

    return config


def create_parser(mode="train"):
    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument(
        "--model",
        default="uncrtaints",  # e.g. 'unet', 'utae', 'uncrtaints',
        type=str,
        help="Type of architecture to use. Can be one of: (utae/unet3d/fpn/convlstm/convgru/uconvlstm/buconvlstm)",
    )
    parser.add_argument(
        "--experiment_name",
        default="my_first_experiment",
        help="Name of the current experiment",
    )

    # fast switching between default arguments, depending on train versus test mode
    if mode == "train":
        parser.add_argument(
            "--res_dir",
            default="./results",
            help="Path to where the results are stored, e.g. ./results for training or ./inference for testing",
        )
        parser.add_argument(
            "--plot_every",
            default=-1,
            type=int,
            help="Interval (in items) of exporting plots at validation or test time. Set -1 to disable",
        )
        parser.add_argument(
            "--export_every",
            default=-1,
            type=int,
            help="Interval (in items) of exporting data at validation or test time. Set -1 to disable",
        )
        parser.add_argument(
            "--resume_at",
            default=0,
            type=int,
            help="Epoch to resume training from (may re-weight --lr in the optimizer) or epoch to load checkpoint from at test time",
        )
    elif mode == "test":
        parser.add_argument(
            "--res_dir",
            default="./inference",
            type=str,
            help="Path to directory where results are written.",
        )
        parser.add_argument(
            "--plot_every",
            default=-1,
            type=int,
            help="Interval (in items) of exporting plots at validation or test time. Set -1 to disable",
        )
        parser.add_argument(
            "--export_every",
            default=1,
            type=int,
            help="Interval (in items) of exporting data at validation or test time. Set -1 to disable",
        )
        parser.add_argument(
            "--resume_at",
            default=-1,
            type=int,
            help="Epoch to load checkpoint from and run testing with (use -1 for best on validation split)",
        )

    parser.add_argument(
        "--encoder_widths",
        default="[128]",
        type=str,
        help="e.g. [64,64,64,128] for U-TAE or [128] for UnCRtainTS",
    )
    parser.add_argument(
        "--decoder_widths",
        default="[128,128,128,128,128]",
        type=str,
        help="e.g. [64,64,64,128] for U-TAE or [128,128,128,128,128] for UnCRtainTS",
    )
    parser.add_argument(
        "--out_conv",
        default=f"[{S2_BANDS}]",
        help="output CONV, note: if inserting another layer then consider treating normalizations separately",
    )
    parser.add_argument(
        "--mean_nonLinearity",
        dest="mean_nonLinearity",
        action="store_false",
        help="whether to apply a sigmoidal output nonlinearity to the mean prediction",
    )
    parser.add_argument(
        "--var_nonLinearity",
        default="softplus",
        type=str,
        help="how to squash the network's variance outputs [relu | softplus | elu ]",
    )
    parser.add_argument(
        "--agg_mode",
        default="att_group",
        type=str,
        help="type of temporal aggregation in L-TAE module",
    )
    parser.add_argument(
        "--encoder_norm",
        default="group",
        type=str,
        help="e.g. 'group' (when using many channels) or 'instance' (for few channels)",
    )
    parser.add_argument(
        "--decoder_norm",
        default="batch",
        type=str,
        help="e.g. 'group' (when using many channels) or 'instance' (for few channels)",
    )
    parser.add_argument(
        "--block_type",
        default="mbconv",
        type=str,
        help="type of CONV block to use [residual | mbconv]",
    )
    parser.add_argument("--padding_mode", default="reflect", type=str)
    parser.add_argument("--pad_value", default=0, type=float)

    # attention-specific parameters
    parser.add_argument("--n_head", default=16, type=int, help="default value of 16, 4 for debugging")
    parser.add_argument("--d_model", default=256, type=int, help="layers in L-TAE, default value of 256")
    parser.add_argument(
        "--positional_encoding",
        dest="positional_encoding",
        action="store_false",
        help="whether to use positional encoding or not",
    )
    parser.add_argument("--d_k", default=4, type=int)
    parser.add_argument("--low_res_size", default=32, type=int, help="resolution to downsample to")
    parser.add_argument(
        "--use_v",
        dest="use_v",
        action="store_true",
        help="whether to use values v or not",
    )

    # set-up parameters
    parser.add_argument("--num_workers", default=0, type=int, help="Number of data loading workers")
    parser.add_argument("--rdm_seed", default=1, type=int, help="Random seed")
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Name of device to use for tensor computations (cuda/cpu)",
    )
    parser.add_argument(
        "--display_step",
        default=10,
        type=int,
        help="Interval in batches between display of training metrics",
    )

    # training parameters
    parser.add_argument(
        "--loss",
        default="MGNLL",
        type=str,
        help="Image reconstruction loss to utilize [l1|l2|GNLL|MGNLL].",
    )
    parser.add_argument(
        "--resume_from",
        dest="resume_from",
        action="store_true",
        help="resume training acc. to JSON in --experiment_name and *.pth chckp in --trained_checkp",
    )
    parser.add_argument(
        "--unfreeze_after",
        default=0,
        type=int,
        help="When to unfreeze ALL weights for training",
    )
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs to train")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument(
        "--chunk_size",
        type=int,
        help="Size of vmap batches, this can be adjusted to accommodate for additional memory needs",
    )
    parser.add_argument("--lr", default=1e-2, type=float, help="Learning rate, e.g. 0.01")
    parser.add_argument(
        "--gamma",
        default=1.0,
        type=float,
        help="Learning rate decay parameter for scheduler",
    )
    parser.add_argument(
        "--val_every",
        default=1,
        type=int,
        help="Interval in epochs between two validation steps.",
    )
    parser.add_argument(
        "--val_after",
        default=0,
        type=int,
        help="Do validation only after that many epochs.",
    )

    # flags specific to SEN12MS-CR and SEN12MS-CR-TS
    parser.add_argument(
        "--use_sar",
        dest="use_sar",
        action="store_true",
        help="whether to use SAR or not",
    )
    parser.add_argument(
        "--pretrain",
        dest="pretrain",
        action="store_true",
        help="whether to perform pretraining on SEN12MS-CR or training on SEN12MS-CR-TS",
    )
    parser.add_argument(
        "--input_t",
        default=3,
        type=int,
        help="number of input time points to sample, unet3d needs at least 4 time points",
    )
    parser.add_argument(
        "--ref_date",
        default="2014-04-03",
        type=str,
        help="reference date for Sentinel observations",
    )
    parser.add_argument(
        "--sample_type",
        default="cloudy_cloudfree",
        type=str,
        help="type of samples returned [cloudy_cloudfree | generic]",
    )
    parser.add_argument(
        "--vary_samples",
        dest="vary_samples",
        action="store_false",
        help="whether to sample different time points across epochs or not",
    )
    parser.add_argument(
        "--min_cov",
        default=0.0,
        type=float,
        help="The minimum cloud coverage to accept per input sample at train time. Gets overwritten by --vary_samples",
    )
    parser.add_argument(
        "--max_cov",
        default=1.0,
        type=float,
        help="The maximum cloud coverage to accept per input sample at train time. Gets overwritten by --vary_samples",
    )
    parser.add_argument(
        "--root1",
        default="/home/data/SEN12MSCRTS",
        type=str,
        help="path to your copy of SEN12MS-CR-TS",
    )
    parser.add_argument(
        "--root2",
        default="/home/data/SEN12MSCRTS",
        type=str,
        help="path to your copy of SEN12MS-CR-TS validation & test splits",
    )
    parser.add_argument(
        "--root3",
        default="/home/data/SEN12MSCR",
        type=str,
        help="path to your copy of SEN12MS-CR for pretraining",
    )
    parser.add_argument(
        "--precomputed",
        default="/home/code/UnCRtainTS/util/precomputed",
        type=str,
        help="path to pre-computed cloud statistics",
    )
    parser.add_argument(
        "--region",
        default="all",
        type=str,
        help="region to (sub-)sample ROI from [all|africa|america|asiaEast|asiaWest|europa]",
    )
    parser.add_argument(
        "--max_samples_count",
        default=int(1e9),
        type=int,
        help="count of data (sub-)samples to take",
    )
    parser.add_argument(
        "--max_samples_frac",
        default=1.0,
        type=float,
        help="fraction of data (sub-)samples to take",
    )
    parser.add_argument(
        "--profile",
        dest="profile",
        action="store_true",
        help="whether to profile code or not",
    )
    parser.add_argument(
        "--trained_checkp",
        default="",
        type=str,
        help="Path to loading a pre-trained network *.pth file, rather than initializing weights randomly",
    )

    # flags specific to uncertainty modeling
    parser.add_argument(
        "--covmode",
        default="diag",
        type=str,
        help="covariance matrix type [uni|iso|diag].",
    )
    parser.add_argument(
        "--scale_by",
        default=1.0,
        type=float,
        help="rescale data within model, e.g. to [0,10]",
    )
    parser.add_argument(
        "--separate_out",
        dest="separate_out",
        action="store_true",
        help="whether to separately process mean and variance predictions or in a shared layer",
    )

    # flags specific for testing
    parser.add_argument(
        "--weight_folder",
        type=str,
        default="./results",
        help="Path to the main folder containing the pre-trained weights",
    )
    parser.add_argument(
        "--use_custom",
        dest="use_custom",
        action="store_true",
        help="whether to test on individually specified patches or not",
    )
    parser.add_argument("--load_config", default="", type=str, help="path of conf.json file to load")

    return parser
