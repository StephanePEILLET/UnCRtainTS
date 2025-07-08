import json
import os
import sys

import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from data.data_module import UnCRtainTS_datamodule
from src.cli import parse_config
from src.logger import (
    checkpoint,
    prepare_output,
    save_results,
)
from src.model import losses, utils
from src.model.learning.weight_init import weight_init
from src.model.model_utils import (
    freeze_layers,
    get_model,
    load_checkpoint,
    load_model,
    save_model,
)
from src.trainer import iterate
from src.utils_training import seed_packages, seed_worker
from utils_misc import config_utils


def main():
    config = parse_config(mode="train")
    # seed everything
    seed_packages(config.rdm_seed)

    # pprint.pprint(config)
    config_utils.print_config_rich(config=config)

    # Chargement des dataloaders
    dm = UnCRtainTS_datamodule(config)

    # instantiate tensorboard logger
    writer = SummaryWriter(os.path.join(os.path.dirname(config.save_dir), "logs", config.experiment_name))
    prepare_output(config)
    device = torch.device(config.device)

    # model definition
    # (compiled model hangs up in validation step on some systems, retry in the future for pytorch > 2.0)
    model = get_model(config)  # torch.compile(get_model(config))

    # set model properties
    model.len_epoch = len(dm.dt_train)

    config.N_params = utils.get_ntrainparams(model)
    # print("\n\nTrainable layers:")
    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(f"\t{name}")
    model = model.to(device)
    # do random weight initialization
    print("\nInitializing weights randomly.")
    model.netG.apply(weight_init)

    if config.trained_checkp and len(config.trained_checkp) > 0:
        # load weights from the indicated checkpoint
        print(f"Loading weights from (pre-)trained checkpoint {config.trained_checkp}")
        load_model(
            config,
            model,
            train_out_layer=True,
            load_out_partly=config.model in ["uncrtaints"],
        )

    with open(os.path.join(config.save_dir, config.experiment_name, "conf.json"), "w") as file:
        file.write(json.dumps(OmegaConf.to_container(config, resolve=True), indent=4))
    print(f"TOTAL TRAINABLE PARAMETERS: {config.N_params}\n")
    print(model)

    # Optimizer and Loss
    model.criterion = losses.get_loss(config)

    # track best loss, checkpoint at best validation performance
    is_better, best_loss = lambda new, prev: new <= prev, float("inf")

    # Training loop
    trainlog = {}

    # Dataloaders
    dl_train = dm.train_dataloader()
    dl_val = dm.val_dataloader()
    dl_test = dm.test_dataloader()

    # resume training at scheduler's latest epoch, != 0 if --resume_from
    begin_at = config.resume_at if config.resume_at >= 0 else model.scheduler_G.state_dict()["last_epoch"]
    for epoch in range(begin_at + 1, config.epochs + 1):
        print(f"\nEPOCH {epoch}/{config.epochs}")

        # put all networks in training mode again
        model.train()
        model.netG.train()

        # unfreeze all layers after specified epoch
        if epoch > config.unfreeze_after and hasattr(model, "frozen") and model.frozen:
            print("Unfreezing all network layers")
            model.frozen = False
            freeze_layers(model.netG, grad=True)

        # re-seed train generator for each epoch anew, depending on seed choice plus current epoch number
        #   ~ else, dataloader provides same samples no matter what epoch training starts/resumes from
        #   ~ note: only re-seed train split dataloader (if config.data.vary_samples), but keep all others consistent
        #   ~ if desiring different runs, then the seeds must at least be config.epochs numbers apart
        if config.data.vary_samples:
            # condition dataloader samples on current epoch count
            f = torch.Generator()
            f.manual_seed(config.rdm_seed + epoch)
            dl_train = torch.utils.data.DataLoader(
                dl_train,
                batch_size=config.batch_size,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=f,
                num_workers=config.num_workers,
            )

        train_metrics = iterate(
            model,
            data_loader=dl_train,
            config=config,
            writer=writer,
            s2_bands=dm.dt_train.s2_channels,
            mode="train",
            epoch=epoch,
            device=device,
        )

        # do regular validation steps at the end of each training epoch
        if epoch % config.val_every == 0 and epoch > config.val_after:
            print("Validation . . . ")

            model.eval()
            model.netG.eval()

            val_metrics, val_img_metrics = iterate(
                model,
                data_loader=dl_val,
                config=config,
                writer=writer,
                s2_bands=dm.dt_val.s2_channels,
                mode="val",
                epoch=epoch,
                device=device,
            )

            # use the training loss for validation
            print("Using training loss as validation loss")
            if "val_loss" in val_metrics:
                val_loss = val_metrics["val_loss"]
            else:
                val_loss = val_metrics["val_loss_ensembleAverage"]

            print(f"Validation Loss {val_loss}")
            print(f"validation image metrics: {val_img_metrics}")
            save_results(
                val_img_metrics,
                os.path.join(config.save_dir, config.experiment_name),
                split=f"val_epoch_{epoch}",
            )
            print(f"\nLogged validation epoch {epoch} metrics to path {os.path.join(config.save_dir, config.experiment_name)}")

            # checkpoint best model
            trainlog[epoch] = {**train_metrics, **val_metrics}
            checkpoint(trainlog, config)
            if is_better(val_loss, best_loss):
                best_loss = val_loss
                save_model(config, epoch, model, "model")
        else:
            trainlog[epoch] = {**train_metrics}
            checkpoint(trainlog, config)

        # always checkpoint the current epoch's model
        save_model(config, epoch, model, f"model_epoch_{epoch}")

        print(f"Completed current epoch of experiment {config.experiment_name}.")

    # following training, test on hold-out data
    print("Testing best epoch . . .")
    load_checkpoint(config, config.save_dir, model, "model")

    model.eval()
    model.netG.eval()

    test_metrics, test_img_metrics = iterate(
        model,
        data_loader=dl_test,
        config=config,
        writer=writer,
        s2_bands=dm.dt_test.s2_channels,
        mode="test",
        epoch=epoch,
        device=device,
    )

    if "test_loss" in test_metrics:
        test_loss = test_metrics["test_loss"]
    else:
        test_loss = test_metrics["test_loss_ensembleAverage"]
    print(f"Test Loss {test_loss}")
    print(f"\nTest image metrics: {test_img_metrics}")
    save_results(
        test_img_metrics,
        os.path.join(config.save_dir, config.experiment_name),
        split="test",
    )
    print(f"\nLogged test metrics to path {os.path.join(config.save_dir, config.experiment_name)}")

    # close tensorboard logging
    writer.close()

    print(f"Finished training experiment {config.experiment_name}.")

    sys.exit()


if __name__ == "__main__":
    main()
