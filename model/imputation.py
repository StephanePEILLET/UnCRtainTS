import os
import sys
import time
import numpy as np
from tqdm import tqdm

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(dirname))

import torch
from model.train_reconstruct import recursive_todevice
from data.utils.process_functions import process_MS
from data.utils.process_functions import process_SAR
from data.utils.process_functions import reverse_process_MS
from model.src.metrics import CloudRemovalDatasetMetrics

S2_BANDS = 10
MAX_PIXEL_INTENSITY_USED_FOR_REVERSE = 10_000


def iterate_one_sample(batch,model, config, device=None):
    # in_S2 shape BS * T * C * H * W => [1, T, C, H, W]
    assert "idx_syn_aleatoire" in batch, "Full sequence inference requires 'idx_syn_aleatoire' in the batch"
    masks = batch["masks"]
    idx_syn_masks = (batch["idx_syn_aleatoire"]).squeeze().numpy().tolist()
    idx_good_frames = batch["idx_good_frames"].squeeze().numpy()
    idx_valid_obs = torch.from_numpy(np.union1d(idx_good_frames, idx_syn_masks).astype(int))
    syn_masks = torch.zeros_like(masks)
    if isinstance(idx_syn_masks, int): # correction si un seul indice afin d'éviter les erreurs lié à array à 0 dimensions
        idx_syn_masks = [idx_syn_masks]

    if isinstance(idx_syn_masks, list) and len(idx_syn_masks) > 0:
        for i in idx_syn_masks:
            syn_masks[:, i, ...] = torch.ones_like(syn_masks[:, i, ...])

    data_S2 = batch["S2"]

    # Application des masques synthétiques sur aux données d'entrée
    if isinstance(idx_syn_masks, list) and len(idx_syn_masks) > 0:
        data_S2 = data_S2 * (1 - syn_masks)

    # Sélection des dates valides (observées ou synthétiques)
    data_S2 = data_S2[:, idx_valid_obs, ...]
    raw_S2 = batch["S2"][:, idx_valid_obs, ...]
    syn_masks = syn_masks[:, idx_valid_obs, ...]
    prob_masks = masks[:, idx_valid_obs, ...]

    in_S2 = torch.from_numpy(process_MS(data_S2))
    if config.use_sar:
        in_S1 = torch.from_numpy(process_SAR(batch["S1"][:, idx_valid_obs, ...]))
    data = torch.cat([in_S2, in_S1], dim=2) if config.use_sar else in_S2
    in_S2_td = torch.tensor(batch["S2 TD"])[idx_valid_obs]

    T = in_S2.shape[1]
    windows = np.lib.stride_tricks.sliding_window_view(np.arange(T), window_shape=config.input_t).tolist()
    idx_targets = [window[len(window) // 2] for window in windows]

    outputs = []
    for window, idx_target in zip(windows, idx_targets):
        # in_S2 shape BS * [window] * C * H * W => [1, [window], C, H, W]
        x = recursive_todevice(data[:, window, ...], device)
        y = recursive_todevice(in_S2[:, idx_target, ...].unsqueeze(1), device)
        in_m = recursive_todevice(syn_masks[:, window, ...].swapaxes(0, 1), device)
        dates = recursive_todevice(in_S2_td[window], device)
        # Prepare model inputs and forward pass
        inputs = {"A": x, "B": y, "dates": dates, "masks": in_m}
        with torch.no_grad():
            model.set_input(inputs)
            model.forward()
            model.get_loss_G()
            model.rescale()
            out = model.fake_B
            out = out[:, :, :S2_BANDS, ...]
            outputs.append(out.cpu())

    outputs = torch.cat(outputs, axis=1)  # Concatenate along time dimension
    preds = reverse_process_MS(outputs, intensity_min=0, intensity_max=MAX_PIXEL_INTENSITY_USED_FOR_REVERSE) # repass in int16 range

    # Recoupage des données pour ne garder que les dates cibles prédites
    targets = raw_S2[:, idx_targets, ...]
    syn_masks = syn_masks[:, idx_targets, ...]
    prob_masks = prob_masks[:, idx_targets, ...]

    # return targets, syn_masks, preds, prob_masks, data_S2
    return {
        "targets": targets,
        "syn_masks": syn_masks,
        "preds": preds,
        "prob_masks": prob_masks,
        "inputs": data_S2
    }

def iterate_full_sequence(model, data_loader, config, device=None):
    # Vérification que l'on se trouve bien dans un contexte d'inférence
    # in_S2 shape BS * T * C * H * W
    # in_S2_td BS * T
    # in_m shape BS * T * C * H * W
    if len(data_loader) == 0:
        raise ValueError("Received data loader with zero samples!")
    assert config.sample_type == "generic"
    assert data_loader.batch_size == 1, "Full sequence inference only implemented for batch size 1"

    # --- start of epoch ---
    t_start = time.time()

    metrics = ["mae", "mse", "rmse", "psnr", "ssim", "r2", "sam"]
    set_metrics = CloudRemovalDatasetMetrics(
        metrics=metrics,
        eval_occluded_observed=True,
        clean_gt_cloudy_pixels=True,
        max_pixel_intensity=MAX_PIXEL_INTENSITY_USED_FOR_REVERSE,
    )    
    for i, batch in enumerate(tqdm(data_loader)):
        batch = iterate_one_sample(batch, model, config, device=device)
        set_metrics.update(
            target=batch["targets"],
            masks=batch["syn_masks"],
            predicted=batch["preds"],
            cloud_masks=batch["prob_masks"],
        )
    
    # --- end of epoch ---
    t_end = time.time()
    total_time = t_end - t_start
    print(f"Epoch time : {total_time:.1f}s")

    return set_metrics.compute()

