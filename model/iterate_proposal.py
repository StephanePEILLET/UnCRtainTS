
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import torchnet as tnt
import matplotlib.pyplot as plt
import numpy as np

# Suppose these imports exist based on your original file context
from src.learning.metrics import avg_img_metrics, img_metrics
from src.utils import prepare_data, plot_img, export, log_train, log_aleatoric, compute_ece, compute_uce_auce, discrete_matshow

S2_BANDS = 10

class IterationManager:
    """
    Class to handle different iteration modes (Train, Val, Inference) to avoid 
    a single monolithic function.
    """
    def __init__(self, model, config, writer=None, device=None):
        self.model = model
        self.config = config
        self.writer = writer
        self.device = device
        self.loss_meter = tnt.meter.AverageValueMeter()
        self.img_meter = avg_img_metrics()

    def reset_meters(self):
        self.loss_meter.reset()
        self.img_meter.reset()

    def prepare_batch(self, batch):
        """Encapsulates data preparation logic."""
        if self.config.sample_type == "cloudy_cloudfree":
            x, y, in_m, dates = prepare_data(batch, self.device, self.config)
        elif self.config.sample_type == "pretrain":
            x, y, in_m = prepare_data(batch, self.device, self.config)
            dates = None
        else:
            raise NotImplementedError
        
        return {"A": x, "B": y, "dates": dates, "masks": in_m}, x, y, in_m, dates

    def train_epoch(self, data_loader, epoch):
        """
        Original iterate logic adapted for Training only.
        """
        self.reset_meters()
        mode = "train"
        t_start = time.time()

        for i, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch} [Train]")):
            step = (epoch - 1) * len(data_loader) + i
            
            # 1. Prepare Data
            inputs, x, y, in_m, _ = self.prepare_batch(batch)
            
            # 2. Forward & Optimize
            self.model.set_input(inputs)
            self.model.optimize_parameters()
            
            # 3. Retrieve outputs for logging
            out = self.model.fake_B.detach().cpu()
            loss_G = self.model.loss_G.item()
            self.loss_meter.add(loss_G)

            # 4. Handle Variance / Uncertainty
            var = self._get_variance(out)
            out = out[:, :, :S2_BANDS, ...]

            # 5. Logging & Plotting (Periodic)
            if step % self.config.display_step == 0:
                self._log_training_step(step, x, out, y, in_m, var)
            
            if self.config.plot_every > 0:
                self._plot_batch(i, epoch, mode, x, y, out, in_m, var)

            plt.close("all")

        return self._end_of_epoch_stats(mode, t_start, step)

    def evaluate(self, data_loader, epoch, mode="val"):
        """
        Original iterate logic adapted for Validation/Test with metrics.
        """
        self.reset_meters()
        t_start = time.time()
        
        # Lists for detailed metrics
        errs, errs_se, errs_ae, vars_aleatoric = [], [], [], []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch} [{mode}]")):
                step = (epoch - 1) * len(data_loader) + i
                
                # 1. Prepare Data
                inputs, x, y, in_m, _ = self.prepare_batch(batch)

                # 2. Forward
                self.model.set_input(inputs)
                self.model.forward()
                self.model.get_loss_G()
                self.model.rescale()
                
                out = self.model.fake_B
                self.loss_meter.add(self.model.loss_G.item())

                # 3. Variance & Output splitting
                var = self._get_variance(out)
                out = out[:, :, :S2_BANDS, ...] 

                # 4. Compute Metrics (Heavy operation)
                batch_stats = self._compute_batch_metrics(y, out, var)
                
                # Aggregate metrics
                for stat in batch_stats:
                    self.img_meter.add(stat['extended_metrics'])
                    if 'covar' in stat: # GNLL specific
                        vars_aleatoric.append(stat['mean_var'])
                        errs.append(stat['error'])
                        errs_se.append(stat['mean_se'])
                        errs_ae.append(stat['mean_ae'])

                # 5. Plotting / Exporting
                self._plot_and_export_val(i, epoch, mode, x, y, out, in_m, var)
                plt.close("all")

        # End of evaluation logging
        metrics = self._end_of_epoch_stats(mode, t_start)
        
        # Advanced Uncertainty Metrics (ECE, etc.)
        if self.config.loss in ["GNLL", "MGNLL"] and len(vars_aleatoric) > 0:
            self._compute_and_log_uncertainty_metrics(
                vars_aleatoric, errs_se, errs, len(data_loader.dataset), mode, step
            )

        return metrics, self.img_meter.value()

    def inference_full_series(self, data_loader, target_date=None):
        """
        NEW: Iterate souhaite.
        - Pas de collection de métriques lourdes.
        - Inférence sur l'ensemble de la série temporelle.
        - Possibilité de faire une inférence sur une date spécifique.
        """
        mode = "inference"
        print(f"Starting inference mode. Target date: {target_date if target_date else 'Full Series'}")
        
        results = []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader, desc="Inference")):
                # 1. Prepare Data
                # Note: We might need a specific prepare_data for full series if shapes differ
                inputs, x, _, in_m, dates = self.prepare_batch(batch)
                
                # Filter by date if requested
                if target_date is not None:
                    # TODO: Implement date filtering logic here based on 'dates' tensor or metadata
                    pass

                # 2. Forward
                self.model.set_input(inputs)
                self.model.forward()
                self.model.rescale() # Ensure output is in correct range
                
                out = self.model.fake_B
                
                # 3. Variance
                var = self._get_variance(out)
                out = out[:, :, :S2_BANDS, ...]

                # 4. Simple Export / Store results (No heavy metrics)
                # Instead of calc metrics, we typically want to return the reconstructed images
                # or save them directly.
                batch_res = {
                    "input": x.cpu(),
                    "output": out.cpu(),
                    "mask": in_m.cpu(),
                    "variance": var.cpu() if var is not None else None
                }
                results.append(batch_res)
                
        return results

    # --- Helper Methods to clean up the main loops ---

    def _get_variance(self, out):
        """Extracts variance from model or output tensor."""
        if hasattr(self.model.netG, "variance") and self.model.netG.variance is not None:
            var = self.model.netG.variance
            if not isinstance(var, torch.Tensor): # Handle if it's already on CPU/Detach
                 pass
            self.model.netG.variance = None # Reset
        else:
            var = out[:, :, S2_BANDS:, ...]
        return var

    def _log_training_step(self, step, x, out, y, in_m, var):
        out_cpu, x_cpu, y_cpu, in_m_cpu = out.cpu(), x.cpu(), y.cpu(), in_m.cpu()
        if self.config.loss in ["GNLL", "MGNLL"]:
            var_cpu = var.cpu() if var is not None else None
            log_train(self.writer, self.config, self.model, step, x_cpu, out_cpu, y_cpu, in_m_cpu, var=var_cpu)
        else:
            log_train(self.writer, self.config, self.model, step, x_cpu, out_cpu, y_cpu, in_m_cpu)

    def _compute_batch_metrics(self, y, out, var):
        """Computes metrics for a batch."""
        stats = []
        batch_size = y.size()[0]
        
        # Pre-process generic variance if needed for GNLL
        covar = None
        if self.config.loss in ["GNLL", "MGNLL"] and len(var.shape) > 5:
            covar = var
            var = var.diagonal(dim1=2, dim2=3).moveaxis(-1, 2)

        for bdx in range(batch_size):
            res = {}
            if self.config.loss in ["GNLL", "MGNLL"]:
                extended_metrics = img_metrics(y[bdx], out[bdx], var=var[bdx])
                res['covar'] = True
                res['mean_var'] = extended_metrics["mean var"]
                res['error'] = extended_metrics["error"]
                res['mean_se'] = extended_metrics["mean se"]
                res['mean_ae'] = extended_metrics["mean ae"]
            else:
                extended_metrics = img_metrics(y[bdx], out[bdx])
            
            res['extended_metrics'] = extended_metrics
            stats.append(res)
        return stats

    def _plot_batch(self, i, epoch, mode, x, y, out, in_m, var):
        """Helper to handle plotting logic."""
        batch_size = y.size()[0]
        # Logic to plot only every config.plot_every is handled inside the loop usually
        # but here we pass the check before calling.
        # This simplifies: we just plot the first element of batch or iterate if strict index matching is needed.
        pass # (Moved verbose plotting logic here if implemented fully)

    def _plot_and_export_val(self, i, epoch, mode, x, y, out, in_m, var):
        """Handles the visualization/export logic for validation/test."""
        batch_size = y.size()[0]
        for bdx in range(batch_size):
            idx = i * batch_size + bdx
            
            if self.config.plot_every > 0 and idx % self.config.plot_every == 0:
                plot_dir = os.path.join(
                    self.config.res_dir,
                    self.config.experiment_name,
                    "plots",
                    f"epoch_{epoch}",
                    f"{mode}",
                )
                # Call existing plotting functions
                plot_img(x[bdx], "in", plot_dir, file_id=idx)
                plot_img(out[bdx], "pred", plot_dir, file_id=idx)
                
                # ... (rest of plotting logic)

    def _end_of_epoch_stats(self, mode, t_start, step=None):
        t_end = time.time()
        total_time = t_end - t_start
        print(f"Epoch time : {total_time:.1f}s")
        metrics = {f"{mode}_epoch_time": total_time}
        metrics[f"{mode}_loss"] = self.loss_meter.value()[0]

        if mode == "train":
            current_lr = self.model.optimizer_G.state_dict()["param_groups"][0]["lr"]
            self.writer.add_scalar("Etc/train/lr", current_lr, step)
            self.model.scheduler_G.step()
        
        return metrics

    def _compute_and_log_uncertainty_metrics(self, vars_aleatoric, errs_se, errs, n_samples, mode, step):
        """Computes ECE, UCE, AUCE."""
        sorted_errors_se = compute_ece(
            vars_aleatoric, errs_se, n_samples, percent=5
        )
        # ... Log logic from original code ...
        uce_l2, auce_l2 = compute_uce_auce(
            self.writer,
            vars_aleatoric,
            errs,
            n_samples,
            percent=5,
            l2=True,
            mode=mode,
            step=step,
        )
        self.img_meter.value()["UCE SE"] = uce_l2.cpu().numpy().item()
        self.img_meter.value()["AUCE SE"] = auce_l2.cpu().numpy().item()

# Wrapper function to maintain backward compatibility if needed
def iterate_v2(model, data_loader, config, writer, mode="train", epoch=None, device=None):
    manager = IterationManager(model, config, writer, device)
    if mode == "train":
        return manager.train_epoch(data_loader, epoch)
    else:
        return manager.evaluate(data_loader, epoch, mode)

