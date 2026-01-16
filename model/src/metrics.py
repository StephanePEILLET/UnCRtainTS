import math
from collections.abc import Callable
from enum import Enum
from functools import partial
from typing import Literal
import numpy as np
import torch
import torchgeometry as tgm
from scipy.stats import linregress
from torch import Tensor
import math
from typing import Dict
from typing import List
from typing import Optional
from torch import Tensor
from torchmetrics.aggregation import MeanMetric


class CloudRemovalMetrics:
    """Compute the metrics used to monitor the training progress or for evaluation."""

    class MetricType(Enum):
        MAE = "mae"
        MSE = "mse"
        RMSE = "rmse"
        PSNR = "psnr"
        SSIM = "ssim"
        R2 = "r2"
        SAM = "sam"

    def __init__(
        self,
        metrics: list[str] | None = None,
        eval_occluded_observed: bool = True,
        clean_gt_cloudy_pixels: bool = True,
        sam_units: str = "rad",
        window_size: int = 5,
        max_pixel_intensity: int = 1,
    ) -> None:
        """Initialize the CloudRemovalMetrics class.

        Args:
        ----
            metrics (list[str] | None): List of metric names to compute
                (mae, mse, rmse, psnr, ssim, r2, sam).
                                        If None, all available metrics are used.
            eval_occluded_observed (bool): If True, also computes metrics separately
                for occluded/observed pixels.
            clean_gt_cloudy_pixels (bool): If True, excludes cloudy pixels
                from the ground truth during evaluation.
            sam_units (str): Units for the SAM metric ("rad" or "deg").
            window_size (int): Window size for SSIM computation.

        """
        # True to evaluate the metrics over all pixels and separately
        # for occluded and observed input pixels;
        # False to evaluate the metrics over all pixels only
        self.eval_occluded_observed = eval_occluded_observed
        self.clean_gt_cloudy_pixels = clean_gt_cloudy_pixels
        self.sam_units = sam_units
        self.window_size = window_size
        self.max_pixel_intensity = max_pixel_intensity
        # Initialize metric functions
        self.metric_fns: dict[CloudRemovalMetrics.MetricType, callable] = {}
        metrics_enum = self._parse_metrics(metrics)
        self._init_metric_functions(metrics_enum)

    @staticmethod
    def _parse_metrics(
        metrics: list[str] | None,
    ) -> list["CloudRemovalMetrics.MetricType"]:
        """Parse and validate the list of metric names into MetricType objects.

        Args:
        ----
            metrics: List of metric names (case-insensitive) or None.

        Returns:
        -------
            Ordered list (without duplicates) of MetricType values.

        Raises:
        ------
            ValueError: If one or more metric names are invalid.

        """
        if metrics is None:
            return list(CloudRemovalMetrics.MetricType)
        allowed = {m.value: m for m in CloudRemovalMetrics.MetricType}
        seen = set()
        ordered: list[CloudRemovalMetrics.MetricType] = []
        invalid: list[str] = []
        for m in metrics:
            key = m.lower()
            if key not in allowed:
                if key not in invalid:  # collect each invalid only once
                    invalid.append(key)
                continue
            enum_val = allowed[key]
            if enum_val not in seen:
                seen.add(enum_val)
                ordered.append(enum_val)
        if invalid:
            msg = (
                f"Unknown metric name(s): {invalid}. ",
                f"Valid metrics are: {sorted(allowed.keys())}",
            )
            raise ValueError(msg)
        return ordered

    def _init_metric_functions(
        self,
        metrics: list["CloudRemovalMetrics.MetricType"],
    ) -> None:
        """Initialize the metric functions based on the provided list of metric types.

        Args:
        ----
            metrics (list[MetricType]): The list of metrics to initialize.

        """
        metric_set = set(metrics)
        MetricType = CloudRemovalMetrics.MetricType  # noqa: N806  # local alias for brevity

        if MetricType.MAE in metric_set:
            self.metric_fns[MetricType.MAE] = lambda p, t: torch.mean(torch.abs(p - t))

        if MetricType.MSE in metric_set:
            self.metric_fns[MetricType.MSE] = lambda p, t: torch.mean(
                torch.square(p - t),
            )

        if MetricType.RMSE in metric_set:
            self.metric_fns[MetricType.RMSE] = lambda p, t: torch.sqrt(
                self.metric_fns[MetricType.MSE](p, t),
            )

        if MetricType.SSIM in metric_set:
            self.metric_fns[MetricType.SSIM] = tgm.losses.SSIM(
                self.window_size,
                reduction="mean",
            )

        if MetricType.PSNR in metric_set:
            self.metric_fns[MetricType.PSNR] = lambda p, t: 20 * torch.log10(
                self.max_pixel_intensity / self.metric_fns[MetricType.RMSE](p, t),
            )

        if MetricType.SAM in metric_set:
            self.metric_fns[MetricType.SAM] = partial(
                self._compute_sam,
                units=self.sam_units,
            )

        if MetricType.R2 in metric_set:
            self.metric_fns[MetricType.R2] = lambda p, t: self.r2_score_torch(
                y_target=t.flatten(),
                y_pred=p.flatten(),
            )

    @staticmethod
    def _compute_sam(
        predicted: Tensor,
        target: Tensor,
        units: Literal["deg", "rad"] = "rad",
    ) -> Tensor:
        """Compute the spectral angle mapper (SAM).

        Averaged over all time steps and batch samples.

        Args:
        ----
            predicted:   torch.Tensor,  (n_frames x C x H x W).
            target:      torch.Tensor,  (n_frames x C x H x W).
            units:       Literal["deg", "rad"].

        Returns:
        -------
            sam_value:   torch.Tensor, (1, ), mean spectral angle [rad].

        """
        dot_product = (predicted * target).sum(dim=1)
        predicted_norm = predicted.norm(dim=1)
        target_norm = target.norm(dim=1)
        # Compute the SAM score for all pixels with vector norm > 0
        flag = torch.logical_and(predicted_norm != 0.0, target_norm != 0.0)
        if torch.any(flag):
            spectral_angles = torch.clamp(
                dot_product[flag] / (predicted_norm[flag] * target_norm[flag]),
                -1,
                1,
            ).acos()
            sam_score = torch.mean(spectral_angles)
            if units == "deg":
                sam_score *= 180 / math.pi
            return sam_score
        return None

    def _compute_pixelwise_metric(
        self,
        metric_name: str,
        metric_fn: Callable,
        predicted: Tensor,
        target: Tensor,
        masks: Tensor,
    ) -> dict[str, float]:
        """Compute a single pixel-wise metric and its variants.

        For occluded and observed regions.

        Args:
        ----
            metric_name (str): The name of the metric (e.g., 'mae', 'rmse').
            metric_fn (callable): The function to compute the metric.
            predicted (Tensor): The predicted tensor of shape (N, C),
                where N is the number of pixels.
            target (Tensor): The target tensor of shape (N, C).
            masks (Tensor): The mask tensor of shape (N, C),
                where 1 indicates occlusion.

        Returns:
        -------
            dict[str, float]: A dictionary containing the computed metric
                              for all pixels,
                              and optionally for occluded and observed pixels.

        """
        metrics = {}
        try:
            metrics[metric_name] = metric_fn(predicted, target)
        except ValueError:
            metrics[metric_name] = np.nan

        if self.eval_occluded_observed:
            occluded_mask = (masks == 1.0).any(dim=1)
            if occluded_mask.any():
                try:
                    metrics[f"{metric_name}_occluded_input_pixels"] = metric_fn(
                        predicted[occluded_mask],
                        target[occluded_mask],
                    )
                except ValueError:
                    metrics[f"{metric_name}_occluded_input_pixels"] = np.nan
            else:
                metrics[f"{metric_name}_occluded_input_pixels"] = np.nan

            observed_mask = (masks == 0.0).all(dim=1)
            if observed_mask.any():
                try:
                    metrics[f"{metric_name}_observed_input_pixels"] = metric_fn(
                        predicted[observed_mask],
                        target[observed_mask],
                    )
                except ValueError:
                    metrics[f"{metric_name}_observed_input_pixels"] = np.nan
            else:
                metrics[f"{metric_name}_observed_input_pixels"] = np.nan

        return metrics

    @staticmethod
    def r2_score_torch(
        y_pred: Tensor,
        y_target: Tensor,
    ) -> Tensor:
        """Compute the R² (coefficient of determination) score using PyTorch.

        The R² score is a statistical measure of how well the regression predictions
        approximate the real data points. An R² of 1 indicates that the predictions
        perfectly fit the data.

        The formula is: R² = 1 - (SS_res / SS_tot)
        where:
        - SS_res (Residual Sum of Squares) is the sum of squared differences:
            Σ(y_target - y_pred)²
        - SS_tot (Total Sum of Squares) is the sum of squared differences from the mean:
            Σ(y_target - mean(y_target))²

        Args:
        ----
            y_pred (Tensor): The tensor containing the predicted values from the model.
            y_target (Tensor): The tensor containing the ground truth (target) values.
                            Must have the same shape as y_pred.

        Returns:
        -------
            Tensor: A scalar tensor containing the R² score.

        """
        if y_pred.shape != y_target.shape:
            msg = (
                "Input tensor shapes must be the same. ",
                f"Got y_pred={y_pred.shape} and y_target={y_target.shape}",
            )
            raise ValueError(msg)

        if y_pred.shape[0] <= 1:
            return None

        result = linregress(y_target.cpu(), y_pred.cpu())
        return result.rvalue**2

    def _compute_imagewise_metrics(
        self,
        predicted: Tensor,
        target: Tensor,
        masks: Tensor,
    ) -> dict[str, float]:
        """Compute image-wise metrics like SSIM.

        Args:
        ----
            predicted (Tensor): The predicted tensor of shape (B*T, C, H, W).
            target (Tensor): The target tensor of shape (B*T, C, H, W).
            masks (Tensor): The mask tensor of shape (B*T, C, H, W).

        Returns:
        -------
            dict[str, float]: A dictionary containing the computed image-wise metrics.

        """
        metrics = {}
        if CloudRemovalMetrics.MetricType.SSIM in self.metric_fns:
            dssim = self.metric_fns[CloudRemovalMetrics.MetricType.SSIM](
                predicted,
                target,
            )
            metrics["ssim"] = 1 - 2 * dssim

            # SSIM per band
            _, C, _, _ = predicted.shape  # noqa: N806
            for band in range(C):
                dssim_band = self.metric_fns[CloudRemovalMetrics.MetricType.SSIM](
                    predicted[:, band : band + 1],
                    target[:, band : band + 1],
                )
                metrics[f"ssim_band_{band}"] = 1 - 2 * dssim_band

            if self.eval_occluded_observed:
                occ_images = (masks == 1.0).any(dim=-1).any(dim=-1).any(dim=-1)
                if occ_images.any():
                    metrics["ssim_images_occluded_input_pixels"] = 1 - 2 * self.metric_fns[
                        CloudRemovalMetrics.MetricType.SSIM
                    ](
                        predicted[occ_images],
                        target[occ_images],
                    )

                    for band in range(C):
                        metrics[f"ssim_images_occluded_input_pixels_band_{band}"] = 1 - 2 * self.metric_fns[
                            CloudRemovalMetrics.MetricType.SSIM
                        ](
                            predicted[occ_images][:, band : band + 1],
                            target[occ_images][:, band : band + 1],
                        )
                else:
                    metrics["ssim_images_occluded_input_pixels"] = np.nan
                    metrics.update(
                        {f"ssim_images_occluded_input_pixels_band_{b}": np.nan for b in range(C)},
                    )

                obs_images = ~occ_images
                if obs_images.any():
                    metrics["ssim_images_observed_input_pixels"] = 1 - 2 * self.metric_fns[
                        CloudRemovalMetrics.MetricType.SSIM
                    ](
                        predicted[obs_images],
                        target[obs_images],
                    )

                    for band in range(C):
                        metrics[f"ssim_images_observed_input_pixels_band_{band}"] = 1 - 2 * self.metric_fns[
                            CloudRemovalMetrics.MetricType.SSIM
                        ](
                            predicted[obs_images][:, band : band + 1],
                            target[obs_images][:, band : band + 1],
                        )
                else:
                    metrics["ssim_images_observed_input_pixels"] = np.nan
                    metrics.update(
                        {f"ssim_images_observed_input_pixels_band_{band}": np.nan for b in range(C)},
                    )
        return metrics

    def _compute_channelwise_metrics(
        self,
        predicted: Tensor,
        target: Tensor,
        masks: Tensor,
    ) -> dict[str, float]:
        """Compute channel-wise metrics like SAM for all, occluded, and observed pixels.

        Args:
        ----
            predicted (Tensor): The predicted tensor of shape (N, C),
                where N is the number of pixels.
            target (Tensor): The target tensor of shape (N, C).
            masks (Tensor): The mask tensor of shape (N, C).

        Returns:
        -------
            dict[str, float]: A dictionary containing the computed channel-wise metrics.

        """
        metrics = {}
        if CloudRemovalMetrics.MetricType.SAM in self.metric_fns:
            sam_score = self.metric_fns[CloudRemovalMetrics.MetricType.SAM](
                predicted,
                target,
            )
            if sam_score is not None:
                metrics["sam"] = sam_score

            if self.eval_occluded_observed:
                occluded_mask = (masks == 1.0).any(dim=1)
                if occluded_mask.any():
                    sam_occ = self.metric_fns[CloudRemovalMetrics.MetricType.SAM](
                        predicted[occluded_mask],
                        target[occluded_mask],
                    )
                    if sam_occ is not None:
                        metrics["sam_occluded_input_pixels"] = sam_occ
                else:
                    metrics["sam_occluded_input_pixels"] = np.nan

                observed_mask = (masks == 0.0).all(dim=1)
                if observed_mask.any():
                    sam_obs = self.metric_fns[CloudRemovalMetrics.MetricType.SAM](
                        predicted[observed_mask],
                        target[observed_mask],
                    )
                    if sam_obs is not None:
                        metrics["sam_observed_input_pixels"] = sam_obs
                else:
                    metrics["sam_observed_input_pixels"] = np.nan
        return metrics

    def _compute_pixelwise_metrics(
        self,
        predicted: Tensor,
        target: Tensor,
        masks: Tensor,
    ) -> dict[str, float]:
        """Compute all requested pixel-wise metrics (MAE, MSE, RMSE, PSNR, R2).

        Args:
        ----
            predicted (Tensor): The predicted tensor of shape (N, C),
                where N is the number of pixels.
            target (Tensor): The target tensor of shape (N, C).
            masks (Tensor): The mask tensor of shape (N, C).

        Returns:
        -------
            dict[str, float]: A dictionary containing all computed pixel-wise metrics.

        """
        metrics = {}
        pixel_metrics_to_compute = {
            CloudRemovalMetrics.MetricType.MAE,
            CloudRemovalMetrics.MetricType.MSE,
            CloudRemovalMetrics.MetricType.RMSE,
            CloudRemovalMetrics.MetricType.PSNR,
            CloudRemovalMetrics.MetricType.R2,
        }
        for metric_type in pixel_metrics_to_compute:
            if metric_type in self.metric_fns:
                new_metrics = self._compute_pixelwise_metric(
                    metric_name=metric_type.value,
                    metric_fn=self.metric_fns[metric_type],
                    predicted=predicted,
                    target=target,
                    masks=masks,
                )
                metrics.update(new_metrics)

        return metrics

    def _compute_pixelwise_per_band_metrics(
        self,
        predicted: Tensor,
        target: Tensor,
        masks: Tensor,
    ) -> dict[str, float]:
        """Compute all requested pixel-wise per band metrics (MAE, MSE, RMSE, PSNR, R2).

        Args:
        ----
            predicted (Tensor): The predicted tensor of shape (N, C),
                where N is the number of pixels.
            target (Tensor): The target tensor of shape (N, C).
            masks (Tensor): The mask tensor of shape (N, C).

        Returns:
        -------
            dict[str, float]: A dictionary containing all computed pixel-wise
                per band metrics.

        """
        _, C = predicted.shape  # noqa: N806
        metrics = {}
        pixel_metrics_to_compute = {
            CloudRemovalMetrics.MetricType.MAE,
            CloudRemovalMetrics.MetricType.MSE,
            CloudRemovalMetrics.MetricType.RMSE,
            CloudRemovalMetrics.MetricType.PSNR,
            CloudRemovalMetrics.MetricType.R2,
        }

        for band in range(C):
            for metric_type in pixel_metrics_to_compute:
                if metric_type in self.metric_fns:
                    new_metrics = self._compute_pixelwise_metric(
                        metric_name=f"{metric_type.value}_band_{band}",
                        metric_fn=self.metric_fns[metric_type],
                        predicted=predicted[:, band : band + 1],
                        target=target[:, band : band + 1],
                        masks=masks[:, band : band + 1],
                    )
                    metrics.update(new_metrics)
        return metrics

    def _prepare_pixelwise_tensors(
        self,
        predicted_img: Tensor,
        target_img: Tensor,
        masks_img: Tensor,
        cloud_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Prepare and reshapes tensors for pixel-wise metric computation.

        This involves flattening the image-like tensors into pixel-lists and optionally
        filtering out pixels that are marked as cloudy in the ground truth.

        Args:
        ----
            predicted_img (Tensor): The predicted tensor of shape (B*T, C, H, W).
            target_img (Tensor): The target tensor of shape (B*T, C, H, W).
            masks_img (Tensor): The mask tensor of shape (B*T, C, H, W).
            cloud_masks (Tensor): The ground truth cloud mask tensor
                (B x T x 1 x H x W).

        Returns:
        -------
            Tuple[Tensor, Tensor, Tensor]: A tuple containing the prepared pixel-wise
                tensors: (predicted_pix, target_pix, masks_pix).

        """
        n_frames, C, H, W = predicted_img.shape  # noqa: N806

        # Optionally filter out cloudy pixels from the ground truth
        if cloud_masks is not None and self.clean_gt_cloudy_pixels:
            cloud_masks = cloud_masks.view(n_frames, 1, H, W)
            # Create a boolean flag to select only non-cloudy
            # pixels from the ground truth
            flag = (cloud_masks.permute(0, 2, 3, 1).reshape(n_frames * H * W) == 0.0).squeeze()
            predicted_pix = predicted_img.permute(0, 2, 3, 1).reshape(
                n_frames * H * W,
                C,
            )[flag]
            target_pix = target_img.permute(0, 2, 3, 1).reshape(n_frames * H * W, C)[flag,]
            masks_pix = masks_img.permute(0, 2, 3, 1).reshape(n_frames * H * W, C)[flag]
        else:
            # If not filtering, flatten all tensors to a pixel-wise representation
            n_pixels = n_frames * H * W
            predicted_pix = predicted_img.permute(0, 2, 3, 1).reshape(n_pixels, C)
            target_pix = target_img.permute(0, 2, 3, 1).reshape(n_pixels, C)
            masks_pix = masks_img.permute(0, 2, 3, 1).reshape(n_pixels, C)

        return predicted_pix, target_pix, masks_pix

    def __call__(
        self,
        target: Tensor,
        masks: Tensor,
        predicted: Tensor,
        cloud_masks: Tensor = None,
    ) -> dict[str, float]:
        """Compute the specified cloud removal metrics.

        Args:
        ----
            target (Tensor): The ground truth tensor (B x T x C x H x W).
            masks (Tensor): The input mask tensor (B x T x 1 x H x W).
            predicted (Tensor): The model's prediction tensor (B x T x C x H x W).
            cloud_masks (Tensor, optional):
                The ground truth cloud mask (B x T x 1 x H x W).

        Returns:
        -------
            dict[str, float]: A dictionary containing the computed metrics.

        """
        # Detach all input tensors from the computation graph to prevent memory leaks
        predicted = predicted.detach()
        target = target.detach()
        masks = masks.detach()
        if cloud_masks is not None:
            cloud_masks = cloud_masks.detach()

        predicted = predicted.to(torch.float32)
        target = target.to(torch.float32)
        masks = masks.to(torch.float32)
        metrics = {}
        B, T, C, H, W = predicted.shape  # noqa: N806
        n_frames = B * T

        # always compute SSIM (image-wise) before any cloud filtering
        predicted_img_orig = predicted.view(n_frames, C, H, W)
        target_img_orig = target.view(n_frames, C, H, W)
        masks_img_orig = masks.view(n_frames, 1, H, W).expand(target_img_orig.shape)
        metrics.update(
            self._compute_imagewise_metrics(
                predicted_img_orig,
                target_img_orig,
                masks_img_orig,
            ),
        )

        # Step 2: apply cloud filtering for pixel-wise metrics
        if self.clean_gt_cloudy_pixels and cloud_masks is not None:
            clear_pixels_mask = 1 - cloud_masks
            predicted = predicted * clear_pixels_mask
            target = target * clear_pixels_mask
            masks = masks * clear_pixels_mask

        # Step 3: prepare tensors and compute remaining metrics
        predicted_img = predicted.view(n_frames, C, H, W)
        target_img = target.view(n_frames, C, H, W)
        masks_img = masks.view(n_frames, 1, H, W).expand(target_img.shape)
        predicted_pix, target_pix, masks_pix = self._prepare_pixelwise_tensors(
            predicted_img,
            target_img,
            masks_img,
            cloud_masks,
        )
        metrics.update(
            self._compute_pixelwise_metrics(predicted_pix, target_pix, masks_pix),
        )
        metrics.update(
            self._compute_pixelwise_per_band_metrics(
                predicted_pix,
                target_pix,
                masks_pix,
            ),
        )
        metrics.update(
            self._compute_channelwise_metrics(predicted_pix, target_pix, masks_pix),
        )

        # Compute pixelwise per band metrics
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = value.item()
            elif value is None:
                metrics[key] = np.nan
            else:
                metrics[key] = value
        return metrics


class CloudRemovalDatasetMetrics:
    """Aggregate cloud removal metrics over multiple samples using torchmetrics' MeanMetric.

    Usage:
        agg = CloudRemovalDatasetMetrics(metrics=["mae", "rmse", "psnr"])  # or None for all
        for batch in dataloader:
            preds = model(...)
            agg.update(target=batch["y"], masks=batch["masks"], predicted=preds, cloud_masks=batch.get("cloud_mask"))
        results = agg.compute()  # dict metric_name -> mean value
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        eval_occluded_observed: bool = True,
        clean_gt_cloudy_pixels: bool = True,
        max_pixel_intensity: int = 1,
        sam_units: str = "rad",
        window_size: int = 5,
        skip_nan: bool = True,
    ) -> None:
        """Args:
        metrics: List of metric names or None for all.
        eval_occluded_observed: Forwarded to CloudRemovalMetrics.
        clean_gt_cloudy_pixels: Forwarded.
        sam_units: "rad" | "deg".
        window_size: SSIM window size.
        skip_nan: If True, ignore NaN values when updating aggregates.
        """
        self.sample_metrics = CloudRemovalMetrics(
            metrics=metrics,
            eval_occluded_observed=eval_occluded_observed,
            clean_gt_cloudy_pixels=clean_gt_cloudy_pixels,
            sam_units=sam_units,
            window_size=window_size,
            max_pixel_intensity=max_pixel_intensity,
        )
        self.skip_nan = skip_nan
        self._aggregators: Dict[str, MeanMetric] = {}

    def _get_or_create_aggregator(self, name: str) -> MeanMetric:
        if name not in self._aggregators:
            self._aggregators[name] = MeanMetric()
        return self._aggregators[name]

    @torch.no_grad()
    def update(
        self,
        *,
        target: Tensor,
        masks: Tensor,
        predicted: Tensor,
        cloud_masks: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """Compute metrics for one sample/batch and update running means.

        Returns the per-sample metrics dict (not averaged)."""
        predicted = predicted.detach()
        target = target.detach()
        masks = masks.detach()
        if cloud_masks is not None:
            cloud_masks = cloud_masks.detach()

        metrics_dict = self.sample_metrics(target=target, masks=masks, predicted=predicted, cloud_masks=cloud_masks)

        for k, v in metrics_dict.items():
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                if self.skip_nan:
                    continue
            value = torch.tensor(v, dtype=torch.float32)
            if torch.isnan(value) or torch.isinf(value):
                if self.skip_nan:
                    continue
            self._get_or_create_aggregator(k).update(value)
        return metrics_dict

    @torch.no_grad()
    def compute(self) -> Dict[str, float]:
        """Return the mean value for each metric accumulated so far."""
        return {k: agg.compute().item() for k, agg in self._aggregators.items()}

    @torch.no_grad()
    def reset(self) -> None:
        for agg in self._aggregators.values():
            agg.reset()

    # Optional: make the class iterable-friendly with state_dict / load_state_dict
    def state_dict(self) -> Dict[str, Dict[str, Tensor]]:
        return {k: agg.state_dict() for k, agg in self._aggregators.items()}

    def load_state_dict(self, state: Dict[str, Dict[str, Tensor]]):
        for k, sd in state.items():
            agg = self._get_or_create_aggregator(k)
            agg.load_state_dict(sd)
