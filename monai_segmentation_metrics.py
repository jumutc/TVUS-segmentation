"""GPU-friendly validation metrics for binary segmentation using MONAI."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from monai.metrics import DiceMetric, MeanIoU, SurfaceDiceMetric


def _resolve_index(idx) -> int:
    if isinstance(idx, torch.Tensor):
        return int(idx.item())
    if isinstance(idx, (np.ndarray, list)):
        return int(idx[0] if len(idx) > 0 else idx)
    return int(idx)


def _binary_one_hot(mask_b1hw: torch.Tensor) -> torch.Tensor:
    """Convert (B, 1, H, W) mask to (B, 2, H, W) one-hot [background, foreground]."""
    fg = (mask_b1hw > 0).float()
    bg = 1.0 - fg
    return torch.cat([bg, fg], dim=1)


class MonaiValidationMetrics:
    """Accumulates IoU, NSD, and Dice on GPU across a validation epoch."""

    def __init__(self, nsd_tolerance: float = 6.0):
        self.nsd_tolerance = nsd_tolerance
        self.reset()

    def reset(self) -> None:
        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch", ignore_empty=True)
        self.iou_metric = MeanIoU(include_background=False, reduction="mean_batch", ignore_empty=True)
        self.nsd_metric = SurfaceDiceMetric(
            class_thresholds=[self.nsd_tolerance],
            include_background=False,
            reduction="mean_batch",
        )

    @staticmethod
    def _prepare_masks(
        output: torch.Tensor,
        val_df,
        idx,
        device: torch.device,
        pred_threshold: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Binarize logits and align prediction with native-resolution ground truth."""
        seg_output = output[:, :1] if output.shape[1] > 1 else output
        pred = (seg_output > pred_threshold).float()

        row_idx = _resolve_index(idx)
        gt_np = np.squeeze(val_df.loc[row_idx]["seg"])
        gt = torch.as_tensor(gt_np, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        if pred.shape[-2:] != gt.shape[-2:]:
            pred = F.interpolate(pred, size=gt.shape[-2:], mode="nearest")

        return pred, gt

    def update(
        self,
        output: torch.Tensor,
        val_df,
        idx,
        device: torch.device,
        pred_threshold: float = 0.0,
    ) -> None:
        pred, gt = self._prepare_masks(output, val_df, idx, device, pred_threshold=pred_threshold)
        pred_oh = _binary_one_hot(pred)
        gt_oh = _binary_one_hot(gt)

        self.dice_metric(y_pred=pred_oh, y=gt_oh)
        self.iou_metric(y_pred=pred_oh, y=gt_oh)
        self.nsd_metric(y_pred=pred_oh, y=gt_oh, spacing=(1.0, 1.0))

    def aggregate(self) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
        """Return mean IoU, NSD, Dice and per-sample score arrays."""
        iou_scores = self.iou_metric.aggregate().detach().cpu().numpy()
        nsd_scores = self.nsd_metric.aggregate().detach().cpu().numpy()
        dice_scores = self.dice_metric.aggregate().detach().cpu().numpy()

        return (
            float(np.nanmean(iou_scores)),
            float(np.nanmean(nsd_scores)),
            float(np.nanmean(dice_scores)),
            iou_scores,
            nsd_scores,
            dice_scores,
        )
