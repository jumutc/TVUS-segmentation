"""
Boundary-aware Tversky loss for binary segmentation.

The boundary is defined in the same way as MONAI NSD / SurfaceDiceMetric with
default settings: binary erosion XOR mask (edge pixels).

Each pixel receives a regression target equal to its Euclidean distance to the
nearest boundary pixel, normalized to [0, 1] per sample by the maximum distance.
The model is expected to predict this normalized distance map via an additional
output channel (passed through sigmoid).

The final loss combines standard Tversky loss with normalized boundary-distance
regression, both in [0, 1]:

    loss = (1 - boundary_lambda) * tversky + boundary_lambda * distance_regression
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import torch
import torch.nn.functional as F
from monai.losses.utils import compute_tp_fp_fn
from monai.networks import one_hot
from monai.transforms.utils import distance_transform_edt
from monai.utils import LossReduction
from torch.nn.modules.loss import _Loss


def _erode_binary(mask: torch.Tensor, spatial_dims: int) -> torch.Tensor:
    """Morphological erosion for soft/binary masks using min-pooling."""
    if spatial_dims == 2:
        return -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    if spatial_dims == 3:
        return -F.max_pool3d(-mask, kernel_size=3, stride=1, padding=1)
    raise ValueError(f"Unsupported spatial_dims={spatial_dims}. Only 2D and 3D are supported.")


def foreground_boundary_mask(mask: torch.Tensor, spatial_dims: int = 2) -> torch.Tensor:
    """
    Extract boundary pixels using MONAI NSD-style erosion XOR.

    Args:
        mask: Binary or soft mask, shape (B, C, H, W) or (B, C, D, H, W).
    """
    hard = mask > 0.5
    eroded = _erode_binary(hard.float(), spatial_dims=spatial_dims) > 0.5
    return (hard != eroded).float()


@torch.no_grad()
def boundary_distance_map(
    target: torch.Tensor,
    spatial_dims: int = 2,
) -> torch.Tensor:
    """
    Build a per-pixel distance map from the ground-truth mask.

    Each pixel stores the Euclidean distance to the nearest boundary pixel.
    Pixels on the boundary receive distance 0.
    """
    distance_map = torch.zeros_like(target)
    for batch_idx in range(target.shape[0]):
        for channel_idx in range(target.shape[1]):
            mask = target[batch_idx, channel_idx]
            boundary = foreground_boundary_mask(mask.unsqueeze(0).unsqueeze(0), spatial_dims=spatial_dims)[0, 0]

            if not boundary.any():
                continue

            dist = distance_transform_edt((~boundary.bool()).unsqueeze(0))[0]
            distance_map[batch_idx, channel_idx] = dist.float()

    return distance_map


def uses_boundary_distance_regression(losses) -> bool:
    """Return True if any configured loss requires a distance regression output channel."""
    return any(getattr(loss, "boundary_lambda", 0.0) > 0.0 for loss in losses)


def resolve_model_out_channels(losses, out_channels: int) -> int:
    """
    Ensure the model emits enough channels for configured losses.

    Boundary distance regression adds one channel on top of the segmentation head.
    For binary segmentation this means ``out_channels=1`` is bumped to ``2``.
    """
    if uses_boundary_distance_regression(losses) and out_channels < 2:
        return 2
    return out_channels


class BoundaryWeightedTverskyLoss(_Loss):
    """
    Tversky loss combined with boundary-distance regression.

    When ``boundary_lambda > 0``, the model input must include one extra channel
    after the segmentation channels that predicts normalized distance to the
    nearest boundary pixel in [0, 1] (sigmoid applied inside the loss). For
    binary segmentation with target shape ``(B, 1, ...)``, the input should have
    shape ``(B, 2, ...)``.
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        alpha: float = 0.5,
        beta: float = 0.5,
        reduction: LossReduction | str = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        soft_label: bool = False,
        boundary_lambda: float = 0.5,
        spatial_dims: int = 2,
        distance_loss: str = "smooth_l1",
        boundary_sigma: float | None = None,  # deprecated, kept for backward-compatible configs
    ) -> None:
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        if not 0.0 <= boundary_lambda <= 1.0:
            raise ValueError("boundary_lambda must be in [0, 1].")
        if distance_loss not in {"l1", "mse", "smooth_l1"}:
            raise ValueError('distance_loss must be one of ["l1", "mse", "smooth_l1"].')
        _ = boundary_sigma

        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.alpha = alpha
        self.beta = beta
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch
        self.soft_label = soft_label
        self.boundary_lambda = float(boundary_lambda)
        self.spatial_dims = spatial_dims
        self.distance_loss = distance_loss

    def _activate_input(self, input: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        return input

    def _prepare_inputs(self, input: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input = self._activate_input(input)

        n_pred_ch = input.shape[1]
        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has differing shape ({target.shape}) from input ({input.shape})")

        return input, target

    def _split_segmentation_and_distance(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.boundary_lambda <= 0.0:
            return input, None

        n_seg_channels = target.shape[1]
        if input.shape[1] == n_seg_channels:
            raise ValueError(
                f"Expected {n_seg_channels + 1} input channels "
                f"(segmentation + distance regression) when boundary_lambda > 0, "
                f"got {input.shape[1]}."
            )
        if input.shape[1] != n_seg_channels + 1:
            raise ValueError(
                f"Expected exactly one distance regression channel. "
                f"Got {input.shape[1]} input channels for {n_seg_channels} segmentation channels."
            )

        seg_input = input[:, :n_seg_channels]
        distance_input = input[:, n_seg_channels:]
        return seg_input, distance_input

    def _tversky_score(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            reduce_axis = [0] + reduce_axis

        tp, fp, fn = compute_tp_fp_fn(input, target, reduce_axis, 1, self.soft_label, False)
        fp *= self.alpha
        fn *= self.beta
        numerator = tp + self.smooth_nr
        denominator = tp + fp + fn + self.smooth_dr
        return 1.0 - numerator / denominator

    @staticmethod
    def _normalize_distance_map(distance_map: torch.Tensor) -> torch.Tensor:
        """Map per-pixel distances to [0, 1] using the per-sample maximum distance."""
        spatial_axes = tuple(range(2, distance_map.ndim))
        max_distance = distance_map.amax(dim=spatial_axes, keepdim=True).clamp_min(1.0)
        return distance_map / max_distance

    def _distance_regression_score(
        self,
        distance_pred: torch.Tensor,
        distance_target: torch.Tensor,
    ) -> torch.Tensor:
        distance_target = self._normalize_distance_map(distance_target)
        distance_pred = torch.sigmoid(distance_pred)

        if self.distance_loss == "l1":
            pixel_loss = F.l1_loss(distance_pred, distance_target, reduction="none")
        elif self.distance_loss == "mse":
            pixel_loss = F.mse_loss(distance_pred, distance_target, reduction="none")
        else:
            pixel_loss = F.smooth_l1_loss(distance_pred, distance_target, reduction="none")

        spatial_axes = tuple(range(2, pixel_loss.ndim))
        return pixel_loss.mean(dim=spatial_axes)

    def _reduce(self, score: torch.Tensor) -> torch.Tensor:
        if self.reduction == LossReduction.SUM.value:
            return torch.sum(score)
        if self.reduction == LossReduction.NONE.value:
            return score
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(score)
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        seg_input_raw = input
        if self.boundary_lambda > 0.0:
            n_seg_channels = target.shape[1]
            if seg_input_raw.shape[1] < n_seg_channels + 1:
                raise ValueError(
                    f"Expected at least {n_seg_channels + 1} raw input channels "
                    f"when boundary_lambda > 0, got {seg_input_raw.shape[1]}."
                )
            distance_input_raw = seg_input_raw[:, n_seg_channels:]
            seg_input_raw = seg_input_raw[:, :n_seg_channels]
        else:
            distance_input_raw = None

        seg_input, target = self._prepare_inputs(seg_input_raw, target)
        tversky_score = self._tversky_score(seg_input, target)

        if self.boundary_lambda <= 0.0:
            return self._reduce(tversky_score)

        if distance_input_raw is None:
            raise RuntimeError("Distance input channel is missing.")

        distance_target = boundary_distance_map(
            target,
            spatial_dims=self.spatial_dims,
        )
        distance_score = self._distance_regression_score(distance_input_raw, distance_target)

        if self.boundary_lambda == 1.0:
            return self._reduce(distance_score)

        combined_score = (1.0 - self.boundary_lambda) * tversky_score + self.boundary_lambda * distance_score
        return self._reduce(combined_score)
