"""3D segmentation metric utilities (MONAI / PyTorch).

Adapted from GS-contrast-flow for nerfstudio x-ray methods.
Uses MONAI metrics for GPU-accelerated computation.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from monai.losses.cldice import SoftclDiceLoss
from scipy.spatial import KDTree
from skimage.morphology import skeletonize
from monai.metrics import (
    compute_average_precision,          # type: ignore[import]
    compute_confusion_matrix_metric,    # type: ignore[import]
    compute_dice,                       # type: ignore[import]
    compute_hausdorff_distance,         # type: ignore[import]
    compute_roc_auc,                    # type: ignore[import]
    get_confusion_matrix,               # type: ignore[import]
)

_soft_cldice_fn = SoftclDiceLoss(iter_=3, smooth=1.0)


def _skeleton_coords(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> np.ndarray | None:
    """Skeletonize *mask* and return point coordinates scaled by *spacing*."""
    sk = skeletonize(mask, method="lee")
    if sk.sum() == 0:
        return None
    coords = np.argwhere(sk).astype(np.float64)
    coords *= np.array(spacing, dtype=np.float64)
    return coords


class SegmentationMetricsComputer:
    """Cached 3D segmentation metric computer.

    Initialised *once* with ground-truth data; reuse across multiple prediction
    volumes and binarisation thresholds to avoid redundant GT device transfers,
    skeletonisations, and KD-tree constructions.
    """

    def __init__(
        self,
        gt: np.ndarray,
        aabb_roi: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> None:
        """
        Args:
            gt: (D, H, W) bool numpy array — ground-truth label.
            aabb_roi: (D, H, W) bool numpy array — AABB region of interest.
            spacing: Voxel spacing ``(sz, sy, sx)`` in mm.
        """
        self._gt_np = gt
        self._aabb_roi_np = aabb_roi
        self._spacing = spacing
        self._gt_sum = int(gt.sum())

        gt_skel = skeletonize(gt, method="lee")
        if gt_skel.sum() > 0:
            gt_coords = np.argwhere(gt_skel).astype(np.float64)
            gt_coords *= np.array(spacing, dtype=np.float64)
            self._gt_skel_coords = gt_coords
            self._gt_skel_tree = KDTree(gt_coords)
        else:
            self._gt_skel_coords = None
            self._gt_skel_tree = None

    def compute(self, pred: torch.Tensor) -> dict[str, float]:
        """Compute threshold-based segmentation metrics for a binary prediction.

        Args:
            pred: (D, H, W) bool tensor on any device.

        Returns:
            dict with keys ``dice``, ``precision``, ``recall``, ``hd95``,
            ``cldice``, ``dist_avg_cl``, ``hd95_cl``.
        """
        pred = pred.to(dtype=torch.bool)
        pred_sum = int(pred.sum().item())

        both_empty = (pred_sum == 0) and (self._gt_sum == 0)
        one_empty = (pred_sum == 0) != (self._gt_sum == 0)

        if both_empty:
            return {
                "dice": 1.0, "precision": 1.0, "recall": 1.0, "hd95": 0.0,
                "cldice": 1.0, "dist_avg_cl": 0.0, "hd95_cl": 0.0,
            }

        pred_bc = pred.unsqueeze(0).unsqueeze(0).float()
        gt_t = torch.from_numpy(self._gt_np).to(device=pred.device, dtype=torch.float32)
        gt_bc = gt_t.unsqueeze(0).unsqueeze(0)

        dice = float(compute_dice(pred_bc, gt_bc, include_background=False).item())

        cm = get_confusion_matrix(y_pred=pred_bc, y=gt_bc, include_background=False)
        precision = float(compute_confusion_matrix_metric("precision", cm).item())
        recall = float(compute_confusion_matrix_metric("recall", cm).item())
        if math.isnan(precision):
            precision = 1.0 if self._gt_sum == 0 else 0.0
        if math.isnan(recall):
            recall = 1.0 if pred_sum == 0 else 0.0

        if one_empty:
            hd95 = float("inf")
        else:
            hd95 = float(
                compute_hausdorff_distance(pred_bc, gt_bc, percentile=95, spacing=self._spacing).item()
            )

        pred_oh = torch.cat([1.0 - pred_bc, pred_bc], dim=1)
        gt_oh = torch.cat([1.0 - gt_bc, gt_bc], dim=1)
        cldice = float((1.0 - _soft_cldice_fn(gt_oh, pred_oh)).item())

        if one_empty:
            dist_avg_cl = float("inf")
            hd95_cl = float("inf")
        else:
            pred_np = pred.cpu().numpy()
            pred_skel_coords = _skeleton_coords(pred_np, self._spacing)
            if pred_skel_coords is None:
                dist_avg_cl = float("inf")
                hd95_cl = float("inf")
            else:
                tree_pred = KDTree(pred_skel_coords)
                d_p2g, _ = self._gt_skel_tree.query(pred_skel_coords, k=1)  # type: ignore[union-attr]
                d_g2p, _ = tree_pred.query(self._gt_skel_coords, k=1)       # type: ignore[union-attr]
                all_dists = np.concatenate([d_p2g, d_g2p])
                dist_avg_cl = float(all_dists.mean())
                hd95_cl = float(np.percentile(all_dists, 95))

        return {
            "dice": dice, "precision": precision, "recall": recall, "hd95": hd95,
            "cldice": cldice, "dist_avg_cl": dist_avg_cl, "hd95_cl": hd95_cl,
        }

    def compute_density(self, density: torch.Tensor) -> dict[str, float]:
        """Compute threshold-free density-based metrics.

        Args:
            density: (D, H, W) float tensor of predicted density values.

        Returns:
            dict with keys ``soft_dice``, ``roc_auc``, ``pr_auc``.
        """
        roi = torch.from_numpy(self._aabb_roi_np).to(device=density.device, dtype=torch.bool)
        gt_t = torch.from_numpy(self._gt_np).to(device=density.device, dtype=torch.bool)

        d_flat = density[roi].float()
        g_flat = gt_t[roi].float()

        d_min = d_flat.min()
        d_max = d_flat.max()
        denom_range = d_max - d_min
        if denom_range > 1e-8:
            I = (d_flat - d_min) / denom_range
        else:
            I = torch.zeros_like(d_flat)

        numerator = 2.0 * (I * g_flat).sum()
        denominator = (I * I).sum() + (g_flat * g_flat).sum() + 1e-8
        soft_dice = float((numerator / denominator).item())

        try:
            roc_auc = float(compute_roc_auc(d_flat, g_flat))  # type: ignore
        except Exception:
            roc_auc = float("nan")
        try:
            pr_auc = float(compute_average_precision(d_flat, g_flat))  # type: ignore
        except Exception:
            pr_auc = float("nan")

        return {"soft_dice": soft_dice, "roc_auc": roc_auc, "pr_auc": pr_auc}
