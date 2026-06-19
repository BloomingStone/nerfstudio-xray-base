"""
Nerfstudio Template Pipeline
"""

import typing
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple, Type

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from xray_base.xray_datamanager import XrayDataManagerConfig
from xray_base.xray_model import XRayModel, XRayModelConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.utils import writer


@dataclass
class Metric3DConfig:
    """Configuration for 3D segmentation metric evaluation."""

    enabled: bool = True
    """If True, compute 3D metrics (Dice/HD95/clDice) during eval."""
    thresholds_absolute: Tuple[float, ...] = (0.0344, )
    """Absolute density thresholds for binarising the predicted volume."""
    thresholds_percentile: Tuple[float, ...] = (0.95, )
    """Percentile-based thresholds (0-1) computed over the ROI volume."""
    eval_time: float = 0.5
    """Normalised wall-clock time (0-1) at which to evaluate 3D metrics.
    Used by K-Planes time planes and D-NeRF temporal distortion."""
    eval_phase: float = 0.0
    """Cardiac phase (0-1) at which to evaluate 3D metrics.
    Used by K-Planes phase grid (when ``use_phase=True``)."""
    chunk_size: int = 1 << 20
    """Number of voxels queried per forward pass when extracting the volume."""
    save_dir: str = ""
    """Directory to save extracted volumes / labels during eval. Empty = don't save.
    Created if it does not exist. Files are named with the eval step."""
    save_items: Tuple[str, ...] = ("volume", "label")
    """Which artefacts to save under save_dir. Options:
    'volume' (predicted density .nii.gz), 'label' (GT label .nii.gz),
    'segmentation' (binarised prediction at each threshold .nii.gz)."""
    save_format: Literal["nii.gz", "npy"] = "nii.gz"
    """File format for saved volumes."""
    eval_only_at_end: bool = True
    """If True, skip 3D metric computation during training and only run once
    after training finishes. This saves GPU memory during training at the cost
    of not monitoring 3D metrics mid-training."""


@dataclass
class XrayPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: XrayPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = field(default_factory=XrayDataManagerConfig)
    """specifies the datamanager config"""
    model: ModelConfig = field(default_factory=XRayModelConfig)
    """specifies the model config"""
    metric3d: Metric3DConfig = field(default_factory=Metric3DConfig)
    """3D segmentation metric configuration."""


class XrayPipeline(VanillaPipeline):
    """Xray Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: XrayPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )

        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                XRayModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])

        # cached 3D metric computer (lazily built on first eval)
        self._metric3d_computer = None
        self._metric3d_label_info = None  # cached label dict

    # ------------------------------------------------------------------
    # 3D metric evaluation
    # ------------------------------------------------------------------
    def _get_label_info(self) -> Optional[dict]:
        """Return the cached 3D label dict from the dataparser metadata."""
        if self._metric3d_label_info is not None:
            return self._metric3d_label_info
        dataset = self.datamanager.train_dataset
        meta = getattr(dataset, "metadata", None)
        label = meta.get("label_3d") if meta else None
        self._metric3d_label_info = label
        return label

    def _build_metric3d_computer(self):
        from xray_base.metric_3d_utils import SegmentationMetricsComputer

        label_info = self._get_label_info()
        if label_info is None:
            return None, None
        gt = label_info["data"].astype(bool)
        aabb = label_info["aabb"].astype(bool)
        affine = label_info["affine"]
        spacing = (float(affine[0, 0]), float(affine[1, 1]), float(affine[2, 2]))
        computer = SegmentationMetricsComputer(gt=gt, aabb_roi=aabb, spacing=spacing)
        return computer, label_info

    def _save_volume(
        self,
        save_dir: str,
        step: Optional[int],
        name: str,
        data: Tensor,
        affine: np.ndarray,
        fmt: str,
    ) -> None:
        """Save a 3D volume to ``save_dir`` as .nii.gz (using *affine*) or .npy."""
        from pathlib import Path

        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        step_tag = f"step{step}" if step is not None else "eval"
        data_np = data.detach().cpu().numpy()
        if fmt == "nii.gz":
            import nibabel as nib

            nii = nib.Nifti1Image(data_np, affine.astype(np.float64))
            nib.save(nii, str(out_dir / f"{name}_{step_tag}.nii.gz"))
        else:
            np.save(str(out_dir / f"{name}_{step_tag}.npy"), data_np)

    def _compute_3d_metrics(self, step: Optional[int] = None) -> Dict[str, float]:
        """Extract a predicted density volume and compute 3D segmentation metrics."""
        cfg = self.config.metric3d
        if not cfg.enabled:
            return {}
        try:
            if self._metric3d_computer is None:
                self._metric3d_computer, label_info = self._build_metric3d_computer()
                if self._metric3d_computer is None:
                    return {}
            else:
                label_info = self._get_label_info()
            assert label_info is not None

            from xray_base.volume_extraction import extract_density_volume

            model = self.model
            field = model.field
            temporal_distortion = getattr(model, "temporal_distortion", None)
            scene_aabb = model.scene_box.aabb
            device = model.device_indicator_param.device

            volume_shape = (
                int(label_info["data"].shape[0]),
                int(label_info["data"].shape[1]),
                int(label_info["data"].shape[2]),
            )
            label_affine = label_info["affine"]
            world_scale = float(getattr(self.datamanager.train_dataset, "metadata", {}).get("world_scale", 1.0))

            has_time_planes = getattr(field, "has_time_planes", False)
            use_phase = getattr(field, "use_phase", False)
            time: Optional[Tensor] = None
            phase: Optional[Tensor] = None
            if has_time_planes or temporal_distortion is not None:
                time = torch.tensor([cfg.eval_time], dtype=torch.float32, device=device)
            if use_phase:
                phase = torch.tensor([cfg.eval_phase], dtype=torch.float32, device=device)

            vol_pred = extract_density_volume(
                field=field,
                volume_shape=volume_shape,
                label_affine=label_affine,
                world_scale=world_scale,
                scene_aabb=scene_aabb,
                device=device,
                batch_size=cfg.chunk_size,
                time=time,
                temporal_distortion=temporal_distortion,
                phase=phase,
            )

            # vol_pred is on CPU (see extract_density_volume); keep all subsequent
            # processing on CPU to minimise GPU memory usage during metric computation.
            aabb_roi = torch.from_numpy(label_info["aabb"]).to(dtype=torch.bool)

            # ---- optional saving of extracted volumes / label ----
            if cfg.save_dir:
                if "volume" in cfg.save_items:
                    self._save_volume(
                        cfg.save_dir, step, "pred_volume", vol_pred, label_affine, cfg.save_format
                    )
                if "label" in cfg.save_items:
                    gt_t = torch.from_numpy(label_info["data"].astype(np.float32))
                    self._save_volume(
                        cfg.save_dir, step, "gt_label", gt_t, label_affine, cfg.save_format
                    )

            # build threshold list
            thresholds: List[Tuple[str, float]] = []
            for thr in cfg.thresholds_absolute:
                thresholds.append((f"thd-{thr:.4f}", float(thr)))
            if cfg.thresholds_percentile:
                vol_roi = vol_pred[aabb_roi]
                if vol_roi.numel() > 0:
                    for pct in cfg.thresholds_percentile:
                        thr_val = float(torch.quantile(vol_roi.cpu(), pct))
                        thresholds.append((f"thd-{pct * 100:.2f}%", thr_val))

            result: Dict[str, float] = {}
            for thr_key, thr_val in thresholds:
                pred = ((vol_pred > thr_val) & aabb_roi).to(dtype=torch.bool)
                metrics = self._metric3d_computer.compute(pred)  # type: ignore[union-attr]
                for name, val in metrics.items():
                    result[f"metric3D/{thr_key}/{name}"] = float(val)
                if cfg.save_dir and "segmentation" in cfg.save_items:
                    self._save_volume(
                        cfg.save_dir, step, f"seg_{thr_key}", pred.to(torch.float32),
                        label_affine, cfg.save_format,
                    )

            density_metrics = self._metric3d_computer.compute_density(vol_pred)  # type: ignore[union-attr]
            for name, val in density_metrics.items():
                result[f"metric3D/density/{name}"] = float(val)
            return result
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Error computing 3D metrics: {e}")
            print(f"Error computing 3D metrics: {e}")
            return {}

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)
        if self.config.metric3d.eval_only_at_end:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN],
                    func=self._compute_and_log_3d_metrics_at_end,
                    update_every_num_iters=None,
                )
            )
        return callbacks

    def _compute_and_log_3d_metrics_at_end(self, step: int) -> None:
        """Compute 3D metrics once after training finishes and log via writer.

        The volume extraction moves each batch to GPU for ``get_density()`` but
        keeps the full position array and output on CPU, so peak GPU memory stays
        low even for large volumes. We still clear the CUDA cache beforehand to
        minimise fragmentation.
        """
        import sys

        print("\n[3D Metrics] Computing final 3D segmentation metrics after training...", flush=True)
        torch.cuda.empty_cache()

        metric3d = self._compute_3d_metrics(step=step)
        if metric3d:
            writer.put_dict(name="Final 3D Metrics Dict", scalar_dict=metric3d, step=step)
            writer.write_out_storage()
            print(f"[3D Metrics] Done. Logged {len(metric3d)} metrics to writer.", flush=True)
            for k, v in metric3d.items():
                print(f"  {k}: {v:.6f}", flush=True)
        else:
            print("[3D Metrics] Skipped (disabled, no label data, or error).", flush=True)
        # Move model back to its original device for viewer interaction
        self.model.to(self.device)

    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[typing.Any] = None, get_std: bool = False
    ):
        """Standard 2D image metrics + 3D segmentation metrics."""
        metrics_dict = super().get_average_eval_image_metrics(step=step, output_path=output_path, get_std=get_std)
        # append 3D metrics (independent of images) — unless eval_only_at_end is set
        if not self.config.metric3d.eval_only_at_end:
            metric3d = self._compute_3d_metrics(step=step)
            metrics_dict.update(metric3d)
        return metrics_dict
