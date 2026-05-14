"""Model for x-ray reconstruction with density-only attenuation."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, List, Tuple, Type, Union, cast

import torch
from torch import Tensor
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.math import intersect_aabb

from xray_base.xray_field import XRayFieldConfig


@dataclass
class XRayModelConfig(ModelConfig):
    """Configuration for the x-ray model."""

    _target: Type = dataclass_field(default_factory=lambda: XRayModel)
    enable_collider: bool = False
    field: XRayFieldConfig = dataclass_field(default_factory=XRayFieldConfig)
    num_samples_per_ray: int = 1024
    background_intensity: float = 1.0
    enable_temporal_distortion: bool = False
    """If True, enable D-NeRF-style deformation field conditioned on time."""
    temporal_distortion_params: Dict[str, Any] = to_immutable_dict({"kind": TemporalDistortionKind.DNERF})
    """Parameters to pass to the temporal distortion module."""
    offset_reg_weight: float = 0.01
    """L2 regularization on deformation offsets; prevents diverging warps."""


class XRayModel(Model):
    """Renders projection-plane x-ray intensity from density predictions."""
    config: XRayModelConfig

    def __init__(self, config: XRayModelConfig, **kwargs):
        self.temporal_distortion = None
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        config = cast(XRayModelConfig, self.config)
        self.field = config.field.setup(aabb=self.scene_box.aabb)
        self.renderer_rgb = RGBRenderer(background_color="white")
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        self.intensity_loss = MSELoss()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)

        if config.enable_temporal_distortion:
            params = dict(config.temporal_distortion_params)
            kind = params.pop("kind")
            self.temporal_distortion = kind.to_temporal_distortion(params)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        groups: Dict[str, List[Parameter]] = {"fields": list(self.field.parameters())}
        if self.temporal_distortion is not None:
            groups["temporal_distortion"] = list(self.temporal_distortion.parameters())
        return groups

    def _render_intensity(self, ray_bundle: RayBundle) -> Dict[str, Union[Tensor, List]]:
        config = cast(XRayModelConfig, self.config)
        num_rays = len(ray_bundle)
        origins = ray_bundle.origins.reshape(-1, 3)
        directions = ray_bundle.directions.reshape(-1, 3)
        aabb = self.scene_box.aabb.flatten().to(origins.device)
        t_min, t_max = intersect_aabb(origins, directions, aabb)
        valid = t_max > t_min

        t_min = torch.where(valid, t_min, torch.zeros_like(t_min))
        t_max = torch.where(valid, t_max, torch.zeros_like(t_max))

        fractions = torch.linspace(0.0, 1.0, config.num_samples_per_ray + 1, device=origins.device)
        span = (t_max - t_min).unsqueeze(-1)
        starts = (t_min.unsqueeze(-1) + span * fractions[:-1]).unsqueeze(-1)
        ends = (t_min.unsqueeze(-1) + span * fractions[1:]).unsqueeze(-1)

        ray_samples = ray_bundle.get_ray_samples(starts, ends)

        # Apply temporal deformation field to warp sampling positions
        if self.temporal_distortion is not None and ray_samples.times is not None:
            offsets = self.temporal_distortion(
                ray_samples.frustums.get_positions(), ray_samples.times
            )
            ray_samples.frustums.set_offsets(offsets)

        field_outputs = self.field(ray_samples)
        density = field_outputs[FieldHeadNames.DENSITY]

        assert ray_samples.deltas is not None, "RaySamples must include deltas for x-ray rendering"
        weights = ray_samples.get_weights(density)
        accumulation = self.renderer_accumulation(weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)

        # Use a white background and zero sample colors so RGB becomes transmittance.
        sample_rgb = torch.zeros((*density.shape[:-1], 3), device=density.device, dtype=density.dtype)
        rgb = self.renderer_rgb(rgb=sample_rgb, weights=weights)

        intensity = rgb[..., :1] * config.background_intensity
        optical_depth = -torch.log(torch.clamp(intensity, min=1e-8))

        outputs = {
            "rgb": rgb.reshape(num_rays, 3),
            "intensity": intensity.reshape(num_rays, 1),
            "optical_depth": optical_depth.reshape(num_rays, 1),
            "depth": depth.reshape(num_rays, 1),
            "accumulation": accumulation.reshape(num_rays, 1),
            "density": density,
        }
        if self.training and self.temporal_distortion is not None and ray_samples.times is not None:
            outputs["offsets"] = offsets  # used for offset_reg loss (keep grad)
            outputs["offset_norm"] = offsets.norm(dim=-1).mean().detach()  # logging only

        return outputs

    def get_outputs(self, ray_bundle: Union[RayBundle, Cameras]) -> Dict[str, Union[Tensor, List]]:
        if isinstance(ray_bundle, Cameras):
            ray_bundle = ray_bundle.generate_rays(camera_indices=0, keep_shape=True)
        return self._render_intensity(ray_bundle)

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        gt = batch["image"].to(self.device)[..., :1]
        metrics_dict = {"psnr": self.psnr(outputs["intensity"], gt)}
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, Tensor]:
        gt = batch["image"].to(self.device)[..., :1]
        pred = outputs["intensity"]
        loss = self.intensity_loss(pred, gt)
        loss_dict: Dict[str, Tensor] = {"xray_loss": loss}

        # Offset regularization: penalize large deformation warps
        if self.training and self.config.offset_reg_weight > 0 and "offsets" in outputs:
            offset_norm = outputs["offsets"].norm(dim=-1).mean()
            loss_dict["offset_reg"] = self.config.offset_reg_weight * offset_norm

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, Tensor]]:
        gt = batch["image"].to(self.device)[..., :1]
        pred = outputs["intensity"]
        metrics_dict = {"psnr": float(self.psnr(pred, gt))}
        gt_rgb = gt.repeat_interleave(3, dim=-1)
        pred_rgb = pred.repeat_interleave(3, dim=-1)
        images_dict = {
            "img": torch.cat([gt_rgb, pred_rgb], dim=1),
            "accumulation": torch.cat([
                colormaps.apply_colormap(outputs["accumulation"]),
            ], dim=1),
            "depth": torch.cat([
                colormaps.apply_depth_colormap(outputs["depth"], accumulation=outputs["accumulation"]),
            ], dim=1),
            "optical_depth": outputs["optical_depth"].repeat_interleave(3, dim=-1),
        }
        return metrics_dict, images_dict
