"""K-Planes model for x-ray attenuation reconstruction."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union, Sequence, cast

import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.encodings import KPlanesEncoding
from nerfstudio.model_components.losses import MSELoss, distortion_loss, interlevel_loss
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, misc
from nerfstudio.utils.math import intersect_aabb

from xray_base.xray_kplanes_field import XRayKPlanesFieldConfig, XRayKPlanesField


@dataclass
class XRayKPlanesModelConfig(ModelConfig):
    """Configuration for the x-ray K-Planes model."""

    _target: Type = field(default_factory=lambda: XRayKPlanesModel)
    enable_collider: bool = False

    # K-Planes grid settings
    grid_base_resolution: Tuple[int, ...] = (128, 128, 128)
    """Base grid resolution. Add a 4th dimension for time (e.g. (128, 128, 128, 25))."""
    grid_feature_dim: int = 32
    """Dimension of feature vectors stored in each grid."""
    multiscale_res: Tuple[int, ...] = (1, 2, 4)
    """Multiplier scales for multi-resolution grid."""

    # Rendering
    num_samples_per_ray: int = 48
    """Number of fine samples per ray."""
    background_intensity: float = 1.0
    """I0 — incident x-ray intensity."""

    # Proposal network settings
    num_proposal_iterations: int = 2
    num_proposal_samples: Tuple[int, ...] = (256, 128)
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"num_output_coords": 8, "resolution": [64, 64, 64]},
            {"num_output_coords": 8, "resolution": [128, 128, 128]},
        ]
    )
    use_same_proposal_network: bool = False
    single_jitter: bool = False
    proposal_warmup: int = 5000
    proposal_update_every: int = 5
    use_proposal_weight_anneal: bool = True
    proposal_weights_anneal_slope: float = 10.0
    proposal_weights_anneal_max_num_iters: int = 1000

    use_phase: bool = False
    """If True, add a parallel phase grid (x, y, z, phase) for periodic cardiac motion.
    Requires the dataparser to provide ``phase`` in ``Cameras.metadata``."""
    phase_grid_resolution: int = 16
    """Number of discrete buckets for the phase axis in the phase grid."""

    # Loss coefficients
    loss_coefficients: Dict[str, float] = to_immutable_dict({
        "xray_loss": 1.0,
        "interlevel": 1.0,
        "distortion": 0.001,
        "plane_tv": 0.01,
        "plane_tv_proposal_net": 0.001,
        "l1_time_planes": 0.001,
        "l1_time_planes_proposal_net": 0.001,
        "time_smoothness": 0.1,
        "time_smoothness_proposal_net": 0.01,
        "l1_phase_planes": 0.001,
        "l1_phase_planes_proposal_net": 0.001,
        "phase_smoothness": 0.1,
        "phase_smoothness_proposal_net": 0.01,
    })


class XRayKPlanesModel(Model):
    """K-Planes model for x-ray reconstruction.

    Uses multi-resolution K-Planes grids to represent a density field,
    with proposal network sampling for efficient ray marching.
    Supports both static (3D) and dynamic (4D + time) modes.
    """
    config: XRayKPlanesModelConfig

    def populate_modules(self):
        config = cast(XRayKPlanesModelConfig, self.config)

        # --- K-Planes field for density ---
        self.field = XRayKPlanesField(
            XRayKPlanesFieldConfig(
                grid_base_resolution=config.grid_base_resolution,
                grid_feature_dim=config.grid_feature_dim,
                multiscale_res=config.multiscale_res,
                use_phase=config.use_phase,
                phase_grid_resolution=config.phase_grid_resolution,
            ),
            aabb=self.scene_box.aabb,
        )

        # --- Proposal networks for coarse-to-fine sampling ---
        self.density_fns = []
        num_prop_nets = config.num_proposal_iterations
        self._proposal_networks = torch.nn.ModuleList()

        def _build_proposal_net(resolution_args: Dict) -> XRayKPlanesField:
            # Proposal networks only do coarse density estimation for sampling,
            # so we intentionally disable the phase grid here to save memory and
            # compute. Only the primary (time) field carries the phase grid.
            return XRayKPlanesField(
                XRayKPlanesFieldConfig(
                    grid_base_resolution=resolution_args["resolution"],
                    grid_feature_dim=resolution_args["num_output_coords"],
                    multiscale_res=(1,),
                    use_phase=False,
                ),
                aabb=self.scene_box.aabb,
            )

        if config.use_same_proposal_network:
            assert len(config.proposal_net_args_list) == 1, \
                "Only one proposal network is allowed with use_same_proposal_network."
            network = _build_proposal_net(config.proposal_net_args_list[0])
            self._proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = config.proposal_net_args_list[
                    min(i, len(config.proposal_net_args_list) - 1)
                ]
                network = _build_proposal_net(prop_net_args)
                self._proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self._proposal_networks])

        # --- Sampler ---
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, config.proposal_warmup], [0, config.proposal_update_every]),
                1,
                config.proposal_update_every,
            )

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=config.num_samples_per_ray,
            num_proposal_samples_per_ray=config.num_proposal_samples,
            num_proposal_network_iterations=config.num_proposal_iterations,
            single_jitter=config.single_jitter,
            update_sched=update_schedule,
            initial_sampler=UniformSampler(single_jitter=config.single_jitter),
        )

        # --- Renderers ---
        self.renderer_rgb = RGBRenderer(background_color="white")
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")

        # --- Loss ---
        self.intensity_loss = MSELoss()

        # --- Metrics ---
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
    
    @property
    def proposal_networks(self) -> Sequence[XRayKPlanesField]:
        res = []
        for net in self._proposal_networks:
            assert isinstance(net, XRayKPlanesField), "Expected proposal network to be XRayKPlanesField"
            res.append(net)
        return res

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            "proposal_networks": list(self._proposal_networks.parameters()),
            "fields": list(self.field.parameters()),
        }

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        config = cast(XRayKPlanesModelConfig, self.config)
        if config.use_proposal_weight_anneal:
            N = config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def _render_intensity(self, ray_bundle: RayBundle) -> Dict[str, Tensor]:
        config = cast(XRayKPlanesModelConfig, self.config)
        num_rays = len(ray_bundle)

        # Compute ray-AABB intersections for nears/fars (required by UniformSampler)
        origins = ray_bundle.origins.reshape(-1, 3)
        directions = ray_bundle.directions.reshape(-1, 3)
        aabb_flat = self.scene_box.aabb.flatten().to(origins.device)
        t_min, t_max = intersect_aabb(origins, directions, aabb_flat)
        valid = t_max > t_min
        ray_bundle.nears = torch.where(valid, t_min, torch.zeros_like(t_min)).unsqueeze(-1)
        ray_bundle.fars = torch.where(valid, t_max, torch.zeros_like(t_max)).unsqueeze(-1)

        # Coarse-to-fine sampling via proposal networks
        density_fns = self.density_fns
        if ray_bundle.times is not None:
            kwargs = {"times": ray_bundle.times}
            if self.field.use_phase and ray_bundle.metadata is not None:
                phase = ray_bundle.metadata.get("phase")
                if phase is not None:
                    kwargs["phase"] = phase
            density_fns = [functools.partial(f, **kwargs) for f in density_fns]
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=density_fns
        )

        field_outputs = self.field(ray_samples)
        density = field_outputs[FieldHeadNames.DENSITY]

        weights = ray_samples.get_weights(density)
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        accumulation = self.renderer_accumulation(weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)

        # X-ray attenuation: I = I0 * exp(-∫μ dx)
        # Volumetric rendering with zero sample color and white background
        # effectively computes transmittance (Beer-Lambert).
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
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        for i in range(config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i]
            )
        return outputs

    def get_outputs(self, ray_bundle: Union[RayBundle, Cameras]) -> Dict[str, Tensor]:
        if isinstance(ray_bundle, Cameras):
            ray_bundle = ray_bundle.generate_rays(camera_indices=0, keep_shape=True)
        return self._render_intensity(ray_bundle)

    def get_metrics_dict(self, outputs, batch) -> Dict[str, Tensor]:
        config = cast(XRayKPlanesModelConfig, self.config)
        gt = batch["image"].to(self.device)[..., :1]

        metrics_dict: Dict[str, Tensor] = {
            "psnr": self.psnr(outputs["intensity"], gt),
        }
        if self.training:
            metrics_dict["interlevel"] = interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            metrics_dict["distortion"] = distortion_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )

            prop_grids = [p.grids for p in self.proposal_networks]
            field_grids = [self.field.grids]
            has_time = len(config.grid_base_resolution) == 4
            use_phase = config.use_phase

            metrics_dict["plane_tv"] = _space_tv_loss(field_grids)
            metrics_dict["plane_tv_proposal_net"] = _space_tv_loss(prop_grids)

            if has_time:
                metrics_dict["l1_time_planes"] = _l1_time_planes(field_grids)
                metrics_dict["l1_time_planes_proposal_net"] = _l1_time_planes(prop_grids)
                metrics_dict["time_smoothness"] = _time_smoothness(field_grids)
                metrics_dict["time_smoothness_proposal_net"] = _time_smoothness(prop_grids)

            if use_phase:
                phase_field_grids = [self.field.phase_grids]
                phase_prop_grids = [p.phase_grids for p in self.proposal_networks]
                metrics_dict["l1_phase_planes"] = _l1_time_planes(phase_field_grids)
                metrics_dict["l1_phase_planes_proposal_net"] = _l1_time_planes(phase_prop_grids)
                metrics_dict["phase_smoothness"] = _phase_smoothness(phase_field_grids)
                metrics_dict["phase_smoothness_proposal_net"] = _phase_smoothness(phase_prop_grids)

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, Tensor]:
        gt = batch["image"].to(self.device)[..., :1]
        pred = outputs["intensity"]
        loss_dict: Dict[str, Tensor] = {"xray_loss": self.intensity_loss(pred, gt)}
        if self.training:
            assert metrics_dict is not None
            for key in self.config.loss_coefficients:
                if key in metrics_dict:
                    loss_dict[key] = metrics_dict[key].clone()
            loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, Tensor]]:
        gt = batch["image"].to(self.device)[..., :1]
        pred = outputs["intensity"]

        # L1 loss
        l1 = float(torch.abs(pred - gt).mean().item())

        # PSNR on single-channel
        psnr = float(self.psnr(pred, gt).item())

        # SSIM and LPIPS require 3-channel images in [0,1] with shape [1, 3, H, W]
        gt_rgb = gt.repeat_interleave(3, dim=-1)  # [H, W, 3]
        pred_rgb = pred.repeat_interleave(3, dim=-1)

        # Move from [H, W, C] to [1, C, H, W]
        gt_rgb_4d = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        pred_rgb_4d = torch.moveaxis(pred_rgb, -1, 0)[None, ...]

        ssim = float(self.ssim(gt_rgb_4d, pred_rgb_4d).item())
        lpips = float(self.lpips(gt_rgb_4d, pred_rgb_4d).item())

        metrics_dict = {
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "l1": l1,
        }

        images_dict: Dict[str, Tensor] = {
            "img": torch.cat([gt_rgb, pred_rgb], dim=1),
            "accumulation": torch.cat([
                colormaps.apply_colormap(outputs["accumulation"]),
            ], dim=1),
            "depth": torch.cat([
                colormaps.apply_depth_colormap(outputs["depth"], accumulation=outputs["accumulation"]),
            ], dim=1),
        }
        return metrics_dict, images_dict


# ---- Regularization losses (adapted from K-Planes) ----

def _compute_plane_tv(t: Tensor, only_w: bool = False) -> Tensor:
    """Total variance across a plane."""
    _, h, w = t.shape
    w_tv = torch.square(t[..., :, 1:] - t[..., :, : w - 1]).mean()
    if only_w:
        return w_tv
    h_tv = torch.square(t[..., 1:, :] - t[..., : h - 1, :]).mean()
    return h_tv + w_tv


def _space_tv_loss(multi_res_grids: List[Sequence[KPlanesEncoding]]) -> Tensor:
    """TV loss over spatial planes (XY, XZ, YZ)."""
    total = torch.tensor(0.0)
    num_planes = 0
    for grids in multi_res_grids:  # grids = ModuleList of KPlanesEncoding
        for encoding in grids:    # encoding = KPlanesEncoding
            planes = encoding.plane_coefs
            for ci, _ in enumerate(planes):
                if len(planes) == 6 and ci in (0, 1, 3):  # 4D: XY, XZ, YZ
                    total = total + _compute_plane_tv(planes[ci])
                    num_planes += 1
                elif len(planes) == 3:  # 3D: XY, XZ, YZ
                    total = total + _compute_plane_tv(planes[ci])
                    num_planes += 1
            total = total.to(encoding.plane_coefs[0].device)
    return total / max(num_planes, 1)


def _l1_time_planes(multi_res_grids: List[Sequence[KPlanesEncoding]]) -> Tensor:
    """L1 regularization over time planes (XT, YT, ZT)."""
    total = torch.tensor(0.0)
    num_planes = 0
    for grids in multi_res_grids:
        for encoding in grids:
            planes = encoding.plane_coefs
            if len(planes) == 6:
                for ci in (2, 4, 5):
                    total = total + planes[ci].abs().mean()
                    num_planes += 1
            total = total.to(encoding.plane_coefs[0].device)
    return total / max(num_planes, 1)


def _time_smoothness(multi_res_grids: List[Sequence[KPlanesEncoding]]) -> Tensor:
    """Smoothness regularization over time planes."""
    total = torch.tensor(0.0)
    num_planes = 0
    for grids in multi_res_grids:
        for encoding in grids:
            planes = encoding.plane_coefs
            if len(planes) == 6:
                for ci in (2, 4, 5):
                    total = total + _compute_plane_tv(planes[ci], only_w=True)
                    num_planes += 1
            total = total.to(encoding.plane_coefs[0].device)
    return total / max(num_planes, 1)


def _phase_smoothness(multi_res_grids: List[Sequence[KPlanesEncoding]]) -> Tensor:
    """Smoothness regularization along the phase axis of phase planes (X-p, Y-p, Z-p).

    For a 4D grid with coords ``(x, y, z, phase)`` the phase-containing planes
    (indices 2, 4, 5) have shape ``(num_components, N_phase, spatial_dim)``.
    This computes TV along the phase axis (dim=1) to encourage smooth transition
    between neighbouring phase buckets.
    """
    total = torch.tensor(0.0)
    num_planes = 0
    for grids in multi_res_grids:
        for encoding in grids:
            planes = encoding.plane_coefs
            if len(planes) == 6:
                for ci in (2, 4, 5):
                    # TV along height = phase axis (dim=1)
                    tv = torch.square(planes[ci][:, 1:, :] - planes[ci][:, :-1, :]).mean()
                    total = total + tv
                    num_planes += 1
            total = total.to(encoding.plane_coefs[0].device)
    return total / max(num_planes, 1)
