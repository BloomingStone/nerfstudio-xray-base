"""K-Planes density field for x-ray attenuation.

Supports dual-time decomposition:
  - Primary 4D grid (x, y, z, time_s) — models contrast agent flow.
  - Optional phase grid (x, y, z, phase) — models periodic cardiac motion,
    using only the 3 planes that contain the phase axis (x-p, y-p, z-p).
    Features from both grids are concatenated before the density MLP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Type

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.encodings import KPlanesEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import Field

try:
    import tinycudann as tcnn
except ImportError:
    pass


def interpolate_ms_features(
    pts: torch.Tensor,
    grid_encodings: Sequence[KPlanesEncoding],
    concat_features: bool,
) -> torch.Tensor:
    """Combines/interpolates features across multiple scales.

    Args:
        pts: Coordinates to query, range [-1, 1]
        grid_encodings: Grid encodings to query (module list of KPlanesEncoding)
        concat_features: Whether to concatenate features at different scales

    Returns:
        Feature vectors
    """
    if concat_features:
        multi_scale_interp = []
        for grid in grid_encodings:
            multi_scale_interp.append(grid(pts))
        return torch.cat(multi_scale_interp, dim=-1)
    else:
        multi_scale_interp = None
        for grid in grid_encodings:
            grid_features = grid(pts)
            multi_scale_interp = grid_features if multi_scale_interp is None else multi_scale_interp + grid_features
        assert multi_scale_interp is not None, "multi_scale_interp should not be None after interpolating features"
        return multi_scale_interp


def interpolate_phase_planes(
    pts: torch.Tensor,
    phase_encodings: Sequence[KPlanesEncoding],
    concat_features: bool = True,
) -> torch.Tensor:
    """Query only the phase-containing planes from 4D KPlanesEncodings.

    For a 4D grid with coords ``(x, y, z, phase)``, the 6 plane
    coefficient combinations are::

        coo_combs = [(0,1)=XY, (0,2)=XZ, (0,3)=X-p, (1,2)=YZ, (1,3)=Y-p, (2,3)=Z-p]

    Only the 3 planes where axis 3 (phase) participates are used
    (indices 2, 4, 5), discarding the pure-spatial planes which
    are redundant with the primary time grid.

    Args:
        pts: (N, 4) coordinates in ``[-1, 1]``; dim 3 is the phase axis.
        phase_encodings: list of :class:`KPlanesEncoding` with ``in_dim=4``.
        concat_features: If ``True`` concatenate across scales, else sum.

    Returns:
        (N, C) feature vectors, where C is the total output dim.
    """
    phase_plane_indices = (2, 4, 5)  # (x,p), (y,p), (z,p)

    if concat_features:
        multi_scale = []
        for encoding in phase_encodings:
            planes = encoding.plane_coefs
            coo_combs = encoding.coo_combs
            output = 1.0  # product identity
            for ci in phase_plane_indices:
                coords = pts[..., coo_combs[ci]].view(1, 1, -1, 2)
                interp = F.grid_sample(
                    planes[ci].unsqueeze(0), coords,
                    align_corners=True, padding_mode="border",
                )
                interp = interp.view(encoding.num_components, -1).T
                output = output * interp
            multi_scale.append(output)
        return torch.cat(multi_scale, dim=-1)
    else:
        multi_scale = None
        for encoding in phase_encodings:
            planes = encoding.plane_coefs
            coo_combs = encoding.coo_combs
            output = 1.0
            for ci in phase_plane_indices:
                coords = pts[..., coo_combs[ci]].view(1, 1, -1, 2)
                interp = F.grid_sample(
                    planes[ci].unsqueeze(0), coords,
                    align_corners=True, padding_mode="border",
                )
                interp = interp.view(encoding.num_components, -1).T
                output = output * interp
            multi_scale = output if multi_scale is None else multi_scale + output
        assert multi_scale is not None
        return multi_scale


@dataclass
class XRayKPlanesFieldConfig(InstantiateConfig):
    """Configuration for the x-ray K-Planes density field."""

    _target: Type = field(default_factory=lambda: XRayKPlanesField)
    grid_base_resolution: Tuple[int, ...] = (128, 128, 128)
    """Base grid resolution. Add a 4th dimension for time (e.g. (128, 128, 128, 25))."""
    grid_feature_dim: int = 32
    """Dimension of feature vectors stored in each grid."""
    multiscale_res: Tuple[int, ...] = (1, 2, 4)
    """Multiplier scales for multi-resolution grid."""
    use_phase: bool = False
    """If True, add a parallel phase grid (x, y, z, phase) for periodic cardiac motion.
    Features from the phase grid are concatenated with the primary time grid features."""
    phase_grid_resolution: int = 16
    """Number of discrete buckets for the phase axis in the phase grid."""


class XRayKPlanesField(Field):
    """A K-Planes field predicting only density for x-ray attenuation.

    Supports three modes depending on configuration:

    * **Static (3D):** ``grid_base_resolution`` has 3 entries.
    * **Dynamic with time (4D):** ``grid_base_resolution`` has 4 entries; the 4th axis is time.
    * **Dynamic with time + phase (4D + phase grid):** as above, plus a separate phase grid
      (``use_phase=True``) that adds 3 phase-planes (x-p, y-p, z-p) whose features are
      concatenated with the time grid features before the density MLP.
    """

    aabb: Tensor

    def __init__(
        self,
        config: XRayKPlanesFieldConfig,
        aabb: Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.config = config
        self.has_time_planes = len(config.grid_base_resolution) == 4
        self.use_phase = config.use_phase

        # ---- Primary grids (time_s axis when has_time_planes) ----
        self._grids = nn.ModuleList()
        for res_mult in config.multiscale_res:
            resolution = [r * res_mult for r in config.grid_base_resolution[:3]]
            if self.has_time_planes:
                resolution += [config.grid_base_resolution[3]]
            self._grids.append(KPlanesEncoding(
                resolution=resolution,
                num_components=config.grid_feature_dim,
            ))

        in_dim = config.grid_feature_dim * len(config.multiscale_res)

        # ---- Phase grid (always 4D: x, y, z, phase) ----
        self._phase_grids = nn.ModuleList()
        if self.use_phase:
            # Use the same multiscale factors for the phase grid
            for res_mult in config.multiscale_res:
                resolution = [r * res_mult for r in config.grid_base_resolution[:3]] + [config.phase_grid_resolution]
                self._phase_grids.append(KPlanesEncoding(
                    resolution=resolution,
                    num_components=config.grid_feature_dim,
                ))
            in_dim += config.grid_feature_dim * len(config.multiscale_res)

        # Tiny MLP: grid features → scalar density logit
        self.sigma_net = tcnn.Network(
            n_input_dims=in_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

    @property
    def grids(self) -> Sequence[KPlanesEncoding]:
        res = []
        for grid in self._grids:
            assert isinstance(grid, KPlanesEncoding), "Expected grid to be KPlanesEncoding"
            res.append(grid)
        return res

    @property
    def phase_grids(self) -> Sequence[KPlanesEncoding]:
        res = []
        for grid in self._phase_grids:
            assert isinstance(grid, KPlanesEncoding), "Expected phase grid to be KPlanesEncoding"
            res.append(grid)
        return res

    def _prepare_positions(self, ray_samples: RaySamples) -> Tensor:
        """Normalize positions and append time if needed. Returns tensor in [-1, 1]."""
        positions = ray_samples.frustums.get_positions()
        # Normalize from world coords to [0, 1] then to [-1, 1]
        normalized = SceneBox.get_normalized_positions(positions, self.aabb)  # [0, 1]
        positions_out = normalized * 2.0 - 1.0  # [-1, 1]

        if self.has_time_planes:
            if ray_samples.times is None:
                # Graceful degradation: viewer / volume extraction may not supply
                # times; fall back to t=0 instead of crashing.
                timestamps = torch.zeros_like(positions_out[..., :1])
            else:
                # Normalize timestamps from [0, 1] to [-1, 1]
                timestamps = ray_samples.times * 2.0 - 1.0
            positions_out = torch.cat((positions_out, timestamps), dim=-1)

        return positions_out

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        # ---- Primary grid features (time_s) ----
        pts = self._prepare_positions(ray_samples)  # [-1, 1] or [-1, 1]^4
        pts_flat = pts.reshape(-1, pts.shape[-1])
        features = interpolate_ms_features(pts_flat, self.grids, concat_features=True)

        # ---- Phase grid features ----
        # Graceful degradation: if phase is missing (e.g. viewer-rendered cameras
        # that only carry {"cam_idx": ...}, or volume extraction without phase),
        # fall back to phase=0 instead of crashing with an AssertionError. This
        # keeps the viewer/eval rendering pipeline alive so images still display.
        if self.use_phase:
            phase = None
            if ray_samples.metadata is not None:
                phase = ray_samples.metadata.get("phase")
            if phase is None:
                phase_flat = torch.zeros(
                    (pts_flat.shape[0], 1), dtype=pts_flat.dtype, device=pts_flat.device
                )
            else:
                # phase is in [0, 1), map to [-1, 1]
                phase_flat = (phase.reshape(-1, 1) * 2.0 - 1.0)  # (N, 1)
            # Spatial part from the primary grid positions (x, y, z)
            spatial = pts_flat[..., :3]  # (N, 3)
            phase_pts = torch.cat([spatial, phase_flat], dim=-1)  # (N, 4)
            phase_feats = interpolate_phase_planes(phase_pts, self.phase_grids, concat_features=True)
            features = torch.cat([features, phase_feats], dim=-1)

        density_before_activation = self.sigma_net(features).reshape(*pts.shape[:-1], 1)

        # Softplus for strictly positive density
        density = torch.nn.functional.softplus(density_before_activation - 1.0)

        return density, density_before_activation

    def density_fn(self, positions: Tensor, times: Optional[Tensor] = None, phase: Optional[Tensor] = None) -> Tensor:
        """Returns only the density. Used by proposal network samplers.

        Args:
            positions: (..., 3) query positions in scene coordinates.
            times: optional (..., 1) time values in [0, 1] for time planes.
            phase: optional (..., 1) phase values in [0, 1] for phase grid.
        """
        if times is not None and len(positions.shape) == 3 and len(times.shape) == 2:
            times = times[:, None]  # [ray, 1] -> [ray, 1, 1] for broadcasting
        if phase is not None and len(positions.shape) == 3 and len(phase.shape) == 2:
            phase = phase[:, None]
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=positions,
                directions=torch.ones_like(positions),
                starts=torch.zeros_like(positions[..., :1]),
                ends=torch.zeros_like(positions[..., :1]),
                pixel_area=torch.ones_like(positions[..., :1]),
            ),
            times=times,
            metadata={"phase": phase} if phase is not None else None,
        )
        density, _ = self.get_density(ray_samples)
        return density

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        return {}
