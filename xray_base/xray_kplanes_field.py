"""K-Planes density field for x-ray attenuation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Type

import torch
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


class XRayKPlanesField(Field):
    """A K-Planes field predicting only density for x-ray attenuation.

    Supports both static (3D) and dynamic (4D with time) modes depending on the
    length of ``grid_base_resolution``.
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

        # Build multi-scale K-Planes grids
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

    def _prepare_positions(self, ray_samples: RaySamples) -> Tensor:
        """Normalize positions and append time if needed. Returns tensor in [-1, 1]."""
        positions = ray_samples.frustums.get_positions()
        # Normalize from world coords to [0, 1] then to [-1, 1]
        normalized = SceneBox.get_normalized_positions(positions, self.aabb)  # [0, 1]
        positions_out = normalized * 2.0 - 1.0  # [-1, 1]

        if self.has_time_planes:
            assert ray_samples.times is not None, \
                "Time planes enabled but no time data in ray_samples"
            # Normalize timestamps from [0, 1] to [-1, 1]
            timestamps = ray_samples.times * 2.0 - 1.0
            positions_out = torch.cat((positions_out, timestamps), dim=-1)

        return positions_out

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        pts = self._prepare_positions(ray_samples)  # [-1, 1] or [-1, 1]^4
        pts_flat = pts.reshape(-1, pts.shape[-1])

        features = interpolate_ms_features(pts_flat, self.grids, concat_features=True)
        density_before_activation = self.sigma_net(features).reshape(*pts.shape[:-1], 1)

        # Softplus for strictly positive density
        density = torch.nn.functional.softplus(density_before_activation - 1.0)

        return density, density_before_activation

    def density_fn(self, positions: Tensor, times: Optional[Tensor] = None) -> Tensor:
        """Returns only the density. Used by proposal network samplers."""
        if times is not None and len(positions.shape) == 3 and len(times.shape) == 2:
            times = times[:, None]  # [ray, 1] -> [ray, 1, 1] for broadcasting
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=positions,
                directions=torch.ones_like(positions),
                starts=torch.zeros_like(positions[..., :1]),
                ends=torch.zeros_like(positions[..., :1]),
                pixel_area=torch.ones_like(positions[..., :1]),
            ),
            times=times,
        )
        density, _ = self.get_density(ray_samples)
        return density

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        return {}
