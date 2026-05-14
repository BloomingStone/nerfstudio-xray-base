"""Density field for x-ray attenuation."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Optional, Tuple, Type

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.fields.base_field import Field


@dataclass
class XRayFieldConfig(InstantiateConfig):
    """Configuration for the x-ray density field."""

    _target: Type = dataclass_field(default_factory=lambda: XRayField)
    hidden_dim: int = 128
    num_layers: int = 5
    density_scale: float = 0.1
    """Initial density scale.  The network can learn to deviate from this."""
    num_frequencies: int = 8
    """Number of frequency bands for positional encoding."""
    max_freq_exp: float = 7.0
    """Maximum frequency exponent (2^max_freq).  With num_frequencies=8 and max=7, the frequencies are [2^0, 2^1, ..., 2^7]."""


class XRayField(Field):
    """Predicts a scalar density / linear attenuation coefficient."""

    aabb: Tensor

    def __init__(self, config: XRayFieldConfig, aabb: Tensor) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.density_scale = nn.Parameter(torch.tensor(config.density_scale, dtype=torch.float32))
        self.config = config

        # Positional encoding: maps (x,y,z) → high-frequency features
        self.position_encoding = NeRFEncoding(
            in_dim=3,
            num_frequencies=config.num_frequencies,
            min_freq_exp=0.0,
            max_freq_exp=config.max_freq_exp,
            include_input=True,
        )

        # Build MLP: input_dim_from_encoding → hidden → ... → 1 (density)
        in_dim = self.position_encoding.get_out_dim()  # 3 + 3*8*2 = 51
        layers = []
        for i in range(config.num_layers - 1):
            out_dim = config.hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))  # final layer → scalar density logit
        self.mlp = nn.Sequential(*layers)

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        positions = ray_samples.frustums.get_positions()
        positions = SceneBox.get_normalized_positions(positions, self.aabb)  # [0, 1]

        # Encode with positional encoding
        flat_positions = self.position_encoding(positions.view(-1, 3))

        density = self.mlp(flat_positions).reshape(*positions.shape[:-1], 1)

        # softplus for smooth, strictly positive density
        density = torch.nn.functional.softplus(density * self.density_scale)
        density = density * ((positions >= 0.0) & (positions <= 1.0)).all(dim=-1, keepdim=True)

        self._sample_locations = positions
        self._density_before_activation = density

        return density, density  # type: ignore[return-value]

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> dict:
        return {}
