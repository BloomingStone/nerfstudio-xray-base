"""Density field for x-ray attenuation."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Optional, Tuple, Type

from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.fields.base_field import Field


@dataclass
class XRayFieldConfig(InstantiateConfig):
    """Configuration for the x-ray density field."""

    _target: Type = dataclass_field(default_factory=lambda: XRayField)
    hidden_dim: int = 64
    num_layers: int = 3
    density_scale: float = 0.01


class XRayField(Field):
    """Predicts a scalar density / linear attenuation coefficient."""

    aabb: Tensor

    def __init__(self, config: XRayFieldConfig, aabb: Tensor) -> None:
        super().__init__()
        self.register_buffer("aabb", aabb)
        self.density_scale = config.density_scale

        layers = []
        in_dim = 3
        for layer_idx in range(max(config.num_layers - 1, 0)):
            out_dim = config.hidden_dim if layer_idx < config.num_layers - 2 else 1
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_idx < config.num_layers - 2:
                layers.append(nn.ReLU())
            in_dim = out_dim
        if config.num_layers <= 1:
            layers = [nn.Linear(in_dim, 1)]
        elif layers and not isinstance(layers[-1], nn.Linear):
            layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        positions = ray_samples.frustums.get_positions()
        normalized = SceneBox.get_normalized_positions(positions, self.aabb)
        selector = ((normalized >= 0.0) & (normalized <= 1.0)).all(dim=-1, keepdim=True)
        normalized = normalized * 2.0 - 1.0
        density_before_activation = self.mlp(normalized.reshape(-1, 3)).reshape(*positions.shape[:-1], 1)
        density = self.density_scale * trunc_exp(density_before_activation)
        density = density * selector
        self._sample_locations = positions
        self._density_before_activation = density_before_activation
        
        # The second return value should be some features, but now we only have density to return, so we 
        # return the density before activation as a _placeholder_.
        return density, density_before_activation

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> dict:
        return {}
