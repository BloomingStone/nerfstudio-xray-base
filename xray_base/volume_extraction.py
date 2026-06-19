"""Extract a 3D density volume from a nerfstudio density field.

The ground-truth label lives in a NIfTI volume with an affine mapping voxel->world (mm).
nerfstudio's scene has been normalized by ``world_scale`` (see rotate_xray_dataparser),
so world-mm coordinates are scaled by ``world_scale`` before being queried in the field.

We query the field by building synthetic :class:`RaySamples` whose frustum origins are the
voxel centres (in scaled scene coords) and calling ``field.get_density``. This reuses
the field's own normalisation / activation / time-plane logic, so the extracted volume
matches what the model actually predicts during rendering.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor


def voxel_world_coords(
    volume_shape: tuple[int, int, int],
    affine: np.ndarray,
) -> Tensor:
    """Return world (mm) coordinates of every voxel centre in *volume_shape*.

    Args:
        volume_shape: (D, H, W) of the label volume.
        affine: (4, 4) voxel->world affine (mm).

    Returns:
        (D*H*W, 3) float32 tensor of world coordinates.
    """
    D, H, W = volume_shape
    # voxel indices (i, j, k) along axes 0, 1, 2 -> world via affine.
    ii = np.tile(np.arange(D, dtype=np.float32)[:, None, None], (1, H, W)).ravel()
    jj = np.tile(np.arange(H, dtype=np.float32)[None, :, None], (D, 1, W)).ravel()
    kk = np.tile(np.arange(W, dtype=np.float32)[None, None, :], (D, H, 1)).ravel()
    pts = np.stack([ii, jj, kk, np.ones(D * H * W, dtype=np.float32)], axis=0)
    world = (affine.astype(np.float32) @ pts)[:3].T  # (N, 3)
    return torch.from_numpy(world)


@torch.no_grad()
def extract_density_volume(
    field,
    volume_shape: tuple[int, int, int],
    label_affine: np.ndarray,
    world_scale: float,
    scene_aabb: Tensor,
    device: str | torch.device,
    batch_size: int = 1 << 20,
    time: Optional[Tensor] = None,
    temporal_distortion=None,
    phase: Optional[Tensor] = None,
) -> Tensor:
    """Query the density field on the GT label voxel grid.

    The field is queried by constructing :class:`RaySamples` whose frustum origins
    are the voxel centres (in scaled scene coordinates) and delegating to
    ``field.get_density``. This keeps normalisation, activation and time handling
    identical to training-time rendering:

      * ``XRayField`` (MLP): ``get_density`` normalises positions to ``[0, 1]`` via
        the scene AABB. Temporal deformation, if any, is applied as frustum offsets
        (mirroring ``XRayModel._render_intensity``).
      * ``XRayKPlanesField``: ``get_density`` → ``_prepare_positions`` normalises
        positions to ``[-1, 1]`` and appends ``times * 2 - 1`` when time planes are
        enabled, so the raw ``time`` value (in ``[0, 1]``) must be passed unchanged
        via ``RaySamples.times``. If the field also has a phase grid (``use_phase``),
        ``phase`` is passed via ``RaySamples.metadata``.

    Args:
        field: the density field.
        volume_shape: (D, H, W).
        label_affine: (4, 4) voxel->world (mm) affine of the GT label.
        world_scale: scalar applied to world (mm) coords to obtain scene coords
            (matches the scaling done in rotate_xray_dataparser).
        scene_aabb: (2, 3) scene box in scaled coordinates — only used for array sizing;
            the field holds its own copy via ``field.aabb``.
        device: torch device.
        batch_size: number of voxels queried per forward pass.
        time: optional (1,) tensor in ``[0, 1]`` — normalised wall-clock time.
            For K-Planes, passed via ``RaySamples.times`` (field normalises to ``[-1,1]``).
            For MLP, used by ``temporal_distortion`` to compute deformation offsets.
        temporal_distortion: optional deformation field (MLP models only); called as
            ``temporal_distortion(positions, times)`` returning offsets.
        phase: optional (1,) tensor in ``[0, 1)`` — cardiac phase. Used by K-Planes phase
            grid (``field.use_phase``). Passed via ``RaySamples.metadata["phase"]``.

    Returns:
        (D, H, W) float32 tensor of predicted density on the GT grid.
    """
    from nerfstudio.cameras.rays import Frustums, RaySamples

    D, H, W = volume_shape
    # Keep positions on CPU; only transfer small batches to GPU for field inference.
    # This avoids holding the full (N, 3) grid on GPU (~700 MiB for 512³) and
    # frees GPU memory for the field + metric computation.
    world = voxel_world_coords(volume_shape, label_affine)           # (N, 3) mm, CPU
    scene_coords = world * float(world_scale)                        # (N, 3) scaled, CPU

    has_time_planes = getattr(field, "has_time_planes", False)
    use_phase = getattr(field, "use_phase", False)
    use_times = (has_time_planes or temporal_distortion is not None) and time is not None

    # Per-voxel time / phase tensors on CPU (expanded once, chunks moved to GPU later).
    t_cpu: Optional[Tensor] = None
    if use_times and time is not None:
        t_cpu = time.cpu().reshape(1, 1).expand(world.shape[0], -1)

    p_cpu: Optional[Tensor] = None
    if use_phase and phase is not None:
        p_cpu = phase.cpu().reshape(1, 1).expand(world.shape[0], -1)

    out = torch.empty(world.shape[0], dtype=torch.float32)           # CPU output
    for start in range(0, world.shape[0], batch_size):
        end = min(start + batch_size, world.shape[0])
        # Move only this chunk to GPU
        chunk_pos = scene_coords[start:end].to(device)               # (B, 3)
        zeros1 = torch.zeros_like(chunk_pos[..., :1])

        frustums = Frustums(
            origins=chunk_pos,
            directions=torch.zeros_like(chunk_pos),
            starts=zeros1,
            ends=zeros1,
            pixel_area=torch.ones_like(zeros1),
        )

        tv = t_cpu[start:end].to(device) if t_cpu is not None else None
        if temporal_distortion is not None and tv is not None:
            offsets = temporal_distortion(frustums.get_positions(), tv)
            frustums.set_offsets(offsets)

        pv = p_cpu[start:end].to(device) if p_cpu is not None else None
        ray_samples = RaySamples(
            frustums=frustums,
            times=tv if has_time_planes else None,
            metadata={"phase": pv} if pv is not None else None,
        )
        density, _ = field.get_density(ray_samples)                  # (B, 1), GPU
        out[start:end] = density.squeeze(-1).float().cpu()           # store on CPU

    return out.reshape(D, H, W)
