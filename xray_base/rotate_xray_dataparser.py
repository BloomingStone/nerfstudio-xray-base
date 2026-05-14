"""Dataparser for rotated x-ray reconstruction data."""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dataclass_field
from itertools import product
from pathlib import Path
from typing import Type, cast

import numpy as np
import torch

from nerfstudio.cameras.cameras import CameraType, Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        stacked = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        stacked = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        stacked = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("axis must be one of X, Y, Z")

    return torch.stack(stacked, dim=-1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    if euler_angles.ndim == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Euler angles must have shape (..., 3).")
    if len(convention) != 3:
        raise ValueError("Convention must contain three axis letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in {"X", "Y", "Z"}:
            raise ValueError(f"Invalid axis {letter} in convention.")

    matrices = [_axis_angle_rotation(axis, angle) for axis, angle in zip(convention, torch.unbind(euler_angles, -1))]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


@dataclass
class RotatedXRayDataParserConfig(DataParserConfig):
    """Configuration for the x-ray dataparser."""

    _target: Type = dataclass_field(default_factory=lambda: RotatedXRayDataParser)
    image_dirname: str = "rotate_dsa"
    json_filename: str = "rotate_dsa.json"


class RotatedXRayDataParser(DataParser):
    """Parse the rotated DSA dataset used for x-ray reconstruction."""

    def __init__(self, config: RotatedXRayDataParserConfig):
        super().__init__(config)
        self.includes_time = False  # x-ray data is static; set True if adding temporal/deformation later

    def _load_json(self) -> tuple[dict, Path]:
        config = cast(RotatedXRayDataParserConfig, self.config)
        data_path = config.data
        if data_path.suffix == ".json":
            json_path = data_path
            data_dir = data_path.parent
        else:
            data_dir = data_path
            json_path = data_dir / config.json_filename

        if not json_path.exists():
            raise FileNotFoundError(f"Could not find x-ray metadata json: {json_path}")
        if not data_dir.exists():
            raise FileNotFoundError(f"Could not find x-ray dataset directory: {data_dir}")

        with open(json_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return metadata, data_dir

    @staticmethod
    def _volume_aabb(affine: np.ndarray, volume_size: list[int], margin: float = 0) -> torch.Tensor:
        """
        Compute the axis-aligned bounding box of a volume given its affine transformation and size.
        Args:
            affine: (4, 4) affine transformation matrix mapping voxel coordinates to world coordinates (in mm).
            volume_size: (3,) size of the volume in voxels, ordered as (X, Y, Z).
            margin: additional margin (in mm) to add to the bounding box on all sides. default is 200mm, The average 
            transverse diameter of the chest is approximately 299 mm for men and 277 mm for women 
        Returns:
            aabb: (2, 3) tensor containing the minimum and maximum (x, y, z) coordinates of the bounding box in world 
            space (in mm).
        """
        
        size = np.asarray(volume_size, dtype=np.float32)
        corners = np.array(list(product([0.0, size[0]], [0.0, size[1]], [0.0, size[2]])), dtype=np.float32)
        corners_h = np.concatenate([corners, np.ones((corners.shape[0], 1), dtype=np.float32)], axis=1)
        world = (affine.astype(np.float32) @ corners_h.T).T[:, :3]
        world_min = world.min(axis=0) - margin
        world_max = world.max(axis=0) + margin
        return torch.tensor(np.stack([world_min, world_max], axis=0), dtype=torch.float32)

    def _cameras_from_metadata(self, metadata: dict) -> Cameras:
        c_arm = metadata["c_arm_geometry"]
        rotate = metadata["rotate_parameters"]
        frames = metadata["frames"]
        num_frames = len(frames)

        convention = rotate["convention"]
        sdd = float(c_arm["sdd"])
        sod = float(c_arm["sod"])
        fx = sdd / float(c_arm["delx"])
        fy = sdd / float(c_arm["dely"])
        width = int(c_arm["width"])
        height = int(c_arm["height"])
        cx = width / 2.0
        cy = height / 2.0

        r_nerfstudio_orient = torch.tensor([
            [-1.0,  0.0, 0.0],
            [0.0,   0.0, 1.0],
            [0.0,   1.0, 0.0],
        ])
        m_nerfstudio_orient = torch.eye(4, dtype=torch.float32)
        m_nerfstudio_orient[:3, :3] = r_nerfstudio_orient

        c2w_list = []
        times = []
        for frame in frames:
            alpha = torch.tensor(float(frame["alpha_degree"]) / 180.0 * torch.pi, dtype=torch.float32)
            beta = torch.tensor(float(frame["beta_degree"]) / 180.0 * torch.pi, dtype=torch.float32)
            angles = torch.stack((-alpha, -beta, torch.tensor(0.0, dtype=torch.float32)))
            m_rotation = torch.eye(4, dtype=torch.float32)
            m_rotation[:3, :3] = euler_angles_to_matrix(angles, convention)
            m_translation = torch.eye(4, dtype=torch.float32)
            m_translation[:3, 3] = torch.tensor([0.0, sod, 0.0], dtype=torch.float32)
            m_c2w = m_rotation @ m_translation @ m_nerfstudio_orient
            c2w_list.append(m_c2w[:3, :4])
            times.append(float(frame["phase"]))

        camera_to_worlds = torch.stack(c2w_list, dim=0)
        return Cameras(
            camera_to_worlds=camera_to_worlds,
            fx=torch.full((num_frames, 1), fx, dtype=torch.float32),
            fy=torch.full((num_frames, 1), fy, dtype=torch.float32),
            cx=torch.full((num_frames, 1), cx, dtype=torch.float32),
            cy=torch.full((num_frames, 1), cy, dtype=torch.float32),
            width=torch.full((num_frames, 1), width, dtype=torch.int64),
            height=torch.full((num_frames, 1), height, dtype=torch.int64),
            camera_type=CameraType.PERSPECTIVE,
            times=torch.tensor(times, dtype=torch.float32).unsqueeze(-1),
        )

    def _generate_dataparser_outputs(self, split: str = "train", **kwargs) -> DataparserOutputs:
        config = cast(RotatedXRayDataParserConfig, self.config)
        metadata, data_dir = self._load_json()
        image_dir = data_dir / config.image_dirname
        image_filenames = sorted(image_dir.glob("*.png"))
        if not image_filenames:
            raise FileNotFoundError(f"No PNG images found in {image_dir}")

        coronary_type = metadata["coronary_type"]
        assert coronary_type in {"LCA", "RCA"}, f"Unsupported coronary type: {coronary_type}"
        print(f"Parsing {len(image_filenames)} images for coronary type {coronary_type} with convention {metadata['rotate_parameters']['convention']}")
        
        affine_key = f"{coronary_type.lower()}_centering_affine"
        volume_size = [int(x) for x in metadata["volume_size"]]
        affine = np.array(metadata[affine_key], dtype=np.float32).reshape(4, 4)
        scene_box = SceneBox(aabb=self._volume_aabb(affine, volume_size))

        cameras = self._cameras_from_metadata(metadata)
        if len(image_filenames) != len(cameras):
            raise RuntimeError(
                f"Image count ({len(image_filenames)}) does not match camera count ({len(cameras)})."
            )

        # ---- Coordinate normalization: bring mm-scale coordinates to roughly [-1, 1] ----
        aabb = scene_box.aabb  # (2, 3), in mm

        # # 1) Apply nerfstudio world-transform (COLMAP → nerfstudio convention)
        # #    Equivalent to _apply_nerfstudio_world_transform in the COLMAP conversion script.
        # #    Flips the camera look-direction so frustums point toward the volume.
        # c2w[:, :, 1:3] *= -1                    # negate Y / Z columns of rotation
        # c2w = c2w[:, [0, 2, 1], :]              # swap rows 1↔2
        # c2w[:, 2, :] *= -1                      # negate new row 2

        # 2) Center & scale to [-1, 1] + translation
        extent = abs(aabb[1] - aabb[0])  # (3,)
        scale = 2.0 / extent.max().item()  # largest dimension → [-1, 1]

        # Rescale scene box
        scene_box = SceneBox(aabb=aabb * scale)

        # Rescale camera positions (translation only)
        c2w = cameras.camera_to_worlds.clone()  # (N, 3, 4)
        c2w[:, :, 3] = c2w[:, :, 3] * scale

        cameras = Cameras(
            camera_to_worlds=c2w,
            fx=cameras.fx,
            fy=cameras.fy,
            cx=cameras.cx,
            cy=cameras.cy,
            width=cameras.width,
            height=cameras.height,
            camera_type=cameras.camera_type,
            times=cameras.times,
        )

        # Store forward transform for exporting back to original coordinates
        dataparser_transform = torch.cat(
            [torch.eye(3, dtype=torch.float32) * scale, torch.zeros(3, 1, dtype=torch.float32)],
            dim=-1,
        )  # (3, 4)
        # ----------------------------------------------------------------------

        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        return DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            dataparser_transform=dataparser_transform,
            dataparser_scale=1.0,
            metadata={
                "xray_metadata": metadata,
                "affine": torch.tensor(affine, dtype=torch.float32),
                "volume_size": torch.tensor(volume_size, dtype=torch.int64),
            },
        )