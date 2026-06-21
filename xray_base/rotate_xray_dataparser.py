"""Dataparser for rotated x-ray reconstruction data."""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dataclass_field
from itertools import product
from pathlib import Path
from typing import Literal, Type, cast

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
    label_3d_filename: str = "label_3d.nii.gz"
    """Filename of the 3D ground-truth label (NIfTI) used for 3D metric evaluation.
    Set to empty string to disable."""
    label_use_aabb_roi: bool = True
    """If True, restrict 3D metrics to the axis-aligned bounding box of the GT label."""
    use_phase_as_time: bool = False
    """If True, use cardiac phase (``frame["phase"]``) as ``Cameras.times`` instead
    of wall-clock time. Useful for synthetic data where only cardiac (periodic)
    motion exists and wall-clock time has no meaningful temporal relationship."""
    eval_mode: Literal["uniform-interval", "all"] = "uniform-interval"
    """How to split the dataset into train/val.
    - ``uniform-interval``: keep first & last frames for training, evenly spread
      ``1 - train_ratio`` frames for validation (recommended for DSA sequences).
    - ``all``: use all frames for both train and val (no split)."""
    train_ratio: float = 0.8
    """Fraction of frames used for training. Only used when ``eval_mode`` is
    ``uniform-interval``."""


class RotatedXRayDataParser(DataParser):
    """Parse the rotated DSA dataset used for x-ray reconstruction."""

    def __init__(self, config: RotatedXRayDataParserConfig):
        super().__init__(config)
        self.includes_time = True  # time info provided in data

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

    @staticmethod
    def _compute_aabb_mask(label: np.ndarray) -> np.ndarray:
        """Return boolean mask of the axis-aligned bounding box of *label* (non-zero voxels)."""
        mask = label > 0
        if not mask.any():
            return np.zeros_like(mask, dtype=bool)
        coords = np.argwhere(mask)
        mn = coords.min(axis=0)
        mx = coords.max(axis=0)
        aabb = np.zeros_like(mask, dtype=bool)
        aabb[mn[0]:mx[0]+1, mn[1]:mx[1]+1, mn[2]:mx[2]+1] = True
        return aabb

    def _load_label_3d(self, data_dir: Path) -> dict | None:
        """Load the 3D ground-truth label NIfTI if it exists.

        Tries ``config.label_3d_filename`` first, then falls back to
        ``coronary_label.nii.gz`` for backward compatibility.

        Returns a dict with keys: ``data`` (D,H,W) uint8, ``affine`` (4,4) float32
        in *mm*, ``aabb`` (D,H,W) bool, or ``None`` if disabled / not found.
        """
        config = cast(RotatedXRayDataParserConfig, self.config)
        if not config.label_3d_filename:
            return None
        # Try configured filename, then known fallback names
        candidates = [data_dir / config.label_3d_filename]
        for fallback_name in ("coronary_label.nii.gz", "label_3d.nii.gz", "LCA_label.nii.gz"):
            fb = data_dir / fallback_name
            if str(fb) != str(candidates[0]) and fb.exists() and fb not in candidates:
                candidates.append(fb)
        label_path = None
        for p in candidates:
            if p.exists():
                label_path = p
                break
        if label_path is None:
            print(f"[RotatedXRayDataParser] 3D label not found at {candidates[0]}, skipping 3D metrics.")
            return None
        import nibabel as nib

        nii = nib.load(str(label_path))
        nii = cast(nib.Nifti1Image, nii)  # for type checker
        assert nii.affine is not None, "NIfTI label must have an affine"
        data = np.asanyarray(nii.get_fdata()).astype(np.uint8)
        aabb = self._compute_aabb_mask(data) if config.label_use_aabb_roi else np.ones_like(data, dtype=bool)
        return {
            "data": data,
            "affine": np.asarray(nii.affine, dtype=np.float32),
            "aabb": aabb,
            "filename": config.label_3d_filename,
        }

    def _cameras_from_metadata(self, metadata: dict):
        """Returns (Cameras, times_phase, times_wall)."""
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
        times_phase = []
        times_wall = []

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
            times_phase.append(float(frame["phase"]))
            times_wall.append(float(frame["time_s"]))

        camera_to_worlds = torch.stack(c2w_list, dim=0)
        
        config = cast(RotatedXRayDataParserConfig, self.config)

        times_wall_t = torch.tensor(times_wall, dtype=torch.float32)
        t_min, t_max = times_wall_t.min(), times_wall_t.max()
        times_wall_norm = (times_wall_t - t_min) / (t_max - t_min + 1e-8)  # [0, 1]

        phase_tensor = torch.tensor(times_phase, dtype=torch.float32).unsqueeze(-1)  # (N, 1)

        if config.use_phase_as_time:
            times = phase_tensor  # phase → Cameras.times
        else:
            times = times_wall_norm.unsqueeze(-1)  # wall-clock time → Cameras.times

        return (
            Cameras(
                camera_to_worlds=camera_to_worlds,
                fx=torch.full((num_frames, 1), fx, dtype=torch.float32),
                fy=torch.full((num_frames, 1), fy, dtype=torch.float32),
                cx=torch.full((num_frames, 1), cx, dtype=torch.float32),
                cy=torch.full((num_frames, 1), cy, dtype=torch.float32),
                width=torch.full((num_frames, 1), width, dtype=torch.int64),
                height=torch.full((num_frames, 1), height, dtype=torch.int64),
                camera_type=CameraType.PERSPECTIVE,
                times=times,
                metadata={"phase": phase_tensor},
            ), 
            torch.tensor(times_phase, dtype=torch.float32), 
            times_wall_norm,
        )

    def _get_split_indices(self, n: int) -> dict[str, list[int]]:
        """Compute train/val/test index lists following UniformIntervalSpliter.

        Keeps first and last frames for training (they correspond to phase-0 and
        define the view-angle span), then evenly spreads validation frames across
        the interior.

        Returns:
            dict with keys ``"train"``, ``"val"``, ``"test"`` (val == test).
        """
        config = cast(RotatedXRayDataParserConfig, self.config)
        if config.eval_mode == "all":
            return {"train": list(range(n)), "val": list(range(n)), "test": list(range(n))}

        target_val = n - int(n * config.train_ratio)
        if target_val <= 0 or n <= 2:
            return {"train": list(range(n)), "val": [], "test": []}

        interior = n - 2                     # frames in (0, n-1)
        val_len = min(target_val, interior)

        if val_len == 1:
            val_set = {n // 2}
        else:
            step = (interior - 1) / (val_len - 1)
            val_set = {int(round(1 + i * step)) for i in range(val_len)}

        val_set = {max(1, min(n - 2, v)) for v in val_set}

        # backfill if rounding produced fewer unique positions than needed
        remaining = sorted(set(range(1, n - 1)) - val_set)
        while len(val_set) < val_len and remaining:
            best = max(remaining, key=lambda x: min(abs(x - v) for v in val_set))
            val_set.add(best)
            remaining.remove(best)

        val_idx = sorted(val_set)[:val_len]
        train_idx = [i for i in range(n) if i not in val_idx]
        return {"train": train_idx, "val": val_idx, "test": val_idx}

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

        cameras, times_phase, times_wall = self._cameras_from_metadata(metadata)
        if len(image_filenames) != len(cameras):
            raise RuntimeError(
                f"Image count ({len(image_filenames)}) does not match camera count ({len(cameras)})."
            )

        # ---- Coordinate normalization: bring mm-scale coordinates to roughly [-1, 1] ----
        aabb = scene_box.aabb  # (2, 3), in mm

        #  scale to [-1, 1] + translation
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
            metadata=cameras.metadata,
        )

        # Store forward transform for exporting back to original coordinates
        dataparser_transform = torch.cat(
            [torch.eye(3, dtype=torch.float32) * scale, torch.zeros(3, 1, dtype=torch.float32)],
            dim=-1,
        )  # (3, 4)
        # ----------------------------------------------------------------------

        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        # ---- Train/Val split -------------------------------------------------
        split_idx = self._get_split_indices(len(image_filenames))
        indices = split_idx[split]  # list[int]

        if not indices:
            # No data for this split — still return an empty-but-valid DataparserOutputs
            # (nerfstudio handles empty eval datasets gracefully.)
            empty_cam = cameras[:0]
            return DataparserOutputs(
                image_filenames=[],
                cameras=empty_cam,
                scene_box=scene_box,
                dataparser_transform=dataparser_transform,
                dataparser_scale=1.0,
                metadata={},
            )

        image_filenames = [image_filenames[i] for i in indices]
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        cameras = cameras[idx_tensor]  # Cameras (TensorDataclass) requires tensor indexing
        times_phase = times_phase[indices] if indices else times_phase[:0]
        times_wall = times_wall[indices] if indices else times_wall[:0]
        # ----------------------------------------------------------------------

        label_3d = self._load_label_3d(data_dir)

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
                "time_s": times_wall,
                "phase": times_phase,
                "label_3d": label_3d,
                # scale factor applied to world (mm) coords to normalize to ~[-1,1]
                "world_scale": torch.tensor(scale, dtype=torch.float32),
            },
        )