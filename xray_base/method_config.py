"""Method specification for x-ray reconstruction."""

from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from xray_base.xray_model import XRayModelConfig
from xray_base.rotate_xray_dataparser import RotatedXRayDataParserConfig
from xray_base.xray_pipeline import XrayPipelineConfig
from xray_base.xray_datamanager import XrayDataManagerConfig


xray_base = MethodSpecification(
    config=TrainerConfig(
        method_name="xray_base",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=XrayPipelineConfig(
            datamanager=XrayDataManagerConfig(
                dataparser=RotatedXRayDataParserConfig(),
                train_num_rays_per_batch=2048,
                eval_num_rays_per_batch=4096,
            ),
            model=XRayModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                num_samples_per_ray=256,
                background_intensity=1.0,
                max_optical_depth=80.0,
            ),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="X-ray attenuation reconstruction method.",
)

