"""Method specification for x-ray reconstruction."""

from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
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
                train_num_rays_per_batch=512,
                eval_num_rays_per_batch=2048,
            ),
            model=XRayModelConfig(
                eval_num_rays_per_chunk=2048,
                num_samples_per_ray=256,
                background_intensity=1.0,
            ),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=10000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=2048),
        vis="viewer",
    ),
    description="Static X-ray attenuation reconstruction.",
)

xray_dynamic = MethodSpecification(
    config=TrainerConfig(
        method_name="xray_dynamic",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=50000,
        mixed_precision=True,
        pipeline=XrayPipelineConfig(
            datamanager=XrayDataManagerConfig(
                dataparser=RotatedXRayDataParserConfig(),
                train_num_rays_per_batch=512,
                eval_num_rays_per_batch=2048,
            ),
            model=XRayModelConfig(
                eval_num_rays_per_chunk=2048,
                num_samples_per_ray=256,
                background_intensity=1.0,
                enable_temporal_distortion=True,
                temporal_distortion_params={
                    "kind": TemporalDistortionKind.DNERF,
                    "mlp_num_layers": 2,
                    "mlp_layer_width": 64,
                },
            ),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=50000),
            },
            "temporal_distortion": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-6, max_steps=50000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=2048),
        vis="viewer",
    ),
    description="Dynamic X-ray reconstruction with temporal deformation field.",
)

