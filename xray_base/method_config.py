"""Method specification for x-ray reconstruction."""

from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import CosineDecaySchedulerConfig, ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.plugins.types import MethodSpecification

from xray_base.xray_model import XRayModelConfig
from xray_base.xray_kplanes_model import XRayKPlanesModelConfig
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
    description="Static X-ray attenuation reconstruction (MLP-based).",
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
    description="Dynamic X-ray reconstruction (MLP + D-NeRF deformation).",
)

# fmt: off
xray_kplanes = MethodSpecification(
    config=TrainerConfig(
        method_name="xray_kplanes",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30001,
        mixed_precision=True,
        pipeline=XrayPipelineConfig(
            datamanager=XrayDataManagerConfig(
                dataparser=RotatedXRayDataParserConfig(time_field="phase"),
                train_num_rays_per_batch=1024,
                eval_num_rays_per_batch=2048,
            ),
            model=XRayKPlanesModelConfig(
                eval_num_rays_per_chunk=4096,
                background_intensity=1.0,
                grid_base_resolution=(128, 128, 128),
                grid_feature_dim=32,
                multiscale_res=(1, 2, 4),
                num_samples_per_ray=48,
                proposal_net_args_list=[
                    {"num_output_coords": 8, "resolution": [64, 64, 64]},
                    {"num_output_coords": 8, "resolution": [128, 128, 128]},
                ],
                loss_coefficients={
                    "xray_loss": 1.0,
                    "interlevel": 1.0,
                    "distortion": 0.001,
                    "plane_tv": 0.01,
                    "plane_tv_proposal_net": 0.0001,
                },
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-12),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-12),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=4096),
        vis="viewer",
    ),
    description="Static X-ray reconstruction (K-Planes grid).",
)

xray_kplanes_dynamic = MethodSpecification(
    config=TrainerConfig(
        method_name="xray_kplanes_dynamic",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30001,
        mixed_precision=True,
        pipeline=XrayPipelineConfig(
            datamanager=XrayDataManagerConfig(
                dataparser=RotatedXRayDataParserConfig(time_field="phase"),
                train_num_rays_per_batch=1024,
                eval_num_rays_per_batch=2048,
            ),
            model=XRayKPlanesModelConfig(
                eval_num_rays_per_chunk=4096,
                background_intensity=1.0,
                grid_base_resolution=(128, 128, 128, 25),
                grid_feature_dim=32,
                multiscale_res=(1, 2, 4),
                num_samples_per_ray=48,
                proposal_net_args_list=[
                    {"num_output_coords": 8, "resolution": [64, 64, 64, 25]},
                    {"num_output_coords": 8, "resolution": [128, 128, 128, 25]},
                ],
                loss_coefficients={
                    "xray_loss": 1.0,
                    "interlevel": 1.0,
                    "distortion": 0.001,
                    "plane_tv": 0.1,
                    "plane_tv_proposal_net": 0.001,
                    "l1_time_planes": 0.001,
                    "l1_time_planes_proposal_net": 0.0001,
                    "time_smoothness": 0.1,
                    "time_smoothness_proposal_net": 0.01,
                },
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-12),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-12),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=4096),
        vis="viewer",
    ),
    description="Dynamic X-ray reconstruction (K-Planes grid + time planes).",
)
# fmt: on

