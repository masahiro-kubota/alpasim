import asyncio
import subprocess
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from alpasim_grpc.v0.common_pb2 import Pose, PoseAtTime, Quat
from alpasim_grpc.v0.common_pb2 import Trajectory as TrajectoryMsg
from alpasim_grpc.v0.common_pb2 import Vec3
from alpasim_grpc.v0.egodriver_pb2 import (
    DriveRequest,
    DriveResponse,
    DriveSessionRequest,
    RolloutCameraImage,
    RolloutEgoTrajectory,
    Route,
    RouteRequest,
)
from omegaconf import DictConfig, OmegaConf
from PIL import Image

import grpc.aio

from ..frame_cache import FrameCache
from ..main import VAMPolicyService


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _config_overrides(cfg: DictConfig, tmp_path: Path, assets_dir: Path) -> DictConfig:
    cfg.model.checkpoint_path = str(assets_dir / "VAM_width_768_pretrained_139k.pt")
    cfg.model.tokenizer_path = str(assets_dir / "VQ_ds16_16384_llamagen_encoder.jit")
    cfg.output_dir = str(tmp_path)
    cfg.port = 0

    return cfg


def _ensure_vavam_assets() -> Path:
    repo_root = _get_repo_root()
    assets_dir = repo_root / "data" / "vavam-driver"
    required_files = [
        assets_dir / "VAM_width_768_pretrained_139k.pt",
        assets_dir / "VQ_ds16_16384_llamagen_encoder.jit",
    ]

    if all(path.exists() for path in required_files):
        return assets_dir

    script_path = repo_root / "data" / "download_vavam_assets.sh"
    if not script_path.exists():
        pytest.skip("VaVAM assets missing and download script not found")

    try:
        subprocess.run(
            ["bash", str(script_path), "--model", "VaVAM-S"],
            check=True,
            cwd=script_path.parent,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"VaVAM assets missing and download failed: {exc}")

    if not all(path.exists() for path in required_files):
        pytest.skip("VaVAM assets missing after attempted download")

    return assets_dir


@pytest.mark.asyncio
async def test_vam_policy_drive_flow(tmp_path: Path) -> None:
    repo_root = _get_repo_root()
    cfg_path = repo_root / "src" / "wizard" / "configs" / "driver" / "vavam.yaml"
    cfg = OmegaConf.load(cfg_path)

    assets_dir = _ensure_vavam_assets()
    cfg = _config_overrides(cfg, tmp_path, assets_dir)

    device = torch.device(cfg.model.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if cfg.model.dtype == "float16" else torch.float32

    loop = asyncio.get_running_loop()
    server = grpc.aio.server()
    service = VAMPolicyService(
        cfg=cfg,
        loop=loop,
        grpc_server=server,
        device=device,
        dtype=dtype,
    )

    session_uuid = "test-session"
    rollout_spec = DriveSessionRequest.RolloutSpec()
    assert (
        len(cfg.inference.use_cameras) == 1
    ), "Only one camera is supported for testing"
    desired_camera_name = cfg.inference.use_cameras[0]
    camera_def = rollout_spec.vehicle.available_cameras.add()
    camera_def.logical_id = desired_camera_name
    camera_def.intrinsics.resolution_h = cfg.inference.image_height
    camera_def.intrinsics.resolution_w = cfg.inference.image_width
    ftheta = camera_def.intrinsics.ftheta_param
    ftheta.principal_point_x = cfg.inference.image_width / 2.0
    ftheta.principal_point_y = cfg.inference.image_height / 2.0
    ftheta.angle_to_pixeldist_poly.extend([0.0, 1.0])
    ftheta.pixeldist_to_angle_poly.extend([0.0, 1.0])
    start_request = DriveSessionRequest(
        session_uuid=session_uuid,
        random_seed=0,
        rollout_spec=rollout_spec,
    )

    try:
        await service.start_session(start_request, None)
        session = service._sessions[session_uuid]

        # Check multi-camera frame caches
        assert isinstance(session.frame_caches, dict)
        assert desired_camera_name in session.frame_caches
        assert isinstance(session.frame_caches[desired_camera_name], FrameCache)
        context = service._context_length
        assert context > 0

        base_ts = 1_000_000
        dt = 100_000

        # Submit egomotion observations one pose at a time (driver expects single pose per call)
        for i in range(context + 1):
            traj_msg = TrajectoryMsg()
            pose_at_time = PoseAtTime(
                pose=Pose(
                    vec=Vec3(x=float(i), y=0.0, z=0.0),
                    quat=Quat(w=1.0, x=0.0, y=0.0, z=0.0),
                ),
                timestamp_us=base_ts + i * dt,
            )
            traj_msg.poses.append(pose_at_time)

            egomotion_request = RolloutEgoTrajectory(
                session_uuid=session_uuid,
                trajectory=traj_msg,
            )
            await service.submit_egomotion_observation(egomotion_request, None)

        height = cfg.inference.image_height
        width = cfg.inference.image_width
        for i in range(context):
            image_array = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_array)
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            frame_start = base_ts + i * dt
            frame_end = frame_start + dt // 2

            image_request = RolloutCameraImage(
                session_uuid=session_uuid,
                camera_image=RolloutCameraImage.CameraImage(
                    logical_id=desired_camera_name,
                    frame_start_us=frame_start,
                    frame_end_us=frame_end,
                    image_bytes=buffer.getvalue(),
                ),
            )
            await service.submit_image_observation(image_request, None)

        route_msg = Route()
        waypoint = route_msg.waypoints.add()
        waypoint.x = float(context + 5)
        waypoint.y = 0.0
        waypoint.z = 0.0
        route_request = RouteRequest(
            session_uuid=session_uuid,
            route=route_msg,
        )
        await service.submit_route(route_request, None)

        drive_request = DriveRequest(
            session_uuid=session_uuid,
            time_now_us=base_ts + (context - 1) * dt,
            time_query_us=base_ts + context * dt,
        )

        response: DriveResponse = await service.drive(drive_request, None)

        assert len(response.trajectory.poses) > 1
        latest_pose = response.trajectory.poses[0]
        # Last submitted pose was at index=context with timestamp base_ts + context * dt
        expected_last_ts = base_ts + context * dt
        expected_last_x = float(context)
        assert latest_pose.timestamp_us == expected_last_ts
        assert pytest.approx(latest_pose.pose.vec.x, rel=1e-5) == expected_last_x
    finally:
        await service.stop_worker()
        if session_uuid in service._sessions:
            del service._sessions[session_uuid]
