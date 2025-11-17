# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""VAM Driver implementation for Alpasim."""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import pickle
import queue
import threading
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import IntEnum
from importlib.metadata import version
from io import BytesIO
from typing import Any, Callable, Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf.dictconfig
import omegaconf.listconfig
import torch
import torch.serialization
from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0 import sensorsim_pb2
from alpasim_grpc.v0.common_pb2 import (
    Empty,
    Pose,
    PoseAtTime,
    Quat,
    SessionRequestStatus,
    Trajectory,
    Vec3,
    VersionId,
)
from alpasim_grpc.v0.egodriver_pb2 import (
    DriveRequest,
    DriveResponse,
    DriveSessionCloseRequest,
    DriveSessionRequest,
    GroundTruthRequest,
    RolloutCameraImage,
    RolloutEgoTrajectory,
    Route,
    RouteRequest,
)
from alpasim_grpc.v0.egodriver_pb2_grpc import (
    EgodriverServiceServicer,
    add_EgodriverServiceServicer_to_server,
)
from omegaconf import OmegaConf
from PIL import Image
from vam.action_expert import VideoActionModelInference
from vam.datalib.transforms import NeuroNCAPTransform

import grpc
import grpc.aio

from .frame_cache import FrameCache, FrameEntry
from .rectification import FthetaToPinholeRectifier, build_rectifier_map
from .schema import VAMDriverConfig

logger = logging.getLogger(__name__)


class DriveCommand(IntEnum):
    """Discrete high-level maneuver commands passed to the VAM."""

    RIGHT = 0
    LEFT = 1
    STRAIGHT = 2


torch.serialization.add_safe_globals(
    # Let torch.load's safe unpickler recreate OmegaConf containers embedded in checkpoints.
    [
        omegaconf.listconfig.ListConfig,
        omegaconf.dictconfig.DictConfig,
    ]
)


def load_inference_VAM(
    checkpoint_path: str,
    device: torch.device | str = "cuda",
    tempdir: Optional[str] = None,
) -> VideoActionModelInference:
    """Custom loader that handles PyTorch 2.6+ weights_only issue."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = ckpt["hyper_parameters"]["vam_conf"].copy()
    config.pop("_target_", None)
    config.pop("_recursive_", None)
    config["gpt_checkpoint_path"] = None
    config["action_checkpoint_path"] = None
    config["gpt_mup_base_shapes"] = None
    config["action_mup_base_shapes"] = None

    vam = VideoActionModelInference(**config)
    state_dict = OrderedDict()
    for key, value in ckpt["state_dict"].items():
        state_dict[key.replace("vam.", "")] = value
    vam.load_state_dict(state_dict, strict=True)
    vam = vam.eval().to(device)
    return vam


def _format_trajs(trajs: torch.Tensor) -> np.ndarray:
    """Normalize VAM trajectory tensor shape to (T, 2)."""

    array = trajs.detach().float().cpu().numpy()
    while array.ndim > 2 and array.shape[0] == 1:
        array = array.squeeze(0)

    if array.ndim != 2:
        raise ValueError(f"Unexpected trajectory shape {array.shape}")

    return array


def _quat_to_yaw(quaternion: Quat) -> float:
    """Extract the yaw component (rotation about +Z) from a quaternion."""

    return np.arctan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def _yaw_to_quat(yaw: float) -> Quat:
    """Create a Z-only rotation quaternion from the provided yaw angle."""

    half_yaw = 0.5 * yaw
    return Quat(w=float(np.cos(half_yaw)), x=0.0, y=0.0, z=float(np.sin(half_yaw)))


def _rig_est_offsets_to_local_positions(
    current_pose_in_local: PoseAtTime, offsets_in_rig: np.ndarray
) -> np.ndarray:
    """Project rig-est displacements onto the local-frame pose anchored by `current_pose`."""

    curr_x = current_pose_in_local.pose.vec.x
    curr_y = current_pose_in_local.pose.vec.y

    curr_quat = current_pose_in_local.pose.quat
    curr_yaw = _quat_to_yaw(curr_quat)

    cos_yaw = np.cos(curr_yaw)
    sin_yaw = np.sin(curr_yaw)

    offsets_array = np.asarray(offsets_in_rig, dtype=float).reshape(-1, 2)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=float)
    rotated_offsets = offsets_array @ rotation.T

    translation = np.array([curr_x, curr_y], dtype=float)
    return rotated_offsets + translation


# Unique queue marker instructing the worker thread to flush and exit.
_SENTINEL_JOB = object()


@dataclass
class DriveJob:
    """Unit of work processed by the background inference worker."""

    session_id: str
    session: "Session"
    command: DriveCommand
    pose: Optional[PoseAtTime]
    timestamp_us: int
    result: asyncio.Future[DriveResponse]


@dataclass
class Session:
    """Represents a VAM session."""

    uuid: str
    seed: int
    debug_scene_id: str

    frame_cache: FrameCache
    available_cameras_logical_ids: set[str]
    desired_cameras_logical_ids: set[str]
    rectifiers: dict[str, Optional[FthetaToPinholeRectifier]] = field(
        default_factory=dict
    )
    poses: list[PoseAtTime] = field(default_factory=list)
    current_command: DriveCommand = DriveCommand.STRAIGHT  # Default to straight

    @staticmethod
    def create(
        request: DriveSessionRequest,
        cfg: VAMDriverConfig,
        context_length: int,
    ) -> Session:
        """Create a new VAM session."""
        debug_scene_id = (
            request.debug_info.scene_id
            if request.debug_info is not None
            else request.session_uuid
        )

        available_cameras_logical_ids: set[str] = set()
        vehicle = request.rollout_spec.vehicle
        if vehicle is None:
            raise ValueError("Vehicle definition is required in DriveSessionRequest")

        camera_specs: dict[
            str, sensorsim_pb2.AvailableCamerasReturn.AvailableCamera
        ] = {}
        for camera_def in vehicle.available_cameras:
            if not camera_def.logical_id:
                raise ValueError(
                    "Logical ID is required for each camera in VehicleDefinition"
                )
            available_cameras_logical_ids.add(camera_def.logical_id)
            camera_specs[camera_def.logical_id] = camera_def

        desired_cameras_logical_ids = set(cfg.inference.use_cameras)
        if not desired_cameras_logical_ids:
            raise ValueError("No cameras specified in inference configuration")
        if not len(desired_cameras_logical_ids) == 1:
            raise ValueError("Only one camera is supported for now.")

        rectifiers = build_rectifier_map(
            cfg.rectification,
            desired_cameras_logical_ids,
            camera_specs,
        )

        session = Session(
            uuid=request.session_uuid,
            seed=request.random_seed,
            debug_scene_id=debug_scene_id,
            frame_cache=FrameCache(context_length),
            available_cameras_logical_ids=available_cameras_logical_ids,
            desired_cameras_logical_ids=desired_cameras_logical_ids,
            rectifiers=rectifiers,
        )

        return session

    def add_image(self, image_tensor: np.ndarray, timestamp_us: int) -> None:
        """Add an image observation."""
        self.frame_cache.add_image(timestamp_us, image_tensor)

    def rectify_image(self, logical_id: str, image: Image.Image) -> Image.Image:
        """Apply rectification for logical_id if configured."""
        rectifier = self.rectifiers.get(logical_id)
        if rectifier is None:
            return image
        image_np = np.array(image)
        rectified = rectifier.rectify(image_np)
        return Image.fromarray(rectified)

    def add_egoposes(self, egoposes: Trajectory) -> None:
        """Add rig-est pose observations in the local frame."""
        self.poses.extend(egoposes.poses)
        self.poses = sorted(self.poses, key=lambda pose: pose.timestamp_us)
        logger.debug(f"poses: {self.poses}")

    def update_command_from_route(
        self,
        route: Route,
        use_waypoint_commands: bool,
        command_distance_threshold: Optional[float] = None,
        min_lookahead_distance: Optional[float] = None,
    ) -> None:
        """Derive command from waypoints using VAM-style logic.
        Note: this is called for RouteRequest and assumed to be in the
        true rig frame.
        Args:
            route: Route containing waypoints in the rig frame.
            use_waypoint_commands: Whether to derive commands from waypoints.
            command_distance_threshold: Lateral distance threshold (meters) for
                determining turn commands. Waypoints beyond this threshold trigger
                LEFT/RIGHT commands.
            min_lookahead_distance: Minimum forward distance (meters) to consider
                a waypoint as the target for command derivation.
        """
        if not use_waypoint_commands or len(route.waypoints) < 1:
            return

        if len(self.poses) == 0:
            return

        if command_distance_threshold is None or min_lookahead_distance is None:
            raise ValueError(
                "command_distance_threshold and min_lookahead_distance must be provided "
                "when use_waypoint_commands is True"
            )

        target_waypoint = None
        for wp in route.waypoints:
            distance = np.hypot(wp.x, wp.y)

            if distance >= min_lookahead_distance:
                target_waypoint = wp
                break

        if target_waypoint is None:
            return

        dy_rig = target_waypoint.y  # already in rig frame (positive is left)

        if dy_rig > command_distance_threshold:
            self.current_command = DriveCommand.LEFT
        elif dy_rig < -command_distance_threshold:
            self.current_command = DriveCommand.RIGHT
        else:
            self.current_command = DriveCommand.STRAIGHT

        logger.debug(
            "Command: %s (lateral displacement: %.2fm)",
            self.current_command.name,
            dy_rig,
        )


def async_log_call(func: Callable) -> Callable:
    """Helper to add logging for gRPC calls (sync or async)."""

    @functools.wraps(func)
    async def async_wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            logger.debug("Calling %s", func.__name__)
            return await func(*args, **kwargs)
        except Exception:  # pragma: no cover - logging assistance
            logger.exception("Exception in %s", func.__name__)
            raise

    return async_wrapped


class VAMPolicyService(EgodriverServiceServicer):
    """VAM Policy service implementing the Alpasim ego driver interface."""

    def __init__(
        self,
        cfg: VAMDriverConfig,
        loop: asyncio.AbstractEventLoop,
        grpc_server: grpc.aio.Server,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Initialize the VAM Policy service.

        Sets up the VAM model, image tokenizer, preprocessing pipeline, and starts
        a background worker thread for batched inference processing.

        Args:
            cfg: Hydra configuration containing model paths and inference settings
            loop: Asyncio event loop for coordinating async operations and scheduling
                futures from the worker thread back to the async gRPC handlers
            grpc_server: gRPC server instance for service registration
            device: PyTorch device (CPU or CUDA) for model execution
            dtype: PyTorch data type for tensor operations
        """

        # Private members
        self._cfg = cfg
        self._loop = loop
        self._grpc_server = grpc_server

        self._device = device
        self._dtype = dtype
        self._image_tokenizer = torch.jit.load(
            cfg.model.tokenizer_path, map_location=self._device
        )
        self._image_tokenizer.to(self._device)
        self._image_tokenizer.eval()

        self._vam = load_inference_VAM(cfg.model.checkpoint_path, self._device)
        self._preproc_pipeline = NeuroNCAPTransform()
        self._use_autocast = self._device.type == "cuda"

        self._max_batch_size = cfg.inference.max_batch_size
        self._job_queue: queue.Queue[DriveJob | object] = queue.Queue()
        self._worker_stop = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker_main,
            name="vam-policy-worker",
            daemon=True,
        )
        self._context_length = self._cfg.inference.context_length
        self._sessions: dict[str, Session] = {}
        self._worker_thread.start()

    async def stop_worker(self) -> None:
        """Signal the worker thread to stop and wait for it to exit."""
        if not self._worker_stop.is_set():
            self._worker_stop.set()
            self._job_queue.put_nowait(_SENTINEL_JOB)
        if self._worker_thread.is_alive():
            await asyncio.to_thread(self._worker_thread.join)

    def _worker_main(self) -> None:
        """Blocking worker loop that batches drive jobs for inference."""
        torch.set_grad_enabled(False)
        while True:
            if self._worker_stop.is_set():
                break

            # Get at least one job
            try:
                job = self._job_queue.get()
            except queue.Empty:
                continue

            # Check if we should stop
            if job is _SENTINEL_JOB:
                break

            batch: list[DriveJob] = [job]

            # Get as many jobs as we can
            stop_after_batch = False
            while len(batch) < self._max_batch_size:
                try:
                    next_job = self._job_queue.get_nowait()
                except queue.Empty:
                    break
                if next_job is _SENTINEL_JOB:
                    stop_after_batch = True
                    break
                batch.append(next_job)

            try:
                logging.info("Running VAM batch of size %s", len(batch))
                responses = self._run_batch(batch)
            except Exception as exc:
                logger.exception("VAM batch failed")
                for pending_job in batch:
                    self._loop.call_soon_threadsafe(
                        pending_job.result.set_exception, exc
                    )
            else:
                logging.info("VAM batch succeeded")
                for pending_job, response in zip(batch, responses, strict=True):
                    self._loop.call_soon_threadsafe(
                        pending_job.result.set_result, response
                    )

            if stop_after_batch:
                break

        # Signal the worker thread to stop
        self._worker_stop.set()
        while True:
            try:
                leftover = self._job_queue.get_nowait()
            except queue.Empty:
                break
            if leftover is _SENTINEL_JOB:
                continue
            self._loop.call_soon_threadsafe(leftover.result.cancel)

    def _tokenize_frames(self, batch: list[DriveJob]) -> None:
        """Tokenize frames for the given batch."""
        frame_entries_to_tokenize: list[tuple[DriveJob, FrameEntry]] = []

        # Which frames are still pending tokenization?
        for job in batch:
            frame_entries_to_tokenize.extend(
                (job, frame_entry)
                for frame_entry in job.session.frame_cache.pending_frames()
            )

        if frame_entries_to_tokenize:
            # Extract out the images and tokenize them.
            images: list[np.ndarray] = []
            owners: list[tuple[DriveJob, FrameEntry]] = []
            for job, frame in frame_entries_to_tokenize:
                images.append(frame.image)
                owners.append((job, frame))

            token_batches = self._tokenize_batch(images)

            # Add the tokens to the frame cache.
            for (_, frame), tokens in zip(owners, token_batches, strict=True):
                frame.tokens = tokens

    def _tokenize_batch(self, images: list[np.ndarray]) -> list[torch.Tensor]:
        if not images:
            return []

        tensors = [self._preproc_pipeline(image) for image in images]
        batch = torch.stack(tensors, dim=0).to(self._device)
        autocast_ctx = (
            torch.amp.autocast(self._device.type, dtype=self._dtype)
            if self._use_autocast
            else nullcontext()
        )
        with torch.no_grad():
            with autocast_ctx:
                token_batch = self._image_tokenizer(batch)
        return [tokens.detach().cpu() for tokens in token_batch]

    def _maybe_plot_batch_results(
        self,
        batch: list[DriveJob],
        context_images: list[list[np.ndarray | None]],
        vam_trajectories: list[np.ndarray],
        alpasim_trajectories: list[Trajectory],
        timestamps: list[int],
        commands: list[DriveCommand],
    ) -> None:
        """Plot all context images and trajectories for debugging."""
        if not self._cfg.plot_debug_images:
            return

        if not self._cfg.output_dir:
            msg = "Output directory is not set but we want to plot output"
            raise ValueError(msg)

        # Plot and save one file per job
        for i, (job, imgs, vam_traj, alpasim_traj, ts, cmd) in enumerate(
            zip(
                batch,
                context_images,
                vam_trajectories,
                alpasim_trajectories,
                timestamps,
                commands,
                strict=True,
            )
        ):
            context_length = len(imgs)

            # Get timestamps for all frames in context
            frame_timestamps = [
                entry.timestamp_us for entry in job.session.frame_cache.entries
            ]

            # Create figure: one row with all context frames + trajectory
            # Layout: 1 row, (context_length + 1) columns
            # [context_frame_0, ..., context_frame_N-1, trajectory]
            plt.figure(figsize=(3 * (context_length + 1), 4))

            # Plot all context frames for this job
            for j in range(context_length):
                subplot_idx = j + 1
                ax_img = plt.subplot(1, context_length + 1, subplot_idx)
                if imgs[j] is not None:
                    ax_img.imshow(imgs[j])

                # Add timestamp to each frame
                frame_ts = frame_timestamps[j] / 1e6  # Convert to seconds
                if j == context_length - 1:
                    # Latest frame: show timestamp and command
                    ax_img.set_title(
                        f"t={frame_ts:.3f}s\ncmd={cmd.name}",
                        fontsize=9,
                    )
                else:
                    # Older frames: show timestamp
                    ax_img.set_title(f"t={frame_ts:.3f}s", fontsize=9)
                ax_img.axis("off")

            # Plot trajectories in the last column
            ax_traj = plt.subplot(1, context_length + 1, context_length + 1)

            # Plot VAM trajectory (raw model output)
            ax_traj.plot(
                vam_traj[:, 0],
                vam_traj[:, 1],
                "b-o",
                label="VAM output",
                markersize=4,
            )

            # Plot Alpasim trajectory (converted to world frame)
            if len(alpasim_traj.poses) > 0:
                alpasim_x = [pose.pose.vec.x for pose in alpasim_traj.poses]
                alpasim_y = [pose.pose.vec.y for pose in alpasim_traj.poses]
                ax_traj.plot(
                    alpasim_x,
                    alpasim_y,
                    "r-s",
                    label="Alpasim trajectory",
                    markersize=4,
                )

            ax_traj.set_xlabel("X (m)", fontsize=8)
            ax_traj.set_ylabel("Y (m)", fontsize=8)
            ax_traj.set_title("Trajectory", fontsize=9)
            ax_traj.legend(fontsize=7)
            ax_traj.grid(True, alpha=0.3)
            ax_traj.axis("equal")

            plt.tight_layout()

            # Create session-specific subfolder
            session_folder = os.path.join(
                self._cfg.output_dir, job.session.debug_scene_id
            )
            os.makedirs(session_folder, exist_ok=True)

            # Save file per job with timestamp
            output_path = os.path.join(session_folder, f"vam_debug_{ts}.png")
            plt.savefig(output_path, dpi=100)
            plt.close()

            # Print timestamp and command info
            logger.info(
                "Job[%s]: timestamp=%.3fs, command=%s (%s), saved to %s",
                i,
                ts / 1e6,
                cmd.name,
                int(cmd),
                output_path,
            )

    def _run_batch(self, batch: list[DriveJob]) -> list[DriveResponse]:
        self._tokenize_frames(batch)

        inputs: list[torch.Tensor] = []
        commands: list[DriveCommand] = []
        timestamps: list[int] = []
        context_images: list[list[np.ndarray | None]] = []

        for job in batch:
            token_window = job.session.frame_cache.latest_token_window()
            tensor = torch.stack(token_window, dim=0)  # (T, C, H, W)
            inputs.append(tensor)
            commands.append(job.command)
            timestamps.append(job.timestamp_us)

            # Get all images and timestamps in the context window
            images_in_context = [
                entry.image for entry in job.session.frame_cache.entries
            ]
            context_images.append(images_in_context)

        visual_tokens = torch.stack(inputs, dim=0).to(self._device)  # (B, T, C, H, W)
        command_tensor = torch.tensor(  # (B, 1)
            [int(cmd) for cmd in commands], device=self._device, dtype=torch.long
        ).unsqueeze(-1)

        autocast_ctx = (
            torch.amp.autocast(self._device.type, dtype=self._dtype)
            if self._use_autocast
            else nullcontext()
        )

        with torch.no_grad():
            with autocast_ctx:
                trajectories = self._vam(visual_tokens, command_tensor, self._dtype)
        trajectories = trajectories.detach().cpu()

        responses: list[DriveResponse] = []
        vam_trajectories: list[np.ndarray] = []
        alpasim_trajectories: list[Trajectory] = []

        for job, traj, now_us in zip(batch, trajectories, timestamps, strict=True):
            np_traj = _format_trajs(traj)
            alpasim_traj = self._convert_vam_trajectory_to_alpasim(
                np_traj, job.pose, now_us
            )
            responses.append(
                DriveResponse(
                    trajectory=alpasim_traj,
                )
            )
            vam_trajectories.append(np_traj)
            alpasim_trajectories.append(alpasim_traj)

        # Plot batch results for debugging
        self._maybe_plot_batch_results(
            batch,
            context_images,
            vam_trajectories,
            alpasim_trajectories,
            timestamps,
            commands,
        )

        return responses

    @async_log_call
    async def start_session(
        self, request: DriveSessionRequest, context: grpc.aio.ServicerContext
    ) -> SessionRequestStatus:
        if request.session_uuid in self._sessions:
            context.abort(
                grpc.StatusCode.ALREADY_EXISTS,
                f"Session {request.session_uuid} already exists.",
            )
            return SessionRequestStatus()

        logger.info(f"Starting VAM session {request.session_uuid}")
        session = Session.create(request, self._cfg, self._context_length)
        self._sessions[request.session_uuid] = session

        return SessionRequestStatus()

    @async_log_call
    async def close_session(
        self, request: DriveSessionCloseRequest, context: grpc.aio.ServicerContext
    ) -> Empty:
        if request.session_uuid not in self._sessions:
            raise KeyError(f"Session {request.session_uuid} does not exist.")

        logger.info(f"Closing session {request.session_uuid}")
        del self._sessions[request.session_uuid]
        return Empty()

    @async_log_call
    async def get_version(
        self, request: Empty, context: grpc.aio.ServicerContext
    ) -> VersionId:
        driver_version = version("alpasim_driver")
        return VersionId(
            version_id=f"vam-driver-{driver_version}",
            git_hash="unknown",
            grpc_api_version=API_VERSION_MESSAGE,
        )

    def _resize_and_crop_image(self, image: Image.Image) -> Image.Image:
        img_w, img_h = image.size

        if img_h != self._cfg.inference.image_height:
            resize_factor = self._cfg.inference.image_height / img_h
            target_width = int(img_w * resize_factor)
            image = image.resize((target_width, self._cfg.inference.image_height))
            img_w, img_h = image.size

        if img_w > self._cfg.inference.image_width:
            left = (img_w - self._cfg.inference.image_width) // 2
            image = image.crop((left, 0, left + self._cfg.inference.image_width, img_h))
        elif img_w < self._cfg.inference.image_width:
            raise ValueError(
                f"Image width {img_w} is less than expected {self._cfg.inference.image_width}"
            )

        return image

    @async_log_call
    async def submit_image_observation(
        self, request: RolloutCameraImage, context: grpc.aio.ServicerContext
    ) -> Empty:
        grpc_image = request.camera_image
        image = Image.open(BytesIO(grpc_image.image_bytes))
        session = self._sessions[request.session_uuid]
        if grpc_image.logical_id not in session.desired_cameras_logical_ids:
            raise ValueError(f"Camera {grpc_image.logical_id} not in desired cameras")

        image = session.rectify_image(grpc_image.logical_id, image)
        image = self._resize_and_crop_image(image)
        session.add_image(np.array(image), grpc_image.frame_end_us)

        return Empty()

    @async_log_call
    async def submit_egomotion_observation(
        self, request: RolloutEgoTrajectory, context: grpc.aio.ServicerContext
    ) -> Empty:
        self._sessions[request.session_uuid].add_egoposes(request.trajectory)
        return Empty()

    @async_log_call
    async def submit_route(
        self, request: RouteRequest, context: grpc.aio.ServicerContext
    ) -> Empty:
        logger.debug("submit_route: waypoint count=%s", len(request.route.waypoints))
        if self._cfg.route is not None:
            self._sessions[request.session_uuid].update_command_from_route(
                request.route,
                self._cfg.route.use_waypoint_commands,
                self._cfg.route.command_distance_threshold,
                self._cfg.route.min_lookahead_distance,
            )
        else:
            self._sessions[request.session_uuid].update_command_from_route(
                request.route,
                use_waypoint_commands=False,
            )
        return Empty()

    @async_log_call
    async def submit_recording_ground_truth(
        self, request: GroundTruthRequest, context: grpc.aio.ServicerContext
    ) -> Empty:
        logger.debug("Ground truth received but not used by VAM")
        return Empty()

    @async_log_call
    async def drive(
        self, request: DriveRequest, context: grpc.aio.ServicerContext
    ) -> DriveResponse:
        if request.session_uuid not in self._sessions:
            raise KeyError(f"Session {request.session_uuid} not found")

        session = self._sessions[request.session_uuid]

        if session.frame_cache.frame_count() < self._context_length:
            empty_traj = Trajectory()
            logger.info(
                "Drive request received with insufficient frames: "
                "got %s frames, need at least %s frames. "
                "Returning empty trajectory",
                session.frame_cache.frame_count(),
                self._context_length,
            )
            return DriveResponse(
                trajectory=empty_traj,
            )

        pose_snapshot = session.poses[-1] if session.poses else None
        logger.debug(f"pose_snapshot: {pose_snapshot}")
        if pose_snapshot is None:
            empty_traj = Trajectory()
            logger.info(
                "Drive request received with no pose snapshot available "
                "(poses list length: %s). Returning empty trajectory",
                len(session.poses),
            )
            return DriveResponse(
                trajectory=empty_traj,
            )

        future: asyncio.Future[DriveResponse] = self._loop.create_future()
        job = DriveJob(
            session_id=request.session_uuid,
            session=session,
            command=session.current_command,
            pose=pose_snapshot,
            timestamp_us=request.time_now_us,
            result=future,
        )
        self._job_queue.put_nowait(job)

        response = await future

        debug_data = {
            "command": int(session.current_command),
            "command_name": session.current_command.name,
            "num_frames": session.frame_cache.frame_count(),
            "num_poses": len(session.poses),
            "trajectory_points": len(response.trajectory.poses),
        }
        response.debug_info.unstructured_debug_info = pickle.dumps(debug_data)

        logging.info("Returning drive response at time %s", request.time_now_us)
        return response

    def _convert_vam_trajectory_to_alpasim(
        self,
        vam_trajectory: np.ndarray,  # Shape: (6, 2) at 2Hz
        current_pose: PoseAtTime,
        time_now_us: int,
    ) -> Trajectory:
        trajectory = Trajectory()

        trajectory.poses.append(current_pose)

        curr_z = current_pose.pose.vec.z

        frequency_hz = self._cfg.trajectory.frequency_hz
        # Convert trajectory output frequency (Hz) to microsecond spacing between poses.
        time_delta_us = int(1_000_000 / frequency_hz)

        local_positions = _rig_est_offsets_to_local_positions(
            current_pose, vam_trajectory
        )
        num_positions = local_positions.shape[0]

        if num_positions == 0:
            return trajectory

        # Pre-compute timestamps and XY deltas between consecutive positions.
        steps = np.arange(1, num_positions + 1, dtype=np.int64)
        timestamps_us = (time_now_us + steps * time_delta_us).tolist()

        previous_positions = np.vstack(
            (
                np.array(
                    [current_pose.pose.vec.x, current_pose.pose.vec.y], dtype=float
                ),
                local_positions[:-1],
            )
        )
        deltas = local_positions - previous_positions
        distances = np.hypot(deltas[:, 0], deltas[:, 1])
        yaws = np.arctan2(deltas[:, 1], deltas[:, 0])

        prev_quat = Quat()
        prev_quat.CopyFrom(current_pose.pose.quat)

        for local_xy, distance, yaw, timestamp_us in zip(
            local_positions,
            distances,
            yaws,
            timestamps_us,
            strict=True,
        ):
            local_x, local_y = map(float, local_xy)
            local_z = curr_z

            if distance > 1e-4:
                quat = _yaw_to_quat(float(yaw))
            else:
                quat = Quat()
                quat.CopyFrom(prev_quat)

            trajectory.poses.append(
                PoseAtTime(
                    pose=Pose(
                        vec=Vec3(x=local_x, y=local_y, z=local_z),
                        quat=quat,
                    ),
                    timestamp_us=timestamp_us,
                )
            )

            prev_quat = quat

        return trajectory

    @async_log_call
    async def shut_down(
        self, request: Empty, context: grpc.aio.ServicerContext
    ) -> Empty:
        logger.info("shut_down requested, scheduling deferred shutdown")
        # Schedule shutdown to happen after RPC completes to avoid CancelledError
        asyncio.create_task(self._deferred_shutdown())
        return Empty()

    async def _deferred_shutdown(self) -> None:
        """Shutdown the server and worker after the shut_down RPC completes.

        This deferred approach prevents the shut_down RPC from cancelling itself
        when stopping the server, which would result in asyncio.exceptions.CancelledError.
        """
        # Small delay to ensure the shut_down RPC response is sent first
        await asyncio.sleep(0.1)
        logger.info("Executing deferred shutdown")
        await self._grpc_server.stop(grace=None)
        await self.stop_worker()


async def serve(cfg: VAMDriverConfig) -> None:
    server = grpc.aio.server()
    loop = asyncio.get_running_loop()

    device = torch.device(cfg.model.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if cfg.model.dtype == "float16" else torch.float32

    service = VAMPolicyService(
        cfg=cfg,
        loop=loop,
        grpc_server=server,
        device=device,
        dtype=dtype,
    )
    add_EgodriverServiceServicer_to_server(service, server)

    address = f"{cfg.host}:{cfg.port}"
    server.add_insecure_port(address)

    await server.start()
    logger.info("Starting VAM driver on %s", address)

    try:
        await server.wait_for_termination()
    finally:
        await service.stop_worker()


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="vam_driver",
)
def main(cfg: VAMDriverConfig) -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if cfg.output_dir:
        os.makedirs(cfg.output_dir, exist_ok=True)
        OmegaConf.save(
            cfg, os.path.join(cfg.output_dir, "vam-driver.yaml"), resolve=True
        )

    asyncio.run(serve(cfg))


if __name__ == "__main__":
    main()
