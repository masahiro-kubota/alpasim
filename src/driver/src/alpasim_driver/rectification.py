# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""Minimal utilities to rectify f-theta camera renders into pinhole frames."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from alpasim_grpc.v0 import sensorsim_pb2

from .schema import RectificationTargetConfig

logger = logging.getLogger(__name__)


def _has_distortion(config: RectificationTargetConfig) -> bool:
    """Return True if any distortion coefficients are provided."""
    return bool(config.radial or config.tangential or config.thin_prism)


class _PinholeCamera:
    """Generates unit rays for every pixel in an image."""

    def __init__(
        self,
        focal_length: Tuple[float, float],
        principal_point: Tuple[float, float],
        resolution_hw: Tuple[int, int],
    ) -> None:
        """Cache basic intrinsics for fast ray generation."""

        self._fx, self._fy = focal_length
        self._cx, self._cy = principal_point
        self._height, self._width = resolution_hw

    def unit_rays_grid(self) -> np.ndarray:
        """Return a (H, W, 3) array of unit rays for each image pixel."""

        xs = np.arange(self._width, dtype=np.float64)
        ys = np.arange(self._height, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(xs, ys)
        x = (grid_x - self._cx) / self._fx
        y = (grid_y - self._cy) / self._fy
        z = np.ones_like(x)
        rays = np.stack((x, y, z), axis=-1)
        norms = np.linalg.norm(rays, axis=-1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        return rays / norms


class _FthetaCamera:
    """Projects unit rays into pixel coordinates using theta→radius polynomials."""

    def __init__(
        self,
        intrinsics: sensorsim_pb2.FthetaCameraParam,
        resolution_hw: Tuple[int, int],
    ) -> None:
        """Precompute polynomial and linear transform state for projections."""

        self._angle_to_pixeldist = np.asarray(
            intrinsics.angle_to_pixeldist_poly, dtype=np.float64
        )
        self._pixeldist_to_angle = np.asarray(
            intrinsics.pixeldist_to_angle_poly, dtype=np.float64
        )
        self._principal_point = np.array(
            [intrinsics.principal_point_x, intrinsics.principal_point_y],
            dtype=np.float64,
        )
        self._max_angle = intrinsics.max_angle if intrinsics.max_angle > 0 else None
        if intrinsics.HasField("linear_cde"):
            linear_c = intrinsics.linear_cde.linear_c
            linear_d = intrinsics.linear_cde.linear_d
            linear_e = intrinsics.linear_cde.linear_e
        else:
            linear_c, linear_d, linear_e = 1.0, 0.0, 0.0
        self._linear_matrix = np.array(
            [[linear_c, linear_d], [linear_e, 1.0]],
            dtype=np.float64,
        )
        self._source_resolution = resolution_hw

    def ray_to_pixel(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project rays (N,3) into pixel coordinates.

        Returns both the pixel coordinates and a boolean validity mask.
        """

        reshaped = rays.reshape(-1, 3).astype(np.float64)
        xy = reshaped[:, :2]
        z = reshaped[:, 2]
        xy_norm = np.linalg.norm(xy, axis=1)

        theta = np.zeros_like(xy_norm)
        positive_z = z > 0.0
        theta[positive_z] = np.arctan2(xy_norm[positive_z], z[positive_z])

        if self._max_angle is not None:
            within_fov = theta <= self._max_angle + 1e-6
        else:
            within_fov = positive_z

        radii = np.polynomial.polynomial.polyval(theta, self._angle_to_pixeldist)
        # Avoid division by zero for rays along the optical axis.
        with np.errstate(divide="ignore", invalid="ignore"):
            scales = np.divide(
                radii, xy_norm, out=np.zeros_like(radii), where=xy_norm > 1e-9
            )

        offsets = xy * scales[:, None]
        pixel_offsets = offsets @ self._linear_matrix.T
        pixels = pixel_offsets + self._principal_point

        height, width = self._source_resolution
        in_image = (
            (pixels[:, 0] >= -0.5)
            & (pixels[:, 0] <= width - 0.5)
            & (pixels[:, 1] >= -0.5)
            & (pixels[:, 1] <= height - 0.5)
        )
        valid = positive_z & within_fov & in_image

        return pixels.reshape(rays.shape[:-1] + (2,)), valid.reshape(rays.shape[:-1])


class FthetaToPinholeRectifier:
    """Caches cv2.remap grids to rectify f-theta renders into a pinhole view."""

    def __init__(
        self,
        source_intrinsics: sensorsim_pb2.FthetaCameraParam,
        source_resolution_hw: Tuple[int, int],
        target_intrinsics: RectificationTargetConfig,
    ) -> None:
        """Precompute remap grids that turn f-theta pixels into pinhole pixels."""

        self._source_resolution = tuple(source_resolution_hw)
        self._target_resolution = tuple(target_intrinsics.resolution_hw)
        self._target_intrinsics = target_intrinsics
        self._map_x, self._map_y = self._build_maps(
            source_intrinsics, target_intrinsics
        )
        self._distort_map_x: Optional[np.ndarray] = None
        self._distort_map_y: Optional[np.ndarray] = None
        if _has_distortion(target_intrinsics):
            (
                self._distort_map_x,
                self._distort_map_y,
            ) = self._build_distortion_maps(target_intrinsics)

    def _build_maps(
        self,
        source_intrinsics: sensorsim_pb2.FthetaCameraParam,
        target_intrinsics: RectificationTargetConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate OpenCV remap tables for the provided camera pair."""

        pinhole = _PinholeCamera(
            target_intrinsics.focal_length,
            target_intrinsics.principal_point,
            self._target_resolution,
        )
        rays = pinhole.unit_rays_grid()
        ftheta = _FthetaCamera(source_intrinsics, self._source_resolution)
        pixels, valid = ftheta.ray_to_pixel(rays)

        map_x = pixels[..., 0].astype(np.float32)
        map_y = pixels[..., 1].astype(np.float32)
        invalid_mask = ~valid
        map_x[invalid_mask] = -1.0
        map_y[invalid_mask] = -1.0

        return map_x, map_y

    def _build_distortion_maps(
        self,
        target_intrinsics: RectificationTargetConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create map to reintroduce OpenCV distortion into the pinhole frame."""

        height, width = self._target_resolution
        fx, fy = target_intrinsics.focal_length
        cx, cy = target_intrinsics.principal_point

        xs = np.arange(width, dtype=np.float64)
        ys = np.arange(height, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(xs, ys)
        distorted_pixels = np.stack((grid_x, grid_y), axis=-1)

        camera_matrix = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        dist = np.zeros(14, dtype=np.float64)
        for idx, value in enumerate(target_intrinsics.radial[:6]):
            dist[[0, 1, 4, 5, 6, 7][idx]] = value
        for idx, value in enumerate(target_intrinsics.tangential[:2]):
            dist[[2, 3][idx]] = value
        for idx, value in enumerate(target_intrinsics.thin_prism[:4]):
            dist[8 + idx] = value

        undistorted = cv2.undistortPoints(
            distorted_pixels.reshape(-1, 1, 2),
            cameraMatrix=camera_matrix,
            distCoeffs=dist,
        ).reshape(-1, 2)
        undistorted_pixels = np.empty_like(undistorted)
        undistorted_pixels[:, 0] = undistorted[:, 0] * fx + cx
        undistorted_pixels[:, 1] = undistorted[:, 1] * fy + cy

        map_x = undistorted_pixels[:, 0].reshape(height, width).astype(np.float32)
        map_y = undistorted_pixels[:, 1].reshape(height, width).astype(np.float32)

        valid = (
            (map_x >= -0.5)
            & (map_x <= width - 0.5)
            & (map_y >= -0.5)
            & (map_y <= height - 0.5)
        )
        map_x[~valid] = -1.0
        map_y[~valid] = -1.0

        return map_x, map_y

    def rectify(self, image: np.ndarray) -> np.ndarray:
        """Return a pinhole-rectified copy of the provided f-theta image."""

        logger.info(
            "Rectifying image %s with source resolution %s",
            image.shape,
            self._source_resolution,
        )

        if (
            image.shape[0] != self._source_resolution[0]
            or image.shape[1] != self._source_resolution[1]
        ):
            raise ValueError(
                "Unexpected source resolution: "
                f"got {(image.shape[0], image.shape[1])}, expected {self._source_resolution}"
            )

        rectified = cv2.remap(
            image,
            self._map_x,
            self._map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if self._distort_map_x is not None and self._distort_map_y is not None:
            rectified = cv2.remap(
                rectified,
                self._distort_map_x,
                self._distort_map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        return rectified


def build_rectifier_map(
    rectification_cfg: Optional[dict[str, RectificationTargetConfig]],
    desired_cameras: set[str],
    camera_lookup: dict[str, sensorsim_pb2.AvailableCamerasReturn.AvailableCamera],
) -> dict[str, Optional[FthetaToPinholeRectifier]]:
    """Instantiate per-camera rectifiers as requested in the Hydra config."""

    rectifiers: dict[str, Optional[FthetaToPinholeRectifier]] = {
        logical_id: None for logical_id in desired_cameras
    }
    if rectification_cfg is None:
        return rectifiers

    for logical_id in desired_cameras:
        target_cfg = rectification_cfg[logical_id]
        camera_proto = camera_lookup[logical_id]

        if camera_proto.intrinsics.WhichOneof("camera_param") != "ftheta_param":
            raise ValueError(f"Camera {logical_id} does not provide f-theta intrinsics")
        rectifiers[logical_id] = FthetaToPinholeRectifier(
            source_intrinsics=camera_proto.intrinsics.ftheta_param,
            source_resolution_hw=(
                int(camera_proto.intrinsics.resolution_h),
                int(camera_proto.intrinsics.resolution_w),
            ),
            target_intrinsics=target_cfg,
        )
        logger.info("Enabled f-theta→pinhole rectification for %s", logical_id)

    return rectifiers
