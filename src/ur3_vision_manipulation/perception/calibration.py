"""Chessboard-style pinhole camera calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import ArrayLike

from .camera import CameraIntrinsics


@dataclass(frozen=True)
class CalibrationResult:
    """Camera calibration result returned by OpenCV."""

    intrinsics: CameraIntrinsics
    rms_reprojection_error: float
    mean_reprojection_error: float


def calibrate_camera(
    object_points: Sequence[ArrayLike],
    image_points: Sequence[ArrayLike],
    image_size: tuple[int, int],
) -> CalibrationResult:
    """Calibrate a camera from corresponding multi-view 3D and 2D points.

    ``image_size`` follows OpenCV ordering: ``(width, height)``. At least three
    views are required so the routine cannot accidentally present a severely
    underconstrained single-view result as a calibration.
    """

    if len(object_points) != len(image_points):
        raise ValueError("object and image point view counts must match")
    if len(object_points) < 3:
        raise ValueError("camera calibration requires at least three views")
    if (
        len(image_size) != 2
        or isinstance(image_size[0], bool)
        or isinstance(image_size[1], bool)
        or not isinstance(image_size[0], (int, np.integer))
        or not isinstance(image_size[1], (int, np.integer))
        or image_size[0] <= 0
        or image_size[1] <= 0
    ):
        raise ValueError("image size must contain positive integer width and height")

    validated_object_points: list[np.ndarray] = []
    validated_image_points: list[np.ndarray] = []
    for view_index, (object_view, image_view) in enumerate(
        zip(object_points, image_points, strict=True)
    ):
        objects = np.asarray(object_view, dtype=np.float32)
        images = np.asarray(image_view, dtype=np.float32)
        if objects.ndim != 2 or objects.shape[1] != 3:
            raise ValueError(f"object points for view {view_index} must have shape (N, 3)")
        if images.ndim == 3 and images.shape[1:] == (1, 2):
            images = images.reshape(-1, 2)
        if images.ndim != 2 or images.shape[1] != 2:
            raise ValueError(f"image points for view {view_index} must have shape (N, 2)")
        if len(objects) != len(images) or len(objects) < 4:
            raise ValueError(
                f"view {view_index} must contain at least four matching object/image points"
            )
        if not np.all(np.isfinite(objects)) or not np.all(np.isfinite(images)):
            raise ValueError(f"view {view_index} points must contain only finite values")
        validated_object_points.append(np.ascontiguousarray(objects))
        validated_image_points.append(np.ascontiguousarray(images))

    rms, matrix, distortion, rotation_vectors, translation_vectors = cv2.calibrateCamera(
        validated_object_points,
        validated_image_points,
        (int(image_size[0]), int(image_size[1])),
        None,
        None,
    )
    if not np.isfinite(rms):
        raise RuntimeError("OpenCV camera calibration returned a non-finite error")

    total_squared_error = 0.0
    total_points = 0
    for objects, images, rvec, tvec in zip(
        validated_object_points,
        validated_image_points,
        rotation_vectors,
        translation_vectors,
        strict=True,
    ):
        projected, _ = cv2.projectPoints(objects, rvec, tvec, matrix, distortion)
        residual = images - projected.reshape(-1, 2)
        total_squared_error += float(np.sum(residual * residual))
        total_points += len(objects)
    mean_error = float(np.sqrt(total_squared_error / total_points))

    intrinsics = CameraIntrinsics(
        matrix,
        distortion,
        image_width=int(image_size[0]),
        image_height=int(image_size[1]),
    )
    return CalibrationResult(intrinsics, float(rms), mean_error)
