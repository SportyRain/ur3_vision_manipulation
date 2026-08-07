"""ArUco detection and marker pose estimation.

The marker object frame is centered on the marker, with +x toward the marker's
right edge, +y toward its bottom edge, and +z normal to the marker plane. The
four corners are ordered top-left, top-right, bottom-right, bottom-left, which
is the order returned by OpenCV's ArUco detector.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ur3_vision_manipulation.geometry import Transform

from .camera import CameraIntrinsics

DEFAULT_ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50


@dataclass(frozen=True)
class ArucoDetection:
    """One detected marker and its TL, TR, BR, BL image corners."""

    marker_id: int
    corners: NDArray[np.float64]


def detect_aruco_markers(
    image: ArrayLike,
    dictionary_id: int = DEFAULT_ARUCO_DICTIONARY,
) -> list[ArucoDetection]:
    """Detect markers, returning an empty list when no marker is present."""

    image_value = np.asarray(image)
    if image_value.ndim not in (2, 3):
        raise ValueError("image must be a grayscale or color image")
    if image_value.ndim == 3 and image_value.shape[2] not in (3, 4):
        raise ValueError("color image must have 3 or 4 channels")
    if image_value.size == 0:
        raise ValueError("image must not be empty")
    if image_value.dtype != np.uint8:
        raise ValueError("image must have uint8 dtype")
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(int(dictionary_id))
    except (cv2.error, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"invalid ArUco dictionary id: {dictionary_id}") from error

    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image_value)
    if ids is None:
        return []
    return [
        ArucoDetection(int(marker_id), np.asarray(marker_corners, dtype=float).reshape(4, 2))
        for marker_id, marker_corners in zip(ids.reshape(-1), corners, strict=True)
    ]


def marker_object_points(marker_size: float) -> NDArray[np.float64]:
    """Return centered marker corners in detector order (TL, TR, BR, BL).

    The marker frame uses +x right and +y down on the marker face. ``marker_size``
    is expressed in the caller's desired translation unit (normally metres).
    """

    if isinstance(marker_size, bool) or not np.isscalar(marker_size):
        raise ValueError("marker size must be a positive finite scalar")
    size = float(marker_size)
    if not np.isfinite(size) or size <= 0:
        raise ValueError("marker size must be a positive finite scalar")
    half = size / 2.0
    return np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        dtype=float,
    )


def estimate_marker_pose(
    corners: ArrayLike,
    marker_size: float,
    intrinsics: CameraIntrinsics,
) -> Transform:
    """Estimate and return ``camera_T_object`` with ``cv2.solvePnP``.

    OpenCV's ``rvec`` and ``tvec`` map object-frame points into the camera
    frame, so no inversion is applied: ``p_camera = camera_T_object @ p_object``.
    """

    if not isinstance(intrinsics, CameraIntrinsics):
        raise TypeError("intrinsics must be a CameraIntrinsics instance")
    image_points = np.asarray(corners, dtype=float)
    if image_points.shape in ((1, 4, 2), (4, 1, 2)):
        image_points = image_points.reshape(4, 2)
    if image_points.shape != (4, 2):
        raise ValueError(f"marker corners must have shape (4, 2), got {image_points.shape}")
    if not np.all(np.isfinite(image_points)):
        raise ValueError("marker corners must contain only finite values")

    success, rotation_vector, translation_vector = cv2.solvePnP(
        marker_object_points(marker_size),
        image_points,
        intrinsics.camera_matrix,
        intrinsics.distortion_coefficients,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("OpenCV solvePnP failed to estimate marker pose")
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    return Transform.from_translation_rotation(
        np.asarray(translation_vector, dtype=float).reshape(3),
        rotation_matrix,
    )
