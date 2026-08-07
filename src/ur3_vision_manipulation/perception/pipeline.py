"""Application-level marker-pose pipeline from an image to the robot base."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import ArrayLike

from ur3_vision_manipulation.geometry import Transform

from .aruco_pose import (
    DEFAULT_ARUCO_DICTIONARY,
    detect_aruco_markers,
    estimate_marker_pose,
)
from .camera import CameraIntrinsics
from .extrinsics import transform_object_to_base


@dataclass(frozen=True)
class MarkerPoseInBase:
    """One marker ID with its camera-frame and base-frame poses."""

    marker_id: int
    camera_T_object: Transform
    base_T_object: Transform


def estimate_marker_poses_in_base(
    image: ArrayLike,
    marker_size: float,
    intrinsics: CameraIntrinsics,
    base_T_camera: Transform,
    dictionary_id: int = DEFAULT_ARUCO_DICTIONARY,
) -> list[MarkerPoseInBase]:
    """Detect markers and return their camera and robot-base poses."""

    results: list[MarkerPoseInBase] = []
    for detection in detect_aruco_markers(image, dictionary_id):
        camera_T_object = estimate_marker_pose(
            detection.corners, marker_size, intrinsics
        )
        results.append(
            MarkerPoseInBase(
                marker_id=detection.marker_id,
                camera_T_object=camera_T_object,
                base_T_object=transform_object_to_base(
                    base_T_camera, camera_T_object
                ),
            )
        )
    return results
