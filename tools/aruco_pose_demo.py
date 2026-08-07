"""Run deterministic synthetic ArUco detection and solvePnP verification."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ur3_vision_manipulation.perception import (
    DEFAULT_ARUCO_DICTIONARY,
    CameraIntrinsics,
    detect_aruco_markers,
    estimate_marker_pose,
    marker_object_points,
)


def main() -> None:
    intrinsics = CameraIntrinsics(
        [[820.0, 0.0, 320.0], [0.0, 815.0, 240.0], [0.0, 0.0, 1.0]],
        np.zeros(5),
        640,
        480,
    )
    marker_size = 0.08
    known_rvec = np.array([0.18, -0.12, 0.07])
    known_translation = np.array([0.03, -0.02, 0.55])
    image_points, _ = cv2.projectPoints(
        marker_object_points(marker_size),
        known_rvec,
        known_translation,
        intrinsics.camera_matrix,
        intrinsics.distortion_coefficients,
    )
    estimated = estimate_marker_pose(image_points, marker_size, intrinsics).to_matrix()
    known_rotation, _ = cv2.Rodrigues(known_rvec)
    translation_error = np.linalg.norm(estimated[:3, 3] - known_translation)
    rotation_error = Rotation.from_matrix(estimated[:3, :3] @ known_rotation.T).magnitude()

    dictionary = cv2.aruco.getPredefinedDictionary(DEFAULT_ARUCO_DICTIONARY)
    marker = cv2.aruco.generateImageMarker(dictionary, 17, 200)
    marker_image = np.full((400, 400), 255, dtype=np.uint8)
    marker_image[100:300, 100:300] = marker
    detections = detect_aruco_markers(marker_image)
    no_marker = detect_aruco_markers(np.full((400, 400), 255, dtype=np.uint8))

    print("SYNTHETIC POSE")
    print(f"known translation: {known_translation.tolist()}")
    print(f"estimated translation: {estimated[:3, 3].tolist()}")
    print(f"translation error (m): {translation_error:.12g}")
    print(f"rotation error (rad): {rotation_error:.12g}")
    print("SYNTHETIC ARUCO")
    print("dictionary: DICT_4X4_50")
    print(f"detected IDs: {[detection.marker_id for detection in detections]}")
    print(f"corner shape: {detections[0].corners.shape if detections else None}")
    print("corner order: top-left, top-right, bottom-right, bottom-left")
    print(f"no-marker detections: {len(no_marker)}")


if __name__ == "__main__":
    main()
