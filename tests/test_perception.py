import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ur3_vision_manipulation.perception import (
    DEFAULT_ARUCO_DICTIONARY,
    CameraIntrinsics,
    calibrate_camera,
    detect_aruco_markers,
    estimate_marker_pose,
    marker_object_points,
)


def _synthetic_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        [[820.0, 0.0, 320.0], [0.0, 815.0, 240.0], [0.0, 0.0, 1.0]],
        np.zeros(5),
        640,
        480,
    )


def _generated_marker(marker_id: int = 17) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(DEFAULT_ARUCO_DICTIONARY)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 200)
    image = np.full((400, 400), 255, dtype=np.uint8)
    image[100:300, 100:300] = marker
    return image


def test_camera_intrinsics_valid():
    intrinsics = _synthetic_intrinsics()

    assert intrinsics.image_size == (640, 480)
    assert intrinsics.camera_matrix.dtype == np.float64
    assert intrinsics.distortion_coefficients.shape == (5,)


def test_camera_intrinsics_invalid_focal_length():
    with pytest.raises(ValueError, match="focal lengths"):
        CameraIntrinsics(np.diag([0.0, 800.0, 1.0]), np.zeros(5), 640, 480)


def test_camera_intrinsics_invalid_shape():
    with pytest.raises(ValueError, match="shape"):
        CameraIntrinsics(np.eye(4), np.zeros(5), 640, 480)


def test_camera_intrinsics_invalid_distortion_shape():
    with pytest.raises(ValueError, match="distortion coefficients"):
        CameraIntrinsics(np.eye(3), np.zeros((2, 2)), 640, 480)


def test_aruco_detects_generated_marker():
    detections = detect_aruco_markers(_generated_marker())

    assert [detection.marker_id for detection in detections] == [17]
    assert detections[0].corners.shape == (4, 2)
    np.testing.assert_allclose(
        detections[0].corners,
        [[100, 100], [299, 100], [299, 299], [100, 299]],
        atol=1.0,
    )


def test_aruco_no_marker_returns_empty():
    assert detect_aruco_markers(np.full((400, 400), 255, dtype=np.uint8)) == []


def test_marker_size_rejected_if_nonpositive():
    with pytest.raises(ValueError, match="positive"):
        marker_object_points(0.0)
    with pytest.raises(ValueError, match="positive"):
        marker_object_points(-0.04)


def test_solvepnp_recovers_synthetic_camera_T_object():
    intrinsics = _synthetic_intrinsics()
    known_rotation_vector = np.array([0.18, -0.12, 0.07])
    known_translation = np.array([0.03, -0.02, 0.55])
    image_points, _ = cv2.projectPoints(
        marker_object_points(0.08),
        known_rotation_vector,
        known_translation,
        intrinsics.camera_matrix,
        intrinsics.distortion_coefficients,
    )

    estimated = estimate_marker_pose(image_points.reshape(4, 2), 0.08, intrinsics)
    estimated_matrix = estimated.to_matrix()
    known_rotation, _ = cv2.Rodrigues(known_rotation_vector)
    translation_error = np.linalg.norm(estimated_matrix[:3, 3] - known_translation)
    rotation_error = Rotation.from_matrix(
        estimated_matrix[:3, :3] @ known_rotation.T
    ).magnitude()

    assert translation_error < 1.0e-8
    assert rotation_error < 1.0e-6


def test_camera_T_object_transform_convention():
    intrinsics = _synthetic_intrinsics()
    known_rotation_vector = np.array([0.1, 0.2, -0.05])
    known_translation = np.array([0.02, 0.01, 0.6])
    object_points = marker_object_points(0.06)
    image_points, _ = cv2.projectPoints(
        object_points,
        known_rotation_vector,
        known_translation,
        intrinsics.camera_matrix,
        intrinsics.distortion_coefficients,
    )
    camera_T_object = estimate_marker_pose(image_points, 0.06, intrinsics)
    known_rotation, _ = cv2.Rodrigues(known_rotation_vector)

    expected_camera_point = known_rotation @ object_points[0] + known_translation
    np.testing.assert_allclose(
        camera_T_object.transform_point(object_points[0]),
        expected_camera_point,
        atol=1.0e-8,
    )


def test_calibration_from_synthetic_views():
    known = _synthetic_intrinsics()
    columns, rows = 9, 6
    grid = np.zeros((columns * rows, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * 0.024
    object_views = []
    image_views = []
    for index in range(12):
        rvec = np.array(
            [
                -0.22 + 0.04 * index,
                0.16 * np.sin(0.7 * index),
                -0.12 + 0.025 * index,
            ]
        )
        tvec = np.array(
            [
                -0.09 + 0.016 * index,
                -0.06 + 0.012 * (index % 5),
                0.55 + 0.025 * (index % 4),
            ]
        )
        projected, _ = cv2.projectPoints(
            grid,
            rvec,
            tvec,
            known.camera_matrix,
            known.distortion_coefficients,
        )
        object_views.append(grid.copy())
        image_views.append(projected.reshape(-1, 2))

    result = calibrate_camera(object_views, image_views, known.image_size)

    np.testing.assert_allclose(
        np.diag(result.intrinsics.camera_matrix)[:2],
        [820.0, 815.0],
        atol=0.2,
    )
    np.testing.assert_allclose(
        result.intrinsics.camera_matrix[:2, 2],
        [320.0, 240.0],
        atol=0.2,
    )
    assert result.rms_reprojection_error < 1.0e-3
    assert result.mean_reprojection_error < 1.0e-3
