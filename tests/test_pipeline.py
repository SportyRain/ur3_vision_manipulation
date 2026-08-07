import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ur3_vision_manipulation.geometry import Transform
from ur3_vision_manipulation.perception import (
    DEFAULT_ARUCO_DICTIONARY,
    CameraIntrinsics,
    estimate_marker_poses_in_base,
    marker_object_points,
)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
MARKER_ID = 23
MARKER_SIZE_M = 0.08
TRANSLATION_TOLERANCE_M = 0.005
ROTATION_TOLERANCE_RAD = 0.05


def _synthetic_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        [[820.0, 0.0, 320.0], [0.0, 815.0, 240.0], [0.0, 0.0, 1.0]],
        np.zeros(5),
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )


def _transform(translation, rotation_vector) -> Transform:
    return Transform.from_translation_rotation(
        translation, Rotation.from_rotvec(rotation_vector).as_matrix()
    )


def _render_marker(
    camera_T_object: Transform,
    intrinsics: CameraIntrinsics,
    marker_id: int = MARKER_ID,
) -> np.ndarray:
    marker_pixels = 240
    dictionary = cv2.aruco.getPredefinedDictionary(DEFAULT_ARUCO_DICTIONARY)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_pixels)
    matrix = camera_T_object.to_matrix()
    rotation_vector, _ = cv2.Rodrigues(matrix[:3, :3])
    image_corners, _ = cv2.projectPoints(
        marker_object_points(MARKER_SIZE_M),
        rotation_vector,
        matrix[:3, 3],
        intrinsics.camera_matrix,
        intrinsics.distortion_coefficients,
    )
    marker_corners = np.array(
        [
            [0, 0],
            [marker_pixels - 1, 0],
            [marker_pixels - 1, marker_pixels - 1],
            [0, marker_pixels - 1],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(
        marker_corners, image_corners.reshape(4, 2).astype(np.float32)
    )
    return cv2.warpPerspective(
        marker,
        homography,
        (intrinsics.image_width, intrinsics.image_height),
        flags=cv2.INTER_NEAREST,
        borderValue=255,
    )


def test_no_marker_returns_empty_result():
    image = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 255, dtype=np.uint8)

    assert (
        estimate_marker_poses_in_base(
            image, MARKER_SIZE_M, _synthetic_intrinsics(), Transform.identity()
        )
        == []
    )


def test_synthetic_aruco_image_to_base_pose_end_to_end():
    intrinsics = _synthetic_intrinsics()
    known_camera_T_object = _transform([0.025, -0.018, 0.58], [0.12, -0.09, 0.04])
    known_base_T_camera = _transform([0.38, -0.16, 0.72], [0.22, -0.11, 0.28])
    known_base_T_object = known_base_T_camera @ known_camera_T_object
    image = _render_marker(known_camera_T_object, intrinsics)

    results = estimate_marker_poses_in_base(
        image, MARKER_SIZE_M, intrinsics, known_base_T_camera
    )

    assert [result.marker_id for result in results] == [MARKER_ID]
    result = results[0]
    expected_from_estimate = known_base_T_camera @ result.camera_T_object
    np.testing.assert_allclose(
        result.base_T_object.to_matrix(), expected_from_estimate.to_matrix(), atol=1.0e-12
    )

    known_matrix = known_base_T_object.to_matrix()
    estimated_matrix = result.base_T_object.to_matrix()
    translation_error = np.linalg.norm(
        estimated_matrix[:3, 3] - known_matrix[:3, 3]
    )
    rotation_error = Rotation.from_matrix(
        estimated_matrix[:3, :3] @ known_matrix[:3, :3].T
    ).magnitude()

    assert translation_error < TRANSLATION_TOLERANCE_M
    assert rotation_error < ROTATION_TOLERANCE_RAD


def test_multiple_marker_ids_stay_associated_with_their_poses():
    intrinsics = _synthetic_intrinsics()
    known_poses = {
        7: _transform([-0.075, -0.01, 0.62], [0.08, -0.06, 0.02]),
        31: _transform([0.075, 0.015, 0.62], [-0.07, 0.05, -0.03]),
    }
    images = [
        _render_marker(pose, intrinsics, marker_id)
        for marker_id, pose in known_poses.items()
    ]
    image = np.minimum.reduce(images)

    results = estimate_marker_poses_in_base(
        image, MARKER_SIZE_M, intrinsics, Transform.identity()
    )
    results_by_id = {result.marker_id: result for result in results}

    assert set(results_by_id) == set(known_poses)
    for marker_id, known_pose in known_poses.items():
        np.testing.assert_allclose(
            results_by_id[marker_id].camera_T_object.to_matrix()[:3, 3],
            known_pose.to_matrix()[:3, 3],
            atol=TRANSLATION_TOLERANCE_M,
        )
