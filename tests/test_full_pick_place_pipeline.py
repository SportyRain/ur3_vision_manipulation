import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ur3_vision_manipulation.geometry import Transform
from ur3_vision_manipulation.perception import (
    DEFAULT_ARUCO_DICTIONARY,
    CameraIntrinsics,
    estimate_marker_poses_in_base,
)
from ur3_vision_manipulation.task_targets import compute_pick_place_targets
from ur3_vision_manipulation.waypoints import (
    WaypointOffsets,
    compute_pick_place_waypoints,
)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
MARKER_ID = 23
MARKER_SIZE_M = 0.08

# Existing synthetic image-to-pose regression bounds for this rendered scene.
CAMERA_TRANSLATION_TOLERANCE_M = 0.005
CAMERA_ROTATION_TOLERANCE_RAD = 0.05
COMPOSITION_ATOL = 1.0e-12
RIGID_TRANSFORM_ATOL = 1.0e-9


def _synthetic_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        [[820.0, 0.0, 320.0], [0.0, 815.0, 240.0], [0.0, 0.0, 1.0]],
        np.zeros(5),
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )


def _rigid_matrix(translation, rotation_vector) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = Rotation.from_rotvec(rotation_vector).as_matrix()
    matrix[:3, 3] = np.asarray(translation, dtype=float)
    return matrix


def _render_marker(
    truth_camera_T_object: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    marker_pixels = 240
    dictionary = cv2.aruco.getPredefinedDictionary(DEFAULT_ARUCO_DICTIONARY)
    marker = cv2.aruco.generateImageMarker(dictionary, MARKER_ID, marker_pixels)
    rotation_vector, _ = cv2.Rodrigues(truth_camera_T_object[:3, :3])
    marker_half_size = MARKER_SIZE_M / 2.0
    truth_marker_corners = np.array(
        [
            [-marker_half_size, -marker_half_size, 0.0],
            [marker_half_size, -marker_half_size, 0.0],
            [marker_half_size, marker_half_size, 0.0],
            [-marker_half_size, marker_half_size, 0.0],
        ],
        dtype=float,
    )
    image_corners, _ = cv2.projectPoints(
        truth_marker_corners,
        rotation_vector,
        truth_camera_T_object[:3, 3],
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
        marker_corners,
        image_corners.reshape(4, 2).astype(np.float32),
    )
    return cv2.warpPerspective(
        marker,
        homography,
        intrinsics.image_size,
        flags=cv2.INTER_NEAREST,
        borderValue=255,
    )


def _assert_transform_matches(
    actual: Transform,
    expected_matrix: np.ndarray,
    *,
    name: str,
) -> float:
    actual_matrix = actual.to_matrix()
    np.testing.assert_allclose(
        actual_matrix,
        expected_matrix,
        atol=COMPOSITION_ATOL,
        rtol=0.0,
        err_msg=name,
    )
    return float(np.max(np.abs(actual_matrix - expected_matrix)))


def _assert_proper_rigid_transform(transform: Transform, *, name: str) -> None:
    matrix = transform.to_matrix()
    assert matrix.shape == (4, 4), f"{name} must have shape (4, 4)"
    assert np.all(np.isfinite(matrix)), f"{name} contains non-finite values"
    np.testing.assert_allclose(
        matrix[3],
        np.array([0.0, 0.0, 0.0, 1.0]),
        atol=RIGID_TRANSFORM_ATOL,
        rtol=0.0,
        err_msg=f"{name} homogeneous row",
    )
    rotation = matrix[:3, :3]
    np.testing.assert_allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=RIGID_TRANSFORM_ATOL,
        rtol=0.0,
        err_msg=f"{name} rotation orthogonality",
    )
    np.testing.assert_allclose(
        np.linalg.det(rotation),
        1.0,
        atol=RIGID_TRANSFORM_ATOL,
        rtol=0.0,
        err_msg=f"{name} rotation determinant",
    )


def test_full_synthetic_image_to_pick_place_waypoints_end_to_end():
    intrinsics = _synthetic_intrinsics()

    # Synthetic test parameters, not physical camera or robot calibration.
    truth_camera_T_object = _rigid_matrix(
        [0.025, -0.018, 0.58], [0.12, -0.09, 0.04]
    )
    base_T_camera_matrix = _rigid_matrix(
        [0.38, -0.16, 0.72], [0.22, -0.11, 0.28]
    )
    desired_base_T_object_matrix = _rigid_matrix(
        [0.28, 0.22, 0.26], [-0.13, 0.19, -0.21]
    )
    object_T_tool_matrix = _rigid_matrix(
        [0.035, -0.012, 0.09], [0.08, 0.14, -0.06]
    )
    offset_matrices = {
        "pick_T_pre_pick": _rigid_matrix(
            [0.018, -0.025, 0.085], [0.06, -0.04, 0.03]
        ),
        "pick_T_lift": _rigid_matrix(
            [-0.012, 0.016, 0.11], [-0.05, 0.07, 0.02]
        ),
        "place_T_pre_place": _rigid_matrix(
            [0.022, 0.014, 0.075], [0.04, 0.05, -0.06]
        ),
        "place_T_retreat": _rigid_matrix(
            [-0.017, 0.021, 0.095], [-0.03, 0.08, 0.04]
        ),
    }

    image = _render_marker(truth_camera_T_object, intrinsics)
    marker_poses = estimate_marker_poses_in_base(
        image=image,
        marker_size=MARKER_SIZE_M,
        intrinsics=intrinsics,
        base_T_camera=Transform.from_matrix(base_T_camera_matrix),
    )

    assert [pose.marker_id for pose in marker_poses] == [MARKER_ID]
    marker_pose = marker_poses[0]
    estimated_camera_T_object = marker_pose.camera_T_object.to_matrix()
    camera_translation_error_m = float(
        np.linalg.norm(
            estimated_camera_T_object[:3, 3] - truth_camera_T_object[:3, 3]
        )
    )
    camera_rotation_error_rad = float(
        Rotation.from_matrix(
            estimated_camera_T_object[:3, :3]
            @ truth_camera_T_object[:3, :3].T
        ).magnitude()
    )
    assert np.isfinite(camera_translation_error_m)
    assert np.isfinite(camera_rotation_error_rad)
    assert camera_translation_error_m < CAMERA_TRANSLATION_TOLERANCE_M
    assert camera_rotation_error_rad < CAMERA_ROTATION_TOLERANCE_RAD

    expected_base_T_object = base_T_camera_matrix @ estimated_camera_T_object
    base_error = _assert_transform_matches(
        marker_pose.base_T_object,
        expected_base_T_object,
        name="base_T_object",
    )

    targets = compute_pick_place_targets(
        marker_pose=marker_pose,
        desired_base_T_object=Transform.from_matrix(desired_base_T_object_matrix),
        object_T_tool=Transform.from_matrix(object_T_tool_matrix),
    )
    expected_pick = expected_base_T_object @ object_T_tool_matrix
    expected_place = desired_base_T_object_matrix @ object_T_tool_matrix
    pick_error = _assert_transform_matches(
        targets.base_T_pick_tool,
        expected_pick,
        name="base_T_pick_tool",
    )
    place_error = _assert_transform_matches(
        targets.base_T_place_tool,
        expected_place,
        name="base_T_place_tool",
    )

    offsets = WaypointOffsets(
        pick_T_pre_pick=Transform.from_matrix(offset_matrices["pick_T_pre_pick"]),
        pick_T_lift=Transform.from_matrix(offset_matrices["pick_T_lift"]),
        place_T_pre_place=Transform.from_matrix(
            offset_matrices["place_T_pre_place"]
        ),
        place_T_retreat=Transform.from_matrix(offset_matrices["place_T_retreat"]),
    )
    waypoints = compute_pick_place_waypoints(targets, offsets)

    expected_waypoints = {
        "pre_pick": expected_pick @ offset_matrices["pick_T_pre_pick"],
        "pick": expected_pick,
        "lift": expected_pick @ offset_matrices["pick_T_lift"],
        "pre_place": expected_place @ offset_matrices["place_T_pre_place"],
        "place": expected_place,
        "retreat": expected_place @ offset_matrices["place_T_retreat"],
    }
    actual_waypoints = {
        "pre_pick": waypoints.base_T_pre_pick_tool,
        "pick": waypoints.base_T_pick_tool,
        "lift": waypoints.base_T_lift_tool,
        "pre_place": waypoints.base_T_pre_place_tool,
        "place": waypoints.base_T_place_tool,
        "retreat": waypoints.base_T_retreat_tool,
    }
    waypoint_errors = {
        name: _assert_transform_matches(
            actual_waypoints[name], expected_matrix, name=name
        )
        for name, expected_matrix in expected_waypoints.items()
    }

    np.testing.assert_array_equal(
        waypoints.base_T_pick_tool.to_matrix(),
        targets.base_T_pick_tool.to_matrix(),
    )
    np.testing.assert_array_equal(
        waypoints.base_T_place_tool.to_matrix(),
        targets.base_T_place_tool.to_matrix(),
    )
    for name, transform in actual_waypoints.items():
        _assert_proper_rigid_transform(transform, name=name)

    assert marker_pose.marker_id == targets.marker_id == waypoints.marker_id == MARKER_ID

    diagnostics = {
        "base": base_error,
        "pick_target": pick_error,
        "place_target": place_error,
        **waypoint_errors,
    }
    diagnostic_text = " ".join(
        f"{name}_max_abs_error={error:.3e}"
        for name, error in diagnostics.items()
    )
    print(
        "E2E_METRICS "
        f"marker_id={waypoints.marker_id} "
        f"camera_translation_error_m={camera_translation_error_m:.12f} "
        f"camera_rotation_error_rad={camera_rotation_error_rad:.12f} "
        f"{diagnostic_text} "
        "all_six_waypoints_proper=True marker_id_preserved=True"
    )
