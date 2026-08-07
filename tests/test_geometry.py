import numpy as np
import pytest

from ur3_vision_manipulation.geometry import (
    Pose,
    Transform,
    identity_rotation,
    matrix_to_quaternion,
    normalize_quaternion,
    quaternion_to_matrix,
    validate_rotation_matrix,
)


def test_rotation_identity():
    np.testing.assert_allclose(identity_rotation(), np.eye(3))
    np.testing.assert_allclose(quaternion_to_matrix([0, 0, 0, 1]), np.eye(3))


def test_rotation_invalid_shape():
    with pytest.raises(ValueError, match="shape"):
        validate_rotation_matrix(np.eye(4))


def test_rotation_rejects_reflection():
    reflection = np.diag([-1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="determinant"):
        validate_rotation_matrix(reflection)


def test_quaternion_normalization():
    np.testing.assert_allclose(normalize_quaternion([0, 0, 0, 2]), [0, 0, 0, 1])


def test_quaternion_zero_rejected():
    with pytest.raises(ValueError, match="non-zero"):
        normalize_quaternion([0, 0, 0, 0])


def test_quaternion_matrix_roundtrip():
    original = normalize_quaternion([0.2, -0.3, 0.1, 0.9])
    matrix = quaternion_to_matrix(original)
    restored = matrix_to_quaternion(matrix)
    np.testing.assert_allclose(quaternion_to_matrix(restored), matrix, atol=1e-12)


def test_transform_identity():
    np.testing.assert_allclose(Transform.identity().to_matrix(), np.eye(4))


def test_transform_inverse():
    transform = Transform.from_translation_rotation(
        [0.4, -0.2, 0.7], [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    )
    np.testing.assert_allclose(
        (transform @ transform.inverse()).to_matrix(), np.eye(4), atol=1e-12
    )


def test_transform_chain():
    base_T_camera = Transform.from_translation_rotation(
        [0.5, 0.0, 0.2], identity_rotation()
    )
    camera_T_object = Transform.from_translation_rotation(
        [0.1, -0.2, 0.3], identity_rotation()
    )
    base_T_object = base_T_camera @ camera_T_object

    expected_translation = np.array([0.6, -0.2, 0.5])
    np.testing.assert_allclose(base_T_object.to_matrix()[:3, 3], expected_translation)


def test_transform_point():
    transform = Transform.from_translation_rotation([1, 2, 3], identity_rotation())
    np.testing.assert_allclose(transform.transform_point([0.5, -1, 2]), [1.5, 1, 5])


def test_pose_matrix_roundtrip():
    pose = Pose([0.25, -0.5, 0.75], [0.1, 0.2, -0.3, 0.9])
    restored = Pose.from_matrix(pose.to_matrix())
    np.testing.assert_allclose(restored.position, pose.position)
    np.testing.assert_allclose(
        quaternion_to_matrix(restored.orientation),
        quaternion_to_matrix(pose.orientation),
        atol=1e-12,
    )


def test_invalid_homogeneous_matrix():
    invalid = np.eye(4)
    invalid[3, 3] = 2.0
    with pytest.raises(ValueError, match="last row"):
        Transform.from_matrix(invalid)
