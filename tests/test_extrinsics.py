import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ur3_vision_manipulation.geometry import Transform
from ur3_vision_manipulation.perception import (
    estimate_base_T_camera,
    transform_object_to_base,
)


def _transform(translation, rotation_vector) -> Transform:
    return Transform.from_translation_rotation(
        translation,
        Rotation.from_rotvec(rotation_vector).as_matrix(),
    )


def _exact_reference_pairs():
    known_base_T_camera = _transform([0.42, -0.18, 0.73], [0.25, -0.15, 0.32])
    camera_T_references = [
        _transform([0.10, 0.02, 0.55], [0.10, -0.05, 0.02]),
        _transform([-0.08, 0.12, 0.68], [-0.12, 0.18, -0.04]),
        _transform([0.04, -0.09, 0.61], [0.07, 0.11, 0.16]),
        _transform([0.15, 0.06, 0.72], [-0.08, -0.14, 0.09]),
    ]
    base_T_references = [
        known_base_T_camera @ camera_T_reference
        for camera_T_reference in camera_T_references
    ]
    return known_base_T_camera, base_T_references, camera_T_references


def test_base_T_object_chain():
    base_T_camera = _transform([0.35, -0.20, 0.62], [0.20, -0.10, 0.25])
    camera_T_object = _transform([0.08, 0.04, 0.48], [-0.12, 0.18, 0.06])
    base_T_object = transform_object_to_base(base_T_camera, camera_T_object)
    object_point = np.array([0.03, -0.02, 0.05])

    expected = (base_T_camera @ camera_T_object).to_matrix()
    np.testing.assert_allclose(base_T_object.to_matrix(), expected, atol=1.0e-12)
    p_camera = camera_T_object.transform_point(object_point)
    p_base_via_camera = base_T_camera.transform_point(p_camera)
    p_base_direct = base_T_object.transform_point(object_point)
    np.testing.assert_allclose(p_base_direct, p_base_via_camera, atol=1.0e-12)


def test_extrinsic_recovers_exact_synthetic_base_T_camera():
    known, base_references, camera_references = _exact_reference_pairs()

    result = estimate_base_T_camera(base_references, camera_references)
    known_matrix = known.to_matrix()
    estimated_matrix = result.base_T_camera.to_matrix()
    translation_error = np.linalg.norm(
        estimated_matrix[:3, 3] - known_matrix[:3, 3]
    )
    rotation_error = Rotation.from_matrix(
        estimated_matrix[:3, :3] @ known_matrix[:3, :3].T
    ).magnitude()

    assert result.sample_count == 4
    assert translation_error < 1.0e-12
    assert rotation_error < 1.0e-12


def test_extrinsic_rejects_empty_samples():
    with pytest.raises(ValueError, match="at least one sample"):
        estimate_base_T_camera([], [])


def test_extrinsic_rejects_mismatched_sample_counts():
    with pytest.raises(ValueError, match="counts must match"):
        estimate_base_T_camera([Transform.identity()], [])


def test_extrinsic_rejects_non_transform_sample():
    with pytest.raises(TypeError, match=r"camera_T_references\[0\]"):
        estimate_base_T_camera([Transform.identity()], [object()])


def test_extrinsic_residual_is_near_zero_for_exact_data():
    _, base_references, camera_references = _exact_reference_pairs()

    result = estimate_base_T_camera(base_references, camera_references)

    assert result.translation_rmse_m < 1.0e-12
    assert result.rotation_rmse_rad < 1.0e-12


def test_extrinsic_handles_deterministic_small_noise():
    known, _, camera_references = _exact_reference_pairs()
    known_matrix = known.to_matrix()
    translation_noise = np.array(
        [
            [0.0005, 0.0, 0.0],
            [-0.0005, 0.0, 0.0],
            [0.0, 0.0004, 0.0],
            [0.0, -0.0004, 0.0],
        ]
    )
    rotation_noise = np.array(
        [
            [0.0010, 0.0, 0.0],
            [-0.0010, 0.0, 0.0],
            [0.0, 0.0008, 0.0],
            [0.0, -0.0008, 0.0],
        ]
    )
    noisy_base_references = []
    for translation_delta, rotation_delta, camera_reference in zip(
        translation_noise, rotation_noise, camera_references, strict=True
    ):
        noisy_base_T_camera = Transform.from_translation_rotation(
            known_matrix[:3, 3] + translation_delta,
            Rotation.from_rotvec(rotation_delta).as_matrix() @ known_matrix[:3, :3],
        )
        noisy_base_references.append(noisy_base_T_camera @ camera_reference)

    result = estimate_base_T_camera(noisy_base_references, camera_references)
    estimated_matrix = result.base_T_camera.to_matrix()
    translation_error = np.linalg.norm(
        estimated_matrix[:3, 3] - known_matrix[:3, 3]
    )
    rotation_error = Rotation.from_matrix(
        estimated_matrix[:3, :3] @ known_matrix[:3, :3].T
    ).magnitude()

    assert translation_error < 1.0e-12
    assert rotation_error < 1.0e-6
    assert result.translation_rmse_m < 0.0006
    assert result.rotation_rmse_rad < 0.0011
