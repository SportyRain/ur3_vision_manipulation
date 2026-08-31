from dataclasses import dataclass

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ur3_vision_manipulation.geometry import Transform
from ur3_vision_manipulation.task_targets import compute_scene_pick_place_targets


@dataclass(frozen=True)
class _SceneObject:
    object_id: str
    base_xyz_m: tuple[float, float, float] | None


def _transform(translation, rotation_vector) -> Transform:
    return Transform.from_translation_rotation(
        translation, Rotation.from_rotvec(rotation_vector).as_matrix()
    )


def test_scene_object_metric_xyz_composes_existing_pick_place_geometry():
    scene_object = _SceneObject("object_007", (0.42, -0.18, 0.31))
    base_R_object = Rotation.from_rotvec([0.24, -0.17, 0.11]).as_matrix()
    desired_base_T_object = _transform(
        [0.28, 0.22, 0.26], [-0.13, 0.19, -0.21]
    )
    object_T_tool = _transform([0.035, -0.012, 0.09], [0.08, 0.14, -0.06])

    result = compute_scene_pick_place_targets(
        scene_object,
        base_R_object,
        desired_base_T_object,
        object_T_tool,
    )

    expected_base_T_object = Transform.from_translation_rotation(
        scene_object.base_xyz_m,
        base_R_object,
    )
    np.testing.assert_allclose(
        result.base_T_pick_tool.to_matrix(),
        (expected_base_T_object @ object_T_tool).to_matrix(),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.base_T_place_tool.to_matrix(),
        (desired_base_T_object @ object_T_tool).to_matrix(),
        atol=1.0e-12,
    )
    assert result.object_id == "object_007"


def test_scene_object_without_metric_xyz_fails_closed():
    scene_object = _SceneObject("object_007", None)

    with pytest.raises(ValueError, match="metric base_xyz_m"):
        compute_scene_pick_place_targets(
            scene_object,
            np.eye(3),
            Transform.identity(),
            Transform.identity(),
        )


def test_scene_object_without_identity_fails_closed():
    scene_object = _SceneObject("   ", (0.42, -0.18, 0.31))

    with pytest.raises(ValueError, match="object_id"):
        compute_scene_pick_place_targets(
            scene_object,
            np.eye(3),
            Transform.identity(),
            Transform.identity(),
        )


def test_scene_object_orientation_must_be_explicitly_valid():
    scene_object = _SceneObject("object_007", (0.42, -0.18, 0.31))
    invalid_rotation = np.diag([1.0, 1.0, 2.0])

    with pytest.raises(ValueError):
        compute_scene_pick_place_targets(
            scene_object,
            invalid_rotation,
            Transform.identity(),
            Transform.identity(),
        )
