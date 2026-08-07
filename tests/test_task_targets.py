import numpy as np
from scipy.spatial.transform import Rotation

from ur3_vision_manipulation.geometry import Transform
from ur3_vision_manipulation.perception import MarkerPoseInBase
from ur3_vision_manipulation.task_targets import compute_pick_place_targets


def _transform(translation, rotation_vector) -> Transform:
    return Transform.from_translation_rotation(
        translation, Rotation.from_rotvec(rotation_vector).as_matrix()
    )


def _synthetic_targets():
    base_T_object = _transform([0.42, -0.18, 0.31], [0.24, -0.17, 0.11])
    desired_base_T_object = _transform(
        [0.28, 0.22, 0.26], [-0.13, 0.19, -0.21]
    )
    object_T_tool = _transform([0.035, -0.012, 0.09], [0.08, 0.14, -0.06])
    marker_pose = MarkerPoseInBase(
        marker_id=27,
        camera_T_object=Transform.identity(),
        base_T_object=base_T_object,
    )
    result = compute_pick_place_targets(
        marker_pose, desired_base_T_object, object_T_tool
    )
    return result, base_T_object, desired_base_T_object, object_T_tool


def test_pick_target_composition_order():
    result, base_T_object, _, object_T_tool = _synthetic_targets()

    expected = base_T_object @ object_T_tool
    reversed_order = object_T_tool @ base_T_object
    np.testing.assert_allclose(
        result.base_T_pick_tool.to_matrix(), expected.to_matrix(), atol=1.0e-12
    )
    assert not np.allclose(
        result.base_T_pick_tool.to_matrix(), reversed_order.to_matrix()
    )


def test_place_target_composition_order():
    result, _, desired_base_T_object, object_T_tool = _synthetic_targets()

    expected = desired_base_T_object @ object_T_tool
    reversed_order = object_T_tool @ desired_base_T_object
    np.testing.assert_allclose(
        result.base_T_place_tool.to_matrix(), expected.to_matrix(), atol=1.0e-12
    )
    assert not np.allclose(
        result.base_T_place_tool.to_matrix(), reversed_order.to_matrix()
    )


def test_marker_id_is_preserved():
    result, _, _, _ = _synthetic_targets()

    assert result.marker_id == 27
