import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ur3_vision_manipulation.geometry import Transform
from ur3_vision_manipulation.task_targets import PickPlaceTargets
from ur3_vision_manipulation.waypoints import (
    GripperState,
    WaypointOffsets,
    compute_pick_sequence,
    compute_pick_place_waypoints,
)


def _transform(translation, rotation_vector) -> Transform:
    return Transform.from_translation_rotation(
        translation, Rotation.from_rotvec(rotation_vector).as_matrix()
    )


def _synthetic_waypoints():
    targets = PickPlaceTargets(
        marker_id=41,
        base_T_pick_tool=_transform([0.43, -0.21, 0.36], [0.22, -0.16, 0.09]),
        base_T_place_tool=_transform([0.25, 0.24, 0.29], [-0.14, 0.18, -0.2]),
    )
    offsets = WaypointOffsets(
        pick_T_pre_pick=_transform([0.018, -0.025, 0.085], [0.06, -0.04, 0.03]),
        pick_T_lift=_transform([-0.012, 0.016, 0.11], [-0.05, 0.07, 0.02]),
        place_T_pre_place=_transform([0.022, 0.014, 0.075], [0.04, 0.05, -0.06]),
        place_T_retreat=_transform([-0.017, 0.021, 0.095], [-0.03, 0.08, 0.04]),
    )
    return targets, offsets, compute_pick_place_waypoints(targets, offsets)


def test_pick_side_waypoint_composition_order():
    targets, offsets, waypoints = _synthetic_waypoints()

    expected_pre_pick = targets.base_T_pick_tool @ offsets.pick_T_pre_pick
    expected_lift = targets.base_T_pick_tool @ offsets.pick_T_lift
    reversed_pre_pick = offsets.pick_T_pre_pick @ targets.base_T_pick_tool
    np.testing.assert_allclose(
        waypoints.base_T_pre_pick_tool.to_matrix(),
        expected_pre_pick.to_matrix(),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        waypoints.base_T_lift_tool.to_matrix(), expected_lift.to_matrix(), atol=1.0e-12
    )
    assert not np.allclose(
        waypoints.base_T_pre_pick_tool.to_matrix(), reversed_pre_pick.to_matrix()
    )


def test_place_side_waypoint_composition_order():
    targets, offsets, waypoints = _synthetic_waypoints()

    expected_pre_place = targets.base_T_place_tool @ offsets.place_T_pre_place
    expected_retreat = targets.base_T_place_tool @ offsets.place_T_retreat
    reversed_retreat = offsets.place_T_retreat @ targets.base_T_place_tool
    np.testing.assert_allclose(
        waypoints.base_T_pre_place_tool.to_matrix(),
        expected_pre_place.to_matrix(),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        waypoints.base_T_retreat_tool.to_matrix(),
        expected_retreat.to_matrix(),
        atol=1.0e-12,
    )
    assert not np.allclose(
        waypoints.base_T_retreat_tool.to_matrix(), reversed_retreat.to_matrix()
    )


def test_pick_and_place_targets_are_preserved():
    targets, _, waypoints = _synthetic_waypoints()

    np.testing.assert_allclose(
        waypoints.base_T_pick_tool.to_matrix(), targets.base_T_pick_tool.to_matrix()
    )
    np.testing.assert_allclose(
        waypoints.base_T_place_tool.to_matrix(), targets.base_T_place_tool.to_matrix()
    )


def test_marker_id_is_preserved():
    targets, _, waypoints = _synthetic_waypoints()

    assert waypoints.marker_id == targets.marker_id


def test_pick_sequence_has_ordered_two_state_gripper_meaning():
    _, _, waypoints = _synthetic_waypoints()

    sequence = compute_pick_sequence(waypoints)

    assert [step.name for step in sequence] == ["pre_pick", "pick", "lift"]
    assert [step.gripper for step in sequence] == [
        GripperState.OFF,
        GripperState.ON,
        GripperState.ON,
    ]


def test_pick_sequence_reuses_existing_waypoint_objects():
    _, _, waypoints = _synthetic_waypoints()

    sequence = compute_pick_sequence(waypoints)

    assert sequence[0].waypoint is waypoints.base_T_pre_pick_tool
    assert sequence[1].waypoint is waypoints.base_T_pick_tool
    assert sequence[2].waypoint is waypoints.base_T_lift_tool


def test_pick_sequence_rejects_non_waypoint_input():
    with pytest.raises(TypeError, match="PickPlaceWaypoints"):
        compute_pick_sequence(None)  # type: ignore[arg-type]
