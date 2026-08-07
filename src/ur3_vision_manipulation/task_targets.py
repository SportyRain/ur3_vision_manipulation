"""Task-space tool targets derived from object poses."""

from __future__ import annotations

from dataclasses import dataclass

from ur3_vision_manipulation.geometry import Transform
from ur3_vision_manipulation.perception.pipeline import MarkerPoseInBase


@dataclass(frozen=True)
class PickPlaceTargets:
    """Pick and place tool poses expressed in the robot base frame."""

    marker_id: int
    base_T_pick_tool: Transform
    base_T_place_tool: Transform


def compute_pick_place_targets(
    marker_pose: MarkerPoseInBase,
    desired_base_T_object: Transform,
    object_T_tool: Transform,
) -> PickPlaceTargets:
    """Compose pick and place tool targets from object-frame tool geometry."""

    return PickPlaceTargets(
        marker_id=marker_pose.marker_id,
        base_T_pick_tool=marker_pose.base_T_object @ object_T_tool,
        base_T_place_tool=desired_base_T_object @ object_T_tool,
    )
