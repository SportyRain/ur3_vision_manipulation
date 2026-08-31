"""Task-space tool targets derived from object poses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from numpy.typing import ArrayLike

from ur3_vision_manipulation.geometry import Transform
from ur3_vision_manipulation.perception.pipeline import MarkerPoseInBase


class SceneObjectLike(Protocol):
    """Minimum scene-object contract needed for metric pick targeting."""

    object_id: str
    base_xyz_m: Sequence[float] | None


@dataclass(frozen=True)
class PickPlaceTargets:
    """Pick and place tool poses expressed in the robot base frame."""

    marker_id: int
    base_T_pick_tool: Transform
    base_T_place_tool: Transform


@dataclass(frozen=True)
class ScenePickPlaceTargets:
    """Pick and place tool poses derived from a stable scene object."""

    object_id: str
    base_T_pick_tool: Transform
    base_T_place_tool: Transform


def _compose_pick_place_tool_targets(
    base_T_object: Transform,
    desired_base_T_object: Transform,
    object_T_tool: Transform,
) -> tuple[Transform, Transform]:
    return (
        base_T_object @ object_T_tool,
        desired_base_T_object @ object_T_tool,
    )


def compute_pick_place_targets(
    marker_pose: MarkerPoseInBase,
    desired_base_T_object: Transform,
    object_T_tool: Transform,
) -> PickPlaceTargets:
    """Compose pick and place tool targets from object-frame tool geometry."""

    base_T_pick_tool, base_T_place_tool = _compose_pick_place_tool_targets(
        marker_pose.base_T_object,
        desired_base_T_object,
        object_T_tool,
    )
    return PickPlaceTargets(
        marker_id=marker_pose.marker_id,
        base_T_pick_tool=base_T_pick_tool,
        base_T_place_tool=base_T_place_tool,
    )


def compute_scene_pick_place_targets(
    scene_object: SceneObjectLike,
    base_R_object: ArrayLike,
    desired_base_T_object: Transform,
    object_T_tool: Transform,
) -> ScenePickPlaceTargets:
    """Compose targets from a metric scene object without inventing orientation.

    The scene-object observer currently provides stable identity and robot-base
    XYZ, but not a verified object orientation.  The caller must therefore
    provide ``base_R_object`` explicitly.  Missing metric position fails closed
    instead of silently promoting a 2D observation into a robot target.
    """

    object_id = str(scene_object.object_id).strip()
    if not object_id:
        raise ValueError("scene object_id must be non-empty")
    if scene_object.base_xyz_m is None:
        raise ValueError("scene object requires metric base_xyz_m for pick targeting")

    base_T_object = Transform.from_translation_rotation(
        scene_object.base_xyz_m,
        base_R_object,
    )
    base_T_pick_tool, base_T_place_tool = _compose_pick_place_tool_targets(
        base_T_object,
        desired_base_T_object,
        object_T_tool,
    )
    return ScenePickPlaceTargets(
        object_id=object_id,
        base_T_pick_tool=base_T_pick_tool,
        base_T_place_tool=base_T_place_tool,
    )
