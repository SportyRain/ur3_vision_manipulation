"""Task-space waypoint poses derived from pick and place targets."""

from __future__ import annotations

from dataclasses import dataclass

from ur3_vision_manipulation.geometry import Transform
from ur3_vision_manipulation.task_targets import PickPlaceTargets, ScenePickPlaceTargets


@dataclass(frozen=True)
class WaypointOffsets:
    """Caller-supplied tool-frame offsets relative to pick and place poses."""

    pick_T_pre_pick: Transform
    pick_T_lift: Transform
    place_T_pre_place: Transform
    place_T_retreat: Transform


@dataclass(frozen=True)
class PickPlaceWaypoints:
    """Pick and place waypoint tool poses expressed in the robot base frame."""

    marker_id: int
    base_T_pre_pick_tool: Transform
    base_T_pick_tool: Transform
    base_T_lift_tool: Transform
    base_T_pre_place_tool: Transform
    base_T_place_tool: Transform
    base_T_retreat_tool: Transform


@dataclass(frozen=True)
class ScenePickPlaceWaypoints:
    """Scene-object Pick/Place waypoints expressed in the robot base frame."""

    object_id: str
    base_T_pre_pick_tool: Transform
    base_T_pick_tool: Transform
    base_T_lift_tool: Transform
    base_T_pre_place_tool: Transform
    base_T_place_tool: Transform
    base_T_retreat_tool: Transform


def compute_pick_place_waypoints(
    targets: PickPlaceTargets,
    offsets: WaypointOffsets,
) -> PickPlaceWaypoints:
    """Compose caller-supplied waypoint offsets with pick and place targets."""

    return PickPlaceWaypoints(
        marker_id=targets.marker_id,
        base_T_pre_pick_tool=targets.base_T_pick_tool @ offsets.pick_T_pre_pick,
        base_T_pick_tool=targets.base_T_pick_tool,
        base_T_lift_tool=targets.base_T_pick_tool @ offsets.pick_T_lift,
        base_T_pre_place_tool=(
            targets.base_T_place_tool @ offsets.place_T_pre_place
        ),
        base_T_place_tool=targets.base_T_place_tool,
        base_T_retreat_tool=targets.base_T_place_tool @ offsets.place_T_retreat,
    )


def compute_scene_pick_place_waypoints(
    targets: ScenePickPlaceTargets,
    offsets: WaypointOffsets,
) -> ScenePickPlaceWaypoints:
    """Reuse the established Pick/Place waypoint composition for a scene object."""

    return ScenePickPlaceWaypoints(
        object_id=targets.object_id,
        base_T_pre_pick_tool=targets.base_T_pick_tool @ offsets.pick_T_pre_pick,
        base_T_pick_tool=targets.base_T_pick_tool,
        base_T_lift_tool=targets.base_T_pick_tool @ offsets.pick_T_lift,
        base_T_pre_place_tool=(
            targets.base_T_place_tool @ offsets.place_T_pre_place
        ),
        base_T_place_tool=targets.base_T_place_tool,
        base_T_retreat_tool=targets.base_T_place_tool @ offsets.place_T_retreat,
    )
