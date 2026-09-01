"""Task-space tool targets derived from object poses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
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


def compute_upright_symmetric_tabletop_scene_pick_place_targets(
    scene_object: SceneObjectLike,
    tabletop_z_m: float,
    object_height_m: float,
    desired_base_T_object: Transform,
    object_T_tool: Transform,
) -> ScenePickPlaceTargets:
    """Compose targets for a bounded upright rotationally symmetric object.

    This contract is intentionally narrower than generic 6D pose estimation.
    The scene observation supplies robot-base X/Y only.  The object is assumed
    to stand upright on a horizontal tabletop, so its center Z is derived from
    explicit tabletop height plus half the explicit object height.  Rotation
    about the upright axis is physically irrelevant for the bounded symmetric
    object, so the task object frame uses the robot-base orientation.

    The observed scene Z is deliberately not promoted into grasp geometry.
    Callers must not use this helper for tilted, asymmetric, or otherwise
    orientation-sensitive objects.
    """

    object_id = str(scene_object.object_id).strip()
    if not object_id:
        raise ValueError("scene object_id must be non-empty")
    if scene_object.base_xyz_m is None:
        raise ValueError("scene object requires metric base_xyz_m for pick targeting")

    scene_xyz = np.asarray(scene_object.base_xyz_m, dtype=float)
    if scene_xyz.shape != (3,):
        raise ValueError(f"scene base_xyz_m must have shape (3,), got {scene_xyz.shape}")
    if not np.all(np.isfinite(scene_xyz)):
        raise ValueError("scene base_xyz_m must contain only finite values")

    tabletop_z = float(tabletop_z_m)
    object_height = float(object_height_m)
    if not np.isfinite(tabletop_z):
        raise ValueError("tabletop_z_m must be finite")
    if not np.isfinite(object_height) or object_height <= 0.0:
        raise ValueError("object_height_m must be finite and positive")

    object_center_xyz = np.array(
        [scene_xyz[0], scene_xyz[1], tabletop_z + 0.5 * object_height],
        dtype=float,
    )
    base_T_object = Transform.from_translation_rotation(object_center_xyz, np.eye(3))
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
