"""Position and orientation pose representation without ROS dependencies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .rotation import matrix_to_quaternion, normalize_quaternion, quaternion_to_matrix
from .transform import validate_homogeneous_matrix


def _validate_position(position: ArrayLike) -> NDArray[np.float64]:
    value = np.asarray(position, dtype=float)
    if value.shape != (3,):
        raise ValueError(f"position must have shape (3,), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("position must contain only finite values")
    return value.copy()


@dataclass(frozen=True)
class Pose:
    """A position and normalized orientation quaternion in ``[x, y, z, w]``."""

    position: NDArray[np.float64]
    orientation: NDArray[np.float64]

    def __init__(self, position: ArrayLike, orientation: ArrayLike) -> None:
        position_value = _validate_position(position)
        orientation_value = normalize_quaternion(orientation)
        position_value.setflags(write=False)
        orientation_value.setflags(write=False)
        object.__setattr__(self, "position", position_value)
        object.__setattr__(self, "orientation", orientation_value)

    def to_matrix(self) -> NDArray[np.float64]:
        """Convert this pose to a rigid 4x4 homogeneous matrix."""

        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = quaternion_to_matrix(self.orientation)
        matrix[:3, 3] = self.position
        return matrix

    @classmethod
    def from_matrix(cls, matrix: ArrayLike) -> Pose:
        """Restore a pose from a rigid 4x4 homogeneous matrix."""

        validated = validate_homogeneous_matrix(matrix)
        return cls(validated[:3, 3], matrix_to_quaternion(validated[:3, :3]))
