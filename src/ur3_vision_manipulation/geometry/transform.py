"""Rigid SE(3) transforms with explicit parent/child frame semantics.

``T_parent_child`` maps coordinates expressed in the child frame into the
parent frame: ``p_parent = T_parent_child @ p_child``. Composition follows the
same notation, for example ``base_T_object = base_T_camera @ camera_T_object``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .rotation import DEFAULT_ATOL, quaternion_to_matrix, validate_rotation_matrix


def _validate_vector(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.copy()


def validate_homogeneous_matrix(
    matrix: ArrayLike, *, atol: float = DEFAULT_ATOL
) -> NDArray[np.float64]:
    """Return a validated rigid 4x4 homogeneous transform matrix."""

    transform = np.asarray(matrix, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"homogeneous matrix must have shape (4, 4), got {transform.shape}")
    if not np.all(np.isfinite(transform)):
        raise ValueError("homogeneous matrix must contain only finite values")
    if not np.allclose(
        transform[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=atol, rtol=0.0
    ):
        raise ValueError("homogeneous matrix last row must be [0, 0, 0, 1]")
    validate_rotation_matrix(transform[:3, :3], atol=atol)
    return transform.copy()


class Transform:
    """A rigid ``T_parent_child`` transform in SE(3)."""

    __slots__ = ("_matrix",)

    def __init__(self, matrix: ArrayLike) -> None:
        validated = validate_homogeneous_matrix(matrix)
        validated.setflags(write=False)
        self._matrix = validated

    @classmethod
    def identity(cls) -> Transform:
        """Return the identity transform."""

        return cls(np.eye(4, dtype=float))

    @classmethod
    def from_translation_rotation(
        cls, translation: ArrayLike, rotation: ArrayLike
    ) -> Transform:
        """Build a transform from translation and a matrix or xyzw quaternion."""

        translation_value = _validate_vector(translation, name="translation")
        rotation_value = np.asarray(rotation, dtype=float)
        if rotation_value.shape == (4,):
            rotation_matrix = quaternion_to_matrix(rotation_value)
        else:
            rotation_matrix = validate_rotation_matrix(rotation_value)
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = rotation_matrix
        matrix[:3, 3] = translation_value
        return cls(matrix)

    @classmethod
    def from_matrix(cls, matrix: ArrayLike) -> Transform:
        """Build a transform from a validated 4x4 homogeneous matrix."""

        return cls(matrix)

    def to_matrix(self) -> NDArray[np.float64]:
        """Return a writable copy of the homogeneous matrix."""

        return self._matrix.copy()

    def inverse(self) -> Transform:
        """Return ``T_child_parent`` for this ``T_parent_child`` transform."""

        rotation = self._matrix[:3, :3]
        translation = self._matrix[:3, 3]
        inverse_matrix = np.eye(4, dtype=float)
        inverse_matrix[:3, :3] = rotation.T
        inverse_matrix[:3, 3] = -(rotation.T @ translation)
        return Transform(inverse_matrix)

    def __matmul__(self, other: object) -> Transform:
        """Compose compatible transforms using parent/child frame order."""

        if not isinstance(other, Transform):
            return NotImplemented
        return Transform(self._matrix @ other._matrix)

    def transform_point(self, point: ArrayLike) -> NDArray[np.float64]:
        """Map a 3D point from the child frame into the parent frame."""

        point_value = _validate_vector(point, name="point")
        return self._matrix[:3, :3] @ point_value + self._matrix[:3, 3]
