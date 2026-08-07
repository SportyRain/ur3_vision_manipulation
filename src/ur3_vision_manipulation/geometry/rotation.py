"""Rotation validation and conversion helpers.

Quaternion values always use the ``[x, y, z, w]`` ordering used by
``scipy.spatial.transform.Rotation``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation as SciPyRotation

DEFAULT_ATOL = 1.0e-9
MIN_QUATERNION_NORM = 1.0e-12


def validate_rotation_matrix(
    matrix: ArrayLike, *, atol: float = DEFAULT_ATOL
) -> NDArray[np.float64]:
    """Return a validated proper 3x3 rotation matrix.

    A valid matrix is finite, orthogonal within ``atol``, and has determinant
    approximately equal to +1. Reflections are therefore rejected.
    """

    rotation = np.asarray(matrix, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation matrix must have shape (3, 3), got {rotation.shape}")
    if not np.all(np.isfinite(rotation)):
        raise ValueError("rotation matrix must contain only finite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol, rtol=0.0):
        raise ValueError("rotation matrix must be orthogonal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=atol, rtol=0.0):
        raise ValueError("rotation matrix determinant must be approximately +1")
    return rotation.copy()


def normalize_quaternion(quaternion: ArrayLike) -> NDArray[np.float64]:
    """Validate and normalize a quaternion ordered as ``[x, y, z, w]``."""

    value = np.asarray(quaternion, dtype=float)
    if value.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("quaternion must contain only finite values")
    norm = float(np.linalg.norm(value))
    if norm <= MIN_QUATERNION_NORM:
        raise ValueError("quaternion norm must be non-zero")
    return value / norm


def quaternion_to_matrix(quaternion: ArrayLike) -> NDArray[np.float64]:
    """Convert an ``[x, y, z, w]`` quaternion to a proper rotation matrix."""

    normalized = normalize_quaternion(quaternion)
    return validate_rotation_matrix(SciPyRotation.from_quat(normalized).as_matrix())


def matrix_to_quaternion(matrix: ArrayLike) -> NDArray[np.float64]:
    """Convert a proper rotation matrix to ``[x, y, z, w]`` ordering."""

    rotation = validate_rotation_matrix(matrix)
    return normalize_quaternion(SciPyRotation.from_matrix(rotation).as_quat())


def identity_rotation() -> NDArray[np.float64]:
    """Return the 3x3 identity rotation."""

    return np.eye(3, dtype=float)
