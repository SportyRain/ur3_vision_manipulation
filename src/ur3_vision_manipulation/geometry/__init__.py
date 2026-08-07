"""Rigid-body geometry using quaternions ordered as ``[x, y, z, w]``."""

from .pose import Pose
from .rotation import (
    identity_rotation,
    matrix_to_quaternion,
    normalize_quaternion,
    quaternion_to_matrix,
    validate_rotation_matrix,
)
from .transform import Transform, validate_homogeneous_matrix

__all__ = [
    "Pose",
    "Transform",
    "identity_rotation",
    "matrix_to_quaternion",
    "normalize_quaternion",
    "quaternion_to_matrix",
    "validate_homogeneous_matrix",
    "validate_rotation_matrix",
]
