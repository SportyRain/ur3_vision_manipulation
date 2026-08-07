"""Validated pinhole-camera intrinsic parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

_VALID_DISTORTION_LENGTHS = frozenset({4, 5, 8, 12, 14})


@dataclass(frozen=True, init=False)
class CameraIntrinsics:
    """OpenCV camera matrix, distortion vector, and calibrated image size.

    Distortion coefficients may be supplied as a one-dimensional vector or an
    OpenCV-style row/column vector. They are stored as a flat vector with one
    of OpenCV's supported lengths: 4, 5, 8, 12, or 14.
    """

    camera_matrix: NDArray[np.float64]
    distortion_coefficients: NDArray[np.float64]
    image_width: int
    image_height: int

    def __init__(
        self,
        camera_matrix: ArrayLike,
        distortion_coefficients: ArrayLike,
        image_width: int,
        image_height: int,
    ) -> None:
        matrix = np.asarray(camera_matrix, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError(f"camera matrix must have shape (3, 3), got {matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("camera matrix must contain only finite values")
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError("camera focal lengths fx and fy must be positive")
        if not np.allclose(matrix[2], [0.0, 0.0, 1.0]):
            raise ValueError("camera matrix last row must be [0, 0, 1]")

        distortion = np.asarray(distortion_coefficients, dtype=float)
        if distortion.ndim == 2 and 1 in distortion.shape:
            distortion = distortion.reshape(-1)
        if distortion.ndim != 1 or distortion.size not in _VALID_DISTORTION_LENGTHS:
            raise ValueError(
                "distortion coefficients must be a vector of length 4, 5, 8, 12, or 14"
            )
        if not np.all(np.isfinite(distortion)):
            raise ValueError("distortion coefficients must contain only finite values")

        if isinstance(image_width, bool) or not isinstance(image_width, (int, np.integer)):
            raise ValueError("image width must be a positive integer")
        if isinstance(image_height, bool) or not isinstance(image_height, (int, np.integer)):
            raise ValueError("image height must be a positive integer")
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image width and height must be positive")

        matrix = matrix.copy()
        distortion = distortion.copy()
        matrix.setflags(write=False)
        distortion.setflags(write=False)
        object.__setattr__(self, "camera_matrix", matrix)
        object.__setattr__(self, "distortion_coefficients", distortion)
        object.__setattr__(self, "image_width", int(image_width))
        object.__setattr__(self, "image_height", int(image_height))

    @property
    def image_size(self) -> tuple[int, int]:
        """Return the OpenCV image size as ``(width, height)``."""

        return self.image_width, self.image_height
