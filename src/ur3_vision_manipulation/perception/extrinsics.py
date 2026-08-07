"""Robot-base mapping and fixed-camera extrinsic estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from ur3_vision_manipulation.geometry import Transform


@dataclass(frozen=True)
class ExtrinsicCalibrationResult:
    """Estimated fixed ``base_T_camera`` and sample residuals."""

    base_T_camera: Transform
    sample_count: int
    translation_rmse_m: float
    rotation_rmse_rad: float


def _require_transform(value: object, *, name: str) -> Transform:
    if not isinstance(value, Transform):
        raise TypeError(f"{name} must be a Transform")
    return value


def transform_object_to_base(
    base_T_camera: Transform,
    camera_T_object: Transform,
) -> Transform:
    """Return ``base_T_object = base_T_camera @ camera_T_object``."""

    base_to_camera = _require_transform(base_T_camera, name="base_T_camera")
    camera_to_object = _require_transform(camera_T_object, name="camera_T_object")
    return base_to_camera @ camera_to_object


def estimate_base_T_camera(
    base_T_references: Sequence[Transform],
    camera_T_references: Sequence[Transform],
) -> ExtrinsicCalibrationResult:
    """Estimate a fixed camera extrinsic from matching reference poses.

    Each pair produces ``base_T_camera = base_T_reference @
    inverse(camera_T_reference)``. Candidate translations are averaged in the
    base frame and candidate rotations are averaged on SO(3) with SciPy.
    Residuals are RMSE values against the resulting average transform.
    """

    if len(base_T_references) != len(camera_T_references):
        raise ValueError("base and camera reference sample counts must match")
    if len(base_T_references) == 0:
        raise ValueError("extrinsic calibration requires at least one sample")

    candidates: list[Transform] = []
    for index, (base_T_reference, camera_T_reference) in enumerate(
        zip(base_T_references, camera_T_references, strict=True)
    ):
        base_reference = _require_transform(
            base_T_reference, name=f"base_T_references[{index}]"
        )
        camera_reference = _require_transform(
            camera_T_reference, name=f"camera_T_references[{index}]"
        )
        candidates.append(base_reference @ camera_reference.inverse())

    candidate_matrices = np.stack([candidate.to_matrix() for candidate in candidates])
    translations = candidate_matrices[:, :3, 3]
    rotations = candidate_matrices[:, :3, :3]
    mean_translation = np.mean(translations, axis=0)
    mean_rotation = Rotation.from_matrix(rotations).mean().as_matrix()
    base_T_camera = Transform.from_translation_rotation(mean_translation, mean_rotation)

    translation_residuals = np.linalg.norm(translations - mean_translation, axis=1)
    rotation_residuals = Rotation.from_matrix(
        rotations @ mean_rotation.T
    ).magnitude()
    translation_rmse = float(np.sqrt(np.mean(np.square(translation_residuals))))
    rotation_rmse = float(np.sqrt(np.mean(np.square(rotation_residuals))))

    return ExtrinsicCalibrationResult(
        base_T_camera=base_T_camera,
        sample_count=len(candidates),
        translation_rmse_m=translation_rmse,
        rotation_rmse_rad=rotation_rmse,
    )
