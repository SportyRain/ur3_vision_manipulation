"""Run deterministic synthetic fixed-camera extrinsic verification."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ur3_vision_manipulation.geometry import Transform
from ur3_vision_manipulation.perception import estimate_base_T_camera


def _transform(translation, rotation_vector) -> Transform:
    return Transform.from_translation_rotation(
        translation,
        Rotation.from_rotvec(rotation_vector).as_matrix(),
    )


def main() -> None:
    known = _transform([0.42, -0.18, 0.73], [0.25, -0.15, 0.32])
    camera_references = [
        _transform([0.10, 0.02, 0.55], [0.10, -0.05, 0.02]),
        _transform([-0.08, 0.12, 0.68], [-0.12, 0.18, -0.04]),
        _transform([0.04, -0.09, 0.61], [0.07, 0.11, 0.16]),
        _transform([0.15, 0.06, 0.72], [-0.08, -0.14, 0.09]),
    ]
    base_references = [known @ reference for reference in camera_references]
    result = estimate_base_T_camera(base_references, camera_references)
    known_matrix = known.to_matrix()
    estimated_matrix = result.base_T_camera.to_matrix()
    translation_error = np.linalg.norm(
        estimated_matrix[:3, 3] - known_matrix[:3, 3]
    )
    rotation_error = Rotation.from_matrix(
        estimated_matrix[:3, :3] @ known_matrix[:3, :3].T
    ).magnitude()

    print("SYNTHETIC EXTRINSIC")
    print("SYNTHETIC ONLY")
    print(f"known translation: {known_matrix[:3, 3].tolist()}")
    print(f"estimated translation: {estimated_matrix[:3, 3].tolist()}")
    print(f"translation error (m): {translation_error:.12g}")
    print(f"rotation error (rad): {rotation_error:.12g}")
    print(f"translation RMSE (m): {result.translation_rmse_m:.12g}")
    print(f"rotation RMSE (rad): {result.rotation_rmse_rad:.12g}")


if __name__ == "__main__":
    main()
