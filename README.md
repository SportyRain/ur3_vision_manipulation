# UR3 Vision-Guided Manipulation

Gate 1 implements a ROS-independent robotics geometry core for rigid 3D poses
and transforms.

## Coordinate convention

- Quaternions use `[x, y, z, w]` ordering.
- `T_parent_child` maps child-frame coordinates into the parent frame.
- `p_parent = T_parent_child @ p_child`
- `base_T_object = base_T_camera @ camera_T_object`

Only proper rigid transforms in SE(3) are accepted; scale, reflection, and
non-finite values are rejected.

## Tests

Run the deterministic Gate 1 tests with the project virtual environment:

```powershell
.venv\Scripts\python.exe -m pytest -q
```
