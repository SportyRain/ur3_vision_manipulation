# Scene Object → Pick Target Software Gate — 2026-08-31

## Scope

```text
MILESTONE = SCENE_OBJECT_TO_PICK_TARGET_SOFTWARE_INTEGRATION
FIRST_UNRESOLVED_BOUNDARY = metric SceneObject → existing pick/place target geometry
PHYSICAL_ACTION = NO
```

## Reuse path

```text
ur3_visual_servoing SceneObject.object_id + base_xyz_m
→ ur3_vision_manipulation existing Transform
→ existing object_T_tool composition
→ ScenePickPlaceTargets
```

The scene observer provides stable identity and robot-base XYZ but does not provide a verified object orientation. The integration therefore requires the caller to supply `base_R_object` explicitly and fails closed when metric XYZ is unavailable. No object orientation is invented.

Existing ArUco `MarkerPoseInBase → PickPlaceTargets` behavior remains unchanged and reuses the same private composition helper.

## Focused evidence

Local source-level focused test against the branch implementation:

```text
new scene-object target tests = 4 passed
existing marker target regression + new tests = 7 passed
```

Covered contracts:

- metric XYZ + explicit valid orientation composes the existing pick/place geometry
- stable `object_id` is preserved
- missing `base_xyz_m` fails closed
- empty object identity fails closed
- invalid rotation fails closed
- existing marker pick target composition remains unchanged
- existing marker place target composition remains unchanged
- existing marker ID preservation remains unchanged

## Status

```text
SCENE_OBJECT_TO_TARGET_ADAPTER = SOFTWARE_VERIFIED
SCENE_OBJECT_6D_ORIENTATION_SOURCE = NOT_VERIFIED / REQUIRED INPUT
REAL_GRASP_POSE = NOT_VERIFIED
REAL_PICK_PLACE_OFFSETS = NOT_VERIFIED
GRIPPER_ACTUATION_RUNTIME = NOT_VERIFIED
REAL_PICK_PLACE_RUNTIME = NOT_VERIFIED
```

This gate does not claim real Pick & Place readiness. The next unresolved boundary after this software adapter is a verified source of object/grasp orientation and grasp geometry for the actual target object/tool.

## Record sync

```text
HISTORY_RECORD_REQUIRED = YES
CURRENT_STATE_UPDATE_REQUIRED = YES
PROGRAM_REGISTRY_UPDATE_REQUIRED = YES
RUNTIME_COMMAND_UPDATE_REQUIRED = NO
RECORD_SYNC_CHECK = PENDING_CENTRAL_REPO_SYNC
```
