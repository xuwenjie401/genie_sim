# Done
1. `free_worker_gripper.py` - gripper keyboard-interactive unit test, done.
2. `free_worker_vertical.py` - vertical torso height adjustment unit test.
   - Physical part works: torso lowers visually when cube is placed low.
   - **Option A (Z-compensation) implemented** — pending verification run.

---

# Current Status: free_worker_vertical.py — Option A v2, needs run

## What Was Implemented (Option A v2: Z-compensation in handle_linear_move)
Three changes vs v1:

1. **`GALBOT_REACH_SAFETY_MARGIN` increased from 0.05 → 0.10** (line ~223):
   - Torso now lowers 5cm more than strictly needed.
   - The IK target in curobo's model is 10cm above the sphere-model boundary instead of 5cm.
   - This gives curobo more comfortable workspace clearance → higher success rate.

2. **Z-compensation moved from `manual_set_command` to `handle_linear_move`** (line ~2129):
   - Compensation is now applied INSIDE the physics thread, AFTER the leg trajectory completes.
   - Uses `self._get_torso_world_z()` to read the ACTUAL current torso height (not predicted).
   - This is more accurate: no sphere-model prediction errors, no timing issues.
   - `target_position = self.data["target_position"].copy(); target_position[2] += dz`

3. **Z-compensation removed from `manual_set_command`**:
   - The uncompensated `ee_translation_goal` (raw cube-in-robot-frame) is stored in data.
   - Compensation happens fresh every physics step based on actual torso position.

4. **Better diagnostic prints**:
   - `[Vertical] follow_cube: cube_world_z=X, ee_goal_z=X (in robot frame, uncompensated)`
   - `[Vertical] handle_linear_move Z-comp: torso=X (default=X), dz=X, curobo_goal_z=X`

## Why v1 May Have Failed
- Compensation was computed from predicted (sphere-model) dz before trajectory ran.
- If the prediction was inaccurate or `_compute_leg_targets_for_height` returned None,
  compensation wasn't applied.
- By computing in `handle_linear_move` after trajectory, we use actual ground-truth torso_z.

## If IK Still Fails After v2
Check these in console:
1. `[Vertical] handle_linear_move Z-comp: dz=X` — is dz > 0? Is curobo_goal_z reasonable?
2. `plan did not converge to a solution: ...` from curobo — what status?
3. If dz=0 when it shouldn't be: check `default_torso_z` is set (printed at init)

Next options if v2 fails:
- Try adding `"disable_collision_links": ["torso_base_link"]` to follow_cube call
- Try clamping cube orientation to identity for IK: `ee_orientation_goal = [1,0,0,0]`
- Further increase `GALBOT_REACH_SAFETY_MARGIN` to 0.15 (more lowering, higher IK target)
- Increase `GALBOT_ARM_REACH_RADIUS` to check if sphere model is too pessimistic

---

# Architecture Reference

## Key Files
- `unit_lab/free_worker_cube.py` — reference (unmodified). Read this first for patterns.
- `unit_lab/free_worker_vertical.py` — current work file.
- `unit_lab/configs/basic_test.yaml` — curobo config: retract_config, lock_joints dict.
- `unit_lab/calibrate_leg_height.py` — calibration script (can re-run to re-verify).
- `source/data_collection/config/robot_cfg/galbot_fixed_dual.json` — robot config.
- `source/data_collection/server/ui_builder.py` — UIBuilder: `_follow_target`, `set_locked_joint_positions`.
- `source/data_collection/server/motion_generator/motion_gen_reacher.py` — CuroboMotion: `update_lock_joints`, `calculate_ik_goal`.

## Thread Model
- Physics thread: `on_physics_step()` → `_step_leg_trajectory()` → curobo steps → `on_command_step()` → `handle_linear_move()`
- Task thread: `TaskManager.run()` every 500 steps → `manual_set_command("follow_cube")` → `blocking_start_server()` → blocks on condition
- Unblock: physics thread sets `data_to_send` → `condition.notify_all()`

## Curobo Lock Joints Flow (Option A)
1. UIBuilder `set_locked_joint_positions(is_right)` reads articulation, but OVERRIDES leg joints
   with `self.leg_joint_defaults` — always DEFAULT values, never physically lowered values.
2. Calls `curoboMotion.update_lock_joints(ids)` with default-valued leg joints.
3. IK goal Z is compensated in `handle_linear_move` after trajectory completes.

## Vertical Adjustment Constants (calibrated 2026-03-11)
- `GALBOT_HEIGHT_PER_LEG_DELTA = 0.67` m/rad — calibrated via `calibrate_leg_height.py`
- `GALBOT_MAX_LEG_DELTA = 0.4` rad — safe tested range
- `GALBOT_REACH_SAFETY_MARGIN = 0.10` m — sphere model conservative margin (v2: increased)
- `GALBOT_LEG_TRAJ_STEPS = 60` — ~1s at 60Hz for gradual leg movement
- Constraint: `joint1 + joint3 ≈ joint2` → parameterize: j1+=delta, j2+=2*delta, j3+=delta
- Sign: **NEGATIVE delta lowers torso** (decreasing joint angles squats down)
- Baseline torso_z at default joints: ~0.9174 m

## Vertical Adjustment Key Methods (all in CommandController in free_worker_vertical.py)
- `_init_vertical_defaults()` — captures default leg joints + default_torso_z after init;
  also sets `ui_builder.leg_joint_defaults` for Option A.
- `_get_torso_world_z()` — XFormPrim("/galbot_one_golf/torso_base_link").get_world_pose()[0][2]
- `_compute_leg_targets_for_height(target_z)` — sphere check, returns joint targets or None
- `_start_leg_trajectory(targets, n_steps=60)` — linspace from current to target over n steps
- `_step_leg_trajectory()` — call each physics step from `on_physics_step`; returns True when done
- `_is_leg_moving()` — True while trajectory active
- `leg_just_finished` flag — set in `_step_leg_trajectory` when done; triggers force-refresh in `handle_linear_move`

## Config Rule
Do NOT modify original config files. Create new files or add config in code with NOTE comments.
