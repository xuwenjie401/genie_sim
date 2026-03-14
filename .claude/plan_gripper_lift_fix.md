# Plan: Fix Post-Grasp Lift Motion Planning Failure

## Context

After the galbot gripper closes on an object, the next motion plan (lift) fails with:
```
Start or End state in collision
plan did not converge to a solution: Invalid Problem
```

The root cause is a **stale cache bug**: `remove_objects_from_world` correctly disables the grasped object in curobo's collision checker and removes it from the world model — but it does NOT remove the object from `CuroboMotion.cached_obstacle_info`. Then, the very first thing `calculate_ik_goal()` does is call `set_obstacles()`, which iterates `cached_obstacle_info` and **reloads** the removed obstacle back into the collision model via `load_collision_model()`. The planner then finds the robot in collision with an object it should no longer see.

### Execution trace

1. `post_close_settle` phase (`free_worker.py:2895`):
   - `ui_builder.remove_objects_from_world(["/World/obstacle1"])`
   - → `CuroboMotion.remove_objects_from_world` disables + removes from `world_model`
   - **BUT** `/World/obstacle1` stays in `CuroboMotion.cached_obstacle_info`
2. `move_left(lift_pose)` → `_hand_moveto` → `_follow_target` → `calculate_ik_goal()`
3. `calculate_ik_goal` calls `set_obstacles()` **first thing** (line 850)
4. `_update_obstacle_poses_fast()` iterates `cached_obstacle_info`, finds `/World/obstacle1`, adds it to updated world config
5. `load_collision_model(...)` reloads obstacle into GPU collision checker
6. `plan_batch(...)` sees start state in collision → "Invalid Problem"

## Files to Modify

- **`unit_lab/free_worker.py`** (single location: `CuroboMotion.remove_objects_from_world`, line ~1039)

## Change

In `CuroboMotion.remove_objects_from_world` (line 1039–1045), also evict the prim path from `CuroboMotion.cached_obstacle_info`:

```python
# BEFORE
def remove_objects_from_world(self, prim_paths):
    for x in prim_paths:
        obs = self.motion_gen.world_model.get_obstacle(x)
        if not obs:
            continue
        self.motion_gen.world_coll_checker.enable_obstacle(enable=False, name=x)
        self.motion_gen.world_model.remove_obstacle(x)

# AFTER
def remove_objects_from_world(self, prim_paths):
    for x in prim_paths:
        obs = self.motion_gen.world_model.get_obstacle(x)
        if not obs:
            continue
        self.motion_gen.world_coll_checker.enable_obstacle(enable=False, name=x)
        self.motion_gen.world_model.remove_obstacle(x)
        CuroboMotion.cached_obstacle_info.pop(x, None)  # prevent re-add by set_obstacles()
```

## Why this is safe

- `cached_obstacle_info` is a class-level dict. Removing the key prevents future `set_obstacles()` calls from re-inserting the object.
- The physical object still exists in USD/Isaac physics — only curobo's collision representation is removed.
- If the object needs to be re-considered for collision later (e.g. next episode), it will be re-added on the next world snapshot/reset via `_extract_cached_obstacles`.

## Verification

Run `unit_lab/free_worker.py` and observe:
1. Lift motion plan should succeed without "Start or End state in collision"
2. Robot should lift the held object
3. Place motion should also succeed
