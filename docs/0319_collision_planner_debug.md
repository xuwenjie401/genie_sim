# 2026-03-19 Collision Planner Debug Note

## Scope

This note explains the motion-planning patch that was applied for the collision issue investigation, why it was done, which files changed, and what the follow-up logs showed.

The first patch was about obstacle loading.

The second patch in the same investigation pass is about attached-object debugging, because the new logs showed that transport is often planned without the grasped object attached to the robot collision model.

## Why the obstacle-loading patch was needed

The initial symptom was:

- planning reported success
- execution still hit large static furniture like the table and shelf

Static inspection showed the main reason:

- `/World/Background` was excluded from obstacle extraction in `source/data_collection/server/motion_generator/motion_gen_reacher.py`
- the planner therefore relied mostly on proxy colliders and whatever non-background meshes were left
- that design likely existed to keep curobo tractable, because loading the full background as meshes can slow planning or make convergence worse

So the goal of the first patch was not "load everything blindly". The goal was:

- include relevant background obstacles again
- keep an escape hatch if performance regresses
- make the obstacle-loading decision visible in logs

## What changed in the first patch

### 1. Background obstacle inclusion became configurable

Before:

- `/World/background`
- `/World/Background`

were always in the ignore list.

After:

- background meshes are included by default
- background meshes can still be filtered so that only collision-enabled-looking background meshes are kept
- both behaviors are now controlled by environment variables

Relevant file:

- `source/data_collection/server/motion_generator/motion_gen_reacher.py`

Relevant knobs:

```bash
GENIESIM_CUROBO_INCLUDE_BACKGROUND_OBSTACLES=1
GENIESIM_CUROBO_BACKGROUND_COLLISION_ONLY=1
```

### 2. Obstacle diagnostics were added

The planner now logs:

- how many meshes were loaded
- how many were skipped by ignore rules
- how many were skipped because collision was explicitly disabled
- how many background meshes were skipped because they did not look collision-enabled
- sample names for loaded and skipped meshes

This was added in the custom `AgibotUsdHelper.get_obstacles_from_stage(...)` path.

The goal was to make the extraction process observable instead of guessing.

### 3. Mesh cache size was increased

Before:

```python
n_obstacle_mesh = 30
```

After:

```python
n_obstacle_mesh = _env_int("GENIESIM_CUROBO_MESH_CACHE_SIZE", 100)
```

Reason:

- once background meshes are allowed back in, the old cache size becomes a likely bottleneck

### 4. Collision activation distance was increased

Before:

```python
collision_activation_distance = 0.01
```

After:

```python
collision_activation_distance = _env_float(
    "GENIESIM_CUROBO_COLLISION_ACTIVATION_DISTANCE",
    0.02,
)
```

Reason:

- 1 cm clearance was too aggressive for this scene
- 2 cm is still modest, but less fragile

### 5. Max obstacle distance became configurable

The old `10.0m` cutoff stayed as the default, but it is no longer hard-coded.

```bash
GENIESIM_CUROBO_MAX_OBSTACLE_DISTANCE=10
```

Reason:

- `3m` looked tempting, but it is risky when using object-origin distance instead of true nearest-point distance

## Files touched in the first patch

- `source/data_collection/server/motion_generator/motion_gen_reacher.py`

## What the new logs proved

The new logs were useful for one thing very clearly:

- background obstacles are being loaded now

Examples from the logs:

- `/World/Background/benchmark_table_017/body/collider_body`
- many other `/World/Background/...` meshes

So the old "background is invisible to the planner" diagnosis is no longer the active blocker in these runs.

However, the logs exposed a second issue:

- `attach_result=False` on every failing transport attempt

That means the grasped object was not attached into curobo's robot collision representation.

When that happens:

- the arm path may still be collision-free
- the carried object is effectively ignored during transport planning
- the object can hit the scene first
- force then transfers into the gripper
- a fragile gripper fails easily

## Follow-up diagnosis from the logs

The important evidence is:

- `attach_result=False`
- `skip_attached=0`
- the target object mesh names appear in the curobo obstacle reload warnings

That combination suggests:

- the target mesh is present in the obstacle-loading phase
- but the attach step still fails when trying to build attached spheres from the world model

The attach path can only fail in a few places:

- no mesh names were provided
- the attached-link sphere budget is zero
- no attach spheres were generated for the requested mesh names

Given the current robot config, the likely real failure is:

- attach lookup cannot find the requested mesh names in the world model seen by the attach function

Update after checking `logs/0319_collision_4.log`:

- the actual failure in this run is earlier than mesh lookup
- `max_spheres=0` for `attached_object`

That means curobo created the extra attached link, but allocated zero collision spheres for it.

The concrete config bug is:

- Galbot defines `extra_links` for `attached_object` and `left_attached_object`
- Galbot also defines `extra_collision_spheres`
- but those attached links were missing from `collision_link_names`

Curobo only allocates per-link sphere storage for links present in `collision_link_names`.

So the result was:

- attached link exists
- attached link sphere count is zero
- attach request fails immediately before any object-mesh lookup

## What changed in the follow-up attach patch

### 1. Attach now resets both attached links and clears stale attached state

Reason:

- the old `attach_obj(...)` only detached the default attached link
- that is unsafe for dual-arm robot configs
- stale `attached_objects` state can also hide meshes from the next obstacle refresh

### 2. Attach now logs exact failure reasons

New logs include:

- link name
- object count
- attached-sphere budget
- per-object sphere counts
- which world model source provided each object
- missing object candidates when lookup fails

### 3. Attach lookup no longer trusts only one world-model reference

It now checks, in order:

- `self.world_cfg`
- `motion_gen.world_model`
- `world_coll_checker.world_model`

Reason:

- the collision world used by planning can be correct while the attach lookup reads a different or stale view

### 4. Controller attach state is no longer updated on attach failure

Before:

- `command_controller.handle_attach_obj()` updated `attach_states` even when curobo attach returned `False`

After:

- `attach_states` only updates when attach actually succeeds

## Update after checking `logs/0319_collision_5.log`

This run is different from `0319_collision_4.log`.

The key evidence is:

- `attach_result=True`
- `attached_sphere_count=47` and later `50`
- place/transport plans report `success`
- visually, the arm seems to route around the shelf, but the carried object still clips it

So the first-order problem is no longer "attach failed".

The most likely remaining bug is a frame mismatch in the attach pose source.

### Why that mismatch is plausible

Galbot is configured with:

- kinematic end effector: `right_gripper_tcp_link` / `left_gripper_tcp_link`
- attached collision link parent: `right_gripper_base_link` / `left_gripper_base_link`

Before the patch, `ui_builder.attach_objs()` always passed the solver's end-effector pose into `attach_obj(...)`.

For Galbot, that means:

- the pose source came from the TCP link
- but attached spheres were later registered onto `attached_object`
- and that extra link is anchored at the gripper base link, not the TCP

If the TCP-to-base offset is not zero, the attached collision proxy will move with the robot but in the wrong relative pose.

That matches the observed symptom:

- the arm appears to avoid the shelf
- the carried object still collides in a way that feels "strange"

## Follow-up patch for `0319_collision_5.log`

The attach path now resolves the pose from the configured curobo extra-link definition instead of blindly trusting the UI request pose.

Concrete change in `motion_gen_reacher.py`:

- read `kinematics.extra_links[link_name].parent_link_name`
- read the current parent-link pose from the USD stage
- convert that pose into robot-local coordinates
- compose it with the configured `fixed_transform`
- use that resolved pose as the attach frame for sphere fitting
- log the final attach pose source and resolved transform

This makes the attached-object sphere frame consistent with the link that curobo actually uses for `attach_spheres_to_robot(...)`.

New logs to watch for:

- `Attach pose resolved: ... parent_link=right_gripper_base_link ...`
- `Attach pose source: ... source=config_parent_link:...`

If those appear and the object still clips, the next likely issue is no longer attach-frame mismatch. At that point the next candidate is path ranking: successful paths are still selected mainly by joint-motion change, not by maximum clearance.

## Update after checking `logs/0319_exit.log`

This log shows a different regression:

- both runs reach `Attach!!!!`
- the process dies immediately after the attach request starts
- there is no Python traceback
- the app falls through the main loop `finally:` path and closes the simulation app

That pattern is more consistent with a native crash or hard abort during the new attach-pose resolution path than with a normal Python exception.

The timing is especially suspicious because the crash happens between:

- `Fast obstacle update completed`
- the next attach-detail log

The risky code introduced in that window was the new motion-generator-side parent-link pose resolver using `get_prim_world_pose(...)` and `Pose.from_matrix(...)`.

To de-risk that path, the follow-up change moves attach-pose resolution back into `ui_builder.py` using plain numpy pose math:

- read the configured `parent_link_name`
- query that link pose through `XFormPrim(...).get_world_pose()`
- convert it into robot-local coordinates in numpy
- compose the configured `fixed_transform`
- pass that resolved pose into `curoboMotion.attach_obj(...)`

At the same time, `motion_gen_reacher.attach_obj(...)` was simplified again to trust the caller-provided pose and avoid doing the extra parent-link pose reconstruction internally.

This keeps the frame fix while removing the new native-crash suspect from the attach hot path.

This matters because the old behavior could make later state playback or debugging look as if the object was attached when it was not.

### 5. Attached links are now forced into collision-link allocation

The runtime config now ensures:

- `attached_object` and `left_attached_object` are present in `collision_link_names`
- both links have `extra_collision_spheres`
- both links get `self_collision_buffer` entries
- both links get symmetric self-collision-ignore entries with same-side gripper links

Reason:

- without inclusion in `collision_link_names`, curobo allocates zero spheres for the attached link
- that is exactly what `logs/0319_collision_4.log` showed

## Files touched in the follow-up attach patch

- `source/data_collection/server/motion_generator/motion_gen_reacher.py`
- `source/data_collection/server/command_controller.py`

## Current code-selection behavior

The planner currently does **not** rank successful trajectories by clearance.

In `source/data_collection/server/motion_generator/motion_gen_reacher.py`, successful paths are filtered by pose error, then ranked using `sort_by_difference_js(...)`.

That means the final choice is biased toward:

- smaller cumulative joint change

not toward:

- larger obstacle clearance
- safer carried-object sweep

So even after the attach issue is fixed, path choice is still "lowest joint-change among successful solutions", not "safest successful solution".

## Practical next step after this patch

Run the same task again and inspect the new attach logs.

The key questions for the next run are:

- does initialization now report nonzero sphere counts for `attached_object` and `left_attached_object`
- does attach now succeed
- if it still fails after nonzero sphere allocation, which world-model source was missing the requested mesh
- if it succeeds, does `skip_attached` become non-zero on later obstacle refreshes
- after attach succeeds, are collisions still coming from path ranking rather than missing attached geometry

## Useful environment knobs

```bash
GENIESIM_CUROBO_INCLUDE_BACKGROUND_OBSTACLES=1
GENIESIM_CUROBO_BACKGROUND_COLLISION_ONLY=1
GENIESIM_CUROBO_OBSTACLE_DIAGNOSTICS=1
GENIESIM_CUROBO_OBSTACLE_DIAGNOSTIC_LIMIT=20
GENIESIM_CUROBO_MESH_CACHE_SIZE=100
GENIESIM_CUROBO_COLLISION_ACTIVATION_DISTANCE=0.02
GENIESIM_CUROBO_MAX_OBSTACLE_DISTANCE=10
```

## Summary

The first patch fixed the world-obstacle visibility problem by reintroducing background obstacles in a controlled, diagnosable way.

The new logs then showed the next real blocker:

- transport planning is often missing the grasped object because attach is failing

The follow-up patch therefore focuses on:

- exact attach failure diagnosis
- safer attach-state handling
- avoiding false "attached" bookkeeping
