# Pick & Place Pipeline — Motion Gen Collision Deep-Dive

*Last updated: 2026-03-19. Based on `develop/collision_world` branch.*

---

## 1. Pipeline Overview

```
Client (planner)                         Server (Isaac Sim / CuRobo)
─────────────────────────────────────    ──────────────────────────────────────────
PickStage.select_pose()                  CuroboMotion.__init__()
  → grasp filter chain                     → load robot_cfg YAML
  → curobo batch-IK check                  → build WorldConfig from stage
  → sort by joint cost                     → warmup motion_gen

PickStage.generate_action_sequence()
  → MOVE → GRASP → PICKUP

  gRPC: move(target, is_right)           UIBuilder.follow_target()
                                           → set_obstacles() (fast pose update)
                                           → curoboMotion.caculate_ik_goal()
                                           → motion_gen.plan_batch()

  gRPC: attach_obj(prim_paths, is_right) UIBuilder.attach_objs()
                                           → _get_ee_pose(is_right, is_local=True)
                                           → curoboMotion.attach_obj(...)
                                           → attach_objects_to_robot(...)

PlaceStage.select_pose()                 (same curobo instance, object now attached)
  → place filter chain
  → curobo batch-IK check

PlaceStage.generate_action_sequence()
  → MOVE → PLACE → PICKUP (reverse)

  gRPC: move(target, is_right)           UIBuilder.follow_target()
                                           → set_obstacles() — attached object
                                           →   now excluded from world obstacles
                                           → plan_batch() with attached spheres live
```

---

## 2. Pick Stage Filter Chain

**File:** `source/data_collection/client/planner/action/grasp.py` → `PickStage.select_pose()`

| Step | Filter | Keeps / Drops |
|------|--------|---------------|
| 1 | Z-percentile | Drop grasps outside `[lower_z_pct, upper_z_pct]` height range |
| 2 | Upright-grasp filter | Drop upside-down grasps (optional, per config) |
| 3 | Random downsample | Subsample to ~300 candidates |
| 4 | Simple Isaac IK | Drop poses where Isaac's fast IK fails |
| 5 | Next-stage viability | Drop if corresponding place target is not reachable |
| 6 | Near-point search | If pool empty, regenerate with offset search |
| 7 | **CuRobo batch-IK** | Final collision-aware IK pass (10 poses per batch, 32 seeds) |
| 8 | Pre-grasp generation | Optionally prepend pre-approach pose |
| 9 | Sort by cost | `joint_pos_dist` + Jacobian score |

### CuRobo IK batch (step 7)

```
CUROBO_BATCH_SIZE = 10
num_ik_seeds = 32
collision_activation_distance = 0.02 m   (env: GENIESIM_CUROBO_COLLISION_ACTIVATION_DISTANCE)
```

IK uses `motion_gen.ik_solver.solve_batch()`. At this point the world model already includes background obstacles but **no attached object** (nothing is grasped yet). The extra collision links `attached_object` / `left_attached_object` are present but their spheres are all at radius `-10.0` (inactivated).

---

## 3. Place Stage Filter Chain

**File:** `source/data_collection/client/planner/action/place.py` → `PlaceStage.select_pose()`

| Step | Filter |
|------|--------|
| 1 | Generate aligned target poses from interaction elements |
| 2 | Upright constraint filter |
| 3 | Sort by distance to passive object center |
| 4 | Simple Isaac IK check |
| 5 | **CuRobo batch-IK** (same call as pick, but object is now attached) |
| 6 | Optional pre-place approach pose |
| 7 | Humanoid posture sort (Galbot only) |
| 8 | Final sort: center-distance × joint cost |

**Critical difference from pick:** by the time `PlaceStage.select_pose()` runs, `attach_obj` has already been called. The CuRobo IK in step 5 therefore includes attached-object collision spheres.

---

## 4. Collision World Setup

### 4.1 Initialization

**File:** `motion_gen_reacher.py` → `CuroboMotion.__init__()` (line ~539)

```
MotionGen.load_from_robot_config(
    collision_checker_type = CollisionCheckerType.MESH
    use_cuda_graph         = True
    collision_cache        = {"obb": 30, "mesh": n_obstacle_mesh}   # default 100
    collision_activation_distance = 0.02
    num_trajopt_seeds = 4
    num_graph_seeds   = 4
    num_ik_seeds      = 32
)
```

### 4.2 Obstacle Extraction

**File:** `motion_gen_reacher.py` → `_extract_cached_obstacles()` (line ~360)

Always-ignored prefixes (hardcoded):
```
/World/target
/World/Xform_01
/World/GroundPlane
/World/Environment_01
/curobo
/World/GroundPlane_01
/World/Meshes
/World/Root/Meshes
/World/Objects/part
/base_cube
virtual_fixed_joint
<robot_prim_path>
```

Background (`/World/Background`, `/World/background`) behaviour depends on env vars:

| Env var | Default | Meaning |
|---------|---------|---------|
| `GENIESIM_CUROBO_INCLUDE_BACKGROUND_OBSTACLES` | `1` (True) | Include background meshes at all |
| `GENIESIM_CUROBO_BACKGROUND_COLLISION_ONLY` | `1` (True) | If included, only keep background prims that have `CollisionAPI` schema |

So **by default, background is included but filtered to collision-enabled prims only.**

If a table or cabinet is in `/World/Background` but its meshes do not have `CollisionAPI` applied, `BACKGROUND_COLLISION_ONLY=1` will silently skip them. Turning it off (`=0`) includes all background meshes regardless.

### 4.3 Per-Frame Pose Update

**File:** `motion_gen_reacher.py` → `set_obstacles()` / `_update_obstacle_poses_fast()` (line ~696)

On every `caculate_ik_goal` call, obstacles are NOT re-extracted from stage (geometry stays cached). Only poses are refreshed:

1. Read `robot_reference_prim` transform → get `r_T_w` (world-to-robot inverse)
2. For each cached prim, read current world pose, apply `r_T_w`
3. Skip if distance > `GENIESIM_CUROBO_MAX_OBSTACLE_DISTANCE` (default 10 m)
4. Skip if path starts with any `attached_objects` entry (object is carried)
5. Load updated WorldConfig into `world_coll_checker`
6. Reset graph planner buffer

### 4.4 Env Knobs Reference

```bash
GENIESIM_CUROBO_INCLUDE_BACKGROUND_OBSTACLES=1   # include /World/Background meshes
GENIESIM_CUROBO_BACKGROUND_COLLISION_ONLY=1      # filter to CollisionAPI-tagged only
GENIESIM_CUROBO_OBSTACLE_DIAGNOSTICS=1           # verbose load/skip logging
GENIESIM_CUROBO_OBSTACLE_DIAGNOSTIC_LIMIT=20     # max sample lines per category
GENIESIM_CUROBO_MESH_CACHE_SIZE=100              # max mesh obstacles in CuRobo cache
GENIESIM_CUROBO_COLLISION_ACTIVATION_DISTANCE=0.02  # clearance margin in metres
GENIESIM_CUROBO_MAX_OBSTACLE_DISTANCE=10         # skip obstacles farther than this
```

---

## 5. Attached-Object Collision Chain

This is the most bug-prone section. Each step below has been a failure point at some point during `develop/collision_world`.

### 5.1 Robot YAML Config (galbot_fixed_right.yml)

```yaml
extra_links:
  attached_object:
    parent_link_name: right_gripper_base_link   # ← see Question 1 below
    link_name: attached_object
    fixed_transform: [0, 0, 0, 1, 0, 0, 0]     # identity: no offset
    joint_type: FIXED
    joint_name: attach_joint
  left_attached_object:
    parent_link_name: left_gripper_base_link
    ...

extra_collision_spheres:
  attached_object: 50
  left_attached_object: 50
```

**Critical:** `attached_object` and `left_attached_object` must also be in `collision_link_names`. If they are absent, CuRobo allocates 0 spheres → every attach call fails silently with `max_spheres=0`. The runtime patch in `_ensure_attached_collision_config()` forces them in at startup.

### 5.2 Attach Call Flow

```
gRPC: AttachObj(obj_prims, is_right)
  → CommandController.handle_attach_obj()
      collect all UsdGeom.Mesh paths under obj_prims
      → UIBuilder.attach_objs(mesh_paths, is_right)
          link_name = "attached_object" (right) or "left_attached_object" (left)
          position, rotation = _get_ee_pose(is_right, is_local=True)
              # Returns TCP link pose in robot-local frame
              # Uses kinematics_solver[key]._articulation_kinematics_solver
          → CuroboMotion.attach_obj(mesh_paths, link_name, position, rotation)
              detach_object_from_robot("left_attached_object")
              detach_object_from_robot("attached_object")
              clear attached_objects list
              set_obstacles()   ← refresh world before attach
              build ee_pose = Pose(position, rotation)
              → attach_objects_to_robot(...)
                  ee_pose_inv = ee_pose.inverse()
                  for each mesh path:
                      find obstacle in world_cfg / world_model / world_coll_checker
                      get_bounding_spheres(n, pre_transform=ee_pose_inv)
                      disable_obstacle(mesh_path)   ← remove from world
                      add mesh_path to attached_objects list
                  attach_spheres_to_robot(sphere_tensor, link_name)
```

### 5.3 Known Frame Mismatch Bug (partially fixed, may recur)

Galbot config:
- CuRobo kinematic EE: `right_gripper_tcp_link`
- `attached_object` parent: `right_gripper_base_link`
- `_get_ee_pose` → returns **TCP** pose (not gripper_base_link)

When attach runs, `ee_pose_inv` (TCP-frame inverse) is used to transform object sphere positions into the local frame of what CuRobo thinks is the `attached_object` origin — which is `gripper_base_link`.

If TCP offset ≠ 0 relative to `gripper_base_link`, the attached spheres ride in the wrong position relative to the physical gripper during transport.

**See Question 1 below for the fix recommendation.**

### 5.4 Transport Planning with Attached Object

During `PlaceStage` motion planning:
1. `set_obstacles()` skips any prim whose path is in `self.attached_objects`
2. `attached_object` link has live spheres from step 5.2
3. `plan_batch()` treats the arm + those spheres as a single collision body

**What can still go wrong:**
- The pose-source mismatch (5.3) makes spheres offset → object clips geometry the arm avoids
- `sort_by_difference_js` picks the lowest joint-change path, not the highest-clearance path
- If attach failed silently, planning succeeds but the carried object is invisible to the planner

---

## 6. `view_debug_world` — What It Actually Shows

**File:** `motion_gen_reacher.py:1043`

```python
def view_debug_world(self):
    if self.debug:
        self.visualize_robot_spheres()   # current robot link spheres
        self.visualize_obstacles()       # obstacles from self.world_cfg
```

`visualize_obstacles()` iterates `self.world_cfg.objects` and converts each to bounding spheres (200 per obstacle).

**`self.world_cfg` is assigned here:**

```python
# set_obstacles() → _update_obstacle_poses_fast()
obstacle = world_config.get_collision_check_world()
self.world_cfg = obstacle   # NOTE: this world_config only affects visualization
self.motion_gen.world_coll_checker.load_collision_model(obstacle, ...)
```

So `view_debug_world` shows exactly what went into the last `load_collision_model` call — i.e., the snapshot from the last `set_obstacles()`.

---

## 7. Question Answers

### Q1: Should `parent_link_name` be `xxx_tcp_link` instead of `xxx_gripper_base_link`?

**Short answer: YES, change it to `right_gripper_tcp_link` / `left_gripper_tcp_link`.**

**Why:**

The attach frame pose passed into `attach_obj()` comes from `_get_ee_pose()`, which calls Isaac's `_articulation_kinematics_solver.compute_end_effector_pose()`. Isaac's kinematics solver uses `ee_link: right_gripper_tcp_link` (from the YAML). So the reported pose is the **TCP** link pose.

Inside `attach_objects_to_robot`, this pose is inverted and used as `pre_transform_pose` when calling `obs.get_bounding_spheres(...)`. That transforms each mesh vertex from world space into the "attach-link-local" space before fitting spheres.

CuRobo then fixes those spheres rigidly to the `attached_object` extra_link. That link's kinematic origin is `parent_link_name`.

If `parent_link_name = right_gripper_base_link` but the pose passed was computed for TCP, there is an offset between the two that is silently baked into every sphere position. During motion, the spheres move as if anchored at TCP offset but are actually displaced by (TCP - base_link).

Concretely for Galbot: the TCP is typically ~5–10 cm beyond the gripper base along the approach axis. The attached spheres will be shifted that distance forward relative to the physical object.

**Two equivalent fixes:**
1. Change `parent_link_name` to `right_gripper_tcp_link` in the YAML. Zero offset between what `_get_ee_pose` reports and where CuRobo anchors the spheres.
2. Keep `right_gripper_base_link` but change `_get_ee_pose` to return the `gripper_base_link` pose instead of the TCP pose. Less preferred because it touches kinematics solver setup.

**Option 1 is the clean fix** — the extra link's origin should match the coordinate frame used to compute `ee_pose` in `attach_objs`.

After making the change, confirm with logs:
```
Attach pose resolved: ... parent_link=right_gripper_tcp_link ...
attached_sphere_count=47
```
Then use `view_debug_world` (requires `debug=True`) to visually confirm the red spheres sit around the object, not offset forward.

---

### Q2: `view_debug_world` shows spheres only on some objects; table/cabinet missing — is that the cause?

**Almost certainly yes — but the root cause matters for the fix.**

What `view_debug_world` shows is exactly what `world_coll_checker` loaded at the last `set_obstacles()`. If table/cabinet are absent from the sphere visualization:

#### Scenario A: Background prims were never extracted into the cache

Symptoms: obstacle count is low from the start, and `ObstacleDiag:initial_cache` log shows 0 loaded_mesh for background paths.

Root causes:
- `GENIESIM_CUROBO_INCLUDE_BACKGROUND_OBSTACLES=0` or omitted and default changed
- Table/cabinet prim paths don't start with `/World/background` or `/World/Background` (note case) — check exact paths

Diagnosis:
```bash
# In run env, check:
echo $GENIESIM_CUROBO_INCLUDE_BACKGROUND_OBSTACLES
```
Then search for the prim prefix:
```
ObstacleDiag:initial_cache loaded_mesh: [...]
```

#### Scenario B: Prims are in `/World/Background` but have no `CollisionAPI`

Symptoms: `BACKGROUND_COLLISION_ONLY=1` (default), many `skip_background_no_collision` entries in log.

Fix: set `GENIESIM_CUROBO_BACKGROUND_COLLISION_ONLY=0` to include all background meshes regardless of CollisionAPI tag. This may slow planning.

Alternatively, apply `CollisionAPI` to the table/cabinet in the USD scene.

#### Scenario C: Prims were extracted but exceeded `n_obstacle_mesh` cache limit

Default is 100. If there are many other scene meshes loaded first, table/cabinet indices exceed the CuRobo mesh cache and are silently dropped.

Symptom: loaded_mesh count reaches exactly 100 in the log, and table/cabinet appear in `skip_mesh_extract_none` or simply never show in `updated samples`.

Fix: increase `GENIESIM_CUROBO_MESH_CACHE_SIZE=200` (or higher).

#### Scenario D: Distance filter skips them

The `max_obstacle_distance=10.0 m` filter computes distance from robot origin to the *mesh's world-space origin*. If a large table's origin is far even though its surface is close, it gets skipped.

Check logs for `skip_distance samples: [/World/Background/table ...]`.

Fix: increase `GENIESIM_CUROBO_MAX_OBSTACLE_DISTANCE=15`.

#### Summary table

| What you see in logs | Root cause | Fix |
|----------------------|-----------|-----|
| `skip_background_no_collision` hits for table prim | No CollisionAPI on mesh | Set `BACKGROUND_COLLISION_ONLY=0`, or add CollisionAPI in scene |
| loaded_mesh = 100 exactly, table absent | Cache size too small | Increase `MESH_CACHE_SIZE` |
| `skip_distance` for table | Mesh origin > 10 m from robot | Increase `MAX_OBSTACLE_DISTANCE` |
| Table in `skip_ignore_substring` | Prim path not matching expected prefixes | Check actual prim path prefix |
| Table present in debug spheres but arm still collides | Path ranking issue (joint-cost not clearance) | See §8 below |

---

## 8. Remaining Known Issue: Path Ranking

Even with world obstacles correct and attach working, the planner still picks **lowest joint-change path among all successful candidates**, not the highest-clearance one.

```python
# motion_gen_reacher.py:1273
sorted_indices = sort_by_difference_js(
    filtered_paths,
    weights=self.tensor_args.to_device(dof_weights),
)
self.cmd_plan = paths[sorted_indices[0]]
```

A low joint-change path can still thread through a gap that barely clears for the arm but not for the carried object (whose attached spheres may be slightly larger than expected). The fix would be to add a clearance metric as a tiebreaker or primary sort criterion, but that requires custom trajectory evaluation.

---

## 9. Quick Checklist for Place-Stage Collision Debugging

1. **Logs show `attach_result=True`?**
   - No → check `max_spheres` in init log, check `collision_link_names` for `attached_object`
   - Yes, but `attached_sphere_count` is very low → `extra_collision_spheres` value too small

2. **`ObstacleDiag:update_pose` shows `skip_attached` > 0?**
   - Yes → object is correctly excluded from world during transport (good)
   - No → object was never in `attached_objects` list (attach failed earlier)

3. **Table/cabinet visible in `view_debug_world`?**
   - No → see Q2 scenarios above

4. **Collision happens but spheres look correct in debug view?**
   - Likely path-ranking issue (§8) or attach-frame offset (Q1)

5. **Attach spheres visually offset from physical object?**
   - Parent link mismatch — fix `parent_link_name` per Q1
