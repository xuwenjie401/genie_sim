# Current Pick & Place Pipeline Audit For CuRobo Place Collisions

Date: 2026-03-19

Scope:
- This note is based on the current workspace code state.
- Focus is the runtime pick/place pipeline that ends in `curobo MotionGen`, especially the place stage with an attached object.
- The main question is why some place trajectories still execute with visible collision.

## Executive Summary

The current workspace has two high-probability suspicious areas for your Galbot place-collision issue:

1. The attached-object collision frame for Galbot looks inconsistent.
   - CuRobo plans with `ee_link = *_gripper_tcp_link` in the Galbot YAMLs.
   - But the attached extra link is still defined as a child of `*_gripper_base_link` with identity transform.
   - The runtime attach path currently uses the solver end-effector pose from `_get_ee_pose(...)`, which is the TCP pose, to generate attached-object spheres.
   - If spheres are generated in TCP frame but interpreted as `gripper_base` frame, attached-object collision will be spatially wrong.

2. Missing background obstacles in `view_debug_world()` can absolutely produce collision trajectories.
   - `view_debug_world()` visualizes CuRobo's filtered obstacle world, not the full USD scene.
   - If the table or cabinet are absent there, CuRobo is likely not planning against them.
   - The current obstacle extraction has several filters: only meshes, optional background filtering, background collision-schema filtering, and max-distance filtering.

There is also a third important point:

3. In the current checked-in place execution path, the passive place target is not automatically removed from the obstacle world.
   - Pick has a `KeepClose` obstacle-removal path.
   - Place has only a commented-out TODO.
   - So current place planning is trying to respect the receptacle/container as an obstacle unless some other custom logic removes it.

## End-To-End Runtime Pipeline

### 1. Place candidate generation

Place pose generation starts in [place.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/planner/action/place.py#L95).

Current flow:

- Read current object pose and current gripper pose, then compute `gripper2obj` from the live grasp relation.
- Generate candidate object poses from aligned active/passive placement elements.
- Convert candidate object poses into candidate gripper poses.
- Optionally filter upside-down poses.
- Sort by receptacle center proximity, then truncate to `center_sort_num` candidates.
- Run Isaac Sim "Simple" IK first.
- Run CuRobo `AvoidObs` IK second.
- Optionally validate `pre_insert` poses if `use_pre_place` is enabled.
- Rank survivors:
  - G2 uses a humanlike ranking.
  - Other robots, including Galbot, use joint-distance ranking plus a final center-based re-rank.

Important code points:

- Simple IK filter: [place.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/planner/action/place.py#L154)
- AvoidObs IK filter: [place.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/planner/action/place.py#L165)
- Fallback when no place pose survives: [place.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/planner/action/place.py#L224)
- Optional pre-place/pre-insert filtering: [place.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/planner/action/place.py#L230)
- Final ranking for non-G2 robots: [place.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/planner/action/place.py#L335)

Implication:

- If place candidates survive `AvoidObs` IK but runtime planning still collides, the issue is likely in runtime world modeling, attached-object collision modeling, or trajectory ranking.
- If no candidates survive `AvoidObs` IK, the issue is earlier and the runtime planner is not the main suspect.

### 2. Place action sequence

The default place action sequence is generated in [place.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/planner/action/place.py#L360).

Current default behavior:

- First action:
  - `Action(target_pose_canonical, None, place_transform_up, "AvoidObs")`
  - `place_transform_up = [0, 0, 0.05]`
- Second action:
  - usually just open gripper with `Action(None, gripper_cmd, I, "Simple")`

Important detail:

- This is not a full multi-stage place approach by default.
- It is basically one `AvoidObs` move to the target pose with a fixed `+5 cm` world-frame translation, then gripper open.
- No default place-time `goal_offset`, no default place-time `path_constraint`, and no default `from_current_pose=True` in the standard place action.

References:

- `place_transform_up`: [place.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/planner/action/place.py#L375)
- main place `AvoidObs` action: [place.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/planner/action/place.py#L377)
- default `Simple` open: [place.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/planner/action/place.py#L400)

Implication:

- The runtime plan is not forced to use a careful "approach above target, descend, release" structure unless task config explicitly adds it.
- That can make place more sensitive to attached-object collision accuracy.

### 3. Client execution

Execution goes through [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L720).

Current behavior:

- Extra planning params are read from the action:
  - `goal_offset`
  - `path_constraint`
  - `offset_and_constraint_in_goal_frame`
  - `disable_collision_links`
  - `from_current_pose`
- If `remove_obstacles=True`, the passive object is removed from the CuRobo obstacle world before motion.
- But in current code, place does not enable that path by default.

Key lines:

- pick-only removal in `KeepClose`: [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L735)
- commented-out place removal TODO: [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L739)
- actual remove call: [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L763)
- motion execution with forwarded params: [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L801)

After a successful pick close, the client marks the passive object as attached and calls the attach RPC:

- attach after pick close: [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L877)
- RPC wrapper: [client.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/client.py#L339)

### 4. Robot client and RPC handoff

The client packages pose moves in [omni_robot.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/omni_robot.py#L186).

For `AvoidObs`:

- `AvoidObs` is remapped to backend `ObsAvoid`.
- The same motion params are forwarded to the server:
  - `goal_offset`
  - `path_constraint`
  - `offset_and_constraint_in_goal_frame`
  - `disable_collision_links`
  - `from_current_pose`

References:

- `move_pose(...)`: [omni_robot.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/omni_robot.py#L186)
- `AvoidObs -> ObsAvoid`: [omni_robot.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/omni_robot.py#L211)
- server `moveto(...)` call: [omni_robot.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/omni_robot.py#L291)

On the server:

- `handle_linear_move()` receives those fields and routes to CuRobo.
- UIBuilder `_follow_target(...)` forwards them into `caculate_ik_goal(...)`.

References:

- `_follow_target(...)`: [ui_builder.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/ui_builder.py#L240)
- place planner setup uses `step=32`: [ui_builder.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/ui_builder.py#L462)

## CuRobo MotionGen Collision Settings

### 1. MotionGen initialization

The main runtime planner is created in [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L539).

Current important settings:

| Setting | Current value | Source |
| --- | --- | --- |
| `collision_checker_type` | `MESH` | [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L587) |
| `num_trajopt_seeds` | `4` | [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L589) |
| `num_graph_seeds` | `4` | [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L590) |
| `num_ik_seeds` | `32` | [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L591) |
| `num_batch_ik_seeds` | `32` | [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L592) |
| `interpolation_dt` | `0.01` | [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L593) |
| `interpolation_steps` | `5000` | [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L594) |
| `collision_cache` | `{"obb": 30, "mesh": env/default 100}` | [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L595) |
| `optimize_dt` | `True` | [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L596) |
| `trajopt_tsteps` | `32` from UIBuilder | [ui_builder.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/ui_builder.py#L471), [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L598) |
| `num_trajopt_noisy_seeds` | `1` | [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L599) |
| `num_batch_trajopt_seeds` | `1` | [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L600) |
| `collision_activation_distance` | env/default `0.02` | [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L568), [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L601) |

Warmup:

- `parallel_finetune=True`
- `batch=CUROBO_BATCH_SIZE`
- `CUROBO_BATCH_SIZE = 10`

References:

- batch constant: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L48)
- warmup: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L609)

### 2. Plan config

Runtime planning uses this `MotionGenPlanConfig`:

- `enable_graph=True`
- `enable_opt=True`
- `need_graph_success=True`
- `enable_graph_attempt=5`
- `max_attempts=40`
- `enable_finetune_trajopt=True`
- `parallel_finetune=True`
- `time_dilation_factor=1.0`
- `ik_fail_return=5`
- `success_ratio=0.5`

Reference:

- [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L618)

What this means:

- Planning is not a pure straight-line check.
- It is a graph + optimization pipeline.
- But once multiple candidates succeed, the final path selection is not clearance-aware.

### 3. Final successful-path selection is not collision-margin-aware

After `plan_batch(...)` succeeds, the code:

- filters paths by position error
- filters by rotation error
- falls back to all paths if the filters remove everything
- sorts by joint-space difference
- picks the first one

References:

- `plan_batch(...)`: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L1246)
- error filtering: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L1257)
- Galbot weight vector: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L1271)
- final selected path: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L1279)

Implication:

- If several collision-free or near-feasible plans exist, the chosen plan is biased toward joint change, not toward maximum obstacle clearance.
- With attached-object collision approximated by spheres, this can still pick a visually risky path.

### 4. Path constraints and collision disabling

`caculate_ik_goal(...)` supports:

- `goal_offset`
- `path_constraint`
- `offset_and_constraint_in_goal_frame`
- `disable_collision_links`
- `from_current_pose`

References:

- function definition: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L1147)
- partial-pose path metric: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L1220)
- collision-link disabling: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L1238)

Current place action usually does not use these features.

By contrast, pick-up after grasp often does use:

- `goal_offset`
- `path_constraint`
- `from_current_pose=True`

Reference:

- [grasp.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/planner/action/grasp.py#L556)

Implication:

- Pick has a more structured constrained lift.
- Place, by default, does not.

## World Obstacle Modeling

### 1. Initial obstacle extraction

Obstacle extraction starts in `AgibotUsdHelper.get_obstacles_from_stage(...)` and `_extract_cached_obstacles(...)`.

References:

- obstacle extraction helper: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L147)
- initial cache build: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L360)

Current important filters:

- only `UsdGeom.Mesh` prims are considered
- objects under the robot path are ignored
- some fixed stage helper paths are ignored
- background objects can be ignored globally
- with `background_collision_only=True`, background meshes are only kept if they expose collision schema / collision-enabled state
- far obstacles are skipped if distance from robot exceeds `GENIESIM_CUROBO_MAX_OBSTACLE_DISTANCE` default `10.0 m`

References:

- mesh-only filter: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L239)
- collision-disabled filter: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L248)
- background collision-only filter: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L263)
- ignored prefixes and background toggle: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L367)
- env defaults: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L560)
- max-distance filter during fast update: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L796)

### 2. Fast obstacle update at runtime

Each refresh uses `_update_obstacle_poses_fast(...)`.

Key behavior:

- iterates cached obstacle geometries
- skips any obstacle whose prim path starts with an attached object path
- skips invalid prims
- skips obstacles beyond `max_obstacle_distance`
- only reloads the collision model if any obstacle pose changed

References:

- skip attached objects: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L766)
- distance filter: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L797)
- collision model reload only on `has_update`: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L705)

Implication:

- After attach, attached meshes are intentionally removed from the static world and then ignored in future world refreshes.
- From that point on, collision safety for the grasped object depends on attached spheres on the robot, not on the original world obstacle mesh.

### 3. What `view_debug_world()` actually shows

`view_debug_world()` calls:

- `visualize_robot_spheres()`
- `visualize_obstacles()`

References:

- [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L1043)
- obstacle visualization: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L871)
- robot visualization: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L888)

Important interpretation:

- `visualize_obstacles()` visualizes `self.world_cfg.objects`.
- That is the planner's filtered CuRobo world, not the full simulator scene.
- So if a table or cabinet is not shown there, CuRobo usually is not reasoning about it.

Very important nuance:

- `remove_objects_from_world(...)` directly disables and removes obstacles from `motion_gen.world_model`.
- That path does not refresh `self.world_cfg`.
- So obstacle visualization is a strong proxy, but not a perfect one after manual removals.

Reference:

- direct removal path: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L1338)

Practical reading:

- If a background table/cabinet is absent from debug world from the beginning, that is a serious red flag and can directly explain collision.
- If an attached object still appears after attach, that can just be stale visualization state.

## Attached-Object Collision Modeling

### 1. Runtime attached-link configuration patch

At startup, `_ensure_attached_collision_config(...)` patches the robot config so attached links are collision-enabled:

- ensures `attached_object` and `left_attached_object` are in `collision_link_names`
- ensures `extra_collision_spheres[attached_link] = 30`
- sets self-collision-ignore entries between attached links and nearby gripper links

Reference:

- [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L111)

This matters for Galbot because the raw YAMLs currently do not list attached links in `collision_link_names`.

References:

- right YAML missing attached links in raw list: [galbot_fixed_right.yml](/home/agxi/RealityLab/genie_sim/source/data_collection/config/curobo/configs/robot/galbot_fixed_right.yml#L15)
- left YAML missing attached links in raw list: [galbot_fixed_left.yml](/home/agxi/RealityLab/genie_sim/source/data_collection/config/curobo/configs/robot/galbot_fixed_left.yml#L15)

### 2. Attach RPC path

Current attach flow:

1. Client marks attached object after successful pick close.
2. Client calls attach RPC with the object prim path.
3. Server expands the object prim into mesh prims.
4. UIBuilder gets current end-effector pose and calls `curoboMotion.attach_obj(...)`.

References:

- client attach after pick: [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L881)
- RPC server attach entry: [grpc_server.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/grpc_server.py#L541)
- mesh expansion in command controller: [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L1348)
- UIBuilder attach: [ui_builder.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/ui_builder.py#L481)

### 3. What attach does inside CuRobo

`attach_obj(...)` and `attach_objects_to_robot(...)` do this:

- detach any existing attached object links
- clear `attached_objects`
- refresh static obstacles
- build `ee_pose` from UIBuilder-provided position/quaternion
- apply `world_objects_pose_offset = [0, 0, 0.005, 1, 0, 0, 0]`
- invert the pose and use it as `pre_transform_pose`
- fit bounding spheres for each object mesh in that transformed frame
- disable the original obstacle
- optionally remove it from the world model
- call `attach_spheres_to_robot(...)`

References:

- attach entry point: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L1386)
- pose offset and attach call: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L1403)
- pose inversion path: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L1427)
- sphere fitting with `pre_transform_pose=ee_pose`: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L1474)
- raw sphere packing: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L1485)
- removal from world model: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L1486)
- final attach to robot: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L1510)

Core implication:

- After attach, the obstacle mesh is no longer protecting you.
- Only the attached spheres protect you.
- If the attached frame is wrong, the planner can think the object is elsewhere and allow a colliding path.

## Galbot-Specific Attach-Frame Analysis

### 1. Current Galbot config is frame-inconsistent

Galbot CuRobo config currently says:

- end effector link is `right_gripper_tcp_link` / `left_gripper_tcp_link`
- attached extra link parent is `right_gripper_base_link` / `left_gripper_base_link`
- attached extra link fixed transform is identity

References:

- `ee_link` in right YAML: [galbot_fixed_right.yml](/home/agxi/RealityLab/genie_sim/source/data_collection/config/curobo/configs/robot/galbot_fixed_right.yml#L13)
- right attached extra link: [galbot_fixed_right.yml](/home/agxi/RealityLab/genie_sim/source/data_collection/config/curobo/configs/robot/galbot_fixed_right.yml#L256)
- `ee_link` in left YAML: [galbot_fixed_left.yml](/home/agxi/RealityLab/genie_sim/source/data_collection/config/curobo/configs/robot/galbot_fixed_left.yml#L13)
- left attached extra link: [galbot_fixed_left.yml](/home/agxi/RealityLab/genie_sim/source/data_collection/config/curobo/configs/robot/galbot_fixed_left.yml#L256)

But UIBuilder currently gets attach pose from `_get_ee_pose(...)`, and for Galbot that solver end-effector is the TCP link:

- `_get_ee_pose(...)`: [ui_builder.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/ui_builder.py#L265)
- Galbot end-effector prim path uses TCP: [galbot_fixed_dual.json](/home/agxi/RealityLab/genie_sim/source/data_collection/config/robot_cfg/galbot_fixed_dual.json#L82)

The Galbot URDF shows that TCP is not the same as gripper base:

- `*_gripper_tcp_link` is a fixed child of `*_gripper_base_link`
- transform is `xyz="0.13996 0 0"` and `rpy="-1.5707963267948966 0 0"`

Reference:

- right TCP joint: [galbot_fixed_dual.urdf](/home/agxi/RealityLab/genie_sim/source/data_collection/config/robot_cfg/galbot/galbot_fixed_dual.urdf#L972)
- left TCP joint: [galbot_fixed_dual.urdf](/home/agxi/RealityLab/genie_sim/source/data_collection/config/robot_cfg/galbot/galbot_fixed_dual.urdf#L1245)

So the current combination is:

- runtime attach spheres are generated from TCP pose
- attached link frame is configured as gripper-base frame
- fixed transform is identity

That is the exact mismatch to check first.

### 2. Recommendation for Galbot

Yes, for the current workspace logic, the attached-object link frame should match the frame used to generate attached spheres.

There are two coherent choices:

1. Preferred simple fix:
   - set attached extra-link parent to `*_gripper_tcp_link`
   - keep attached extra-link fixed transform as identity

2. Equivalent but less clean fix:
   - keep parent as `*_gripper_base_link`
   - set attached extra-link fixed transform equal to the URDF base->TCP transform

What is not coherent is the current Galbot combination:

- parent = `*_gripper_base_link`
- fixed transform = identity
- runtime attach pose = TCP pose

Why I prefer option 1 for Galbot:

- it matches the current attach caller
- it matches `ee_link`
- it removes duplicated transform bookkeeping
- it makes debug interpretation easier

One caution:

- G2 also uses a base-link parent for attached links, but there the YAML encodes a non-identity fixed transform to align the attached frame with the true grasp frame.
- Galbot currently does not.

Reference:

- G2 attached-link pattern: [G2_omnipicker_fixed_dual.yml](/home/agxi/RealityLab/genie_sim/source/data_collection/config/curobo/configs/robot/G2_omnipicker_fixed_dual.yml#L398)

## Specific Suspicious Points For Place Collisions

### Suspicion A: Missing background obstacles

This can directly explain collision with tables, cabinets, large scene fixtures, and receptacle furniture.

What to verify:

- In debug logs, inspect `[ObstacleDiag:initial_cache]`.
- Check whether table/cabinet paths appear under `loaded_mesh` or under `skip_background_no_collision`.
- Confirm whether the object path is under `/World/background` or `/World/Background`.
- Confirm whether those prims have `UsdPhysics.CollisionAPI` / collision-enabled state.

Relevant code:

- background filter: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L263)
- background prefixes: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L50)

### Suspicion B: Attached spheres are offset or rotated incorrectly

This is the highest-priority Galbot-specific issue.

Expected symptom:

- robot body and world obstacles look correct in debug world
- but the held object spheres are visibly offset relative to the grasped object
- planner produces paths that look collision-free only if you ignore the real held geometry

Relevant code:

- UIBuilder supplies TCP pose: [ui_builder.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/ui_builder.py#L488)
- sphere fitting uses that pose as frame basis: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L1477)

### Suspicion C: Place target remains an obstacle

In current checked-in code, place does not automatically remove the passive target from the CuRobo world.

That means:

- if the container/receptacle is modeled too conservatively, place may fail early
- if the planner still succeeds, it is because it found some feasible corridor around the container
- this is separate from the missing-background issue

Relevant code:

- place removal TODO is commented out: [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L739)

### Suspicion D: `collision_activation_distance = 0.02` may be too small for coarse attached spheres

This is a secondary risk, not my first suspect.

Why:

- attached collision is approximated with spheres
- surface fit radius is `0.005`
- activation distance is only `0.02`
- with a poorly aligned sphere model, a 2 cm activation band is not enough to compensate

References:

- activation distance: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L568)
- attached sphere fit radius: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L1407)

### Suspicion E: Final trajectory ranking prefers joint economy, not clearance

This is a secondary suspect when multiple valid paths exist.

If collision margins are thin, the chosen result can still look bad because the planner chooses the minimum-joint-change path, not the max-clearance path.

Relevant code:

- final sort by `sort_by_difference_js(...)`: [motion_gen_reacher.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/motion_generator/motion_gen_reacher.py#L1273)

## What I Would Check First With You

### Check 1: Galbot attach frame

Verify immediately after attach:

- are the robot spheres for the attached object centered at the true grasped object pose?
- are they rotated/offset as if they were attached to `gripper_base` instead of TCP?

If they are offset by roughly the Galbot base->TCP transform, that basically confirms the bug.

### Check 2: Background world inclusion

Enable debug world and inspect:

- table
- cabinet
- large receptacle support surfaces

If they do not appear, inspect the log lines from obstacle diagnostics.

Likely explanations:

- background prim lacks collision schema while `GENIESIM_CUROBO_BACKGROUND_COLLISION_ONLY=1`
- path filtered by background prefix logic
- mesh extraction skipped
- distance filtered out

### Check 3: Receptacle obstacle policy during place

Decide whether the passive place object should remain an obstacle during the place approach.

Current code says:

- pick can optionally remove it
- place does not remove it by default

If your place style needs insertion into a container or close contact with a receptacle, the current obstacle policy may be too strict unless the scene geometry is very accurate.

## Bottom-Line Answers To Your Two Questions

### 1. Should Galbot attached-object parent link be `*_tcp_link` instead of `*_gripper_base_link`?

My answer: yes, given the current workspace attach path, that is the cleanest fix.

More precise answer:

- either make the attached link frame be TCP
- or keep base as parent but set the attached link fixed transform to the real base->TCP transform

Do not keep the current mix of:

- attach pose generated from TCP
- attached link frame defined as base
- identity fixed transform

### 2. If `view_debug_world()` does not show background table/cabinet, can that be the reason for collision?

My answer: yes, absolutely.

If those obstacles are absent from `view_debug_world()` from the start, CuRobo is very likely not planning against them, and that alone can produce collision trajectories.

But it is probably not the only issue if your collision is specifically bad during place with a grasped object. For Galbot, I would treat these as two independent suspects:

- missing background obstacles
- attached-object frame mismatch

Both can exist at the same time.

