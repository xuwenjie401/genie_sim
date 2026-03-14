# Gripper Physics Control: Why free_worker.py Succeeded Where the Server Pipeline Did Not

## 1. Background

The galbot gripper is a **closed-loop linkage gripper** that Isaac Sim does not support natively.
The workaround is to drive one joint (`left_gripper_r_knuckle_joint`) via a USD `PhysicsDriveAPI`
and let the coupled joint follow through the constraint.
This makes correct drive-mode management essential: the wrong mode at the wrong time either loses
the grasp or causes an explosive release.

---

## 2. The Old Server Pipeline (`parallel_gripper.py` + `command_controller.py`)

### 2.1 Close: velocity-drive, tiny force cap

```python
# parallel_gripper.py  forward("close")
drive.GetStiffnessAttr().Set(0)          # velocity-only mode
drive.GetMaxForceAttr().Set(0.16)        # ~0.16 N cap
target_joint_velocities[...] = 40       # 40 rad/s closing velocity
ArticulationAction(joint_velocities=...) # one-shot command
```

| Attribute | Value | Effect |
|-----------|-------|--------|
| stiffness | 0 | pure velocity drive, no position restoring force |
| maxForce  | 0.16 N | tiny — enough to grip, not enough to hold |
| targetVelocity | 40 rad/s | USD drive keeps re-asserting this every step |

### 2.2 Stop: one-shot zero-velocity ArticulationAction

```python
# parallel_gripper.py  instant_stop()
ArticulationAction(joint_velocities=[0, 0])   # applied once
```

**Fatal flaw**: `ArticulationAction` is a one-shot command processed by Isaac Sim's
articulation controller.
The USD `DriveAPI` attributes (`stiffness=0`, `maxForce=0.16`, `targetVelocity=40`) **persist
in the physics cache independently** of what `apply_action` does.
On the very next physics substep, PhysX re-reads those drive attributes and re-applies 0.16 N
of closing torque — the stop never sticks.

### 2.3 Open: position-drive, strong force

```python
# parallel_gripper.py  forward("open")
drive.GetMaxForceAttr().Set(100)
target_joint_positions[...] = joint_opened_positions
target_joint_velocities[...] = 40
ArticulationAction(joint_positions=..., joint_velocities=...)
```

Open works fine because 100 N easily overcomes any residual force.
But if the close drive was never truly stopped, the bottle has been squeezed continuously
during lift and place — building up contact stress.
When the open command fires with 100 N, all that stored stress releases at once → **explosive ejection**.

### 2.4 handle_set_gripper_state: command-level, not physics-step-level

```python
# command_controller.py
def handle_set_gripper_state(self):
    if self.gripper_state != state:
        self._set_gripper_state(...)      # issued once
    if is_reached:
        self.make_gripper_stop(isRight)   # instant_stop() — also one-shot
        self.gripper_state = ""
```

Once `is_reached` clears the command, nothing further touches the drive.
The USD drive is left in whatever state the close command set it.

### 2.5 Summary of old-pipeline flaws

| Problem | Root Cause |
|---------|-----------|
| Gripper keeps closing during lift/place | USD drive `targetVelocity=40` re-asserts every step; one-shot stop doesn't persist |
| Bottle ejected explosively on open | Continuous squeeze → contact stress builds; sudden 100 N open releases everything at once |
| Bottle floats after release | `_hold_left_gripper_at_grasp` position hold causes rigid body to sleep; no sleep prevention |

---

## 3. The New free_worker.py Pipeline

### 3.1 Core insight: ArticulationAction vs USD DriveAPI

| Layer | Persistence | Controls |
|-------|-------------|---------|
| `ArticulationAction` via `apply_action` | **One-shot** — overridden next step | Joint position/velocity targets in Isaac Sim's controller |
| `UsdPhysics.DriveAPI` attributes | **Persistent** — PhysX reads them every substep | Stiffness, damping, maxForce of the PD drive |

The old code only ever touched the one-shot layer.
The new code changes **both layers** together.

### 3.2 After grasp: switch to position hold via DriveAPI

```python
# free_worker.py  _hold_left_gripper_at_grasp()
drive.GetStiffnessAttr().Set(5000)   # switch: velocity → position drive
drive.GetMaxForceAttr().Set(10.0)    # moderate force — hold without over-squeezing

# Read where the fingers actually stopped (bottle thickness determines this)
current = robot.get_joint_positions(joint_indices=[ctrl_dof, mirror_dof])

# Apply position target at that exact position
robot.apply_action(ArticulationAction(joint_positions=[current[0], current[1]]))
```

This changes the **USD drive mode** from velocity-control (stiffness=0) to
position-control (stiffness=5000).
PhysX now actively holds the finger position every substep — no continuous closing force.

### 3.3 Maintain hold every physics step, even during pause

```python
# free_worker.py  _update_state_machine()  — runs before the pause guard
if self.phase in ("post_close_settle", "wait_lift", "wait_place"):
    self._hold_left_gripper_at_grasp()
if self.paused:
    return
```

Because the N-key pause only stops the FSM, `on_physics_step` still runs.
The pre-pause guard re-asserts `stiffness=5000` and the position target on **every single physics
step** — across all three hold phases and during interactive pauses.
No other code path can override it.

### 3.4 Open at place: gradual, no teleport

```python
# wait_place_gripper_open — re-issues open every 20 steps, no force-open teleport
if step % 20 == 0:
    self.set_left_gripper("open")   # drive: stiffness=0, maxForce=100, vel=40
```

`forward("open")` overwrites the USD drive:
- `stiffness` → 0
- `maxForce` → 100
- position target → `joint_opened_positions`
- velocity target → 40 rad/s

Because `_hold_left_gripper_at_grasp` is **not** in the pre-pause guard for this phase,
the open command is no longer overridden.
The fingers move gradually to open position — contact force between fingers and bottle
dissipates naturally rather than in one explosive step.

### 3.5 Sleep prevention for the bottle

```python
# free_worker.py  _apply_physics_to_obstacle()
if not kinematic:
    physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_rb_api.GetSleepThresholdAttr().Set(0.0)
```

Set **before** any simulation runs.
`sleepThreshold=0` means PhysX can never put this body to sleep: it always integrates gravity.
Setting it at release time (after the body is already sleeping) is too late.

---

## 4. Comparison Table

| Aspect | Old Server Pipeline | New free_worker.py |
|--------|--------------------|--------------------|
| Close drive | velocity, stiffness=0, maxForce=0.16 N | same — correct for grasping |
| Stop after close | one-shot `ArticulationAction(vel=0)` — **does not persist** | switches USD drive to stiffness=5000, position hold — **persistent** |
| Hold during lift/place | nothing (drive re-asserts vel=40) | `_hold_left_gripper_at_grasp()` every physics step including pauses |
| Open at place | 100 N position drive, one-shot — explosive | 100 N position drive, re-issued every 20 steps, **no force teleport** |
| Bottle sleep | no prevention → bottle floats after release | `sleepThreshold=0` set at physics init → always awake |
| Box collision | dynamic rigid body, convexDecomposition fills open top | kinematic, convexDecomposition — does not move under impact |

---

## 5. Key Takeaways

1. **`ArticulationAction` is one-shot; `DriveAPI` attributes are persistent.**
   To permanently change gripper behavior you must write the USD drive attributes, not just call `apply_action`.

2. **After grasping, switch drive mode from velocity → position.**
   Velocity-drive with a tiny maxForce keeps squeezing indefinitely.
   Position-drive at the current position holds the grasp without building stress.

3. **Re-assert the hold every physics step, before any pause check.**
   Interactive pauses (N-key) still run `on_physics_step`. Without the pre-pause guard, a paused simulation lets the drive revert.

4. **Open gradually; never teleport fingers while the bottle is between them.**
   `_force_left_gripper_open_pose` (direct joint position assignment) generates large impulses.
   The position-control open command moves fingers smoothly.

5. **Prevent rigid body sleep before grasping, not after.**
   `sleepThreshold=0` must be set when the rigid body is created.
   Setting it on a sleeping body has no immediate effect.
