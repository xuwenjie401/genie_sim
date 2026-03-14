# Gripper Hold Persistence in `free_worker_gripper.py`

## Conclusion

The sentence

> after gripper reached, it set stiffness(5000), use "persistent" position control to make gripper holds where it is

is only **partly correct**.

It is correct if "persistent" means:

1. the code switches the USD gripper drive into a nonzero-stiffness hold mode, and
2. the code keeps re-applying the hold target on every physics step during the carry phases.

It is **not** correct if it means:

1. a single `ArticulationAction(joint_positions=...)` is itself persistent, or
2. setting stiffness once automatically guarantees that the articulation controller will keep the same target forever.

`ArticulationAction` is still a one-shot command interface. The persistence in this unit test comes from the **control loop**, not from that one call alone.

---

## 1. What happens during close

In [`source/data_collection/server/controllers/parallel_gripper.py`](/home/agxi/RealityLab/genie_sim/source/data_collection/server/controllers/parallel_gripper.py#L326), the `"close"` branch does:

```python
drive.GetStiffnessAttr().Set(0)
drive.GetMaxForceAttr().Set(target_force)
target_action = ArticulationAction(joint_velocities=target_joint_velocities)
```

Important points:

1. `stiffness = 0` means the drive is not acting like a position spring.
2. The close command is sent as a velocity action.
3. `instant_stop()` later only sends zero joint velocities; it does not restore a hold-mode stiffness by itself.

So if you only did:

1. close with velocity mode,
2. one-shot stop with zero velocity,

then that would not be a robust "hold at grasp width" design.

---

## 2. What `_hold_left_gripper_at_grasp()` really does

In [`unit_lab/free_worker_gripper.py`](/home/agxi/RealityLab/genie_sim/unit_lab/free_worker_gripper.py#L2937), `_hold_left_gripper_at_grasp()` does two distinct things:

### 2.1 It changes the USD drive mode

```python
drive.GetStiffnessAttr().Set(5000)
drive.GetMaxForceAttr().Set(10.0)
```

This is important. It changes the gripper from the previous close behavior into a hold behavior with nonzero stiffness and a moderate force cap.

This part is more persistent than `ArticulationAction`, because these are drive attributes living on the USD/PhysX side.

### 2.2 It samples the current joint positions and sends them back as targets

```python
current = robot.get_joint_positions(...)
robot.apply_action(ArticulationAction(joint_positions=target_positions))
```

This means:

1. the current finger width at the instant of grasp is measured,
2. that measured width becomes the desired hold target,
3. the articulation controller is asked to hold that target.

But this `apply_action(...)` is still a one-shot call.

So the exact statement is:

- the function establishes a hold target,
- but it does not become persistent merely because `apply_action(...)` was called once.

---

## 3. Where the real persistence comes from

The key logic is in [`unit_lab/free_worker_gripper.py`](/home/agxi/RealityLab/genie_sim/unit_lab/free_worker_gripper.py#L2979):

```python
if self.phase in ("post_close_settle", "wait_lift", "wait_place"):
    self._hold_left_gripper_at_grasp()
if self.paused:
    return
```

This means:

1. during `post_close_settle`,
2. during `wait_lift`,
3. during `wait_place`,

the code calls `_hold_left_gripper_at_grasp()` on **every physics step**.

And it does this **before** the pause guard, so even when the FSM is paused with the keyboard, the hold function is still re-run.

This is the real reason the hold behaves persistently in practice.

So the correct interpretation is:

- not "one position action persists forever",
- but "the system keeps reasserting the same hold policy every physics step".

---

## 4. Why your confusion is valid

Your reading is correct:

1. `ArticulationAction` is one-shot.
2. `_hold_left_gripper_at_grasp()` does not show an explicit long-lived controller object with internal memory.
3. If you read only that function, the phrase "persistent position control" sounds too strong.

The missing piece is the phase loop in `_update_state_machine()`, which repeatedly calls the hold function.

Without that loop, the statement would be much weaker and probably inaccurate.

---

## 5. Comparison with the old server pipeline

In [`source/data_collection/server/command_controller.py`](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L727), `make_gripper_stop()` is:

```python
action = self.gripper_L.instant_stop()
self.robot.apply_action(action)
```

and `instant_stop()` in [`source/data_collection/server/controllers/parallel_gripper.py`](/home/agxi/RealityLab/genie_sim/source/data_collection/server/controllers/parallel_gripper.py#L366) is only:

```python
ArticulationAction(joint_velocities=0.0)
```

That is just a one-shot velocity command. It does not explicitly create a grasp-width hold target, and it does not install the repeated physics-step reassertion loop that the unit test has.

So compared with the old server path, the unit test is genuinely doing something stronger.

---

## 6. Strict wording that matches the code

If we want a precise wording, it should be:

> After grasp completion, `free_worker_gripper.py` switches the gripper drive to a nonzero-stiffness hold mode, samples the current finger positions as the hold target, and re-applies that hold target every physics step during the carry phases. The hold is persistent because the state machine continuously reasserts it, not because a single `ArticulationAction` is persistent by itself.

---

## 7. Bottom line

So the previous conclusion is:

1. **not fully wrong**,
2. but **too compressed**, and
3. easy to misread.

The safest final judgment is:

- "persistent hold" is true at the **behavior level**,
- but not because `ArticulationAction` itself is persistent,
- rather because the code repeatedly restores the hold target and drive mode every physics step.
