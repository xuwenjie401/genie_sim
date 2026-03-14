# `free_worker_gripper.py` 里的夹爪保持到底是不是“persistent”

## 结论

这句话：

> after gripper reached, it set stiffness(5000), use "persistent" position control to make gripper holds where it is

只能算 **部分正确**。

如果这里的 "persistent" 指的是：

1. 代码把 USD 里的 gripper drive 切换成了非零 stiffness 的保持模式；
2. 在搬运阶段的每个 physics step 都重新施加一次保持目标；

那这个说法基本成立。

但如果它想表达的是：

1. 单次 `ArticulationAction(joint_positions=...)` 本身就是持久控制；
2. 只要设了一次 stiffness=5000，后面就会自动一直稳定保持；

那这个说法就不准确。

`ArticulationAction` 仍然是 **one-shot** 接口。这个 unit test 里的“持续保持”来自 **控制循环持续重发**，不是来自那一次 `apply_action(...)` 本身。

---

## 1. close 阶段实际做了什么

在 [`source/data_collection/server/controllers/parallel_gripper.py`](/home/agxi/RealityLab/genie_sim/source/data_collection/server/controllers/parallel_gripper.py#L326) 的 `"close"` 分支里，核心逻辑是：

```python
drive.GetStiffnessAttr().Set(0)
drive.GetMaxForceAttr().Set(target_force)
target_action = ArticulationAction(joint_velocities=target_joint_velocities)
```

这里要注意：

1. `stiffness = 0`，说明它不是典型的位置弹簧保持。
2. close 是通过速度命令发出去的。
3. 后面的 `instant_stop()` 只是发一个零速度动作，并不会自动把 drive 切回“按抓取宽度保持”的模式。

所以如果只有下面这两步：

1. velocity 模式 close，
2. 然后 one-shot 零速度 stop，

那它并不能算一个很强的“稳定保持抓取宽度”的方案。

---

## 2. `_hold_left_gripper_at_grasp()` 真正做了什么

在 [`unit_lab/free_worker_gripper.py`](/home/agxi/RealityLab/genie_sim/unit_lab/free_worker_gripper.py#L2937) 里，`_hold_left_gripper_at_grasp()` 实际包含两个动作。

### 2.1 先改 USD drive 模式

```python
drive.GetStiffnessAttr().Set(5000)
drive.GetMaxForceAttr().Set(10.0)
```

这一步很关键。它把夹爪从前面的 close 行为切到一个非零 stiffness 的 hold 模式，并且把力上限调到中等值。

这一部分比 `ArticulationAction` 更“持久”，因为它改的是 USD/PhysX 侧 drive 属性。

### 2.2 再读取当前关节位置，并把这个位置作为目标重新发出去

```python
current = robot.get_joint_positions(...)
robot.apply_action(ArticulationAction(joint_positions=target_positions))
```

它的含义是：

1. 先读取当前手指停下来的实际位置；
2. 把这个实际位置当成“保持目标”；
3. 告诉 articulation controller 去保持这个目标。

但这个 `apply_action(...)` 本身依然是 one-shot。

所以更准确的说法是：

- 这个函数确实建立了一个 hold target；
- 但“持久”不是因为这一条 action 天生会永久生效。

---

## 3. 真正的 persistent 来自哪里

关键逻辑在 [`unit_lab/free_worker_gripper.py`](/home/agxi/RealityLab/genie_sim/unit_lab/free_worker_gripper.py#L2979)：

```python
if self.phase in ("post_close_settle", "wait_lift", "wait_place"):
    self._hold_left_gripper_at_grasp()
if self.paused:
    return
```

这表示在下面几个阶段里：

1. `post_close_settle`
2. `wait_lift`
3. `wait_place`

代码会在 **每一个 physics step** 都调用 `_hold_left_gripper_at_grasp()`。

而且它发生在 pause 判断之前，所以即使你键盘暂停了 FSM，这个 hold 逻辑还是会继续跑。

这才是它在行为上看起来“持续保持”的真正原因。

所以正确理解应该是：

- 不是“某一条 position action 会一直自动存在”；
- 而是“系统每个 physics step 都在重新声明同一个 hold 策略”。

---

## 4. 你为什么会觉得前面的结论可疑

你的困惑完全合理，因为单看 `_hold_left_gripper_at_grasp()`，确实会得到下面这个判断：

1. `ArticulationAction` 是 one-shot；
2. 这个函数里看不到一个长期驻留的控制器对象；
3. 所以把它直接描述成 “persistent position control” 会让人误以为“一次调用永久保持”。

真正补全这个结论的，是 `_update_state_machine()` 里的重复调用逻辑。

如果没有那段循环，这个结论就会明显站不住。

---

## 5. 和旧 server pipeline 的差别

在 [`source/data_collection/server/command_controller.py`](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L727) 里，`make_gripper_stop()` 只是：

```python
action = self.gripper_L.instant_stop()
self.robot.apply_action(action)
```

而 [`source/data_collection/server/controllers/parallel_gripper.py`](/home/agxi/RealityLab/genie_sim/source/data_collection/server/controllers/parallel_gripper.py#L366) 里的 `instant_stop()` 本质只是：

```python
ArticulationAction(joint_velocities=0.0)
```

这就是一个 one-shot 零速度命令。它没有明确建立“按当前抓取宽度保持”的目标，也没有 unit test 里这种“每个 physics step 重发 hold”的机制。

所以相对旧 server pipeline，这个 unit test 的 hold 逻辑确实更强，也更接近你测试到的稳定效果。

---

## 6. 最严格、最贴代码的表述

如果要写成完全贴合代码的表述，可以这样说：

> After grasp completion, `free_worker_gripper.py` switches the gripper drive to a nonzero-stiffness hold mode, samples the current finger positions as the hold target, and re-applies that hold target every physics step during the carry phases. The hold is persistent because the state machine continuously reasserts it, not because a single `ArticulationAction` is persistent by itself.

对应中文可以理解为：

> 在抓取完成后，`free_worker_gripper.py` 会把 gripper drive 切换到非零 stiffness 的保持模式，读取当前手指位置作为保持目标，并在搬运相关阶段的每个 physics step 重新施加这个目标。它之所以在行为上表现为持续保持，是因为状态机在持续重申这个 hold，而不是因为单次 `ArticulationAction` 本身具有持久性。

---

## 7. 最终判断

所以，对之前那段总结的最稳妥评价是：

1. **不算错**；
2. 但 **压缩过头了**；
3. 它省略了最关键的一点：persistent 不是来自单次 action，而是来自 physics-step 级别的持续重发。
