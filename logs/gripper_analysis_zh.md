# 夹爪物理控制：为什么 free_worker.py 成功而服务器流水线失败

## 1. 背景

Galbot 夹爪是一种**闭环连杆式夹爪**，Isaac Sim 原生不支持此类结构。
解决方案是通过 USD `PhysicsDriveAPI` 驱动一个关节（`left_gripper_r_knuckle_joint`），
另一个耦合关节通过约束跟随运动。
因此，正确的驱动模式管理至关重要：在错误时刻使用错误模式，要么导致抓取失败，要么引发爆炸式释放。

---

## 2. 旧版服务器流水线（`parallel_gripper.py` + `command_controller.py`）

### 2.1 关闭（夹紧）：速度驱动 + 极小力上限

```python
# parallel_gripper.py  forward("close")
drive.GetStiffnessAttr().Set(0)          # 纯速度模式
drive.GetMaxForceAttr().Set(0.16)        # ~0.16 N 上限
target_joint_velocities[...] = 40       # 40 rad/s 夹紧速度
ArticulationAction(joint_velocities=...) # 一次性指令
```

| 属性 | 值 | 效果 |
|------|----|----|
| stiffness | 0 | 纯速度驱动，无位置恢复力 |
| maxForce  | 0.16 N | 极小——勉强能夹住，无法持续保持 |
| targetVelocity | 40 rad/s | USD 驱动每个物理步都会重新施加 |

### 2.2 停止：一次性零速度 ArticulationAction

```python
# parallel_gripper.py  instant_stop()
ArticulationAction(joint_velocities=[0, 0])   # 仅施加一次
```

**致命缺陷**：`ArticulationAction` 是一次性指令，由 Isaac Sim 的关节控制器处理。
USD `DriveAPI` 属性（`stiffness=0`、`maxForce=0.16`、`targetVelocity=40`）
**在物理缓存中独立持久存在**，不受 `apply_action` 影响。
下一个物理子步，PhysX 重新读取这些驱动属性，重新施加 0.16 N 的夹紧力矩——停止指令根本没有持续效果。

### 2.3 打开：位置驱动 + 强力

```python
# parallel_gripper.py  forward("open")
drive.GetMaxForceAttr().Set(100)
target_joint_positions[...] = joint_opened_positions
target_joint_velocities[...] = 40
ArticulationAction(joint_positions=..., joint_velocities=...)
```

打开操作本身没问题，因为 100 N 足以克服任何残余力。
但如果夹紧驱动从未真正停止，瓶子在整个搬运和放置过程中都持续被挤压，
接触应力不断积累。当打开指令以 100 N 触发时，所有积累的应力瞬间释放 → **爆炸式弹射**。

### 2.4 handle_set_gripper_state：指令层而非物理步层

```python
# command_controller.py
def handle_set_gripper_state(self):
    if self.gripper_state != state:
        self._set_gripper_state(...)      # 仅发出一次
    if is_reached:
        self.make_gripper_stop(isRight)   # instant_stop() —— 同样是一次性
        self.gripper_state = ""
```

一旦 `is_reached` 清除了指令，就不再有任何代码接触驱动。
USD 驱动被遗留在夹紧指令设置的状态下。

### 2.5 旧流水线缺陷总结

| 问题 | 根本原因 |
|------|--------|
| 搬运/放置时夹爪持续夹紧 | USD 驱动 `targetVelocity=40` 每步重新施加；一次性停止指令无法持久 |
| 打开时瓶子爆炸式弹射 | 持续挤压积累接触应力；100 N 打开力瞬间释放所有应力 |
| 释放后瓶子漂浮 | 位置保持导致刚体进入睡眠状态；无睡眠预防机制 |

---

## 3. 新版 free_worker.py 流水线

### 3.1 核心认知：ArticulationAction vs USD DriveAPI

| 层次 | 持久性 | 控制内容 |
|------|--------|--------|
| `ArticulationAction` via `apply_action` | **一次性** —— 下一步被覆盖 | Isaac Sim 控制器中的关节位置/速度目标 |
| `UsdPhysics.DriveAPI` 属性 | **持久** —— PhysX 每个子步读取 | PD 驱动的刚度、阻尼、最大力 |

旧代码只操作一次性层。新代码**同时修改两个层**。

### 3.2 抓取后：通过 DriveAPI 切换为位置保持

```python
# free_worker.py  _hold_left_gripper_at_grasp()
drive.GetStiffnessAttr().Set(5000)   # 切换：速度驱动 → 位置驱动
drive.GetMaxForceAttr().Set(10.0)    # 适中力 —— 保持抓取不过度挤压

# 读取手指实际停止位置（由瓶子厚度决定）
current = robot.get_joint_positions(joint_indices=[ctrl_dof, mirror_dof])

# 在该精确位置施加位置目标
robot.apply_action(ArticulationAction(joint_positions=[current[0], current[1]]))
```

这将 **USD 驱动模式**从速度控制（stiffness=0）切换为位置控制（stiffness=5000）。
PhysX 现在每个子步都主动保持手指位置——不再有持续的夹紧力。

### 3.3 每个物理步维持保持，即使在暂停中

```python
# free_worker.py  _update_state_machine()  —— 在暂停守卫之前运行
if self.phase in ("post_close_settle", "wait_lift", "wait_place"):
    self._hold_left_gripper_at_grasp()
if self.paused:
    return
```

N 键暂停只停止有限状态机，`on_physics_step` 仍然运行。
暂停前守卫在**每一个物理步**重新施加 `stiffness=5000` 和位置目标——
覆盖三个保持阶段以及交互式暂停期间的所有步骤。
没有其他代码路径能覆盖它。

### 3.4 放置时打开：渐进式，无传送

```python
# wait_place_gripper_open —— 每 20 步重新发出打开指令，无强制传送
if step % 20 == 0:
    self.set_left_gripper("open")   # 驱动：stiffness=0, maxForce=100, vel=40
```

`forward("open")` 覆盖 USD 驱动：
- `stiffness` → 0
- `maxForce` → 100
- 位置目标 → `joint_opened_positions`
- 速度目标 → 40 rad/s

因为 `_hold_left_gripper_at_grasp` **不在**此阶段的暂停前守卫中，
打开指令不会被覆盖。手指逐渐移动到打开位置——
手指与瓶子之间的接触力自然消散，而不是在一个爆炸性步骤中释放。

### 3.5 瓶子的睡眠预防

```python
# free_worker.py  _apply_physics_to_obstacle()
if not kinematic:
    physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_rb_api.GetSleepThresholdAttr().Set(0.0)
```

在**任何仿真运行之前**设置。
`sleepThreshold=0` 意味着 PhysX 永远不能让该刚体进入睡眠：它始终积分重力。
在释放时（刚体已经在睡眠之后）设置为时已晚。

---

## 4. 对比表

| 方面 | 旧版服务器流水线 | 新版 free_worker.py |
|------|----------------|-------------------|
| 夹紧驱动 | 速度驱动，stiffness=0，maxForce=0.16 N | 相同 —— 抓取时正确 |
| 夹紧后停止 | 一次性 `ArticulationAction(vel=0)` —— **不持久** | 切换 USD 驱动为 stiffness=5000，位置保持 —— **持久** |
| 搬运/放置时保持 | 无（驱动重新施加 vel=40） | 每个物理步调用 `_hold_left_gripper_at_grasp()`，包括暂停期间 |
| 放置时打开 | 100 N 位置驱动，一次性 —— 爆炸式 | 100 N 位置驱动，每 20 步重新发出，**无强制传送** |
| 瓶子睡眠 | 无预防 → 释放后漂浮 | 物理初始化时设置 `sleepThreshold=0` → 始终保持活跃 |
| 箱子碰撞 | 动态刚体，convexDecomposition 填充开口顶部 | kinematic，convexDecomposition —— 不因撞击移动 |

---

## 5. 关键结论

1. **`ArticulationAction` 是一次性的；`DriveAPI` 属性是持久的。**
   要永久改变夹爪行为，必须写入 USD 驱动属性，而不仅仅是调用 `apply_action`。

2. **抓取后，将驱动模式从速度切换为位置。**
   速度驱动加上极小的 maxForce 会无限期持续挤压。
   在当前位置使用位置驱动可以在不积累应力的情况下保持抓取。

3. **每个物理步重新施加保持，在任何暂停检查之前。**
   交互式暂停（N 键）仍然运行 `on_physics_step`。没有暂停前守卫，暂停的仿真会让驱动恢复。

4. **渐进式打开；手指夹住瓶子时不要传送手指。**
   `_force_left_gripper_open_pose`（直接关节位置赋值）会产生大冲量。
   位置控制打开指令让手指平滑移动。

5. **在抓取之前防止刚体睡眠，而不是在抓取之后。**
   `sleepThreshold=0` 必须在刚体创建时设置。
   在已经睡眠的刚体上设置它没有立即效果。
