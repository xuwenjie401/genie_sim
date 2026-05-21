# Isaac Sim 单进程 Teleop/Manipulation 踩坑记录

本文总结 Galbot SLAM 手操采集脚本开发过程中踩过的坑，重点覆盖 Isaac Sim 物理、直接移动 root pose、Curobo MotionGen、gripper 状态机、attach 逻辑和交互式 viewport。目标是后续再做类似单进程 teleop 程序时，不要重复掉进同一批坑。

相关代码：

- `source/data_collection/scripts/run_slam_data_collection.py`
- `source/data_collection/slam_collection/base_motion.py`
- `source/data_collection/slam_collection/vertical_lift.py`
- `source/data_collection/slam_collection/manipulation_control.py`
- `source/data_collection/slam_collection/gripper_control.py`
- `source/data_collection/slam_collection/posture_hold.py`
- `source/data_collection/tasks/diy/slam/galbot_slam_home_b.json`

## 1. 直接 set root pose 不是物理底盘

当前底盘移动是直接更新 robot root pose，本质接近 kinematic teleport。虽然每帧位姿连续变化，看起来像全向底盘，但它不是通过轮子、base velocity 或 articulation action 让 PhysX 连续推着机器人走。

这会带来一个关键后果：

- 夹爪、瓶子、桌面之间的接触约束会被反复硬搬动
- 摩擦力、接触 solver、夹爪 force 只能改善小幅滑动
- 当 root pose 被硬更新时，抓住的轻物体仍可能滑落或抖出

经验结论：

- 做 SLAM/manual collection 时，优先接受 root pose kinematic 移动，并在抓取后显式 attach target
- 不要期待单靠调大摩擦系数解决所有滑落
- 如果目标是严格物理交互，再考虑连续 base velocity、kinematic target、轮子模型或 PhysX joint 方案

## 2. attach 比摩擦更可靠

当机器人 root pose 是直接 set 的时候，最稳的抓取搬运方案是：

1. gripper close
2. 等 close motion 稳定进入 holding
3. 检查 grasp target 和 TCP 距离是否合理
4. 记录 `tcp_to_object = inv(tcp_pose) @ object_pose`
5. 每帧用 `object_pose = tcp_pose @ tcp_to_object` 跟随
6. open gripper 时 detach，并恢复物理状态

注意点：

- attach 时机不要放在按键按下瞬间，应该放在 close 状态机确认 `closing -> holding` 后
- open 时先 detach，再执行开夹爪
- attach 后最好把目标物体设置为 kinematic，并清零 `physics:velocity` 和 `physics:angularVelocity`
- detach 时恢复原来的 `physics:kinematicEnabled`，再清零速度
- 加一个 `attach_distance_threshold`，否则空抓或抓歪时会把远处目标硬吸到手上

如果要用 PhysX fixed joint 或 D6 joint，也要遵守同样的状态机边界：close 稳定后创建 joint，open 前销毁 joint。

## 3. gripper holding 不能只锁当前关节位置

一开始的 gripper 状态机是：

- close 阶段用低 force velocity close，避免猛夹
- 检测位置变化小于阈值后，latch 当前 gripper joint position
- holding 阶段用 position drive 保持 latch 位置

这个方案能防止夹爪继续乱动，但 attach target 后，视觉上可能出现“手指弹开一点”的感觉。原因是 holding 只是锁住接触瞬间的位置，没有持续闭合趋势。

更好的 holding 策略：

- close 阶段仍用较小 `close_max_force`，保持柔和探测
- holding 阶段提高 `hold_max_force`
- holding 阶段设置 `hold_stiffness` 和 `hold_damping`
- holding target 在 latch position 基础上加一个很小的 close bias
- bias 要按 `closed_velocities` 的方向加，并 clip 到 `closed_positions`

推荐把这些做成配置参数：

```json
"gripper": {
    "close_max_force": 0.2,
    "hold_stiffness": 10000.0,
    "hold_damping": 1000.0,
    "hold_max_force": 30.0,
    "hold_close_position_bias": 0.03
}
```

调参经验：

- 看起来夹得太深，降低 `hold_close_position_bias` 到 `0.015`
- 仍然松开，升到 `0.05`
- 物体被弹飞，先降低 `hold_max_force` 或 bias，而不是提高 close 阶段 force

## 4. Curobo 的 robot model 必须同步当前 vertical/lock joints

Galbot 的升降腿会改变 torso/arm base 高度。如果 Curobo 内部 robot model 仍使用默认 lock joints，高度会和 Isaac 真实 articulation 不一致。

典型现象：

- 按 J 后，抓取 pose 看起来整体偏低或偏高
- 规划目标似乎落在物体下面
- lift 或 reset 在当前高度下表现奇怪

修复原则：

- 每次 arm planning 前读取 Isaac articulation 当前 joint positions
- 找出 Curobo config 里的 locked joints
- 调用 motion generator 更新 lock joints
- 同步更新 Curobo kinematics lock joints

这一步要在每次 `_plan_to_base_pose()`、`_plan_lift()`、`_plan_reset()` 前做，而不是只在启动时做。

## 5. Curobo collision world 不要启动时全局构建

从 stage mesh 生成 Curobo `WorldConfig` 很重，特别是 home scene 里有大量物体和复杂 mesh。如果程序一启动就基于全局 mesh 计算 collision world，会导致进入 teleop 前卡很久。

推荐策略：

- Curobo MotionGen 懒初始化，第一次 manipulation 按键触发时再初始化
- 初始 MotionGen 跳过全局 obstacle extraction
- 每次规划前只加载局部 collision world
- 局部范围用 TCP start 到 goal 的线段距离筛选
- 只包含明确前缀下的可交互物体，例如 `/World/SceneObjectPlacer/PlacedObjects`
- 限制 `max_obstacles`
- 简化 mesh faces，例如 `max_mesh_faces = 128`
- 排除 robot、自身 target helper、viewport helper、Curobo helper prim

配置建议：

```json
"local_collision": {
    "enabled": true,
    "radius": 0.45,
    "max_obstacles": 12,
    "max_mesh_faces": 128,
    "include_prefixes": [
        "/World/SceneObjectPlacer/PlacedObjects"
    ]
}
```

不要把“全局场景碰撞越完整越好”当成默认原则。对交互式 teleop 来说，规划延迟比理论完整性更容易毁掉体验。

## 6. grasp target 是否作为 obstacle 要分清阶段

抓取阶段对 target object 的 collision 处理很微妙。

如果直接规划 TCP 到最终接触 pose，并且 gripper collision 仍完整启用，把 grasp target 作为普通 obstacle 可能导致无解，因为目标就是要接触它。

但如果无脑把 grasp target 忽略掉，也会出现：

- approach 方向穿过物体
- 手腕或手指从物体内部绕出来
- Curobo 规划看起来“避障成功”，视觉上却不合理

建议：

- 不要在 collision extraction 里硬编码永久忽略 grasp target
- 通过 action 阶段显式传 `ignore_prim_paths`
- 对 `move_grasp`，优先使用更合理的 grasp pose 和 approach 筛选，必要时配合 pre-grasp
- 对 `lift` 和 `move_place`，通常应忽略已经 attach 的 grasp target，否则规划会把手上物体当成外部障碍
- 如果未来要严格避免搬运时撞环境，应该给 attached object 单独做 robot-side collision representation，而不是继续把原物体 mesh 当 world obstacle

## 7. lift 应使用 partial constraint

lift 不是普通的 “move TCP 到新 pose”。它通常应该保持当前姿态，只沿某个方向平移一段距离。

如果用完整 pose constraint，容易出现：

- `Partial position between start and goal is not equal`
- pose cost metric 更新失败
- Curobo 无法规划 lift
- lift 时末端姿态被不必要地改变

经验做法：

- 读取当前 TCP pose
- 以当前 pose 作为 target
- 用 `goal_offset` 表示 lift offset
- 使用 `path_constraint` 放开 lift 方向，锁住横向平移
- `from_current_pose=True`
- 明确 `offset_and_constraint_in_goal_frame=False`，让 offset 在当前约定的 frame 下解释

对默认 Z 向 lift，position weights 类似 `[1, 1, 0]`，意思是锁住 X/Y，放开 Z。

## 8. interaction pose 不能照搬原流程排序

原 manipulation 流程里的 grasp pose selection 往往服务于自动任务，可能会结合 IK、joint cost、历史动作和不同机器人假设。单进程手操里直接复用会遇到：

- 抓取点偏高
- approach 方向别扭
- 空抓时 TCP upside down
- 左臂从身体内侧或反方向接近

Galbot 左臂 teleop 里更实用的筛选：

- 禁止 upside-down grasp
- 过滤过强 top-down approach
- 过滤朝机器人内部接近的 pose
- 适度偏好物体中下部，而不是总取上方候选
- 在剩余候选中再按当前 TCP 距离排序

推荐配置化：

```json
"grasp_vertical_threshold_deg": 20.0,
"grasp_reject_towards_robot": true,
"grasp_towards_robot_max_dot": 0.0,
"grasp_preferred_height_percentile": 45.0,
"grasp_distance_weight": 1.0,
"grasp_approach_weight": 0.35,
"grasp_height_weight": 0.25
```

## 9. place pose 必须使用实时 gripper-object 关系

JSON 里只应该配置 grasp target 和 place target，不应该预先写死 TCP pose。place 阶段需要根据当前真实抓取结果计算：

```python
gripper_to_object = inv(object_pose_world) @ current_tcp_pose_world
target_gripper_pose = target_object_pose @ gripper_to_object
```

这点很重要：

- 抓取时可能有偏差
- 物体可能在 close 或 attach 时轻微移动
- base/vertical/arm 都可能改变当前 TCP pose
- 直接用固定 place TCP pose 会导致放置偏移累积

## 10. helper UI 和 debug prim 可能污染相机画面

Isaac 里要区分 viewport UI、debug draw、USD helper prim 和 camera sensor render。

踩过的坑：

- 原点处调试坐标轴或球形 helper 被 camera 记录
- viewport selection outline 或可见 helper prim 混进画面
- 第三人称相机用来观察，但不小心被当成采集相机

建议：

- 不要把调试坐标轴、sphere、target helper 放在可记录相机视野内
- 不需要的 helper prim 直接隐藏或删除
- 每帧清空 selection，避免选中高亮进入观察画面
- 记录用相机和操作观察 viewport 分离
- third-person viewport camera 可以跟随 robot base，但不要加入 ROS bag camera list

## 11. articulation 需要持续 hold

如果只是初始化时设置关节角，Galbot 的 torso、arms、head 可能会在重力、碰撞或 base movement 后慢慢偏离。特别是直接移动 root pose 时，小的接触扰动会更明显。

建议：

- 对固定关节和非操作臂使用 posture hold
- arm motion active 时，暂停正在由 Curobo 控制的 arm joints 的 posture hold
- gripper active 时，暂停 gripper joints 的 posture hold
- arm motion 完成后，把 posture hold target 更新为当前 joint positions
- vertical lift 改变 leg joints 后，也要更新 hold target

这能避免“稍微受力后机器人姿态被重力拉倒不再复原”的问题。

## 12. main loop 顺序会影响稳定性

推荐单进程 teleop loop 顺序：

1. 读取 keyboard command 和 edge-triggered actions
2. 更新 base root pose
3. 更新 vertical lift joint targets
4. 更新 viewport follow camera
5. gripper state machine step
6. Curobo arm motion step
7. attach/follow target，最好在 arm motion step 之后
8. posture hold apply
9. `world.step(render=False)`
10. 按 render FPS 调用 `world.render()`

关键点：

- attach follow 如果在 arm motion step 前执行，会落后一帧
- root pose 更新后再读 TCP pose，才能让 attached object 跟随新 base pose
- posture hold 要避开正在被 Curobo 或 gripper 控制的 joints

## 13. 单进程不等于没有边界

这次 SLAM 手操流程不需要复用原 manipulation 的 gRPC/protobuf/TaskGenerator/DataCollectionAgent，因为：

- 键盘控制是实时交互，不是自动 policy
- 不需要多进程 client/server round trip
- 不想引入 protobuf 生成和 gRPC 维护成本

但单进程仍然需要一个清晰的 simulation-side controller 边界：

- keyboard 只产生 command/action
- controller 负责状态机、规划、attach/detach
- Isaac Sim stage/articulation 操作集中在主循环上下文
- 不要从任意线程直接改 stage 或 articulation

这个边界比 gRPC 轻，但仍然能防止状态散落。

## 14. 推荐日志和验收点

交互式程序最好把关键状态打日志：

- Curobo 是否懒初始化
- 每次局部 collision world 加载了多少 objects
- grasp candidates 输入数量和过滤后数量
- Curobo 每个候选是否 planning 成功
- attach 是否成功
- attach skip 时 target 到 TCP 距离是多少
- gripper open/close/holding 状态切换
- lift 是否使用 partial constraint

smoke test 顺序建议：

1. 启动脚本，确认没有启动时长时间卡在 mesh collision extraction
2. W/A/S/D/Q/E 控制底盘，确认 root pose 连续移动
3. R/F 控制 vertical lift，确认 Curobo 后续规划高度正确
4. H 空抓 move，确认 TCP pose 不 upside down
5. J move grasp，确认 grasp approach 自然
6. I close，确认进入 holding 后 attach 成功
7. L lift，确认物体跟随
8. W/A/S/D 移动底盘，确认物体不滑落
9. K move place，确认使用当前 gripper-object 关系
10. U open，确认 detach 后物体释放
11. O reset，确认 arm 回到 reset target

## 15. 什么时候不要用这些 shortcut

这些经验适合“手操采集、SLAM 数据、可视化真实感优先”的场景。以下情况要更谨慎：

- 要评估真实 grasp physics 成功率
- 要训练接触丰富的 manipulation policy
- 要让 base dynamics 与物体接触严格物理一致
- 要把抓取物作为规划中的 attached collision object 严格避障

这些情况下应优先做完整物理建模：

- base 连续驱动
- gripper/object 材质和 solver 参数调校
- PhysX joint 或 constraint-based attach
- Curobo attached object collision frame 校准
- 更严格的 pre-grasp、approach、retreat 阶段

对当前 Galbot SLAM 手操采集，最实用的组合是：

- root pose 连续 kinematic 移动
- 局部懒加载 Curobo collision world
- grasp pose 方向筛选
- close 后 TCP attach
- gripper holding close bias
- posture hold 稳定 articulation
