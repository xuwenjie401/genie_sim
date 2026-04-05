# Data Collection 帧率退化排查结论（agent-A）

日期：2026-04-04

Problem：
- `source/data_collection` 在长时间连续采集时，系统会随着 episode 数增长出现明显“卡顿”。
- 当前需要回答的问题不是“如何修”，而是“帧率/响应退化更可能来自哪一层：Python 逻辑、资源生命周期、录制链路，还是 Isaac Sim 引擎层”。

Phenomenon：
- 随着 episode 增长，Isaac Sim UI 中机械臂动作的时间间隔变长，肉眼可见动作越来越慢。
- 保存出来的 observations 视频看起来仍然较流畅，但这不能证明运行时没有退化，因为视频是按采样帧和固定 FPS 重新组织后的结果。
- 一个直接业务影响是：夹爪关闭完成判定会在系统变慢后更容易提前触发，导致夹爪尚未真正夹紧就被认为已经 close。

范围：
- 本文基于当前工作区中 `source/data_collection` 的代码静态排查。
- 本文聚焦的问题是：`source/data_collection` 在连续运行多个 episode 后，Isaac Sim UI 中机械臂动作间隔变长、系统出现持续“卡顿”的原因。
- 本文不是修复方案文档，只记录本轮排查结论、证据与未证实假设。

## Agent-A 执行摘要

- 结论主要基于代码路径分析与结构推断，不包含 runtime profiler、Isaac Sim 引擎内部 trace、GPU profiler 或 ROS bag 实测统计。

约束说明：
- “已排除”表示在当前 Python 代码层没有找到足以解释持续退化的强证据，不等于从引擎层百分之百否定。
- “高嫌疑”表示当前代码证据最集中、最值得优先验证的方向。- 结论主要基于代码路径分析与结构推断，不包含 runtime profiler、Isaac Sim 引擎内部 trace、GPU profiler 或 ROS bag 实测统计。

约束说明：
- “已排除”表示在当前 Python 代码层没有找到足以解释持续退化的强证据，不等于从引擎层百分之百否定。
- “高嫌疑”表示当前代码证据最集中、最值得优先验证的方向。

短结论：

- 当前没有发现 Python 代码里一个会随着 episode 数持续增长的明显 `O(n)` per-frame 路径。
- 对于这条具体 task，`USD` 动态物体累积不是主要退化源。对象集合是有限的，更像早期平台化开销，而不是会贯穿 200 个 episode 持续线性恶化的主因。
- 需要修正一个关键误判：`if not self.camera_graph_path:` 这个守卫并不能证明 camera publisher 只初始化一次。因为 `_on_reset()` 会在每个 episode 把 `camera_graph_path` 和 `ros_publishers` 清空，下一轮录制仍会再次初始化 camera 发布链路。
- 当前最强嫌疑不是普通 Python 列表增长，而是 camera render-product / Replicator writer / ROS camera publisher 在 episode 间重复初始化但未看到对应清理。
- 次强嫌疑是录制与 extract 带来的 CPU 与磁盘争用；对象缓存、soft reset、OmniGraph 编辑残留属于后续验证项。

当前嫌疑排序：

1. camera render-product / Replicator writer / ROS camera publisher 跨 episode 重复初始化但未清理。
2. 录制与 extract 带来的 CPU + 磁盘争用。
3. soft reset，没有 hard reset world/timeline。
4. 固定 graph path 上的 OmniGraph 编辑残留。
5. GPU 显存碎片化。

## 已排除或降级的假设

### 1. USD 物体累积不是这条 task 的主要帧率退化源

结论：
- 对这条具体 task，动态对象集合是有限的，不支持“每个 episode 都往 Stage 持续堆出一整套新物体”这一说法。
- 它更像对象缓存策略带来的有限上限开销，而不是长期持续恶化的主因。

证据：
- task 模板里的 scene candidate 是有限集合，见 [left_place_cola_can_into_box_galbot_v1.json](../source/data_collection/tasks/diy/single_task/left_place_cola_can_into_box_galbot_v1.json)。
- task generator 会把与 task object 同资产的 scene object 排除掉，见 [task_generate.py#L355-L362](../source/data_collection/client/layout/task_generate.py#L355-L362) 和 [task_generate.py#L429-L432](../source/data_collection/client/layout/task_generate.py#L429-L432)。
- 默认 `prim_path` 固定由 `object_id` 派生为 `/World/Objects/<object_id>`，见 [base.py#L55](../source/data_collection/client/agent/base.py#L55)。
- 服务端加对象时，如果该 `prim_path` 已存在，就不会再次 `add_reference_to_stage(...)`，见 [command_controller.py#L2178-L2187](../source/data_collection/server/command_controller.py#L2178-L2187)。

对这条 task 的具体估计：
- task-related object 固定为 2 个。
- scene candidate 在排除与 task object 同资产后，剩余唯一候选约 13 个。
- 因此动态对象总上限约为 15 个，而不是随 episode 无界增长。

补充说明：
- episode 结束时确实没有 delete prim，而是把对象移到 `[99, 99, 0]`，见 [omniagent.py#L1058-L1069](../source/data_collection/client/agent/omniagent.py#L1058-L1069)。
- 这说明系统在做对象缓存复用，但这件事本身更像“有限集合缓存”，不足以单独解释长期持续退化。

### 2. `frame_status / gripper_action_status / timing_stats` 不是主因

结论：
- 这些容器没有显示出跨 episode 不断积累并进入 per-frame 热路径的模式。

证据：
- `_on_reset()` 会清空 `frame_status`、playback 缓存，并重置 gripper action tracking，见 [command_controller.py#L1934-L1950](../source/data_collection/server/command_controller.py#L1934-L1950)。
- timing 统计只按固定 key 聚合均值和总时间，没有显式进入高复杂度路径。

### 3. Python 显式 `on_physics_step` 循环里，没有发现随 episode 增长的明显 `O(n)` 操作

结论：
- 代码层的每帧显式循环没有表现出“episode 越多，遍历对象越多”的结构。
- 这不能证明系统不会退化，只能说明“退化更可能来自底层资源积累导致常数项变大”，而不是表层 Python 复杂度升高。

证据：
- `on_physics_step()` 里主要是 UIBuilder、可选 hold gripper、`on_command_step()` 和 `ros_publishers` tick，见 [command_controller.py#L829-L859](../source/data_collection/server/command_controller.py#L829-L859)。
- 当前没有看到一个与 episode 数直接相关的 Python 容器遍历。

### 4. `camera_list` append-only 是泄漏信号，但不是当前最强的代码层退化解释

结论：
- 即使某些 camera 相关状态存在 append-only 行为，只要没有进入 per-frame 遍历，就不足以解释持续退化。
- 这类状态更像“资源未清理”的旁证，而不是单独的根因。

## 关键修正结论

### 为什么“`if not self.camera_graph_path` 使 camera publisher 只初始化一次”这个判断不成立

这是本轮排查里最重要的修正点。

结论：
- 这个守卫只在“单次 reset 之间”有效，不在“跨 episode”上成立。
- 因为 `_on_reset()` 每个 episode 都会把 `self.ros_publishers = []` 和 `self.camera_graph_path = []` 清空，所以下一轮 `start_recording()` 时，camera 初始化分支会再次执行。

证据链：
- `start_recording()` 里 camera 初始化的守卫是 `if not self.camera_graph_path:`，见 [command_controller.py#L1429-L1468](../source/data_collection/server/command_controller.py#L1429-L1468)。
- `_on_reset()` 在 episode reset 时显式清空 `self.ros_publishers` 和 `self.camera_graph_path`，见 [command_controller.py#L1948-L1949](../source/data_collection/server/command_controller.py#L1948-L1949)。
- `omni_robot.reset()` 会调用 `client.reset()`，见 [omni_robot.py#L214-L221](../source/data_collection/client/robot/omni_robot.py#L214-L221)；服务端 reset 最终走到 `_on_reset()`。

因此：
- “Python 列表没有增长”不等于“底层 camera 资源没有重复初始化”。
- 更大的风险是：列表被清空了，但底层 Isaac/Replicator/ROS writer 资源并未同步 detach 或销毁。

## 当前最可信的退化原因排序

### 1. Camera render-product / Replicator writer / ROS camera publisher 重复初始化但未清理

这是当前最强嫌疑。

证据：
- `_init_camera()` 每次都会 `Camera.initialize()`，见 [base.py#L54-L75](../source/data_collection/server/ros_publisher/base.py#L54-L75)。
- RGB / depth publisher 都会 `writer.attach([render_product])`，见 [camera.py#L118-L139](../source/data_collection/server/ros_publisher/camera.py#L118-L139) 和 [camera.py#L241-L250](../source/data_collection/server/ros_publisher/camera.py#L241-L250)。
- `stop_recording()` 只调用 `remove_graph(self.graph_path)`，而 `graph_path` 只包含 `RobotTFActionGraph`、`RobotJointActionGraph`、`ClockActionGraph`，见 [command_controller.py#L1505-L1513](../source/data_collection/server/command_controller.py#L1505-L1513)。
- 当前没有看到 camera writer / render product 的显式 detach 或销毁路径。

推断：
- 即使 Python 侧的 `camera_graph_path` 被 reset，底层 camera 发布链路仍可能在 Isaac / Replicator 层累积。
- 这类累积会直接放大每帧渲染、图像发布和录制的常数项，更符合“episode 越跑越慢”的现象。

### 2. 录制与 extract 带来的 CPU 与磁盘争用

这是当前第二强嫌疑。

证据：
- 录制使用 `ros2 bag record -a`，见 [command_controller.py#L1508-L1513](../source/data_collection/server/command_controller.py#L1508-L1513) 上游逻辑和当前录制分支。
- 成功 episode 后会启动 extract 子进程，extract 内部又会开 `ProcessPoolExecutor(max_workers=os.cpu_count() or 4)`，并执行图像落盘与 `ffmpeg`，这意味着 CPU 和磁盘 I/O 都可能持续竞争。

推断：
- 即使 recorder 进程管理本身没有明显泄漏，长期运行下的录制与后处理并发仍可能明显拉低 Isaac Sim 主循环。

### 3. Soft reset，没有 hard reset world/timeline

这是保留项，但证据弱于前两项。

证据：
- 当前 reset 路径是服务端 `_on_reset()` 清理 Python 侧状态，并没有看到 stop/play timeline 或 world hard reset 的动作，见 [command_controller.py#L1934-L1950](../source/data_collection/server/command_controller.py#L1934-L1950)。

推断：
- 如果 PhysX 接触缓存、碰撞缓存、内部 graph 或 sensor 状态需要 hard reset 才能彻底释放，那么 soft reset 可能会让运行时常数项逐渐变大。
- 但这条目前缺少直接代码证据，只能列为次级引擎层假设。

### 4. 固定 graph path 上的 OmniGraph 编辑残留

这是保留项。

证据：
- `publish_tf()` / `publish_joint()` 会反复对固定 graph path 执行 `og.Controller.edit(...)`。
- 当前没有看到对 camera graph 类似资源的系统性回收。

推断：
- 即使 graph path 固定，OmniGraph 内部是否完全复用节点、是否残留附着关系，还需要引擎层验证。

### 5. GPU 显存碎片化

这是当前最弱假设。

结论：
- 代码层没有直接证据支持，只能作为长时间运行常见问题保留。

## 对“对象移到 `[99, 99, 0]`”的判断

结论：
- 这段逻辑更像对象缓存策略，而不是每轮严格的销毁重建。
- 它能解释“为什么对象不 delete prim”，但不能单独解释长期持续 FPS 退化。

证据：
- episode 结束时对象只是被挪走，见 [omniagent.py#L1058-L1069](../source/data_collection/client/agent/omniagent.py#L1058-L1069)。
- 下一轮 add object 时，如果 Stage 中已有同名 `prim_path`，就不会再 `add_reference_to_stage(...)`，见 [command_controller.py#L2178-L2187](../source/data_collection/server/command_controller.py#L2178-L2187)。

附带风险：
- 这种缓存策略会导致 pose / rigid body / collision 状态复用逻辑变得脆弱。
- 它是状态残留问题的重要来源，但对这个 task 不像是持续退化的最强解释。

## 证据索引

最关键的代码位置如下：

- camera 初始化守卫与重复进入路径：
  - [command_controller.py#L1429-L1468](../source/data_collection/server/command_controller.py#L1429-L1468)
- stop recording 的清理范围：
  - [command_controller.py#L1505-L1513](../source/data_collection/server/command_controller.py#L1505-L1513)
- reset 时清空 `ros_publishers` 与 `camera_graph_path`：
  - [command_controller.py#L1934-L1950](../source/data_collection/server/command_controller.py#L1934-L1950)
- camera 初始化：
  - [base.py#L54-L75](../source/data_collection/server/ros_publisher/base.py#L54-L75)
- RGB / depth writer attach：
  - [camera.py#L118-L139](../source/data_collection/server/ros_publisher/camera.py#L118-L139)
  - [camera.py#L241-L250](../source/data_collection/server/ros_publisher/camera.py#L241-L250)
- 已存在 prim 时不重复 add：
  - [command_controller.py#L2178-L2187](../source/data_collection/server/command_controller.py#L2178-L2187)
- episode 结束时把对象移到 `[99, 99, 0]`：
  - [omniagent.py#L1058-L1069](../source/data_collection/client/agent/omniagent.py#L1058-L1069)
- scene object 采样时排除与 task object 同资产对象：
  - [task_generate.py#L355-L362](../source/data_collection/client/layout/task_generate.py#L355-L362)
  - [task_generate.py#L429-L432](../source/data_collection/client/layout/task_generate.py#L429-L432)
- 默认对象 `prim_path`：
  - [base.py#L55](../source/data_collection/client/agent/base.py#L55)

## 尚未验证的引擎层假设

以下条目目前没有足够代码证据，只能列为待验证项：

1. PhysX 长时间运行后，碰撞检测缓存、接触点历史或其他内部状态在 soft reset 下未完全释放。
2. OmniGraph 固定 graph path 上的 `og.Controller.edit(...)` 在引擎内部留下残留节点或附着关系。
3. CUDA / GPU 显存碎片化导致长时间运行后分配效率下降。

这些方向都需要 runtime profiler、Isaac 内部 trace 或更细粒度的性能计数器验证，不能仅靠当前 Python 代码静态阅读下定论。


## Agent-B 执行摘要

Agent-B 对 Agent-A 的排查文档进行了独立交叉验证。

与 Agent-A 的一致点：
- USD 物体累积不是这条 task 的主因（有限集合，上限 ~15 个）。
- `frame_status` / `gripper_action_status` / `timing_stats` 不是主因。
- Python 显式 `on_physics_step` 循环无随 episode 增长的 O(n) 遍历。
- Camera render-product / Replicator writer 重复初始化未清理是当前最强嫌疑。

Agent-B 的独立补充（Agent-A 未覆盖或未展开的内容）：
1. **`_on_reset` → `_get_observation` → `_capture_camera` → `Camera().initialize()` + `camera_list.append()` 链路**：每 episode 对每个 camera 重复创建 Isaac `Camera` 对象并 append 到 `ui_builder.camera_list`，该列表从未被 clear/pop/遍历，是纯 Python 内存泄漏（不影响帧率但可确认资源管理缺失）。
2. **`publish_noised_rgb` 比 `publish_rgb` 更重**：它会 `rep.create.render_product()` 创建全新 render product + `annotator.attach(rp)`，而非复用 `camera._render_product_path`。当 `noised_probability > 0` 时，每 episode 累积的渲染开销更大。
3. **`_init_camera` 返回的 `camera_graph`（gate_path 列表）在 `command_controller.py:1466` 赋值后未存储到任何成员变量**，直接丢失引用。即使想在 stop 时清理这些 OmniGraph gate node，也没有引用可用。
4. **工作区有未提交改动**：`handle_task_status` 末尾从 `self.object_asset_dict = {}` 改成了 `self._cleanup_episode_objects()`，但该方法在全 repo 中**没有定义**。如果 `task_name is not None`（正常采集路径），运行到这行会 `AttributeError`。
5. **夹爪 `check_gripper_state` 的异步协程泄漏**：每次 `forward()` 都 `asyncio.ensure_future()` 但不 cancel 前一个。虽然不是帧率退化的主因，但会在事件循环中堆积残留协程。
6. **`ros_node_initialized` 守卫只控制 `ServerNode` 的创建**（`command_controller.py:1498-1500`），不控制 camera pipeline 的创建。两者是独立路径。

对 Agent-A 嫌疑排序的调整意见：
- 排序一致，无需调整。
- 但建议将嫌疑 1 进一步细分为两个子项（见下方"细化"部分），因为修复策略不同。

---

## 对嫌疑 1 的细化：Camera 资源泄漏的两条独立路径

Agent-A 将 camera 资源泄漏作为一个整体描述。Agent-B 认为需要区分两条独立的泄漏路径，因为它们的修复方式不同。

### 路径 A：Replicator writer attach 无对应 detach

资源类型：`rep.writers` + `SyntheticData gate node` + `annotator`（仅 noised 路径）

创建时机：每 episode 的 `startRecording` → `_init_camera()` → `publish_rgb()` / `publish_depth()` / `publish_noised_rgb()` 等

关键代码：
- `publish_rgb`：`rep.writers.get(rv + "ROS2PublishImage").attach([render_product])`，见 [camera.py#L127-L134](../source/data_collection/server/ros_publisher/camera.py#L127-L134)
- `publish_depth`：同上模式，见 [camera.py#L241-L248](../source/data_collection/server/ros_publisher/camera.py#L241-L248)
- `publish_noised_rgb`：`rep.create.render_product()` + `annotator.attach(rp)`，见 [camera.py#L144-L146](../source/data_collection/server/ros_publisher/camera.py#L144-L146)

清理缺失：
- `stopRecording` 只 `remove_graph(self.graph_path)`，`graph_path` = `["/World/RobotTFActionGraph", "/World/RobotJointActionGraph", "/ClockActionGraph"]`，不包含 camera 相关 graph/writer。见 [command_controller.py#L1513](../source/data_collection/server/command_controller.py#L1513)。
- 没有任何地方调用 `writer.detach()` 或 `annotator.detach()`。

引用丢失：
- `_init_camera` 返回 `(camera_graph, ros_nodes)`，但调用方只使用了 `ros_nodes`（追加到 `self.ros_publishers`），`camera_graph`（包含 gate_path）赋值后未存储。见 [command_controller.py#L1466-L1467](../source/data_collection/server/command_controller.py#L1466-L1467)。

累积量估计（此 task）：
- 3 个 camera，每 camera 的 publish 列表为 `["rgb:/<name>_rgb"]`（noised_probability=0.0，所以走 `publish_rgb` 而非 `publish_noised_rgb`）。
- 每 episode 累积 3 个 Replicator writer + 3 个 SyntheticData gate node。
- 200 episodes ≈ 600 个 writer 仍 attach 在 3 个 render product 上。

### 路径 B：`Camera().initialize()` 在 `_on_reset` 中重复调用

资源类型：Isaac `Camera` sensor 对象 + 底层 render product 初始化

创建时机：每 episode 的 `_on_reset()` → `_get_observation()` → `_capture_camera()` → `ui_builder._on_capture_cam()` → `Camera(prim_path=...).initialize()`

关键代码：
- [ui_builder.py#L183-L186](../source/data_collection/server/ui_builder.py#L183-L186)：每次 `_on_capture_cam` 都 `Camera(prim_path=self._currentCamera, resolution=resolution).initialize()` 并 append 到 `camera_list`。
- [command_controller.py#L1943](../source/data_collection/server/command_controller.py#L1943)：`_on_reset` 调用 `_get_observation()`，对每个 camera 触发上述链路。

与路径 A 的区别：
- 路径 A 发生在 `startRecording` 阶段，是关于 ROS publishing pipeline 的资源泄漏。
- 路径 B 发生在 `_on_reset` 阶段，是关于 Camera sensor 本身的重复初始化。
- 路径 B 的 `camera_list` 从未被遍历，所以 Python 层没有 per-frame 影响。但 `Camera.initialize()` 对同一 prim_path 反复调用是否在 Isaac 引擎内部产生累积，需要 runtime 验证。

累积量估计（此 task）：
- 3 个 camera × (1 次 reset + 1 次 startRecording 中的 `_init_camera`) = 每 episode 6 次 `Camera.initialize()` 调用。
- `camera_list` 200 episodes 后约 600-1200 个 `Camera` 对象（取决于 `_get_observation` 调用频率）。

---

## 对嫌疑 2 的补充：录制与 extract 争用的具体机制

Agent-A 提到了 CPU/磁盘争用但未展开细节。补充如下。

### extract 子进程的资源消耗

证据：
- `handle_task_status` 中成功 episode 后启动 `extract_and_convert_data.py` 子进程，见 [command_controller.py#L1648-L1664](../source/data_collection/server/command_controller.py#L1648-L1664)。
- `MAX_EXTRACT_PROCESS_NUM = 2`，最多并发 2 个提取进程。见 [command_controller.py#L49](../source/data_collection/server/command_controller.py#L49)。
- 等待提取进程空位时有 120s 超时，超时后 kill 最老的进程。见 [command_controller.py#L1630-L1644](../source/data_collection/server/command_controller.py#L1630-L1644)。

注意：`handle_task_status` 在 `on_command_step` 中执行，而 `on_command_step` 运行在 Isaac Sim 的物理步回调内。如果等待提取进程空位的 `while True: ... time.sleep(0.1)` 循环阻塞了较长时间，它不会直接导致帧率退化（因为是在特定 command 处理期间），但会导致 episode 之间的切换延迟增大。

### 录制期间的 I/O 竞争

- `ros2 bag record -a` 录制所有 topic，包括 3 个 camera 的 rgb 图像流。见 [command_controller.py#L1362](../source/data_collection/server/command_controller.py#L1362)。
- 录制进程与 Isaac Sim 主进程共享磁盘 I/O 带宽。
- 如果磁盘是 HDD 或多 episode 录制数据在同一分区，I/O 延迟可能随数据量增长。

---

## Agent-A 未提及的附加发现

### 1. 工作区有未提交改动引入了一个必崩的 bug

`handle_task_status` 末尾的 `self._cleanup_episode_objects()` 在全 repo 中没有定义（HEAD 版本是 `self.object_asset_dict = {}`）。

证据：
```bash
git diff HEAD -- source/data_collection/server/command_controller.py
# -        self.object_asset_dict = {}
# +        self._cleanup_episode_objects()
```
- `CommandController` 没有父类（`class CommandController:` 无继承），见 [command_controller.py#L67](../source/data_collection/server/command_controller.py#L67)。
- `grep -r '_cleanup_episode_objects' .` 在全 repo 中只有这一处调用，无定义。

影响：如果当前工作区的代码被用于采集，每次 `handle_task_status`（每 episode 结束时）都会 `AttributeError`，导致 task status 处理失败。但由于 `on_command_step` 有异常处理（或者实际运行时可能用的是其他分支），这可能被静默吞掉了。需要确认。

### 2. 夹爪异步协程无取消机制

`parallel_gripper.py:363` 每次 `forward()` 都 `asyncio.ensure_future(check_gripper_state())`，但不 cancel 前一个。

证据：
- `forward()` 开头只设 `self.is_reached = False`（line 307/327），不管前一个协程状态。见 [parallel_gripper.py#L286-L364](../source/data_collection/server/controllers/parallel_gripper.py#L286-L364)。
- 没有 `self._gripper_task` 成员变量来追踪当前协程。

影响：不是帧率退化的主因，但随 episode 增长可能在事件循环中堆积数十个未完成的协程。更关键的是，旧协程可能在新 `forward()` 之后意外设置 `is_reached = True`，导致逻辑竞争。

### 3. `_on_reset` 中 `_get_observation` 的副作用

`_on_reset()` 调用 `_get_observation()`（line 1943），后者对每个 camera 调用 `_capture_camera()` → `_on_capture_cam()`。

`_on_capture_cam` 的副作用：
- 每次调用都 `Camera(prim_path=...).initialize()` 创建新的 Camera wrapper。见 [ui_builder.py#L183-L184](../source/data_collection/server/ui_builder.py#L183-L184)。
- 无条件 append 到 `camera_list` 和 `camera_prim_list`（never cleared）。见 [ui_builder.py#L185-L186](../source/data_collection/server/ui_builder.py#L185-L186)。
- 但 `camera_list` 和 `camera_prim_list` 在整个代码库中**从未被读取**（只有 append，没有任何 `for ... in camera_list` 或 `camera_list[i]`）。

这说明 `_on_capture_cam` 的设计可能本意是做 Camera 缓存以避免重复创建，但实际上每次都在创建新对象而非查找已有对象。如果改为先检查 `prim_path` 是否已在 `camera_prim_list` 中，则可以避免重复 `Camera.initialize()`。

---

## Fix Plan 摘要（供后续 Agent 直接上手）

### Fix 1（优先级最高）：Camera 资源生命周期管理

目标：消除 camera render-product / writer / annotator 的跨 episode 累积。

有两种互补策略，建议先做 1a，如不够再加 1b：

**1a. 让 camera pipeline 只初始化一次（最小改动）**

思路：`_on_reset` 中不清空 `camera_graph_path`，让 `if not self.camera_graph_path:` 守卫生效。

改动：
- `command_controller.py` `_on_reset()`：删除 `self.camera_graph_path = []` 这行（当前 line 1949）。
- 同时也不清空 `self.ros_publishers = []`（当前 line 1948），否则 camera publisher 的 Python 引用丢失但底层资源仍在跑。
- 如果 `ros_publishers` 不清空，需要确保 `stopRecording` 时停止 tick（可以加一个 `self._recording_active = False` 标志让 `on_physics_step` 跳过 tick，`startRecording` 时 set True）。

局限：不适用于每 episode 需要变化 camera noise 参数的 task。但当前 task 的 `noised_probability = 0.0`，适用。

**1b. 正确 detach 再重建（通用方案）**

思路：追踪 writer/annotator 引用，`stopRecording` 时调用 detach 并删除 gate node。

改动：
- `command_controller.py`：新增 `self._camera_writers = []` 和 `self._camera_render_products = []` 成员变量。
- `ros_publisher/camera.py`：修改 `publish_rgb` / `publish_depth` / `publish_noised_rgb` 等，返回 `(gate_path, writer)` 或 `(gate_path, writer, render_product, annotator)` 元组。
- `ros_publisher/base.py` `_init_camera`：收集所有返回的资源引用，一并返回。
- `command_controller.py` `startRecording`：存储返回的资源引用到成员变量。
- `command_controller.py` `stopRecording`：新增 `_cleanup_camera_resources()` 方法，对每个 writer 调用 `writer.detach()`，对 noised 路径的 annotator 调用 `annotator.detach()`，删除 gate node prim。

### Fix 2：夹爪检测改用仿真步数 + 协程取消

目标：让夹爪关闭判定不受帧率退化影响，同时防止协程泄漏。

改动：
- `parallel_gripper.py` `__init__`：增加 `self._physics_step_event = asyncio.Event()`、`self._physics_step_count = 0`、`self._gripper_task = None`。
- `parallel_gripper.py` 新增 `notify_physics_step()`：递增计数器并 set event。
- `parallel_gripper.py` `forward()`：
  - 开头 cancel 旧 task：`if self._gripper_task and not self._gripper_task.done(): self._gripper_task.cancel()`
  - `check_gripper_state()` 内部：将 `await asyncio.sleep(0.1)` 替换为等待 N 个物理步（如每 6 步 ≈ 0.1s @60Hz）。退出条件从 `n > 50` 改为物理步数超限。
  - 末尾：`self._gripper_task = asyncio.ensure_future(check_gripper_state())`
- `command_controller.py` `on_physics_step()`：调用 `self.gripper_L.notify_physics_step()` 和 `self.gripper_R.notify_physics_step()`（需判空）。

### Fix 3（可选）：修复工作区中的 `_cleanup_episode_objects` 崩溃

改动：将 `command_controller.py:1680` 的 `self._cleanup_episode_objects()` 还原为 `self.object_asset_dict = {}`，或者实现该方法。

### Fix 4（可选）：`_on_capture_cam` 避免重复创建 Camera 对象

改动：在 `ui_builder.py` `_on_capture_cam()` 中，先检查 `self._currentCamera` 是否已在 `self.camera_prim_list` 中，如果已存在则复用已有 Camera 对象而非重新创建。

---

## 证据索引（Agent-B 补充，不重复 Agent-A 已列出的）

- `_on_reset` 调用 `_get_observation` 导致 Camera 重复创建：
  - [command_controller.py#L1943](../source/data_collection/server/command_controller.py#L1943)
  - [ui_builder.py#L183-L186](../source/data_collection/server/ui_builder.py#L183-L186)
- `camera_list` 从未被读取（只 append）：
  - [ui_builder.py#L69-L70](../source/data_collection/server/ui_builder.py#L69-L70)（初始化）
  - [ui_builder.py#L185-L186](../source/data_collection/server/ui_builder.py#L185-L186)（唯一写入点）
- `_init_camera` 返回的 `camera_graph` 未被存储：
  - [command_controller.py#L1466](../source/data_collection/server/command_controller.py#L1466)
- `publish_noised_rgb` 创建额外 render product：
  - [camera.py#L144](../source/data_collection/server/ros_publisher/camera.py#L144)
- 工作区未提交改动引入未定义方法：
  - `git diff HEAD -- source/data_collection/server/command_controller.py` 中 line 1680
- 夹爪协程无取消机制：
  - [parallel_gripper.py#L347-L363](../source/data_collection/server/controllers/parallel_gripper.py#L347-L363)
- extract 子进程管理：
  - [command_controller.py#L1630-L1664](../source/data_collection/server/command_controller.py#L1630-L1664)
- `ros_node_initialized` 只控制 ServerNode，不控制 camera：
  - [command_controller.py#L1498-L1500](../source/data_collection/server/command_controller.py#L1498-L1500)

