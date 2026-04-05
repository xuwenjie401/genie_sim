# Data Collection 收尾效率优化分析与计划（agent-A）

日期：2026-04-05

Problem：
- `source/data_collection` 的正常 episode 在收尾阶段出现明显停顿。
- 用户观察到当日志出现如下几行时，Isaac Sim 主采集流程会暂停较久：
  - `[rosbag2_cpp]: Writing remaining messages from cache to the bag. It may take a while`
  - `[rosbag2_recorder]: Event publisher thread: Exiting`
  - `[rosbag2_recorder]: Recording stopped`
- 当前需要回答的问题不是“马上怎么改”，而是：
  - 这段停顿的主因在哪里。
  - 是否有可能把这部分放到后台执行。
  - 如果要做到“当前 episode 收尾”和“下一个 episode layout/reset/load”重叠，改动难度主要在哪些地方。

Phenomenon：
- 在给定日志里，`stop recording` 相关日志很快出现，但 Isaac Sim 主流程并没有立即继续。
- 具体时序见 [episode_finish_example.log](../logs/episode_finish_example.log)：
  - `03:30:58` 发送 SIGINT，见 [episode_finish_example.log#L13](../logs/episode_finish_example.log#L13)
  - 随后马上出现 `Recording stopped`，见 [episode_finish_example.log#L16](../logs/episode_finish_example.log#L16)
  - 直到 `03:31:28` 才打印 `forcing SIGKILL and continuing`，见 [episode_finish_example.log#L17](../logs/episode_finish_example.log#L17)
- 这说明“录制逻辑上已经停止”和“主流程解除阻塞”不是同一时刻。

范围：
- 本文基于当前工作区中 `source/data_collection` 与 `logs/episode_finish_example.log` 的静态排查。
- 本文聚焦“episode 收尾阶段的停顿”，不讨论长期 FPS 退化问题。
- 本文只记录 agent-A 的分析、证据与 plan，不实现代码。

## Agent-A 执行摘要

- 当前主停顿不是 extract。extract 已经在 `task_status` 阶段以子进程后台启动。
- 当前主停顿来自 `stop_recording()` 的同步等待路径。更准确地说，是 recorder 进程退出等待，而不是 bag 逻辑停止本身。
- 代码现状下，`stop_recording` 被设计成一个同步 barrier：
  - 客户端必须等 `stop_recording()` RPC 返回。
  - gRPC 服务端必须等 `handle_get_observation(stopRecording=True)` 完成。
  - `handle_get_observation()` 又会同步等待 `_stop_recording_processes(... wait_timeout=30.0)`。
- 因此，“看到 `Recording stopped` 以后主流程还卡很久”与当前代码完全一致，不是偶发现象。
- 理论上可以把部分收尾移到后台，但分两层：
  1. 低风险优化：缩短或重构 stop 的等待逻辑，让 bag metadata 一旦 ready 就尽快返回。
  2. 高收益但高改动：允许 episode N 的 finalize 在后台进行，同时 episode N+1 进入 reset/layout/load。
- 第二层不是简单把一个 `wait()` 改成线程就行，因为当前 controller 里大量 episode 上下文是全局共享状态，下一轮 reset/load 会覆盖上一轮 extract 所需数据。

短结论：

- 可以优化，而且第一步就有较大收益空间。
- 但不建议直接做“完整后台化并立刻开始下一个 episode”，除非先把 episode 上下文快照和 recorder 生命周期从 controller 的全局状态里拆开。
- 推荐先做 stop path 优化，再决定是否推进真正的跨 episode 重叠。

## 关键结论

### 1. 当前主停顿来自同步 stop barrier，而不是 extract

结论：
- 现在的主停顿发生在 `self.robot.client.stop_recording()` 返回之前。
- extract 启动更晚，而且已经是后台子进程，不是这 30 秒停顿的主因。

证据：
- 客户端在每个 episode 结束后，先同步调用 `stop_recording()`，见 [omniagent.py#L1058](../source/data_collection/client/agent/omniagent.py#L1058)。
- 只有 `stop_recording()` 返回后，客户端才继续清场、发 `task_status`，见 [omniagent.py#L1059-L1075](../source/data_collection/client/agent/omniagent.py#L1059-L1075)。
- 服务端 `get_observation(stopRecording=True)` 会阻塞等待 `blocking_start_server(...)` 返回，见 [grpc_server.py#L415-L434](../source/data_collection/server/grpc_server.py#L415-L434)。
- stop 分支中，服务端同步调用 `_stop_recording_processes(... wait_timeout=30.0)`，见 [command_controller.py#L1586-L1609](../source/data_collection/server/command_controller.py#L1586-L1609)。
- extract 是在 `task_status` 成功路径里作为子进程启动的，见 [command_controller.py#L1728-L1746](../source/data_collection/server/command_controller.py#L1728-L1746)。

推断：
- 日志中 `Recording stopped` 之后仍卡 30 秒，说明 recorder 逻辑停止很快，但 `process.wait(timeout=30.0)` 没有及时返回。
- 当前真正浪费的时间，是“等待 recorder 进程退出”这一层，而不是“等待 bag 停止写入”这一层。

### 2. 当前设计下，下一 episode 的 layout/reset/load 没机会提前开始

结论：
- 现有主流程是严格串行的。
- 不改协议或服务端状态机的话，下一 episode 的初始化无法与上一 episode 的收尾重叠。

证据：
- 客户端 `run()` 每个 task 的收尾顺序是：
  1. `stop_recording()`
  2. 清场挪走对象
  3. `send_task_status(...)`
  4. 下一轮 `load_task(...)`
  见 [omniagent.py#L1058-L1075](../source/data_collection/client/agent/omniagent.py#L1058-L1075) 和 [omniagent.py#L920-L928](../source/data_collection/client/agent/omniagent.py#L920-L928)。
- `load_task()` 开头会执行 `self.reset()`，见 [omniagent.py#L920-L928](../source/data_collection/client/agent/omniagent.py#L920-L928)。
- 服务端 `_on_reset()` 若发现 `self.process` 非空，会再次同步 stop recorder，见 [command_controller.py#L2016-L2023](../source/data_collection/server/command_controller.py#L2016-L2023)。

因此：
- 即使客户端不显式等待 `stop_recording()`，只要下一轮调用了 reset，服务端仍会把你拦住。
- 所以“直接开始下一个 episode 的 layout 加载初始化”不是简单前移几行代码能实现的。

### 3. `Recording stopped` 出现不等于当前 barrier 可以解除

结论：
- 从目前实现看，服务端是否返回，不以 `rosbag2_recorder` 打印 `Recording stopped` 为准，而以 Python 子进程是否退出、metadata 是否 ready 为准。

证据：
- `_stop_recording_processes()` 在发送 SIGINT 后，会对 recorder 进程执行 `process.wait(timeout=wait_timeout)`，见 [command_controller.py#L502-L559](../source/data_collection/server/command_controller.py#L502-L559)。
- 如果超时，它才进一步检查 `metadata.yaml` 是否已生成，并决定是否强制 `SIGKILL`，见 [command_controller.py#L527-L546](../source/data_collection/server/command_controller.py#L527-L546)。
- 在 stop 分支里，这个等待时间当前固定为 `30.0s`，见 [command_controller.py#L1590-L1594](../source/data_collection/server/command_controller.py#L1590-L1594)。

推断：
- 从日志时序看，`Recording stopped` 在 `03:30:58` 已出现，但直到 `03:31:28` 才继续，最符合的解释是：metadata 很可能早已可用，但服务端先傻等了整整 30 秒，超时后才进入 metadata 检查和强杀流程。
- 这是根据日志和代码路径做的推断，不是 runtime trace 直接证明。

## 当前最值得做的优化方向

### 1. 优先优化 stop path，而不是先改 extract

结论：
- 第一优先级应是缩短 `stop_recording()` 的同步阻塞时间。
- 这一步收益大、风险相对可控，而且不要求立刻重构 episode 生命周期。

建议方向：
- 不再无条件先等满 `wait_timeout=30s`。
- 改成：
  - 发送 SIGINT 后短周期轮询：
    - recorder 是否退出
    - `metadata.yaml` 是否生成
  - 一旦 metadata ready，立即结束前台等待。
  - lingering recorder 进程放到更短的后续清理逻辑里处理，必要时直接 `SIGKILL`。

原因：
- 当前 `recording_ready_for_extraction` 本身就是以 metadata 是否 ready 为核心判断，见 [command_controller.py#L1596-L1605](../source/data_collection/server/command_controller.py#L1596-L1605) 和 [command_controller.py#L1655-L1663](../source/data_collection/server/command_controller.py#L1655-L1663)。
- 既然 extract 能否开始最终看 metadata，那么 stop 阶段的前台等待就不该优先绑定在 recorder OS 进程何时自然退出上。

### 2. extract 已经后台化，不是当前第一瓶颈

结论：
- “结束时直接开始下一个 episode，同时把 extract 放后台”这件事，其实 extract 这一半已经基本在后台。
- 真正没后台化的是 stop/finalize barrier。

证据：
- `task_status` 中会启动 `extract_and_convert_data.py` 子进程，见 [command_controller.py#L1728-L1746](../source/data_collection/server/command_controller.py#L1728-L1746)。
- 该脚本内部再做 ros bag extract、convert、filter，见 [extract_and_convert_data.py](../source/data_collection/server/recording/extract_and_convert_data.py)。

因此：
- 如果目标是“尽快开始下一条 episode”，首刀不应该先砍 extract，而应该先砍 stop barrier。

## 真正做“后台 finalize + 前台下一 episode”时的难点

### 1. controller 当前使用的是单份 episode 全局状态

结论：
- 要让上一条 episode 的 finalize/extract 后台跑，同时下一条 episode 开始 reset/load，必须先把上一条 episode 的导出上下文做快照。
- 不然下一轮 episode 会覆盖上一轮需要写盘的数据。

证据：
- `task_status` 里构造 extract 输入时，直接读取 controller 当前成员变量：
  - `self.path_to_save`
  - `self.camera_info_list`
  - `self.frame_status`
  - `self.gripper_action_status`
  - `self.light_config`
  - `self.object_asset_dict`
  - `self.task_name`
  见 [command_controller.py#L1667-L1697](../source/data_collection/server/command_controller.py#L1667-L1697)。
- `_on_reset()` 会清理其中一部分关键状态，例如 `frame_status`、playback 状态、gripper action tracking，见 [command_controller.py#L2031-L2038](../source/data_collection/server/command_controller.py#L2031-L2038)。

风险：
- 如果上一条 episode 还没把这些上下文固定下来，就开始下一条 reset/load，那么上一条 extract 可能读到被污染的新状态，或者直接丢数据。

### 2. recorder 生命周期当前和 reset/start_recording 强耦合

结论：
- 想让下一条 episode 提前开始，必须重新定义 recorder 状态机。
- 现在 recorder 只要还在 `self.process` 里，reset 和下一次 start_recording 都会把它当成必须同步清理的旧进程。

证据：
- `startRecording` 前若发现 `self.process` 非空，会先 stop stale recorder，见 [command_controller.py#L1339-L1347](../source/data_collection/server/command_controller.py#L1339-L1347)。
- `reset` 也会在 `self.process` 非空时同步 stop，见 [command_controller.py#L2016-L2023](../source/data_collection/server/command_controller.py#L2016-L2023)。

推断：
- 如果要做真正异步化，需要把 recorder 至少拆成：
  - `recording`
  - `stopping`
  - `finalized`
  - `cleaned`
- 并明确哪些状态下允许 reset，哪些状态下允许下一轮 start_recording。

### 3. 不能在 bag 未真正 finalize 前就开始下一轮录制相关动作

结论：
- 如果在上一条 bag 还没确认 finalize 完成时，就开始下一轮 reset/layout/load，存在把下一条 episode 的动作混进上一条 bag 的风险。

推断依据：
- 当前 recorder 是 `ros2 bag record -a`，见 [command_controller.py#L1413](../source/data_collection/server/command_controller.py#L1413)。
- 它是全 topic 录制，不是基于 episode id 的 topic 隔离模式。

因此：
- “提前开始下一 episode”最安全的边界，不是 `Recording stopped` 日志出现，而应是“bag payload 与 metadata 已经达到了可独立抽取的 finalized 条件”。

## 推荐的分阶段 Plan

### Phase 0. 先加观测，不改行为

目标：
- 把 stop 过程拆成更细的耗时点，确认实际时间花在：
  - SIGINT 后 recorder 自然退出
  - metadata 生成
  - 强杀 lingering 进程
  - stop RPC 返回

建议埋点：
- 发送 SIGINT 的时间。
- 首次发现 `metadata.yaml` 存在的时间。
- recorder `poll()` 变为 exited 的时间。
- 执行 `SIGKILL` 的时间。
- `handle_get_observation(stopRecording)` 返回时间。

产出：
- 一轮真实采集的 stop latency 分解表。

### Phase 1. 低风险优化 stop barrier

目标：
- 不改变整体 episode 生命周期。
- 只减少 `stop_recording()` 的前台阻塞。

建议做法：
- 重写 `_stop_recording_processes()` 的等待策略：
  - 不再直接 `process.wait(timeout=30.0)`。
  - 改为短轮询 `process.poll()` + `_wait_for_recording_metadata(...)`。
  - metadata ready 后立即返回 `recording_ready=True`。
  - 对 lingering process 走快速 kill 和回收。
- 保留当前 `task_status` 和 extract 路径不动。

预期收益：
- 大概率直接消掉日志里这 30 秒级空等。

风险：
- 需要确认 metadata ready 的时刻是否足够代表 bag 数据已完整可读。
- 这一步最好结合实际 bag 抽取验证。

### Phase 2. 把 episode 上下文改为显式快照

目标：
- 为后续异步 finalize 做准备。

建议做法：
- 在 stop 完成后、reset 前，把当前 episode 需要导出的上下文整体封装成 `episode_context`：
  - recording path
  - task name
  - frame status
  - gripper action status
  - camera info
  - light config
  - object asset dict
  - metric config
  - scene / robot metadata
- 后续 `task_status`、extract、清理逻辑都改读这份快照，而不是直接读 controller 的当前成员变量。

预期收益：
- 解开“上一条 episode 后处理”和“下一条 episode reset/load”之间的状态覆盖问题。

### Phase 3. 引入异步 finalize 状态机

目标：
- 在 stop barrier 缩短后，进一步让非关键路径后台化。

建议做法：
- 将 stop 拆成两个层次：
  1. `front-end stop`
     - 保证 bag 已达到 finalized 条件
     - 允许前台继续
  2. `background finalize`
     - lingering process kill/reap
     - 记录日志
     - 启动 extract
- 允许 episode N+1 在 episode N 进入 `finalized` 后开始 reset/layout/load。

注意：
- 不建议让 episode N+1 早于 `finalized` 边界开始。
- 否则容易发生 bag 污染和上下文串写。

### Phase 4. 评估是否还需要“更激进的重叠”

目标：
- 确认是否值得继续追求“stop 后立即 load next layout”。

评估标准：
- 如果 Phase 1 已经把主要停顿降到可接受范围，未必需要推进更重的生命周期重构。
- 如果 stop 仍显著拖慢吞吐，再考虑更激进的跨 episode 并行。

## 改动难度评估

### 难度较低

- stop 等待策略优化。
- stop 流程埋点。
- metadata ready 优先返回的策略验证。

原因：
- 主要改动集中在 [command_controller.py](../source/data_collection/server/command_controller.py) 的 recorder 收尾路径。
- 不必立即改客户端协议和 extract 架构。

### 难度中等

- 把 extract 所需数据改成显式 episode context 快照。
- 清理 `task_status` 对 controller 全局状态的直接读取。

原因：
- 要梳理哪些字段在 reset/load 中会被覆盖。
- 但仍属于服务端内部重构，边界相对清晰。

### 难度较高

- 真正允许“上一条 finalize 在后台，下一条 layout/reset/load 在前台”。

原因：
- 需要重做 recorder 状态机。
- 需要明确 finalize 完成边界。
- 需要验证不会把下一条 episode 动作录进上一条 bag。
- 需要检查 reset/start_recording/task_status 三条路径对 `self.process` 和 episode 状态的假设。

## 证据索引

最关键的代码位置如下：

- 客户端收尾顺序：
  - [omniagent.py#L1058-L1075](../source/data_collection/client/agent/omniagent.py#L1058-L1075)
- 客户端下一轮 load/reset：
  - [omniagent.py#L920-L928](../source/data_collection/client/agent/omniagent.py#L920-L928)
- gRPC stop observation 的阻塞路径：
  - [grpc_server.py#L415-L434](../source/data_collection/server/grpc_server.py#L415-L434)
- stop recording 分支：
  - [command_controller.py#L1586-L1609](../source/data_collection/server/command_controller.py#L1586-L1609)
- `_stop_recording_processes()`：
  - [command_controller.py#L502-L559](../source/data_collection/server/command_controller.py#L502-L559)
- `task_status` 构造 episode 导出上下文：
  - [command_controller.py#L1667-L1746](../source/data_collection/server/command_controller.py#L1667-L1746)
- `reset` 遇到 active recorder 会同步 stop：
  - [command_controller.py#L2016-L2023](../source/data_collection/server/command_controller.py#L2016-L2023)
- `startRecording` 遇到 stale recorder 会同步 stop：
  - [command_controller.py#L1339-L1347](../source/data_collection/server/command_controller.py#L1339-L1347)
- recorder 启动命令：
  - [command_controller.py#L1413](../source/data_collection/server/command_controller.py#L1413)
- 参考日志：
  - [episode_finish_example.log#L13-L18](../logs/episode_finish_example.log#L13-L18)

## 给 Agent-B 的交接建议

建议 Agent-B 优先做两件事：

1. 复核 Phase 1 的核心判断：
   - `Recording stopped` 很早出现，但 stop RPC 被 `process.wait(timeout=30.0)` 拖住，这个结论是否还有遗漏路径。
2. 基于这份文档，独立检查真正推进 Phase 2/3 时的隐藏耦合：
   - 还有哪些 episode 状态没有被本文列入快照清单。
   - 除了 `reset` 与 `start_recording` 外，是否还有其他路径会因 `self.process` 非空而重新阻塞。

本轮结论：
- “能不能放后台做”的答案是：能，但建议分阶段推进。
- 第一阶段先优化 stop barrier；真正的跨 episode 重叠是后续结构性改造，不是小改。

---

## Agent-B 执行摘要

日期：2026-04-05

独立复核范围：
- 完整阅读 `logs/episode_finish_example.log`
- 独立走读 `source/data_collection/` pipeline：`command_controller.py`、`omniagent.py`、`grpc_server.py`、`extract_ros_bag.py`
- 对 Agent-A 全部结论逐条交叉验证

### 对 Agent-A 结论的认同

以下结论经我独立验证，完全同意，不再重复论证：

1. **主停顿来自 `_stop_recording_processes()` 的同步 `process.wait(timeout=30.0)`**，而非 extract。Extract 已经是后台子进程。
2. **`Recording stopped` 不等于 barrier 解除**。服务端判定条件是进程退出 + metadata ready，不是 rosbag2 的日志输出。
3. **当前 episode 生命周期严格串行**，下一 episode 的 `load_task()` → `reset()` → `start_recording()` 无法提前开始。
4. **`_on_reset()` 和 `startRecording` 的 stale recorder 清理路径**会二次阻塞——这是绕不过去的。
5. **Phase 0 先加观测、Phase 1 优化 stop path** 的优先级判断正确。
6. **controller 全局状态耦合**是做跨 episode 重叠的核心障碍。

### 补充分析：Agent-A 未覆盖的点

#### 1. 30 秒停顿的真正根因：不是 bag I/O，是 ROS2 节点销毁

Agent-A 的结论停留在 “recorder 进程没有及时退出”。我进一步定位了原因。

证据：
- 日志 L14-L16 显示 `Writing remaining messages` → `Event publisher thread: Exiting` → `Recording stopped` 三行在 SIGINT 后**立即**出现（同一秒 `1775359858`）。
- 这意味着 **bag cache flush 和 metadata.yaml 写入在 SIGINT 后几乎瞬间完成**。
- 但 L17 显示进程直到 30 秒后仍未退出，被 SIGKILL 强杀。

推断：
- rosbag2 的数据层面（flush + metadata）在收到 SIGINT 后很快完成。
- 但 ROS2 节点销毁（DDS participant teardown、shared memory cleanup）在某些环境下会挂住，特别是在 Isaac Sim 这种有大量 DDS 参与者的进程空间里。
- 这 30 秒的等待，实际上是在等一个**对数据完整性毫无意义的 ROS2 基础设施清理**。

意义：
- 这大幅降低了 Phase 1 的风险。一旦 `metadata.yaml` 存在，bag 数据已经完整可用，此时 SIGKILL 进程不会损坏数据。
- Agent-A 的 Phase 1 建议中说 “需要确认 metadata ready 的时刻是否足够代表 bag 数据已完整可读”——基于上述分析，答案是**是的，足够**。

#### 2. `ros2 bag record -a` 是一个低成本高收益的优化点

Agent-A 完全没提到 topic 过滤这个方向。

现状：
- 录制命令是 `ros2 bag record -o {recording_path} -a`，见 [command_controller.py#L1413](../source/data_collection/server/command_controller.py#L1413)。
- `-a` 录制所有 topic，包括 Isaac Sim 内部的大量系统 topic（`/parameter_events`、`/rosout`、各种 service 相关 topic 等）。

而 `RosExtrater` 实际只读取以下几类 topic，见 [extract_ros_bag.py#L789-L860](../source/data_collection/server/recording/extract_ros_bag.py#L789-L860)：
- `sensor_msgs/msg/Image`：depth/semantic 相机图像
- `sensor_msgs/msg/CompressedImage`：RGB 压缩图像
- `sensor_msgs/msg/JointState`：`/joint_states`、`/articulated/*`、`/articulation_action`
- `tf2_msgs/msg/TFMessage`：`/tf`、`/tf_static`
- `std_msgs/msg/String`：少量字符串 topic

建议：
- 把 `-a` 替换为显式 topic 列表。这只是修改一行字符串，零架构改动。
- 减少无用 topic 的录制可以直接降低 cache 大小、减少 flush 耗时和磁盘 I/O。
- 即使在上述分析中 flush 已经很快，减少无用数据仍有好处：减少磁盘空间占用、减少 extract 阶段的 bag 扫描时间。

注意：
- topic 列表需要从 camera pipeline 的 ROS publisher 配置中动态生成，不能写死。因为 camera 数量和名称在每次任务中可能不同。
- 可以在 `startRecording` 阶段，先初始化 ROS publishers，收集实际发布的 topic 名称，再拼入 `ros2 bag record` 命令。

#### 3. Phase 2 快照的简化方案

Agent-A 建议引入显式 `episode_context` 快照，列了 9 个字段。我认为方向正确，但实现可以更简单。

观察：
- `handle_task_status()` 成功路径里已经在把所有 extract 所需的上下文序列化到文件：
  - `recording_info.json`（包含所有 task_info 字段），见 [command_controller.py#L1698-L1700](../source/data_collection/server/command_controller.py#L1698-L1700)
  - `frame_state.json`，见 [command_controller.py#L1701-L1706](../source/data_collection/server/command_controller.py#L1701-L1706)
  - `metric_config.json`，见 [command_controller.py#L1707-L1709](../source/data_collection/server/command_controller.py#L1707-L1709)
- `extract_and_convert_data.py` 是通过文件路径参数读取这些 JSON 的，**不**依赖 controller 运行时状态。

因此：
- 不需要引入新的 `episode_context` 数据结构。
- 只需要把这三个 JSON 的写入时机**从 `handle_task_status()` 前移到 `stop_recording()` 返回后、`send_task_status()` 之前**。
- 写完文件后，controller 的全局状态就可以安全地被下一 episode 覆盖，extract 子进程从文件读取即可。
- 唯一的例外是 `isSuccess` 判定需要在 `send_task_status()` 才知道，但可以用一个简单的标记文件或延迟删除来处理。

### 对 Agent-A 观点的存疑

#### 1. Phase 0（埋点）是否真的需要单独做一轮

Agent-A 建议先做 Phase 0 纯埋点，跑一轮采集拿 latency 分解表。

存疑：
- 从现有日志已经能推断出完整的时间分布：SIGINT 后 flush 瞬间完成（<1s），进程挂死 30s 后被 SIGKILL。这个 pattern 在给定日志中已经足够清晰。
- 如果 Phase 0 的唯一目的是确认 “metadata.yaml 是否在 `Recording stopped` 之前/之后立即出现”，一个简单的 `ls -la metadata.yaml` + `stat` 在 SIGINT 后加一行 log 即可，不需要完整的 Phase 0 定义。
- 建议：把 Phase 0 的关键埋点（metadata.yaml 出现时间）**合入 Phase 1 的实现**中，作为 Phase 1 的前置验证步骤，而不是独立 Phase。这样可以少跑一轮完整采集。

#### 2. Phase 1 的轮询策略可以更简单

Agent-A 建议改 `_stop_recording_processes()` 为 “短轮询 `process.poll()` + `_wait_for_recording_metadata()`”。

优化建议：
- 不需要同时轮询进程状态和 metadata。根据上面的根因分析，进程退出是不可靠的（会挂在 ROS2 cleanup），metadata 才是唯一可靠的完成信号。
- 更直接的策略：
  1. 发送 SIGINT
  2. 轮询 `metadata.yaml` 是否出现（短超时，比如 5-10s）
  3. metadata 出现 → 立即返回 `recording_ready=True`，同时把 lingering process 放入一个 “待清理” 列表
  4. 在下一次 `startRecording` 或 `_on_reset()` 时，对 “待清理” 列表中的进程做非阻塞 `poll()` + 必要时 SIGKILL
- 这比交替轮询两个条件更简单，也更符合实际行为模式。

### Agent-B 的补充 Plan

在 Agent-A 的 Phase 0-4 基础上，增加以下补充：

#### Phase 0.5：Topic 过滤（可与 Phase 1 并行）

目标：
- 减少 bag 数据量和无用 topic，降低磁盘占用和 extract 扫描时间。

做法：
- 在 `handle_get_observation(startRecording=True)` 中，初始化 ROS publishers 后，收集所有实际发布的 topic 名称。
- 将 `ros2 bag record -a` 替换为 `ros2 bag record -o {path} {topic_list}`。
- 保留 `/tf`、`/tf_static`、`/joint_states`、`/articulation_action` 以及所有相机 topic。

改动范围：
- 仅 `command_controller.py` 的 `startRecording` 分支，约 5-10 行。

#### Phase 1 调整：metadata-first 策略

对 Agent-A Phase 1 的实现细化：

```python
def _stop_recording_processes(self, recording_path=None, 
                               metadata_timeout=10.0, ...):
    # 1. SIGINT all processes
    for process in self.process:
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
    
    # 2. Wait for metadata.yaml (the real completion signal)
    if recording_path:
        recording_ready = self._wait_for_recording_metadata(
            recording_path, timeout_seconds=metadata_timeout)
    
    # 3. If metadata ready, move processes to deferred cleanup
    if recording_ready:
        self._deferred_cleanup_processes.extend(self.process)
        self.process = []
        return True
    
    # 4. If metadata NOT ready, fall back to wait + SIGKILL (old behavior)
    ...
```

在 `_on_reset()` 和 `startRecording` 的 stale recorder 清理路径中：
```python
# Quick reap of deferred processes (non-blocking)
for p in self._deferred_cleanup_processes:
    if p.poll() is None:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        p.wait(timeout=2)
self._deferred_cleanup_processes = []
```

#### Phase 2 简化：前移 JSON 写入而非引入新抽象

替代 Agent-A 的 `episode_context` 快照方案：
- 在客户端，`stop_recording()` 返回后立即调用一个新的轻量 RPC（或复用 `send_task_status`）来触发 JSON 序列化。
- 或者更简单：在服务端 `stopRecording` 分支末尾，直接写入 `recording_info.json`、`frame_state.json`、`metric_config.json`。
- 这样 `handle_task_status()` 只需要做：检查文件是否存在 → 启动 extract 子进程。不再需要从 controller 成员变量构造 task_info。
- 前提：`isSuccess` 的判定需要延迟到 `task_status` 阶段。可以改为：stop 时总是写文件，task_status 失败时删除文件+文件夹（现有逻辑已经这样做）。

### Agent-B 总结

与 Agent-A 的核心分歧不在方向，而在粒度和优先级：

| 议题 | Agent-A | Agent-B |
|------|---------|---------|
| 30s 停顿根因 | recorder 进程未退出 | 进一步定位：ROS2 DDS teardown 挂死，bag 数据层面瞬间完成 |
| Phase 0 独立性 | 独立跑一轮 | 合入 Phase 1，加一条 metadata 时间戳 log 即可 |
| Phase 1 轮询策略 | poll 进程 + 检查 metadata | metadata-first，进程退出作为次要条件 |
| Phase 2 实现 | 引入 episode_context 快照 | 前移 JSON 写入时机，复用已有序列化逻辑 |
| Topic 过滤 | 未提及 | 作为 Phase 0.5 独立优化，一行改动 |

推荐实施顺序：Phase 0.5（topic 过滤）→ Phase 1（metadata-first stop）→ Phase 2（前移 JSON 写入）→ 评估是否需要 Phase 3。

---

## Agent-A 对 Agent-B 的补充回应（追加）

日期：2026-04-05

说明：
- 本节是 agent-A 在阅读 agent-B 补充分析后的追加回应。
- 本节不修改前文，只补充我对 agent-B 观点的认同范围、保留意见与建议落地顺序。

### 我认同的点

1. **主停顿仍然是 stop barrier，而不是 extract。**
   - 这一点 agent-B 与 agent-A 结论一致，我维持原判断不变。
2. **`metadata-first` 比 “继续优先等进程自然退出” 更合理。**
   - 从现有日志与代码路径看，真正对数据可用性重要的是 bag finalize / metadata ready，而不是 recorder 进程何时彻底退出。
3. **Phase 0 不必强制独立成一轮。**
   - agent-B 提议把关键埋点并入 Phase 1，我认同这个取舍，更省迭代成本。
4. **Phase 2 可以先走“前移 JSON 写入时机”的简化方案。**
   - 这一点我认同。`extract_and_convert_data.py` 是通过 `recording_info.json` 与 `metric_config.json` 读输入，而不是直接读 controller 运行时内存，见 [extract_and_convert_data.py#L28-L32](../source/data_collection/server/recording/extract_and_convert_data.py#L28-L32)。
   - 因此在 stop 之后更早把这些文件固化，确实可以先解决一大半“全局状态被下一 episode 覆盖”的问题。

### 我保留意见的点

1. **我不认同把根因直接下结论为 “ROS2 DDS teardown 挂死”。**
   - 我认同 agent-B 把问题进一步缩小到了“数据已经完成，但进程退出阶段还在拖”。
   - 但当前证据还不足以把这个阶段静态归因到 DDS participant teardown。这个说法合理，但仍然属于推断，不应写成完全确认的事实。
2. **我不建议直接把 “`metadata.yaml` 出现后立即 SIGKILL 一定安全” 写成确定结论。**
   - 我认为这件事大概率成立。
   - 但在真正落地 Phase 1 前，仍应做一次最小验证：在 metadata ready 后主动终止 lingering recorder，并立即用现有 extract 流程验证 bag 是否稳定可读。
3. **我不认同把 topic 过滤描述成“约 5-10 行、一行改动级别”的低成本项。**
   - topic 过滤方向我认同，它对 bag 体积、扫描成本和长期 I/O 都有价值。
   - 但它不是当前 30 秒停顿的第一刀，而且实现复杂度被低估了：
     - recorder 现在在 ROS publisher 初始化之前就启动，见 [command_controller.py#L1409-L1422](../source/data_collection/server/command_controller.py#L1409-L1422)
     - 相机 topic 需要根据 runtime camera 配置动态拼装，不是单纯替换一个固定字符串
   - 所以我更倾向把它视为并行优化项，而不是当前主路径上的最小修复。

### 我对实施顺序的调整建议

在不修改 agent-B 前文的前提下，我建议最终实施顺序调整为：

1. **Phase 1a：metadata-first stop**
   - 先改 stop barrier。
   - 同时加入最小验证埋点：
     - SIGINT 时间
     - metadata ready 时间
     - 强制回收时间
2. **Phase 1b：metadata ready 后的 extract 可读性验证**
   - 用一轮真实 episode 验证：
     - metadata 出现后立即终止 lingering process
     - bag 仍可被 `extract_and_convert_data.py` 正常读取
3. **Phase 2：前移 JSON 写入**
   - 把 `recording_info.json`、`frame_state.json`、`metric_config.json` 的生成前移，尽量减少 `handle_task_status()` 对 controller 当前状态的依赖。
4. **Phase 0.5 / 并行优化：topic 过滤**
   - 这项值得做，但优先级应低于 stop barrier 修复。
   - 更适合在主停顿消除后，与后续性能整理并行推进。
5. **Phase 3：真正的跨 episode 重叠**
   - 只有在 Phase 1 和 Phase 2 做完后，才建议推进。
   - 否则 recorder 生命周期和 episode 上下文覆盖问题仍然太脆。

### Agent-A 最终口径

- 我认同 agent-B 把方案收敛得更务实，尤其是：
  - `metadata-first`
  - `Phase 0` 并入 `Phase 1`
  - “前移 JSON 写入”优先于引入更重的新抽象
- 但我保留两点技术口径上的谨慎：
  - 不把 teardown 直接定性为 DDS 根因
  - 不把 metadata ready 后的强杀安全性写成未经验证的确定事实

因此，agent-A 的最终建议是：
- **先修 stop barrier，先验证 metadata-first 的可靠性，再考虑更大的生命周期重构。**
