# Data Collection gRPC 流水线分析

日期：2026-03-26

范围：
- 本文基于当前工作区中 `source/data_collection` 的代码状态。
- 本文分析的问题是：当前 server-client gRPC 流水线是否正在拖慢数据采集效率；如果目标只剩下 automated collection，是否更适合改成单进程实现。
- 本文不是泛化的软件架构评审，而是聚焦在控制流、延迟结构、吞吐约束和迁移取舍上。

问题：
- 对于 `source/data_collection`，当前 server-client gRPC 流水线是否已经明显损害了效率，以至于应该被移除？
- 如果不再需要 simulation 中的 teleoperation，是否应该改成单进程设计？

## 执行摘要

短结论：

- 如果目标范围已经收敛为 automated collection only，那么逐步放弃当前 gRPC 拆分是合理方向。
- 但核心问题并不是“localhost gRPC 天生太慢”。
- 更大的问题是：当前架构把很多本来普通的控制和状态读取，变成了被 Isaac Sim 主循环门控的同步往返请求。
- 实际上，这个系统在逻辑上是两服务，但服务端真正执行仍然通过单个共享命令槽串行化，因此传输边界带来了复杂度，却没有换来多少真实的执行并行度。
- 对你当前缩窄后的使用场景，单进程设计大概率是更好的形态；但真正的收益主要来自减少 round trip 和批量化状态访问，而不只是删掉 protobuf 定义。

我的建议：

1. 不要继续把当前 gRPC 拆分作为 automated-only collection 的默认路径。
2. 也不要把 command/controller 这层抽象整个扔掉。
3. 先把 gRPC 传输层替换成进程内 backend。
4. 然后再处理高频状态轮询和元数据 RPC，因为当前架构的延迟主要在这里被放大。

## 高层结论

当前系统里真正重要的边界有三层：

1. 启动边界。
2. 传输边界。
3. 仿真线程边界。

对于 automated-only 的场景，前两层可以去掉。

但第三层不能无视。Isaac Sim 相关工作仍然需要在正确的运行上下文里发生。所以最合理的目标不是“所有代码随便直接函数调用”，而是“保留一个本地 command/controller 接口，但不再为每一次控制或状态查询支付网络、protobuf 和多进程成本”。

## 当前实际存在的东西

### 1. 逻辑架构

当前运行时是一个 client-server 系统。

- 采集应用在 [run_data_collection.py](/home/agxi/RealityLab/genie_sim/source/data_collection/scripts/run_data_collection.py#L68) 中创建了一个基于 RPC 的 robot。
- 这个 robot 使用 `RpcClient`，它在 [client.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/client.py#L83) 中打开 localhost gRPC channel。
- 服务端在 [grpc_server.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/grpc_server.py#L778) 中启动 gRPC service。
- gRPC server 背后真正持有 Isaac Sim 交互状态的是 `CommandController`，定义在 [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L63)。

也就是说，当前控制路径是：

`DataCollectionAgent -> IsaacSimRpcRobot -> RpcClient -> gRPC -> GrpcServer -> CommandController -> UIBuilder / Isaac Sim / cuRobo`

### 2. 实际启动模型

文档和脚本展示了两种略有不同的启动视图。

README 里描述的是手工双终端工作流：

- 一个终端启动 `data_collector_server.py`。
- 另一个终端启动 `run_data_collection.py`。

参考：
- [README.md](/home/agxi/RealityLab/genie_sim/source/data_collection/README.md#L132)

但当前容器化启动脚本其实已经替你把两个程序一起拉起来了：

- `run_data_collection.sh` 在 [run_data_collection.sh](/home/agxi/RealityLab/genie_sim/source/data_collection/scripts/run_data_collection.sh#L184) 中以 `data_collection_entrypoint.sh` 作为容器 entrypoint。
- 这个 entrypoint 会在 [data_collection_entrypoint.sh](/home/agxi/RealityLab/genie_sim/source/data_collection/scripts/data_collection_entrypoint.sh#L261) 后台启动 `data_collector_server.py`。
- 然后又会在 [data_collection_entrypoint.sh](/home/agxi/RealityLab/genie_sim/source/data_collection/scripts/data_collection_entrypoint.sh#L290) 后台启动 `run_data_collection.py`。

这个区别很重要：

- 系统在逻辑上仍然是双进程、client-server。
- 但在正常 automated workflow 下，它已经更像一个被打包在一起的协调式 bundle，而不是两个真正独立管理的工具。

这会削弱“为了 automated path 必须保留 gRPC 硬边界”的实际价值。

## 端到端命令流

### 1. 客户端发出一次 motion request

客户端发送的是高层 move，不是逐帧 joint streaming。

例如：

- `IsaacSimRpcRobot.move(...)` 最终会在 [omni_robot.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/omni_robot.py#L291) 调用 `self.client.moveto(...)`。
- `RpcClient.moveto(...)` 会在 [client.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/client.py#L160) 里构造 `LinearMoveReq` 并执行同步 RPC。

这一点很关键，因为它说明 gRPC 这一层并不是被用来承载高频伺服控制流。一次 RPC 往往代表的是一大段后续工作。

### 2. 服务端拿到请求之后实际做了什么

gRPC service handler 本身并不直接执行 simulation logic。

以 linear move 为例：

- `armService.linear_move(...)` 会把请求转发到 [grpc_server.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/grpc_server.py#L118) 中的 `blocking_start_server(...)`。
- 真正的交接发生在 [grpc_server.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/grpc_server.py#L137)。

进入 `CommandController` 之后：

- 请求被写入共享字段 `self.data` 和 `self.Command`，位置在 [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L1740)。
- 调用线程随后在条件变量上等待结果返回，位置在 [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L1743)。

真正执行请求的地方，是仿真主循环：

- `on_physics_step()` 每个 step 被调用，定义在 [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L643)。
- 它会进一步调用 [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L665) 中的 `on_command_step()`。
- `on_command_step()` 再根据 `self.Command` 分发到具体 handler，位置在 [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L1660)。
- 命令完成后，再通过 [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L1721) 唤醒等待线程。

所以真实路径其实是：

1. Client 发起 RPC。
2. gRPC 工作线程收到请求。
3. 请求被拷贝进 controller 的共享状态。
4. 调用方阻塞等待。
5. Isaac Sim 主循环走到下一次 `on_physics_step()`。
6. `on_command_step()` 分发并执行该命令。
7. 结果被写回。
8. 等待中的 RPC 线程返回响应。

### 3. physics-step 门控带来固定延迟

服务端默认 `physics_step = 30`，见 [data_collector_server.py](/home/agxi/RealityLab/genie_sim/source/data_collection/scripts/data_collector_server.py#L42)，之后在 [data_collector_server.py](/home/agxi/RealityLab/genie_sim/source/data_collection/scripts/data_collector_server.py#L73) 变成 `physics_dt = 1 / args.physics_step`。

可推导出：

- 一个同步命令在 simulation loop 真正处理前，不可能完成。
- 在默认 30 Hz physics rate 下，每个命令都会有一个不可消除的固定延迟分量，量级大约就是一个 physics tick，也就是约 `33 ms`，这还没算 protobuf 序列化、gRPC 调度和真实仿真工作。

这个固定延迟对长时间 motion plan 影响不大，但对很多小粒度 control/state request 就会变得明显。

## 当前系统比表面看起来更串行

### 1. gRPC 层看起来支持并发

服务端是通过 `ThreadPoolExecutor(max_workers=10)` 创建的，见 [grpc_server.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/grpc_server.py#L788)。

这看起来像是 transport layer 可以接受多个并发请求。

### 2. command controller 实际上把执行串行化了

但 controller 内部只维护一个活跃命令的共享字段：

- `self.data` 在 [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L76)
- `self.Command` 在 [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L77)
- `self.data_to_send` 在 [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L78)

这不是一个真正的多命令执行队列，更像是“单个在途命令槽 + wait/notify 机制”。

实际含义是：

- 即使 gRPC 层能并发接收请求，真正有价值的执行路径仍然被单个 controller 状态机串起来了。
- 所以这个架构在 RPC 层为并发付出了复杂度，但 simulation command path 仍然基本是单线程串行行为。

这不代表代码一定错误，但它确实说明当前 gRPC 拆分并没有按其复杂度换来等比例的真实吞吐收益。

## 开销真正来自哪里

### 1. 纯 gRPC 和 protobuf 开销

这部分确实存在，但不是主导问题。

具体包括：

- protobuf 对象构造
- 序列化与反序列化
- 本地 socket 传输
- gRPC worker 调度

客户端这边还存在一些重复 stub 构造：

- `set_frame_state()` 在 [client.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/client.py#L124) 创建新 stub
- `moveto()` 在 [client.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/client.py#L160) 创建新 stub
- `get_object_pose()` 在 [client.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/client.py#L278) 创建新 stub
- `get_part_dof_joint()` 在 [client.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/client.py#L285) 创建新 stub
- `get_ee_pose()` 在 [client.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/client.py#L379) 创建新 stub

这当然不理想，但相较于每条命令都要等待 simulation loop 的同步结构，它更像是二阶问题。

### 2. 线程切换和阻塞同步

这部分比纯 gRPC 开销更重要。

每条命令现在都要经历：

- 进入 gRPC worker thread
- 把数据写入 controller 共享状态
- 在 condition variable 上阻塞等待
- 等 sim loop 写回结果后恢复执行

这会引入：

- 线程调度开销
- context switch
- lock / condition 开销
- 结果回传开销

对一条很长的 motion 来说，这些代价可以忽略；但对很多小粒度状态读取来说，会不断累积。

### 3. physics-loop 服务延迟

这大概率是同步命令最核心的固定延迟来源。

因为 `on_command_step()` 是从 `on_physics_step()` 里调用的：

- 命令到达后不会立刻被处理
- 命令要等待 simulation loop
- 所以命令延迟会直接受到 physics rate、渲染负载、ROS publishing 负载，以及服务端当前 frame 内其他重任务的影响

这也是为什么只讨论 transport 不够。当前系统不是“RPC 进来就立即改 sim state”，而是“RPC 进来之后进入 wait-until-next-sim-step pipeline”。

### 4. automated workflow 很碎很 chatty

这正是当前设计变得昂贵的地方。

自动采集 agent 会围绕每个 action 发很多 RPC。

运动前：

- `set_frame_state()` 在 [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L743) 被调用
- `remove_objs_from_obstacle()` 可能在 [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L763) 被调用

运动本身：

- `move_pose()` 在 [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L801) 被调用

运动后：

- 再次 `set_frame_state()`，位置在 [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L821)
- gripper action 可能会通过 [omni_robot.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/omni_robot.py#L171) 触发 `set_gripper_state()` 和 `detach_obj()`
- 又一次 `set_frame_state()`，位置在 [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L838)

状态刷新：

- end-effector pose 会通过 [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L851) 的 `get_ee_pose()` 获取
- 接着 `update_objects()` 会在 [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L222) 刷新 object state
- 每个 object 刷新都要通过 [omni_robot.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/omni_robot.py#L337) 走一次 `get_object_pose()`
- articulated part 还要在 [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L238) 再调 `get_part_dof_joint()`

更多元数据：

- 再一次 `set_frame_state()`，位置在 [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L855)
- `attach_obj()` 可能在 [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L884) 被调用
- 最后再一次 `set_frame_state()`，位置在 [omniagent.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/agent/omniagent.py#L888)

也就是说，一个逻辑上的单一 action，实际上很容易包含：

- 1 次高层 motion RPC
- 若干次元数据 RPC
- 若干次状态查询 RPC
- 外加每个 object 一次 object-pose RPC
- 外加每个 articulated part 一次关节状态 RPC

这就是我认为 automated-only collection 最有力的反对当前拆分的理由。

### 5. recording 和 ROS publishing

gRPC 拆分并不是唯一运行时成本。

在启用 recording 时：

- entrypoint 会为服务端增加 `--publish_ros`，见 [data_collection_entrypoint.sh](/home/agxi/RealityLab/genie_sim/source/data_collection/scripts/data_collection_entrypoint.sh#L234)
- `on_physics_step()` 会在 [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L666) tick ROS publishers

所以如果你观察到 recording 时吞吐变差，其中一部分下降本来就会来自：

- ROS bridge 工作
- 传感器发布
- 渲染
- recording 序列化

而不只是 command transport layer。

### 6. motion planning 和 simulation 本身

最重的工作依然在服务端：

- Isaac Sim stepping 在 [data_collector_server.py](/home/agxi/RealityLab/genie_sim/source/data_collection/scripts/data_collector_server.py#L104)
- world rendering 在 [data_collector_server.py](/home/agxi/RealityLab/genie_sim/source/data_collection/scripts/data_collector_server.py#L110)
- cuRobo stepping 在 [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L648) 的 `on_physics_step()` 内部发生

所以把所有运行时成本都归因于 gRPC 是不准确的。

对于长时间 motion、collision-aware planning、scene loading 或 recording 很重的 episode，主要成本仍然在别处。

## 为什么 gRPC 不是所有情况下的主瓶颈

当前系统仍然有一些场景下是“够用”的：

- Motion RPC 是高层粒度，不是逐帧 servo command。
- 客户端正常路径不会按控制频率持续流式传大轨迹。
- 服务端大部分时间依旧花在 simulation、planning、rendering 和 recording 上。
- 当请求少而且粒度大时，localhost gRPC 通常是足够快的。

所以如果工作负载是这样：

- 一次初始化
- 一次大 motion
- 一次 recording
- 一次退出

那 gRPC 层大概率不是第一优先级优化对象。

但 automated collection 的真实模式不是这样。它的调用足够碎，以至于固定的每请求成本开始变得明显。

## 为什么单进程更适合 automated-only collection

如果 teleoperation、远程控制、多客户端访问都不再是当前目标，那么单进程方案的合理性会明显增强。

优点：

- 启动和失败模型更简单
- 调试更直接，stack trace 留在本地
- 普通内部控制不再跨 protobuf / RPC 边界
- 元数据和状态访问延迟更低
- 更容易做状态批量访问
- 更容易做全链路 profiling

当前启动脚本实际上已经把两部分在同一个 container 里打包协同运行了，这进一步削弱了为了 automated path 保持双进程的现实必要性。

## 关键 caveat：单进程本身还不够

这是最重要的细节。

如果你只是简单地做下面这些事：

- 保持同样的 client 逻辑
- 保持同样的 blocking command handoff
- 保持同样的 chatty call pattern
- 只是把 gRPC 换成本地 Python 调用

那么你确实会去掉一部分开销，但不会消除全部结构性延迟。

你仍然会保留：

- 每命令同步边界
- simulation-step 门控
- 重复状态读取
- 重复元数据写入

所以更准确的收益优先级应该是：

1. 减少 round trip。
2. 批量化状态访问。
3. 再去掉 transport/process overhead。

而不是：

1. 只要删掉 gRPC，问题就自然解决。

## 推荐目标架构

### 1. 保留 command/controller 抽象

我倾向于继续保留一个 central simulation-side controller，因为当前代码已经很明确地说明：进入 Isaac Sim 的交互最好有一个受控边界。

`CommandController` 这个概念性接缝本身是对的。问题在于围绕它的 transport 和调用粒度，而不是 controller 这个抽象本身。

### 2. 把 `RpcClient` 替换为与传输无关的 backend

最干净的迁移路径，是把 client robot 这边改造成 backend 接口。

例如：

- `GrpcBackend`：保留当前行为
- `LocalBackend`：直接做进程内 controller 调用

这样做的好处是：

- 初期可以保留高层 client/agent 逻辑不动
- 可以做 A/B 性能对比，而不是一次性全改
- 如果未来真的需要远程工作流，还能保留 gRPC 作为可选路径

### 3. 增加 batch state API

这大概率是 automated collection 收益最高的架构改进。

当前 `update_objects()` 模式之所以昂贵，是因为它按状态项一个个查询。

更好的方式是增加类似：

- `get_world_snapshot()`
- `get_object_poses(object_ids)`
- `get_runtime_state(gripper_pose, object_poses, articulated_joint_states, attachments)`

这样每个 action 的状态刷新可以从很多次同步往返，变成一次本地调用。

### 4. 减少 `set_frame_state()` 的写入频率

当前 agent 在每个 action 周围会多次写 frame metadata。

其中一部分可能确实是 recording 语义需要，但这件事应该被当作 data design 问题重新审视，而不是只当作 control design。

可考虑的改法：

- 只在真正的语义状态转移点写入
- 先在本地积累 stage metadata，再一次性 flush
- 把轻量级内存态更新和持久化事件日志拆开

### 5. 保持 high-level motion command 的粗粒度

这一点其实是当前系统设计里表现好的部分。

`move_pose()` 在 [omni_robot.py](/home/agxi/RealityLab/genie_sim/source/data_collection/client/robot/omni_robot.py#L291) 中转发的是高层 motion request，而不是从 client 侧手工逐步推 joint。

这一设计在 local-backend 版本中也应该保留。

## 迁移选项

### 选项 A：本地传输适配器，控制模式保持不变

描述：

- 保留 `DataCollectionAgent`
- 保留 `IsaacSimRpcRobot` 风格接口
- 用 `LocalClient` 替换 `RpcClient`
- `LocalClient` 通过进程内 controller adapter 调用

优点：

- 迁移风险最低
- 最容易与当前流水线做 A/B 对比
- 直接去掉 protobuf 和多进程边界

缺点：

- 仍然保留当前相当一部分 call chatter
- 很可能仍保留同步边界
- 会有提速，但未必是最大收益

预期效果：

- 是一个好的第一步
- 很可能值得做
- 但不是收益上限

### 选项 B：单进程 + 批量状态快照

描述：

- 先完成选项 A
- 再增加 batch state refresh API
- 把 action 后的 object-by-object polling 替换掉

优点：

- 更接近真实吞吐提升
- 能直接削减 automated path 里最明显的低效段
- 架构仍然清晰

缺点：

- 改动量中等
- 需要调整 agent 侧假设

预期效果：

- 这是我认为收益最高、风险中等的方案

### 选项 C：与 simulation loop 深度耦合的进程内 scheduler

描述：

- 让 automated agent 作为本地任务更紧密地挂到 sim loop 上
- 在可能的地方避免 per-command wait/notify
- 把 action execution 改造成 simulation-driven state machine 或 coroutine

优点：

- 对 automated-only collection 来说，这是长期最优形态
- transport 开销最低
- sim-thread ownership 最清晰

缺点：

- 重写量最大
- 如果做得太快，最容易引入回归

预期效果：

- 收益上限最高
- 但除非你现在就想做较大级别重构，否则不建议作为第一步

## 什么情况下保留 gRPC 仍然有意义

当前拆分仍然有一些合理保留理由：

- 未来 teleoperation
- 远程 planner 或 GUI client
- 多客户端访问
- 与 Isaac Sim crash 隔离进程
- 外部工具集成

如果这些是未来目标，那么更合理的折中方案是：

- 把 gRPC 保留成一个可选 adapter
- 不要继续把它作为 automated collection 的唯一默认路径

## 如果是我来做，我会怎么推进

如果我是按你当前目标去优化这个代码库，我会按这个顺序做：

1. 引入一个与现有 client 接口一致的 local backend。
2. 让 automated collection 默认切到 local backend。
3. 把 gRPC 保留为调试或未来 remote use 的可选开关。
4. 增加 batched world-state snapshot API。
5. 替换 `update_objects()` 中重复的 per-object state polling。
6. 重新评估到底哪些 `set_frame_state()` 写入是真正必要的。
7. 重新 profile，再决定要不要进一步做更深的 scheduler 重构。

这个推进顺序可以在不承担一次性大改风险的前提下，拿到大部分可预期收益。

## 重构前后的测量计划

当前代码里其实已经有 timing infrastructure，在 `CommandController` 里：

- timing storage 在 [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L146)
- timing context 在 [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L153)
- timing report 在 [command_controller.py](/home/agxi/RealityLab/genie_sim/source/data_collection/server/command_controller.py#L193)

这已经是个不错的起点，但还不足以完整回答架构问题。

我会补充测这些指标：

- end-to-end episode runtime
- per-stage runtime
- per-action runtime
- 每个 action 内的 RPC-style call 数量
- 从 request 发出到结果返回的平均值和 p95 延迟
- `update_objects()` 花了多少时间
- object pose refresh 花了多少时间
- `on_physics_step()` 花了多少时间
- recording 开启时 ROS publish path 花了多少时间
- motion planning 时间 vs 等待服务时间

最小 benchmark 矩阵：

- 当前 gRPC split，recording off
- 当前 gRPC split，recording on
- local backend，recording off
- local backend，recording on
- local backend + batched state snapshot，recording off
- local backend + batched state snapshot，recording on

这个矩阵会告诉你：

- 主痛点到底是不是 transport
- 主痛点是不是 state polling
- 主痛点是不是 recording / rendering
- 或者到底是这几者的组合

## 单进程重构的风险

这类重构有真实风险，应该明确写出来。

### 1. Isaac Sim 线程安全假设

当前 controller 的存在，本身就说明 sim 操作不是普通的纯 Python 业务逻辑。一个粗糙的单进程重构，很容易把 sim 调用放到错误执行上下文里。

### 2. 隐性耦合

当前拆分虽然笨重，但它客观上强迫了一定 API discipline。直接改单进程后，如果设计不好，很容易变成跨层随意互调。

### 3. 丢失可选 remote workflow

如果把 gRPC 彻底删掉，以后再想恢复 teleoperation 或 remote tool 接入，成本会更高。

### 4. 重构范围失控

如果你把 transport removal、planner 调整、recording 调整和 controller 重写全部揉在一起，迁移过程会更难验证。

所以第一步最合理的做法，还是 transport-agnostic backend interface。

## 最终建议

如果场景已经缩窄为 automated collection only，那么答案是：是的，当前 server-client gRPC 流水线很可能已经把效率拉低到不再适合作为默认架构。

但更准确的表述应该是：

- 真正的主问题不是 raw gRPC 本身
- 真正的主问题是：跨传输边界的同步 request/response 控制，加上 simulation-step 门控，再叠加 chatty 的状态轮询

因此：

- 改成单进程是合理的
- 保留 controller 边界也是合理的
- 最值得先做的是 local backend + batched state access
- 如果只是删 gRPC 而不减少 round trip，会有收益，但不会彻底解决问题

如果目标是以可控风险拿到最大实际收益，我会选择：

`单进程 automated path + 可选 gRPC adapter + batched state snapshot`

这是我认为最符合当前代码状态和你当前需求的方向。

## 附录：为什么当前设计会让人感觉“比它本来应该的更慢”

如果只用最短的自然语言概括这份分析，就是：

- 正常 automated path 本来就已经在同一个 container 里打包一起跑了，所以进程拆分没有换来多少实际运维灵活性。
- 每条命令都要先过 RPC，再等 sim loop，所以请求并不是到达后立刻执行。
- gRPC server 虽然能接多个请求，但真正有效执行仍然汇聚到一个 controller command slot。
- automated agent 的小粒度 control/state request 足够多，所以固定延迟开始显著。
- 最值得优化的不是“更快的网络”，而是“更少的同步边界”。
