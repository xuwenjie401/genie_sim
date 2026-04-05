# Meta-Task Data Collection Pipeline Upgrade Plan

## 背景

当前 `data_collection` 流程以单个 task template 为输入，先一次性生成该 task 的全部 `saved_task/*.json`，再逐个执行多个 episodes。  
目标是把它扩展为支持一个 meta-task 配置多个 single-task，并为每个 single-task 指定独立的 episode 数。

本文件只覆盖现状分析与实现计划，不包含实际 `run server/client/test` 执行。

---

## 现状分析

### 1. 当前入口只支持单 task template

入口脚本在 `source/data_collection/scripts/run_data_collection.py`。

当前流程是：

1. 读取 `--task_template` 对应 JSON
2. 构造 `TaskGenerator(task_info)`
3. 直接调用 `generate_tasks(...)`
4. 在 `saved_task/<task_name>/` 下生成全部 episode 对应的具体 task json
5. 用 `DataCollectionAgent.run(...)` 顺序执行这些生成后的 task

这意味着：

- 当前没有 meta-task orchestration
- 也没有 per-child-task episode 配置能力
- `saved_task` 是一次性全量预生成

### 2. `saved_task` 的生成时机在 TaskGenerator

`source/data_collection/client/layout/task_generate.py` 中：

- `generate(output_file)` 负责生成单个具体 task
- `generate_tasks(save_path, task_num, task_name)` 会：
  - 清空目标目录
  - 连续生成 `task_num` 个具体 episode task json

这意味着：

- 现有代码天然假设“一个模板对应一个保存目录，一次性生成所有 episode”
- 如果要支持 meta-task 的“按 child task 顺序生成”，不应该直接改成一次性把所有 child 都生成完
- 更合理的改法是在外层 orchestration 控制何时调用 `generate_tasks`

### 3. episode 成功判定点在客户端最稳

`source/data_collection/client/agent/omniagent.py` 的 `DataCollectionAgent.run(...)` 中：

- 每个 generated task file 会被加载并执行
- 执行结束后本地会得到 `success` / `task_success`
- 然后调用 `self.robot.client.send_task_status(...)`

服务端 `source/data_collection/server/command_controller.py` 中：

- 成功时会写 `recording_info.json` 并异步启动 extraction
- 失败时会删掉 recording 目录

这意味着：

- 如果需要“每成功一个 episode 就立即更新统计 JSON”，最稳定、最及时的更新点在客户端 `DataCollectionAgent.run(...)`
- 如果等服务端 extraction 完成才计数，会更慢，而且中断时不满足“运行中不断更新”的要求

### 4. 切换 single-task 时不会重新 init 整个 sim 场景

当前流程中：

- robot/scene 只在启动时通过第一个 generated task 初始化
- 后续 episode 执行时主要通过 `reset()`、重新 `add_object()`、重设 task config 来继续跑

这意味着：

- meta-task 可以在一个 sim session 内顺序切换多个兼容的 single-task
- 但必须保证这些 child task 在房间级场景和机器人配置上兼容
- 否则虽然“逻辑上可以切 task”，实际 scene/robot 初始化参数可能冲突

### 5. 当前没有 meta-task 兼容性校验

仓库中现有 `meta_pick_place_V0.json` 只是一个很轻量的示意格式：

```json
{
    "meta_task": true,
    "tasks": [
        {
            "source/data_collection/tasks/diy/single_task/place_cola_can_into_blue_box_galbot_v1.json": 100,
            "source/data_collection/tasks/diy/single_task/place_cola_can_into_blue_box_galbot_v2.json": 200
        }
    ]
}
```

但当前入口代码没有消费这套结构，也没有校验：

- `scene.scene_id`
- `scene.scene_info_dir`
- `scene.scene_usd`
- `robot.robot_id`
- `robot.robot_cfg`

所以如果直接叠加 meta 运行而不加预检查，任务切换时容易出现不兼容问题。

### 6. shell 启动脚本依赖 task name 做日志目录

`source/data_collection/scripts/run_data_collection.sh` 会尝试从 JSON 中读取 `task` 字段做日志目录命名。

这意味着：

- meta-task 入口仍可沿用当前 shell 方式
- 但 meta-task 自身的日志命名与 child task 的运行身份需要明确约定
- 否则会出现日志目录名和实际 child 执行内容不一致的问题

---

## 已确认的决策

### 1. meta-task 配置结构

采用新结构，不继续沿用旧的 path->episodes dict 风格：

```json
{
    "meta_task": true,
    "tasks": [
        {
            "task_template": "source/data_collection/tasks/diy/single_task/task_a.json",
            "episodes": 100
        },
        {
            "task_template": "source/data_collection/tasks/diy/single_task/task_b.json",
            "episodes": 200
        }
    ]
}
```

原因：

- 结构清晰
- 后续可扩展字段更自然
- 比旧格式更适合做运行时状态映射

### 2. 统计 JSON 存放位置

统计文件放在 meta-task 模板同目录。

例如：

- meta 文件：`source/data_collection/tasks/diy/meta_task/meta_pick_place_V0.json`
- 统计文件：`source/data_collection/tasks/diy/meta_task/meta_pick_place_V0_progress.json`

原因：

- 符合“存储在 meta_task 下”的描述
- 便于直接和配置文件一起查看
- 不依赖运行时临时目录

### 3. 不兼容 child task 的处理策略

采用 `warn and skip`：

- 启动前做兼容性校验
- 不兼容任务记录告警并标记为 skipped
- 继续执行剩余兼容任务

原因：

- 比 fail-fast 更适合批量采集任务
- 比完全不校验更安全

### 4. success 统计口径

采用“客户端 task 执行成功即计数”。

不等待服务端 extraction 完成。

原因：

- 满足“每成功一个 episode 就更新一次”
- 能在运行中持续刷新统计
- 避免异步 extraction 带来的延迟和中断不确定性

---

## 改造范围

### 主要改动文件

- `source/data_collection/scripts/run_data_collection.py`
- `source/data_collection/client/agent/omniagent.py`

### 尽量不动的文件

- `source/data_collection/scripts/data_collector_server.py`
- `source/data_collection/client/layout/task_generate.py`

说明：

- `TaskGenerator` 保持单 task 生成能力不变
- meta-task 的“何时生成、生成哪个 child task”由外层 orchestration 控制
- server 端录制与 extraction 逻辑不需要为这次改造增加复杂度

---

## 实现计划

### 1. 在 run_data_collection.py 增加 meta-task 分流

#### 单任务模式

保持当前行为不变：

1. 读取 single-task template
2. 调用 `TaskGenerator.generate_tasks(...)`
3. 执行 `DataCollectionAgent.run(...)`

#### Meta-task 模式

新增流程：

1. 读取 meta-task json
2. 解析 `tasks` 列表
3. 解析每个 child task 模板路径
4. 读取每个 child task 模板做预检查
5. 生成初始 progress json
6. 锁定本次 meta-run 使用的 scene
7. 按顺序处理每个 child task：
   - 到该 child 时才生成它的 `saved_task`
   - 执行完所有 episodes
   - 更新 progress json
   - 再切到下一个 child task

### 2. 定义 meta-task 路径解析规则

为了兼容不同写法，child `task_template` 路径按以下顺序解析：

1. 如果是绝对路径，直接使用
2. 相对 meta 文件所在目录解析
3. 如果以 `source/data_collection/` 开头，则去掉前缀后按 data_collection 根目录解析
4. 否则按 data_collection 根目录普通相对路径解析

如果全部失败：

- 该 child task 标记为 invalid
- 记录到 progress json
- 不阻塞其他 child task

### 3. 增加 meta-task 兼容性预检查

每个 child task 至少读取以下字段：

- `scene.scene_id`
- `scene.scene_info_dir`
- `scene.scene_usd`
- `robot.robot_id`
- `robot.robot_cfg`

#### 兼容性规则

一个 meta-task 内允许：

- task-related objects 不同
- pick/place candidates 不同
- workspace 采样结果不同
- `recording_setting.num_of_episode` 被外部覆盖

一个 meta-task 内不允许：

- 房间级场景切换
- 机器人类型切换
- scene info 根目录切换

#### scene_usd 锁定策略

由于某些 single-task 里 `scene.scene_usd` 是 list，需要在 meta-run 开始时锁定一个具体场景：

1. 找到第一个兼容 child task
2. 如果它的 `scene_usd` 是列表，只随机选一次具体值
3. 把这个具体 `scene_usd` 作为 `locked_scene_usd`
4. 后续 child task 必须与该值一致或包含该值，否则跳过

这样可以满足：

- 一个 meta-task 不换房间级场景
- 但保留现有模板里 `scene_usd` 为 list 的写法

### 4. 设计 progress json 结构

建议结构如下：

```json
{
    "meta_task_file": "source/data_collection/tasks/diy/meta_task/meta_pick_place_V0.json",
    "status": "running",
    "started_at": "2026-04-05T12:34:56",
    "updated_at": "2026-04-05T12:35:12",
    "locked_scene_usd": "background/home_b/home_b_galbot.usda",
    "totals": {
        "requested_episodes": 300,
        "attempted_episodes": 12,
        "successful_episodes": 9,
        "failed_episodes": 3,
        "skipped_tasks": 1
    },
    "tasks": [
        {
            "task_template": "source/data_collection/tasks/diy/single_task/task_a.json",
            "runtime_task_id": "task_a",
            "requested_episodes": 100,
            "generated_episodes": 10,
            "attempted_episodes": 10,
            "successful_episodes": 8,
            "failed_episodes": 2,
            "status": "running",
            "skip_reason": ""
        },
        {
            "task_template": "source/data_collection/tasks/diy/single_task/task_b.json",
            "runtime_task_id": "task_b",
            "requested_episodes": 200,
            "generated_episodes": 0,
            "attempted_episodes": 0,
            "successful_episodes": 0,
            "failed_episodes": 0,
            "status": "pending",
            "skip_reason": ""
        }
    ]
}
```

#### 状态建议

顶层 `status`：

- `pending`
- `running`
- `completed`
- `completed_with_skips`
- `failed`

每个 child task `status`：

- `pending`
- `running`
- `completed`
- `skipped_invalid`
- `skipped_incompatible`
- `failed_to_generate`

### 5. progress json 的更新时机

progress json 需要在运行中不断更新，并考虑中断安全。

#### 更新时机

1. meta-task 预扫描完成后
2. 某个 child task 开始前
3. 某个 child task 的 `saved_task` 生成完成后
4. 每个 episode 执行完成后
5. 某个 child task 全部完成后
6. 整个 meta-run 结束后

#### 写盘方式

采用原子更新：

1. 先写临时文件
2. 再 `os.replace(tmp, target)`

这样即使中断，也不会留下半截 JSON。

### 6. `saved_task` 生成策略改为按 child task 延迟生成

当前单任务模式是一次性生成全部 episode。  
meta-task 模式下改成：

- 进入 child task A 时，才生成 `saved_task/<meta_name>/<order>_<task_stem>/`
- child task A 全部 episodes 跑完后
- 再生成 child task B 对应的 `saved_task`

#### 目录建议

使用：

```text
saved_task/<meta_name>/<order>_<task_file_stem>/
```

例如：

```text
saved_task/meta_pick_place_V0/00_place_cola_can_into_blue_box_galbot_v1/
saved_task/meta_pick_place_V0/01_left_place_cola_can_into_box_galbot_v2/
```

原因：

- 避免不同 child task 之间目录冲突
- 明确执行顺序
- 不依赖 child JSON 内部 `task` 字段是否和文件名一致

### 7. 扩展 DataCollectionAgent.run

当前 `DataCollectionAgent.run(...)` 只负责执行一个目录下的 generated task files。  
需要最小扩展为支持外层 meta orchestrator 统计。

#### 建议新增能力

给 `run(...)` 增加可选参数：

- `episode_result_callback`
- 或者返回结构化统计结果

#### callback 触发时机

每个 generated episode 执行结束后调用一次，传出：

- `task_file`
- `success`
- `task_folder`
- `task_template`
- `episode_index`

外层 meta orchestrator 收到后：

- 累加当前 child task 统计
- 累加 meta 总统计
- 立刻刷新 progress json

#### 保持不变的行为

- 现有单任务调用方式仍然可用
- server 端 `send_task_status(...)` 继续走原逻辑
- 录制与 extraction 行为不改

### 8. 首个 child task 负责启动时 robot/scene 初始化

当前 `run_data_collection.py` 在开始时会：

1. 读取生成后的第一个 episode json
2. 从中提取 startup robot / scene 配置
3. 初始化 `IsaacSimRpcRobot`

meta 模式下继续沿用这个模型，但改为：

1. 先找到第一个兼容、可运行的 child task
2. 只生成该 child 的 `saved_task`
3. 读取其第一个 generated episode 作为 startup task
4. 初始化 robot / scene
5. 后续 child 不再重新 init 整个 robot，只更新 task reset config 并继续运行

### 9. 失败与跳过策略

#### child task 模板无效

例如：

- 路径不存在
- JSON 解析失败
- 缺字段

处理：

- 记录 `skipped_invalid`
- 写入 `skip_reason`
- 继续后续任务

#### child task 与 locked scene/robot 不兼容

处理：

- 记录 `skipped_incompatible`
- 写入 `skip_reason`
- 继续后续任务

#### child task 生成失败

如果 `TaskGenerator.generate_tasks(...)` 最终没有生成出可执行 episode：

- 标记 `failed_to_generate`
- 继续后续任务

#### 某个 generated episode 执行失败

处理：

- 仅该 episode 计入 `failed_episodes`
- 继续当前 child task 的后续 episodes

### 10. 对 shell 启动脚本的影响

`run_data_collection.sh` 不需要新增参数。  
仍然通过：

```bash
--task <meta_task.json>
```

来启动。

#### 日志目录

shell 脚本当前会读取 JSON 里的 `task` 字段生成日志目录。  
对于 meta-task，建议：

- 若 meta JSON 没有 `task` 字段，则回退到 meta 文件名 stem

这样日志目录仍然稳定。

本次改造可不强制修改 shell 脚本，只要确认 meta 文件名可接受即可。

---

## 风险点

### 1. child task 内部 `task` 字段和文件名不一致

已在现有样例中看到这种情况。

风险：

- 直接用 `task_info["task"]` 做目录或进度标识，可能混淆

解决：

- meta 运行身份统一使用 child 模板文件名 stem
- `task` 字段只作为 task description 的一部分，不作为 meta orchestration 主键

### 2. scene_usd 为 list 的场景随机性

风险：

- 如果每个 child task 各自随机，meta-run 实际会切房间场景

解决：

- 首个 child 锁定具体 `scene_usd`
- 后续 child 只能复用这个锁定值

### 3. success 统计与 extraction 成功不是同一口径

风险：

- progress json 的 `successful_episodes` 代表“任务执行成功”
- 不代表 extraction 已完全完成

解决：

- 明确文档定义 success 口径
- 如未来有需要，再扩展 second metric

---

## 建议测试方案

本次只做实现计划，不实际执行 server/client/test。  
后续实施时建议验证以下内容。

### 纯 Python 单测

对拆出来的 helper 做测试：

- meta schema 解析
- child task 路径解析
- 兼容性校验
- locked scene 选择逻辑
- progress json 原子更新逻辑

### stub/mocked 级验证

对 orchestrator 做轻量模拟：

- 2 个兼容 child task 顺序执行
- 验证 second child 的 `saved_task` 只在 first child 完成后生成
- 验证每个 episode 完成后 progress json 递增

### 手工冒烟

准备一个小 meta-task：

- 2 个兼容 single-task
- 每个 1 到 2 个 episode
- 再混入 1 个不兼容 task

验证：

- 兼容任务能按顺序运行
- 不兼容任务会告警并跳过
- progress json 会持续更新
- 最终统计符合预期

---

## 最终实施清单

1. 在 `run_data_collection.py` 增加 meta-task 解析与 orchestration
2. 增加 child task 路径解析 helper
3. 增加 child task 兼容性校验 helper
4. 增加 progress json 数据结构与原子写盘 helper
5. 在 meta 模式下按 child task 延迟生成 `saved_task`
6. 在 `DataCollectionAgent.run(...)` 中增加 episode 结果回调或结构化返回
7. 在每个 episode 成功/失败后更新 progress json
8. 保持 single-task 原流程兼容
9. 不改 server 主体行为，不做实际 `run-server-client-test`

---

## 结论

本次改造的核心不在于重写 task generator，而在于：

- 在入口增加 meta-task orchestration
- 把 `saved_task` 生成粒度从“整个 meta 一次性全量生成”改为“按 child task 顺序延迟生成”
- 把 episode 级成功统计挂在客户端执行完成时更新
- 对 child task 的 scene/robot 兼容性做预检查并支持 warn-and-skip

这样可以在尽量少改动现有采集链路的前提下，实现 meta-task 批量采集能力。
