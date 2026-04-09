# Meta-Task Usage

## 目的

本文说明如何测试 meta-task 数据采集流程，包括：

- meta-task 配置格式
- 如何手动分开启动 server 和 client
- 如何用现有脚本一键启动
- 运行后会在哪些位置看到进度与生成结果

---

## 1. 前置条件

测试前请确认以下环境已经准备好：

- `SIM_ASSETS` 已设置
- `ISAACSIM_HOME` 已设置
- 当前代码已包含 meta-task 支持改动
- 推荐从 `source/data_collection` 目录启动，这样 `saved_task/` 路径最直观

示例：

```bash
cd /home/agxi/RealityLab/genie_sim/source/data_collection
export SIM_ASSETS=/path/to/GenieSimAssets
export ISAACSIM_HOME=/path/to/isaac-sim
```

如果你要录制数据，还需要：

- ROS/bridge 相关环境正常
- server 启动时带 `--publish_ros`
- client 启动时带 `--use_recording`

---

## 2. Meta-Task 配置格式

推荐使用新格式：

```json
{
    "meta_task": true,
    "tasks": [
        {
            "task_template": "source/data_collection/tasks/diy/single_task/left_place_cola_can_into_box_galbot_v1.json",
            "episodes": 100
        },
        {
            "task_template": "source/data_collection/tasks/diy/single_task/left_place_cola_can_into_box_galbot_v2.json",
            "episodes": 200
        }
    ]
}
```

仓库里的参考文件：

```text
source/data_collection/tasks/diy/meta_task/meta_pick_place_V0.json
```

说明：

- `episodes` 会覆盖 child single-task 里的 `recording_setting.num_of_episode`
- child task 自己的 `camera_list`、`fps`、`task_metric`、`robot` reset 配置继续生效
- 一个 meta-task 内不应该切房间级 scene；代码会做 scene/robot 兼容性检查

---

## 3. 推荐方式：手动分开启动 server 和 client

这是最适合调试 meta-task 的方式。

### 终端 1：启动 server

不录制版本：

```bash
cd /home/agxi/RealityLab/genie_sim/source/data_collection
${ISAACSIM_HOME}/python.sh scripts/data_collector_server.py \
  --enable_physics \
  --enable_curobo
```

录制版本：

```bash
cd /home/agxi/RealityLab/genie_sim/source/data_collection
${ISAACSIM_HOME}/python.sh scripts/data_collector_server.py \
  --enable_physics \
  --enable_curobo \
  --publish_ros
```

Headless 版本：

```bash
cd /home/agxi/RealityLab/genie_sim/source/data_collection
${ISAACSIM_HOME}/python.sh scripts/data_collector_server.py \
  --enable_physics \
  --enable_curobo \
  --publish_ros \
  --headless
```

建议：

- server 启动后先等待 10 到 15 秒，再启动 client
- headless 模式建议多等几秒

### 终端 2：启动 client

不录制版本：

```bash
cd /home/agxi/RealityLab/genie_sim/source/data_collection
${ISAACSIM_HOME}/python.sh scripts/run_data_collection.py \
  --task_template tasks/diy/meta_task/meta_pick_place_V0.json
```

录制版本：

```bash
cd /home/agxi/RealityLab/genie_sim/source/data_collection
${ISAACSIM_HOME}/python.sh scripts/run_data_collection.py \
  --task_template tasks/diy/meta_task/meta_pick_place_V0.json \
  --use_recording
```

如果 server 不在默认地址，也可以显式指定：

```bash
cd /home/agxi/RealityLab/genie_sim/source/data_collection
${ISAACSIM_HOME}/python.sh scripts/run_data_collection.py \
  --client_host localhost:50051 \
  --task_template tasks/diy/meta_task/meta_pick_place_V0.json \
  --use_recording
```

---

## 4. 一键方式：使用现有启动脚本

如果你只是想快速验证，也可以直接使用：

```bash
cd /home/agxi/RealityLab/genie_sim/source/data_collection
./scripts/run_data_collection.sh \
  --task tasks/diy/meta_task/meta_pick_place_V0.json
```

录制版本默认开启；如果不想录制：

```bash
cd /home/agxi/RealityLab/genie_sim/source/data_collection
./scripts/run_data_collection.sh \
  --task tasks/diy/meta_task/meta_pick_place_V0.json \
  --no-record
```

Headless 版本：

```bash
cd /home/agxi/RealityLab/genie_sim/source/data_collection
./scripts/run_data_collection.sh \
  --task tasks/diy/meta_task/meta_pick_place_V0.json \
  --headless
```

说明：

- 这个脚本会自动起 server 和 client
- 如果 client 退出，脚本会 cleanup 并停掉 server
- 更适合常规验证，不如手动双终端方式方便定位问题

---

## 5. 运行后会看到什么

### progress json

meta-task 运行时会持续更新：

```text
source/data_collection/tasks/diy/meta_task/meta_pick_place_V0_progress.json
```

其中会记录：

- 每个 child task 请求的 episodes 数
- 实际生成了多少个 `saved_task` json
- 已尝试多少个 episodes
- 成功多少个 episodes
- 失败多少个 episodes
- 哪些 child task 被兼容性检查跳过

### saved_task 目录

如果你是从 `source/data_collection` 目录启动，meta-task 生成结果会放在：

```text
source/data_collection/saved_task/<meta_name>/<order>_<task_file_stem>/
```

例如：

```text
source/data_collection/saved_task/meta_pick_place_V0/00_left_place_cola_can_into_box_galbot_v1/
source/data_collection/saved_task/meta_pick_place_V0/01_left_place_cola_can_into_box_galbot_v2/
```

特点：

- 不会一开始就把所有 child task 一次性全生成
- 只有当前 child task 轮到执行时，才会生成对应目录

### recording_data

如果开启录制，录制数据仍然走现有路径：

```text
source/data_collection/recording_data/
```

---

## 6. 单任务兼容性

当前改动保留了原有 single-task 启动方式。

也就是说下面的命令仍然可用：

```bash
cd /home/agxi/RealityLab/genie_sim/source/data_collection
${ISAACSIM_HOME}/python.sh scripts/run_data_collection.py \
  --task_template tasks/diy/single_task/left_place_apple_into_box_galbot_v1.json \
  --use_recording
```

此时：

- 不会进入 meta-task 分支
- 仍然使用 single-task 自己 JSON 里的 `recording_setting.num_of_episode`

---

## 7. 建议测试顺序

建议按下面顺序测试：

1. 先用手动双终端方式跑一个 single-task，确认 server/client 基本链路正常
2. 再跑一个只有 2 个 child task、每个 1 到 2 个 episodes 的 meta-task
3. 确认 progress json 会持续更新
4. 确认第二个 child task 的 `saved_task` 是在第一个 child task 结束后才出现
5. 最后再开录制和 headless

---

## 8. 最短可用命令

如果你现在就要开始测 meta-task，直接用下面这一组。

终端 1：

```bash
cd /home/agxi/RealityLab/genie_sim/source/data_collection
${ISAACSIM_HOME}/python.sh scripts/data_collector_server.py \
  --enable_physics \
  --enable_curobo \
  --publish_ros
```

终端 2：

```bash
cd /home/agxi/RealityLab/genie_sim/source/data_collection
${ISAACSIM_HOME}/python.sh scripts/run_data_collection.py \
  --task_template tasks/diy/meta_task/meta_pick_place_V0.json \
  --use_recording
```

如果你只是先验证逻辑，不录数据，把两条命令里的：

- server 侧 `--publish_ros` 去掉
- client 侧 `--use_recording` 去掉

就可以了。
