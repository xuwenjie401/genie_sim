# Autorun Data Collection Usage

本文说明如何使用 `source/data_collection/scripts/autorun/` 下的自动采集脚本，包括：

- 手动启动一轮采集
- 手动停止当前采集
- 注册并启用 `systemd` user service
- 取消注册该 service

当前默认任务配置位于：

```text
source/data_collection/scripts/autorun/autorun_config.sh
```

其中默认任务为：

```text
tasks/diy/meta_task/galbot_meta_pick_place_V1.json
```

## 1. 手动启动一轮采集

直接执行：

```bash
cd /home/agxi/RealityLab/genie_sim
./source/data_collection/scripts/autorun/run_autorun_data_collection.sh
```

也可以用绝对路径执行：

```bash
/home/agxi/RealityLab/genie_sim/source/data_collection/scripts/autorun/run_autorun_data_collection.sh
```

脚本会自动完成以下动作：

1. 检查 `source/data_collection/recording_data/` 是否非空。
2. 如果非空，把目录内现有内容移动到：

```text
/home/agxi/Datasets/galbot_sim/raw/autorun_<task-name>_<mmdd>_<HHMM>/
```

3. `sleep 60s`
4. 启动 server
5. 再 `sleep 60s`
6. 启动 client

脚本内部会自动加载这些环境：

```bash
source /home/agxi/miniconda3/etc/profile.d/conda.sh
conda activate issac
source /home/agxi/RealityLab/genie_sim/source/data_collection/assets.bashrc
source /home/agxi/RealityLab/genie_sim/source/data_collection/ros2_env.bashrc
```

## 2. 手动停止当前采集

如果你是前台直接运行 `run_autorun_data_collection.sh`，可以直接按：

```bash
Ctrl+C
```

也可以从另一个终端执行停止脚本：

```bash
/home/agxi/RealityLab/genie_sim/source/data_collection/scripts/autorun/stop_autorun_data_collection.sh
```

停止逻辑是：

1. 优先停止 client
2. 等 client 通过现有 RPC 退出逻辑通知 server
3. 再等待 server 退出
4. 只有超时后才升级为更强的终止信号

## 3. 查看日志

每次运行的日志目录位于：

```text
/home/agxi/RealityLab/genie_sim/source/data_collection/logs/autorun/<timestamp>/
```

其中常见文件包括：

- `autorun.log`
- `server.log`
- `client.log`

最新一次运行的日志软链接位于：

```text
/home/agxi/RealityLab/genie_sim/source/data_collection/logs/autorun/latest
```

## 4. 修改任务配置

如果你要切换任务文件，编辑：

```text
/home/agxi/RealityLab/genie_sim/source/data_collection/scripts/autorun/autorun_config.sh
```

重点字段：

```bash
TASK_TEMPLATE="tasks/diy/meta_task/galbot_meta_pick_place_V1.json"
PRE_SERVER_SLEEP_SECONDS=60
PRE_CLIENT_SLEEP_SECONDS=60
```

修改这个配置文件后，不需要重新安装脚本。

## 5. 注册并启用 systemd user service

service 文件模板位于：

```text
/home/agxi/RealityLab/genie_sim/source/data_collection/scripts/autorun/genie-sim-autorun.service
```

安装命令：

```bash
cd /home/agxi/RealityLab/genie_sim/source/data_collection/scripts/autorun
./install_user_service.sh
```

这会做以下事情：

1. 将 `genie-sim-autorun.service` 复制到：

```text
~/.config/systemd/user/genie-sim-autorun.service
```

2. 执行：

```bash
systemctl --user daemon-reload
systemctl --user enable genie-sim-autorun.service
```

如果你想安装后立刻启动：

```bash
cd /home/agxi/RealityLab/genie_sim/source/data_collection/scripts/autorun
./install_user_service.sh --now
```

如果你希望它在“开机后即使没有登录也自动启动”，还需要执行：

```bash
sudo loginctl enable-linger agxi
```

启用后，日常控制命令如下：

```bash
systemctl --user start genie-sim-autorun.service
systemctl --user stop genie-sim-autorun.service
systemctl --user restart genie-sim-autorun.service
systemctl --user status genie-sim-autorun.service
journalctl --user -u genie-sim-autorun.service -f
```

## 6. 取消注册 service

如果你不再需要开机自启动，可以执行：

```bash
systemctl --user stop genie-sim-autorun.service
systemctl --user disable genie-sim-autorun.service
rm -f ~/.config/systemd/user/genie-sim-autorun.service
systemctl --user daemon-reload
systemctl --user reset-failed
```

如果你还想取消“用户在未登录时也能运行 user service”的设置，再执行：

```bash
sudo loginctl disable-linger agxi
```

注意：

- `disable` 只是取消开机自动启动，不会自动删除 service 文件。
- `stop` 是停止当前正在运行的服务。
- `rm ~/.config/systemd/user/genie-sim-autorun.service` 才是把已注册的 service 文件删除。

## 7. 常用操作总结

手动启动一轮采集：

```bash
/home/agxi/RealityLab/genie_sim/source/data_collection/scripts/autorun/run_autorun_data_collection.sh
```

手动停止当前采集：

```bash
/home/agxi/RealityLab/genie_sim/source/data_collection/scripts/autorun/stop_autorun_data_collection.sh
```

注册 service：

```bash
cd /home/agxi/RealityLab/genie_sim/source/data_collection/scripts/autorun
./install_user_service.sh
sudo loginctl enable-linger agxi
```

启动 service：

```bash
systemctl --user start genie-sim-autorun.service
```

停止 service：

```bash
systemctl --user stop genie-sim-autorun.service
```

取消注册 service：

```bash
systemctl --user stop genie-sim-autorun.service
systemctl --user disable genie-sim-autorun.service
rm -f ~/.config/systemd/user/genie-sim-autorun.service
systemctl --user daemon-reload
systemctl --user reset-failed
sudo loginctl disable-linger agxi
```
