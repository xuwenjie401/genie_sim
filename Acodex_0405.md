# 开机自启动 运行server & client

## Background

references:
我现在手动启动仿真数据采集的操作步骤是
1) 终端1
cd RealityLab/genie_sim/source/data_collection && conda activate issac
source ros2_env.bashrc
python scripts/data_collector_server.py --enable_physics --enable_curobo --publish_ros --headless

2) 终端2
cd RealityLab/genie_sim/source/data_collection && conda activate issac
python scripts/run_data_collection.py --task_template tasks/diy/meta_task/galbot_meta_pick_place_V1.json --use_recording


由于采集程序运行时间久之后，会因为一些主板/GPU散热之类的硬件问题 导致电脑重启(不要尝试解决这个问题了) (也不用考虑我的程序会运行报错，现在很稳定)。所以我希望有开机自启动的脚本运行 数据采集流程

## NOTES
1) 每次开机时，你需要检查
/home/agxi/RealityLab/genie_sim/source/data_collection/recording_data
下是否有文件，如果有，把该目录下所有文件移动到
/home/agxi/Datasets/galbot_sim/raw/
下面的新建文件夹内部，命名规则如下: autorun_galbot_meta_pick_place_V1_0407_1720
即autorun_(task-name)_month&date_hour&minute (时间为本次运行时的时间)

2) 这个(或可能不止一个?)开机自启动的脚本，在运行时先sleep 60s, 然后启动server，sleep 60s, 启动client.
且注意 我需要在脚本中配置任务文件

3) 这个自启动运行的进程，必须可以被我较为方便地手动关闭

