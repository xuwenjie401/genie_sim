# source/data_collection pipeline中加入场景光照的多样性

## 参考
1. 流程代码入口:
source/data_collection/scripts/data_collector_server.py
source/data_collection/scripts/run_data_collection.py

2. 示例场景 以及 主要想改的光源
1) /home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/background/home_b/home_b_galbot.usda
/World/Light/Light_00/DiskLight_02/DiskLight_02
2) /home/agxi/.cache/modelscope/hub/datasets/agibot_world/GenieSimAssets/background/restaurant_00/background_01.usda
/World/Light/RectLight_04

3) 关于改光照条件的形式，我个人想的是只改intensity这个属性，你如果有更好的建议可以提

## Goal
1) 对于一个具体的task (例如 source/data_collection/tasks/diy/single_task/left_place_cola_can_into_box_galbot_v2.json), 我个人猜测你的实现方式可能需要在task-json中加入一个light-prim name的字段，和一个 intensity字段（也可能是min & max），
随后以改动最小为目标，比如我设想的是，每个episode结束后，你直接random intensity

## Note
你在查阅代码以及指定具体plan时，发现了预期外的问题，可以多与我讨论决策

