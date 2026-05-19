# SLAM Data Collection

现在希望类比 source/data_collection/ 下原本的 manipulation 数据采集流程，
参考文件:
1) source/data_collection/scripts/data_collector_server.py
2) source/data_collection/scripts/run_data_collection.py
3) source/data_collection/tasks/diy/single_task/left_place_cola_can_into_box_galbot_s3c2v1.json


制作一套 SLAM数据采集流程，具体要求如下
- 目前只针对galbot 机器人资产做适配
- 底盘没有轮子，需要以其他形式控制机器人看似合理的连续移动 （全向底盘）
- 通过键盘输入，实时控制底盘移动，不需要有自动policy
- 录制图像/深度图数据，以及记录 各相机/坐标系的轨迹（位姿序列）真值


## NOTES
- 本机上对应的 conda 环境为 issac

