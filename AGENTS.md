# Current Task
## Reference Files
1) data collection pipeline:
source/data_collection/  omniagent.py, command_controller.py ...
2) task_generate:
source/data_collection/client/layout/task_generate.py
3) specific task file as example:
source/data_collection/tasks/diy/single_task/left_place_cola_can_into_box_galbot_v1.json


## Problem
1) current load_task pipeline, only shuffle objects-layout using task-generate, but the robot_init_pose(with random noise), and init_arm_pose(with random noise) will be calculated and fixed at the beginning.
i want robot's pose and joint's init value, have a same reset module when load new task


## Plan


