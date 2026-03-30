# Current Task
## Reference Files
1) data collection and recording pipeline:
source/data_collection/server/command_controller.py   handle_observation, recording ... etc.
2) already collected data
dirs under source/data_collection/recording_data
3) collected data checker and visualization
source/data_collection/scripts/visualize_recording_data.py
4) specific task-config
source/data_collection/tasks/geniesim_2025/place_object_into_box_of_specific_color/galbot/place_cola_can_into_blue_box_galbot_v1.json
and yes, we are only using galbot now

## Problem
for every arm's gripper, we recorded the two revolute-joint values, but actually, we also need a binary value to express gripper-action, i.e. whether the gripper "is closing" or "is opening". we mark this value as "GA"
for example, first we initialize and select a grasp pose (GA==1) --> arm moves (GA==1)--> gripper closing (GA==0) --> grasping object and hold to lift/ move to place pose (GA==0 all the time to hold) --> gripper opening(GA==1) --> arms reset and moves (GA==1).

But now in our source/data_collection pipeline, we didn't record this value yet, just revolute-joints.

## Plan
1) we need to add a new entry in our source/data_collection pipeline to record binary Gripper Action;
2) we need to visualize and check this, too;
3) for the data we already collected, figure out a script to deal with them, calculate this binary-gripper-action from continuous joint-states. (Note that you should consider the whole trajector-gripper_joint-trends to mark frame-joint_action accurately) (tips: i observed that for some data, when arm arrived at place-pose, the gripper opened a little but not completely opened, and then arm moves to reset while gripper openning at the same time, for this situation, GA==1 start at the first "opened a littlel")


