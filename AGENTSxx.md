## Long-Term Goal
genie_sim is a simulation data_collection and evaluation workflow based on isaac-sim for VLA training and testing. But in a long future, we will only focus on data-collection part, to achieve the goal ----

Use galbot's relevant asset to effciently automatically collect navigation->manipulation->navigation->manipulation task-template data with various high-quality object&background assets.
A very specific task would be like "go to the drawer-->open the drawer(could with torso height change)-->pick the medicine bottle-->navigate back-->place the medicine bottle into box/plate/hands..."


## Progress
1. We already have plenty of high-quality object&background assets, and basic workflow for fixed-robot pick&place on a table, refer to codes at "source/data_collection/...", this workflow works well for agibot-G1/G2.
But 1)navigate; 2)open the drawer(could with torso height change); 3) effcience(success rate, multi world parallel, collected-data format converted from ros2 bags)  are not achieved yet.
2. Galbot now stuck at gripper's grasp and performance, and grasp-pose humanlike-filter.Now We're focusing on fix previous server-pipeline's gripper issues.


## Done in Detail
1. free_worker_gripper.py as a unit_test (rename from previous free_worker.py since the job is done), has achieved acceptable gripper performance and keyboard-carb-interactive control mode


## TODO

### Current Task
1. now free_worker_vertical.py is just a copy of free_worker_cube.py, but it's created to do this task: 
In previous pipeline, robot lock its waist/leg joints [for galbot, specifically as leg_joint1/2/3/4/5 (but from now on, when we talk about waist/leg dofs, we only care about leg_joint1, leg_joint2 and leg_joint3, because leg_joint4/5 are for yaw/roll, this task cares only vertical change, i.e. pitch)], and only plans(motion gen) arm joints to do pick and place.

now, the task-name "vertical" stands for: if some targets, can't be reached because of height rather than horizontal/x-y distance, we add a new stage, to change the torso's height by controlling leg_joint1/2/3, and then apply the previous pipeline.

and here are some important rules/tips you should remember:
1) this task may need to change some config-files or read extra configs from new file, you cannot modify the original config-files, you can only create-new-files or create some configs in code, and comment-NOTE in codes(notice me) when you do this, or you could ask me to modify or do-config-files-change;
2) to change the torso's height, you cannot control leg_joint1/2/3 without constraints, the main principle is: torso needs to be approximately vertical, excessive leaning is very bad for real-robot's leg joints. 

[!! To achieve this, a simple approach is to use the following approximate constraint:
joint1 + joint3 ≈ joint2
This helps keep the torso approximately vertical.

If you want to use other methods—for example, treating the full leg joint chain as an arm in motion_gen and solving IK with torch_base_link as the end effector plus an additional rotation constraint—that is also acceptable, as long as you have a solid reason for doing so.

Also, if you choose to use the simple formula above, you may want to write another small test script to accurately determine the mathematical relationship between torso height and leg_joint1/2/3
!!]

3) In the current code, the configuration defines a default robot posture, including the leg joints. This posture should be treated as the default reference state (e.g. for reset and height adjustment decisions priority), and no torso height increase beyond this default state is needed.

An additional idea for your plan:
You may use a sphere radius as a simple way to check whether a target's height is reachable.
However, if you prefer to use IK or other methods, I am also happy with a different solution.

4) This free_worker_vertical.py is a unit test, so "obstacle1-bottle" and "obstacle2-box" in the current code can remain as collision obstacles. However, for the grasp target, you could use a collision-free cube so that I can drag it around easily in Isaac Sim just like free_worker_cube.py did

## Long-Term Memory

Before doing meaningful work in this repo, first read these memory files:
- `/home/agxi/.codex/memories/genie_sim_architecture.md`
- `/home/agxi/.codex/memories/genie_sim_active_work.md`

Purpose:
- avoid rereading the whole tree on every turn;
- keep the main `source/data_collection` pipeline map stable across sessions;
- preserve the current vertical-height experiment constraints and the fact that some pipeline files already contain in-progress user/Claude edits.

Memory maintenance rules:
- When you finish a task that changes the real understanding of the pipeline, update the relevant memory file in addition to the code.
- Prefer adding concise, durable facts: entrypoints, control flow, invariants, active constraints, risky files, calibration constants, and known integration boundaries.
- Do not dump transient logs, speculative ideas, or large debugging transcripts into memory.
- If the user says a previous agent partially solved something and they are still testing, record the current state carefully and avoid treating the worktree as clean.
